import operator
from typing import TypedDict, List, Dict, Annotated, Literal, Optional
from langgraph.graph import StateGraph, START, END
from openai import OpenAI
from pprint import pprint
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import time
from datetime import datetime

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

token = os.getenv("MGA_API_KEY")
client = OpenAI(base_url="https://chat.int.bayer.com/api/v2", api_key=token)
model = 'o4-mini'

llm = ChatOpenAI(
    openai_api_base="https://chat.int.bayer.com/api/v2",
    openai_api_key=token,
    model=model,
    temperature=0.7
)

# Configuration for the workflow
MAX_RETRIES = 3
QUALITY_THRESHOLD = 0.6
MAX_CONTEXT_TOKENS = 2000  # Approximate token limit for context
MAX_ITERATIONS = 20

# ============================================================================
# STATE DEFINITION
# ============================================================================

class BlogWorkflowState(TypedDict):
    """Enhanced state with proper context management"""
    # Core workflow
    topic: str
    plan: List[str]
    pending: List[str]
    drafts: List[Dict[str, str]]
    
    # Context engineering
    current_context: str  # Summarized context for agents
    context_window: List[Dict[str, str]]  # Recent drafts only
    
    # Quality control
    quality_scores: Dict[str, float]  # section -> score
    needs_revision: List[str]  # sections that need rewriting
    
    # Workflow management
    iteration_count: int
    errors: List[str]
    status: str  # "planning", "writing", "validating", "compiling", "complete"
    
    # Final output
    final_post: str

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def count_tokens(text: str) -> int:
    """Approximate token count (rough estimation)"""
    return len(text.split()) * 1.3

def truncate_context(context: str, max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    """Truncate context to fit within token limit"""
    tokens = count_tokens(context)
    if tokens <= max_tokens:
        return context
    
    # Truncate to approximate token count
    words = context.split()
    target_words = int(max_tokens / 1.3)
    truncated = " ".join(words[:target_words])
    return truncated + "\n...[context truncated]..."

def log_event(message: str):
    """Log workflow events with timestamps"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

# ============================================================================
# AGENT 1: PLANNER
# ============================================================================

def planner_node(state: BlogWorkflowState) -> Dict:
    """
    Creates a structured plan for the blog post.
    
    CONTEXT ENGINEERING:
    - Only uses the topic (minimal context)
    - Produces a clear, actionable plan
    """
    log_event("🎯 AGENT: Planner - Creating blog structure")
    
    topic = state['topic']
    
    try:
        # LLM call with minimal, focused context
        prompt = f"""Create a detailed blog post plan for the topic: "{topic}"

Output exactly 5 section titles in this format:
1. [Section Title]
2. [Section Title]
...

Requirements:
- Start with an engaging introduction
- Include 3 main content sections
- End with a strong conclusion
- Keep titles clear and specific"""

        response = llm.invoke([
            SystemMessage(content="You are an expert content strategist. Create clear, logical blog post structures."),
            HumanMessage(content=prompt)
        ])
        
        # Parse the plan
        plan_text = response.content
        plan_list = [line.strip() for line in plan_text.split('\n') if line.strip() and line[0].isdigit()]
        
        if len(plan_list) < 3:
            raise ValueError("Plan must have at least 3 sections")
        
        log_event(f"✅ Plan created with {len(plan_list)} sections")
        
        return {
            "plan": plan_list,
            "pending": plan_list.copy(),  # Copy to avoid mutation
            "drafts": [],  # Initialize empty drafts list
            "status": "writing",
            "iteration_count": 0,
            "errors": [],
            "quality_scores": {},
            "needs_revision": [],
            "context_window": [],
            "current_context": f"Topic: {topic}\nPlan: {', '.join(plan_list)}"
        }
        
    except Exception as e:
        log_event(f"❌ Planner error: {str(e)}")
        return {
            "errors": [f"Planner failed: {str(e)}"],
            "status": "error"
        }

# ============================================================================
# AGENT 2: CONTEXT MANAGER
# ============================================================================

def context_manager_node(state: BlogWorkflowState) -> Dict:
    """
    Manages context to keep it relevant and within token limits.
    
    CONTEXT ENGINEERING CORE:
    - Summarizes old content
    - Keeps recent context in full
    - Ensures context stays under token limit
    """
    log_event("🧠 AGENT: Context Manager - Optimizing context")
    
    drafts = state.get('drafts', [])
    topic = state.get('topic', '')
    plan = state.get('plan', [])
    
    # Keep only last 3 drafts in full detail
    CONTEXT_WINDOW_SIZE = 3
    
    if len(drafts) <= CONTEXT_WINDOW_SIZE:
        # All drafts fit in context window
        context_window = drafts.copy()
        current_context = f"Topic: {topic}\n\n"
        for draft in drafts:
            current_context += f"{draft['section']}\n{draft['content']}\n\n"
    else:
        # Need to summarize older content
        old_drafts = drafts[:-CONTEXT_WINDOW_SIZE]
        recent_drafts = drafts[-CONTEXT_WINDOW_SIZE:]
        
        # Create summary of old content
        old_content = "\n\n".join([f"{d['section']}: {d['content']}" for d in old_drafts])
        
        try:
            summary_prompt = f"""Summarize the following blog sections in 2-3 sentences, capturing key points:

{old_content}"""
            
            summary_response = llm.invoke([
                SystemMessage(content="You create concise, informative summaries."),
                HumanMessage(content=summary_prompt)
            ])
            
            summary = summary_response.content
            
        except Exception as e:
            log_event(f"⚠️ Summarization failed, using truncation: {str(e)}")
            summary = truncate_context(old_content, max_tokens=500)
        
        # Build optimized context
        current_context = f"Topic: {topic}\n\nPrevious sections (summarized):\n{summary}\n\n"
        current_context += "Recent sections (full detail):\n\n"
        for draft in recent_drafts:
            current_context += f"{draft['section']}\n{draft['content']}\n\n"
        
        context_window = recent_drafts.copy()
    
    # Final truncation if still too long
    current_context = truncate_context(current_context, MAX_CONTEXT_TOKENS)
    
    log_event(f"✅ Context optimized: {len(context_window)} drafts in window")
    
    return {
        "current_context": current_context,
        "context_window": context_window
    }

# ============================================================================
# AGENT 3: WRITER
# ============================================================================

def writer_node(state: BlogWorkflowState) -> Dict:
    """
    Writes a single section with proper context.
    
    CONTEXT ENGINEERING:
    - Receives only optimized context
    - Focuses on one section at a time
    - Uses minimal relevant information
    """
    log_event("✍️ AGENT: Writer - Drafting section")
    
    # Get next section to write
    pending = state['pending'].copy()  # No mutation!
    if not pending:
        return {"status": "validating"}
    
    section_to_write = pending[0]
    new_pending = pending[1:]  # Create new list
    
    topic = state['topic']
    current_context = state.get('current_context', '')
    iteration = state.get('iteration_count', 0)
    
    # Retry logic
    for attempt in range(MAX_RETRIES):
        try:
            log_event(f"📝 Writing: {section_to_write} (attempt {attempt + 1})")
            
            # Focused prompt with only relevant context
            prompt = f"""Write an engaging, informative paragraph for this blog section.

{current_context}

SECTION TO WRITE: {section_to_write}

Requirements:
- 150-250 words
- Engaging and clear writing
- Connect logically with previous sections
- Provide valuable insights
- No fluff or filler

Write the paragraph now:"""

            response = llm.invoke([
                SystemMessage(content="You are an expert blog writer. Write clear, engaging, informative content."),
                HumanMessage(content=prompt)
            ])
            
            content = response.content.strip()
            
            # Validate minimum length
            if len(content.split()) < 50:
                raise ValueError("Content too short")
            
            # Update drafts
            drafts = state.get('drafts', []).copy()
            drafts.append({
                "section": section_to_write,
                "content": content,
                "attempt": attempt + 1
            })
            
            log_event(f"✅ Section completed: {section_to_write}")
            
            return {
                "pending": new_pending,
                "drafts": drafts,
                "iteration_count": iteration + 1
            }
            
        except Exception as e:
            log_event(f"⚠️ Write attempt {attempt + 1} failed: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                # Final attempt failed
                errors = state.get('errors', []).copy()
                errors.append(f"Failed to write '{section_to_write}' after {MAX_RETRIES} attempts")
                return {
                    "errors": errors,
                    "status": "error"
                }
            time.sleep(1)  # Brief pause before retry

# ============================================================================
# AGENT 4: QUALITY VALIDATOR
# ============================================================================

def validator_node(state: BlogWorkflowState) -> Dict:
    """
    Validates quality of written content.
    
    CONTEXT ENGINEERING:
    - Only evaluates the latest draft
    - Uses clear quality criteria
    """
    log_event("🔍 AGENT: Validator - Checking quality")
    
    drafts = state['drafts']
    if not drafts:
        return {"status": "compiling"}
    
    latest_draft = drafts[-1]
    section = latest_draft['section']
    content = latest_draft['content']
    
    try:
        # Quality check prompt
        prompt = f"""Evaluate this blog section on a scale of 0-10:

SECTION: {section}

CONTENT:
{content}

Criteria:
- Clarity and readability (0-10)
- Depth of insight (0-10)
- Engagement factor (0-10)
- Grammar and style (0-10)

Output ONLY a single number (the average score): X.X"""

        response = llm.invoke([
            SystemMessage(content="You are a strict content quality evaluator."),
            HumanMessage(content=prompt)
        ])
        
        # Parse score
        score_text = response.content.strip()
        score = float(score_text) / 10.0  # Normalize to 0-1
        
        quality_scores = state.get('quality_scores', {}).copy()
        quality_scores[section] = score
        
        log_event(f"📊 Quality score for '{section}': {score:.2f}")
        
        # Check if revision needed
        needs_revision = state.get('needs_revision', []).copy()
        pending = state.get('pending', []).copy()
        
        if score < QUALITY_THRESHOLD:
            log_event(f"⚠️ Section needs revision: {section}")
            needs_revision.append(section)
            # Add back to pending for rewrite
            if section not in pending:
                pending.insert(0, section)
        
        # Determine next status
        if pending:
            next_status = "writing"
        elif needs_revision:
            next_status = "writing"
        else:
            next_status = "compiling"
        
        return {
            "quality_scores": quality_scores,
            "needs_revision": needs_revision,
            "pending": pending,
            "status": next_status
        }
        
    except Exception as e:
        log_event(f"⚠️ Validation failed: {str(e)}, proceeding anyway")
        # If validation fails, proceed without it
        return {
            "status": "compiling" if not state.get('pending') else "writing"
        }

# ============================================================================
# AGENT 5: COMPILER
# ============================================================================

def compiler_node(state: BlogWorkflowState) -> Dict:
    """
    Assembles the final blog post.
    
    CONTEXT ENGINEERING:
    - Uses all drafts but in organized format
    - Adds structure and polish
    """
    log_event("📦 AGENT: Compiler - Assembling final post")
    
    drafts = state['drafts']
    topic = state['topic']
    plan = state['plan']
    
    try:
        # Build the post
        final_post = f"# {topic}\n\n"
        
        # Organize drafts by plan order
        draft_dict = {d['section']: d['content'] for d in drafts}
        
        for section in plan:
            if section in draft_dict:
                final_post += f"## {section}\n\n"
                final_post += f"{draft_dict[section]}\n\n"
        
        # Add quality report
        quality_scores = state.get('quality_scores', {})
        if quality_scores:
            avg_quality = sum(quality_scores.values()) / len(quality_scores)
            final_post += f"\n---\n*Average Quality Score: {avg_quality:.2f}*\n"
        
        log_event("✅ Compilation complete!")
        
        return {
            "final_post": final_post,
            "status": "complete"
        }
        
    except Exception as e:
        log_event(f"❌ Compilation error: {str(e)}")
        return {
            "errors": state.get('errors', []) + [f"Compilation failed: {str(e)}"],
            "status": "error"
        }

# ============================================================================
# ROUTER (SUPERVISOR)
# ============================================================================

def router_node(state: BlogWorkflowState) -> Literal["write", "context", "validate", "compile", "error"]:
    """
    Intelligent routing based on state.
    
    This is the orchestration layer that decides what happens next.
    """
    log_event("🎯 ROUTER: Deciding next action")
    
    status = state.get('status', 'writing')
    iteration = state.get('iteration_count', 0)
    pending = state.get('pending', [])
    errors = state.get('errors', [])
    
    # Check for error conditions
    if errors:
        log_event("❌ Error detected, halting workflow")
        return "error"
    
    # Check iteration limit
    if iteration >= MAX_ITERATIONS:
        log_event("⚠️ Max iterations reached, forcing compilation")
        return "compile"
    
    # Route based on status
    if status == "writing":
        if pending:
            # Need to update context before writing
            log_event("→ Routing to: Context Manager")
            return "context"
        else:
            log_event("→ Routing to: Validator")
            return "validate"
    
    elif status == "validating":
        log_event("→ Routing to: Validator")
        return "validate"
    
    elif status == "compiling" or status == "complete":
        log_event("→ Routing to: Compiler")
        return "compile"
    
    else:
        # Default to context manager
        return "context"

def after_context_router(state: BlogWorkflowState) -> Literal["write", "validate"]:
    """Router specifically after context management"""
    pending = state.get('pending', [])
    if pending:
        return "write"
    return "validate"

# ============================================================================
# BUILD THE WORKFLOW GRAPH
# ============================================================================

def build_workflow() -> StateGraph:
    """Construct the complete workflow graph"""
    
    workflow = StateGraph(BlogWorkflowState)
    
    # Add all nodes (router is NOT a node, just a routing function)
    workflow.add_node("planner", planner_node)
    workflow.add_node("context_manager", context_manager_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("compiler", compiler_node)
    
    # Set entry point
    workflow.set_entry_point("planner")
    
    # Define the flow with conditional routing directly from nodes
    # After planner, route based on state
    workflow.add_conditional_edges(
        "planner",
        router_node,
        {
            "context": "context_manager",
            "write": "writer",
            "validate": "validator",
            "compile": "compiler",
            "error": END
        }
    )
    
    # After context manager, decide to write or validate
    workflow.add_conditional_edges(
        "context_manager",
        after_context_router,
        {
            "write": "writer",
            "validate": "validator"
        }
    )
    
    # After writing, route based on state
    workflow.add_conditional_edges(
        "writer",
        router_node,
        {
            "context": "context_manager",
            "write": "writer",
            "validate": "validator",
            "compile": "compiler",
            "error": END
        }
    )
    
    # After validation, route based on state
    workflow.add_conditional_edges(
        "validator",
        router_node,
        {
            "context": "context_manager",
            "write": "writer",
            "validate": "validator",
            "compile": "compiler",
            "error": END
        }
    )
    
    # Compiler ends the workflow
    workflow.add_edge("compiler", END)
    
    return workflow.compile()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the complete workflow"""
    
    print("\n" + "="*80)
    print("MULTI-AGENT BLOG WRITER - WITH BEST PRACTICES")
    print("="*80 + "\n")
    
    # Build the workflow
    app = build_workflow()
    
    # Initial input
    inputs = {
        "topic": "The Future of Artificial Intelligence in Healthcare"
    }
    
    print(f"📝 Topic: {inputs['topic']}\n")
    print("Starting workflow...\n")
    
    # Run the workflow
    start_time = time.time()
    
    try:
        # Stream events for visibility
        for event in app.stream(inputs, {"recursion_limit": MAX_ITERATIONS + 5}):
            node_name = list(event.keys())[0]
            print(f"\n{'='*80}")
            print(f"NODE EXECUTED: {node_name}")
            print(f"{'='*80}\n")
        
        # Get final state
        final_state = app.invoke(inputs, {"recursion_limit": MAX_ITERATIONS + 5})
        
        elapsed_time = time.time() - start_time
        
        # Display results
        print("\n" + "="*80)
        print("WORKFLOW COMPLETE")
        print("="*80 + "\n")
        
        print(f"⏱️ Total time: {elapsed_time:.2f} seconds")
        print(f"🔄 Iterations: {final_state.get('iteration_count', 0)}")
        print(f"📊 Sections written: {len(final_state.get('drafts', []))}")
        
        if final_state.get('quality_scores'):
            avg_quality = sum(final_state['quality_scores'].values()) / len(final_state['quality_scores'])
            print(f"⭐ Average quality: {avg_quality:.2f}")
        
        if final_state.get('errors'):
            print(f"\n⚠️ Errors encountered: {len(final_state['errors'])}")
            for error in final_state['errors']:
                print(f"  - {error}")
        
        print("\n" + "="*80)
        print("FINAL BLOG POST")
        print("="*80 + "\n")
        print(final_state.get('final_post', 'No post generated'))
        
    except Exception as e:
        print(f"\n❌ Workflow failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()