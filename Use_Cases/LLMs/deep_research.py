import dspy
import os
from tavily import TavilyClient
from dotenv import load_dotenv
from typing import List

# ... (all your setup code, load_dotenv, check_llm_connection, lm configuration remains the same) ...
# ... (The TavilySearch, GenerateSearchQueries, SynthesizeAndAnswer, and GenerateFinalReport classes are perfect as they are) ...
# Load environment variables from .env file
load_dotenv()

def check_llm_connection():
    """
    Performs a simple "Hello World" test to verify the LLM connection.
    """
    try:
        class BasicQA(dspy.Signature):
            """Ask a simple question, get a simple answer."""
            question = dspy.InputField()
            answer = dspy.OutputField()

        hello_world_predictor = dspy.Predict(BasicQA)
        # Using a date relevant to our current time in the Netherlands.
        question = "What day of the week was August 13, 2025?"
        
        result = hello_world_predictor(question=question)

        if result and result.answer:
            print(f"✅ LLM Connection Successful!")
            print(f"   Question: {question}")
            print(f"   LLM Answer: {result.answer}")
            return True
        else:
            print("❌ LLM Connection Failed: Received an empty response.")
            return False

    except Exception as e:
        print(f"❌ LLM Connection Failed.")
        print(f"   Error details: {e}")
        return False

# Configure the LLM
lm = dspy.LM("openrouter/moonshotai/kimi-k2:free", api_key=os.getenv("OPENROUTER_API_KEY"), api_base="https://openrouter.ai/api/v1")
dspy.configure(lm=lm)
if not check_llm_connection():
    exit() # Exit if the LLM is not working

class TavilySearch(dspy.Retrieve):
    def __init__(self, k=3, api_key=None):
        super().__init__(k=k)
        if api_key is None:
            api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("You must provide a Tavily API key or set the TAVILY_API_KEY environment variable.")
        self.client = TavilyClient(api_key=api_key)

    def forward(self, query_or_queries, k=None):
        k = k if k is not None else self.k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        
        results = []
        print(f"\n Searching Tavily for: {queries}...")
        for query in queries:
            response = self.client.search(query=query, search_depth="basic", max_results=k)
            results.extend([d['content'] for d in response['results']])
        
        print(f"Found {len(results)} snippets of content.")
        return results

class GenerateSearchQueries(dspy.Signature):
    """Generate a list of 3-5 specific search queries to research a topic."""
    research_topic:str = dspy.InputField(desc="The high-level topic to be researched.")
    search_queries:List[str] = dspy.OutputField(desc="A list of 3-5 specific questions for a search engine.")

class SynthesizeAndAnswer(dspy.Signature):
    """Given a question and search context, provide a concise answer."""
    question = dspy.InputField(desc="The question to be answered.")
    search_context = dspy.InputField(desc="Relevant text from a web search.")
    answer = dspy.OutputField(desc="A concise, 2-3 sentence answer to the question.")

class GenerateFinalReport(dspy.Signature):
    """Compile a list of questions and answers into a final research report."""
    research_topic = dspy.InputField()
    qa_pairs = dspy.InputField(desc="A list of questions and their corresponding answers.")
    report = dspy.OutputField(desc="A comprehensive and well-structured research report.")



class DeepResearchAgent(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.planner = dspy.Predict(GenerateSearchQueries)
        self.synthesizer = dspy.ChainOfThought(SynthesizeAndAnswer)
        self.reporter = dspy.ChainOfThought(GenerateFinalReport)
        self.retrieve = retriever

    def forward(self, research_topic):
        print(f"Starting deep research for topic: '{research_topic}'")
        
        try:
            # Step 1: Plan the search queries
            planned_queries_result = self.planner(research_topic=research_topic)
            planned_queries = planned_queries_result.search_queries
            
            # REFINEMENT 3: Robust check for list type
            if not isinstance(planned_queries, List):
                print("LLM failed to return a list. Using the raw output as a single query.")
                # Attempt to rescue by treating the whole output as one query
                planned_queries = [str(planned_queries)]

            print(f"Planned Queries: {planned_queries}")

        except Exception as e:
            print(f"Failed to generate search queries from LLM. Error: {e}")
            return dspy.Prediction(report="Could not complete research because the planning stage failed.")

        # Step 2 & 3: Search and Synthesize for each query
        qa_pairs = []
        for query in planned_queries:
            # REFINEMENT 4: Call your retriever directly. It returns a list of strings.
            context_from_web = self.retrieve(query) 
            
            print(f"\n Answering question: '{query}'")
            answer_result = self.synthesizer(question=query, search_context=context_from_web)
            
            qa = {"question": query, "answer": answer_result.answer}
            qa_pairs.append(qa)
            print(f" Answer: {qa['answer']}")

        # Step 4: Generate the final report
        print("\nGenerating final report...")
        formatted_qa = "\n\n".join([f"Question: {p['question']}\nAnswer: {p['answer']}" for p in qa_pairs])
        final_report = self.reporter(research_topic=research_topic, qa_pairs=formatted_qa)
        
        return dspy.Prediction(report=final_report.report)
    

tavily_retriever = TavilySearch(k=3)

research_agent = DeepResearchAgent(retriever=tavily_retriever)

research_topic = "The psychological effects of striving for constant happiness."
result = research_agent(research_topic=research_topic)

print("\n\n--- FINAL REPORT ---")
print(result.report)