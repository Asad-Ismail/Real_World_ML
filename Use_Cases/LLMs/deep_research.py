import dspy
import os
from tavily import TavilyClient
from dotnet import load_dotenv

# Load environment variables from .env file
load_dotenv()

def check_llm_connection():
    """
    Performs a simple "Hello World" test to verify the LLM connection.
    """
    try:
        # 1. Define a simple signature for a basic task
        class BasicQA(dspy.Signature):
            """Ask a simple question, get a simple answer."""
            question = dspy.InputField()
            answer = dspy.OutputField()

        # 2. Create a dspy.Predict module with this signature
        hello_world_predictor = dspy.Predict(BasicQA)

        # 3. Ask a trivial question
        # This is a lightweight call that should always work if the connection is good.
        question = "What day of the week was , August 13, 2025?"
        
        result = hello_world_predictor(question=question)

        # 4. Check the result and print a status message
        if result and result.answer:
            print(f"✅ LLM Connection Successful!")
            print(f"   Question: {question}")
            print(f"   LLM Answer: {result.answer}")
            return True
        else:
            print("❌ LLM Connection Failed: Received an empty response.")
            return False

    except Exception as e:
        # Catch any exception, from authentication errors to network issues
        print(f"❌ LLM Connection Failed.")
        print(f"   Error details: {e}")
        return False


openrouter_model_name = "mistralai/mistral-7b-instruct:free"
openrouter_lm = dspy.OpenAI(
    model=openrouter_model_name,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    api_base="https://openrouter.ai/api/v1",
    headers={"HTTP-Referer": "http://localhost:3000"},
    max_tokens=2500
)

dspy.settings.configure(openrouter_lm)

check_llm_connection()

# os.environ["TAVILY_API_KEY"] = "YOUR_TAVILY_API_KEY"

# 2. Create the custom retriever class
class TavilySearch(dspy.Retrieve):
    def __init__(self, k=3, api_key=None):
        """
        A retriever that uses the Tavily Search API.

        Args:
            k (int): The number of top search results to retrieve.
            api_key (str): The Tavily API key. If None, it will use the TAVILY_API_KEY environment variable.
        """
        super().__init__(k=k)
        if api_key is None:
            api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("You must provide a Tavily API key or set the TAVILY_API_KEY environment variable.")
        self.client = TavilyClient(api_key=api_key)

    def forward(self, query_or_queries, k=None):
        """
        Search with Tavily and return the top k results.

        Args:
            query_or_queries (str or list[str]): The query or list of queries to search for.
            k (int, optional): The number of results to retrieve. Defaults to self.k.
        """
        k = k if k is not None else self.k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        
        # Perform the search for each query
        results = []
        for query in queries:
            response = self.client.search(query=query, search_depth="basic")
            # We are getting the content from the search results to feed the LLM
            results.extend([d['content'] for d in response['results'][:k]])
            
        return results
    

class GenerateSearchQueries(dspy.Signature):
    """Generate a list of 3-5 specific search queries to research a topic."""
    research_topic = dspy.InputField(desc="The high-level topic to be researched.")
    search_queries = dspy.OutputField(
        desc="A list of 3-5 specific questions for a search engine."
    )

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


# Instantiate your new Tavily retriever
tavily_retriever = TavilySearch(k=3) # Retrieve top 3 results per query

# Configure DSPy settings with the LM and your new retriever
dspy.settings.configure(lm=openrouter_lm, rm=tavily_retriever)


class DeepResearchAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        # These modules will now be backed by prompts optimized for web search
        self.planner = dspy.ChainOfThought(GenerateSearchQueries)
        self.searcher_synthesizer = dspy.ChainOfThought(SynthesizeAndAnswer)
        self.reporter = dspy.ChainOfThought(GenerateFinalReport)

    def forward(self, research_topic):
        planned_queries = self.planner(research_topic=research_topic).search_queries
        qa_pairs = []
        
        for query in planned_queries:
            # THIS LINE NOW USES TAVILY AUTOMATICALLY!
            context_from_web = dspy.retrieve(query) 
            
            answer_result = self.searcher_synthesizer(question=query, search_context=context_from_web)
            qa_pairs.append({"question": query, "answer": answer_result.answer})
            print(f"Question: {query}\nAnswer: {answer_result.answer}\n---\n")

        # ... rest of the class is the same ...
        formatted_qa = "\n\n".join([f"Question: {p['question']}\nAnswer: {p['answer']}" for p in qa_pairs])
        final_report = self.reporter(research_topic=research_topic, qa_pairs=formatted_qa)
        return dspy.Prediction(report=final_report.report)

# --- Now you can run it ---
# research_agent = DeepResearchAgent()
# result = research_agent(research_topic="The future of decentralized social media.")
# print(result.report)