import os
from typing import List
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()


def load_documents() -> List[dict]:
    """
    Load .txt documents from the data folder.
    Returns: List of file contents
    """
    results = []

    data_folder = os.path.join(os.path.dirname(__file__), "..", "data")
    data_folder = os.path.abspath(data_folder)

    for filename in os.listdir(data_folder):
        print("Found file:", filename)
        if filename.endswith((".txt", ".eve")):
            file_path = os.path.join(data_folder, filename)
            print("-> Loading:", filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:  # skip empty files
                    results.append({
                        "content": content,
                        "metadata": {"source": filename}
                })

    return results


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    def __init__(self):
        """Initialize the RAG assistant."""
        # Initialize LLM - check for available API keys in order of preference
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

        # Initialize vector database
        self.vector_db = VectorDB()

        # Create RAG prompt template
        self.prompt_template = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Use the context to answer the question.
If the answer is not in the context, say: "I don't know based on the available documents."

Context:
{context}

Question:
{question}

Answer:
""")

        # Create the chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("RAG Assistant initialized successfully")

    def _initialize_llm(self):
        """
        Initialize the LLM by checking for available API keys.
        Tries OpenAI, Groq, and Google Gemini in that order.
        """
        # Check for OpenAI API key
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0.0,
            )

        else:
            raise ValueError(
                "No valid API key found. Please set one of: OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

    def add_documents(self, documents: List) -> None:
        """Add documents to the knowledge base."""
        if not documents:
            print("No documents to add.")
            return
        print(f"Adding {len(documents)} documents to vector DB...")
        self.vector_db.add_documents(documents)
        print("Documents added successfully.")

    def invoke(self, input_text: str, n_results: int = 3) -> str:
      raw_results = self.vector_db.search(input_text, n_results= n_results)["documents"]

      if not raw_results:
        return "I don't know based on the available documents."

    # FIX: flatten list of lists
      search_results = [item[0] if isinstance(item, list) else item for item in raw_results]

      context = "\n\n".join(search_results)

      llm_answer = self.chain.invoke({
        "context": context,
        "question": input_text
      })
      return llm_answer


    # Fix: main() is calling assistant.query(), so create alias
    def query(self, question: str):
        return self.invoke(question)


def main():
    """Main function to demonstrate the RAG assistant."""
    try:
        # Initialize the RAG assistant
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        # Load sample documents
        print("\nLoading documents...")
        sample_docs = load_documents()
        print(f"Loaded {len(sample_docs)} sample documents\n")
        for doc in sample_docs:
          print(" -", doc["metadata"]["source"], ":", doc["content"][:50], "...")
        assistant.add_documents(sample_docs)

        while True:
            question = input("Enter a question or 'quit' to exit: ")

            if question.lower() == "quit":
                break

            result = assistant.query(question)
            print("\nAnswer:\n", result, "\n")

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
