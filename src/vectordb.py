import os
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):

        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    # ---------------------------------------------------------
    # 1️⃣ SIMPLE CHUNKING FUNCTION
    # ---------------------------------------------------------
    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:

        words = text.split()
        chunks = []
        current = []

        for w in words:
            if len(" ".join(current)) + len(w) + 1 <= chunk_size:
                current.append(w)
            else:
                chunks.append(" ".join(current))
                current = [w]

        if current:
            chunks.append(" ".join(current))

        return chunks

    # ---------------------------------------------------------
    # 2️⃣ ADD DOCUMENTS TO CHROMADB
    # ---------------------------------------------------------
    def add_documents(self, documents: List) -> None:

        print(f"Processing {len(documents)} documents...")

        all_ids = []
        all_texts = []
        all_metadatas = []

        id_counter = 0

        for i, doc in enumerate(documents):
            # ✅ Safe handling of dicts and strings
            if isinstance(doc, dict):
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
            else:  # If doc is just a string
                content = str(doc)
                metadata = {}

            chunks = self.chunk_text(content)

            print(f"Document {i}: {len(chunks)} chunks")

            for j, chunk in enumerate(chunks):
                chunk_id = f"doc_{i}_chunk_{j}"

                all_ids.append(chunk_id)
                all_texts.append(chunk)
                all_metadatas.append(metadata)

                id_counter += 1

        # Create embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(all_texts).tolist()

        print("Storing in ChromaDB...")
        self.collection.add(
            ids=all_ids,
            documents=all_texts,
            embeddings=embeddings,
            metadatas=all_metadatas,
        )

        print("Documents added to vector database")

    # ---------------------------------------------------------
    # 3️⃣ SEARCH FUNCTION
    # ---------------------------------------------------------
    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:

        print(f"Searching for: {query}")

        query_embedding = self.embedding_model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        return {
            "documents": results.get("documents", []),
            "metadatas": results.get("metadatas", []),
            "distances": results.get("distances", []),
            "ids": results.get("ids", []),
        }
