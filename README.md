# 🤖 RAG-Based AI Assistant (AAIDC – Project 1)

## 📌 Overview

This project implements a **Retrieval-Augmented Generation (RAG) based AI assistant** that can answer user questions using **custom documents**. Instead of relying only on a language model’s internal knowledge, the system retrieves relevant context from documents and then generates grounded responses.

In simple terms: **it’s an AI assistant that understands *your* documents and answers questions based on them**.

This project was built as part of the **AAIDC learning program**, with a focus on understanding **real-world RAG pipelines, data flow, and system behavior**.

---

## 🎯 What This Project Does

The AI assistant can:

* 📄 Load documents (text files)
* ✂️ Split documents into smaller chunks
* 🧠 Generate embeddings and store them in a vector database
* 🔍 Retrieve the most relevant chunks for a user query
* 💬 Generate answers using retrieved context + an LLM

This demonstrates the **end-to-end RAG workflow** used in production AI systems.

---

## 🏗️ System Architecture (High Level)

```
User Question
     ↓
Query Embedding
     ↓
Vector Database Search
     ↓
Relevant Document Chunks
     ↓
Prompt + Context
     ↓
LLM Response
```

---

## 🧩 Implementation Details

### 1️⃣ Document Loading

* Documents are loaded from the `data/` directory
* Each document is read and stored with metadata

### 2️⃣ Text Chunking

* Documents are split into smaller chunks for better retrieval
* Chunking improves semantic search accuracy

### 3️⃣ Vector Database

* Embeddings are generated using a transformer-based embedding model
* Chunks are stored in **ChromaDB** for similarity search

### 4️⃣ Similarity Search

* User queries are embedded
* Top-k relevant chunks are retrieved based on vector similarity

### 5️⃣ RAG Prompting

* Retrieved context is injected into a structured prompt
* The LLM is instructed to answer strictly using the provided context

### 6️⃣ Answer Generation

* The LLM generates a final response grounded in retrieved documents

---

## 🛠️ Tech Stack

* **Python**
* **LangChain** (prompting & chaining)
* **ChromaDB** (vector database)
* **Hugging Face / OpenAI embeddings**
* **LLMs** (OpenAI / Groq / Google – configurable)

---

## 📂 Project Structure

```
rt-aaidc-project1-template/
├── src/
│   ├── app.py        # Main RAG application
│   └── vectordb.py   # Vector DB logic (chunking, search)
├── data/             # Input documents
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Configure environment variables

Create a `.env` file and add your API key:

```env
OPENAI_API_KEY=your_key_here
# or
GROQ_API_KEY=your_key_here
# or
GOOGLE_API_KEY=your_key_here
```

### 3️⃣ Add your documents

Place `.txt` files inside the `data/` directory.

### 4️⃣ Run the application

```bash
python src/app.py
```

---

## 🧪 Example Queries

* "What is [topic from my documents]?"
* "Explain [concept]"
* "How does [process] work?"

---

## 🎓 Learning Outcomes

Through this project, I:

* Gained hands-on experience with **RAG systems**
* Understood **embeddings, vector search, and retrieval quality**
* Built an **end-to-end AI pipeline**, not just a demo
* Learned how real AI systems handle **data flow and reliability**

---

## 🔮 Future Improvements

* Support for PDF documents
* Advanced chunking strategies
* Better evaluation and logging
* UI integration

---

## 👤 Author

**Akalya Radhakrishnan**
---

⭐ *This project focuses on learning and system understanding rather than perfection. Feedback and suggestions are welcome.*
