# 🤖 AI Knowledge Assistant

An AI-powered document question answering application built using **Retrieval-Augmented Generation (RAG)**. The application enables users to upload multiple PDF documents, build a semantic knowledge base, and interact with them through natural language conversations. It leverages **Google Gemini**, **LangChain**, **Hugging Face Embeddings**, and **ChromaDB** to deliver accurate, context-aware responses.

---

## 🚀 Features

- 📄 Upload and process multiple PDF documents
- 🧠 Retrieval-Augmented Generation (RAG) pipeline
- 🤖 Context-aware question answering using Google Gemini API
- 🔍 Semantic similarity search with Hugging Face Embeddings
- 🗂️ ChromaDB vector database for efficient document retrieval
- ✂️ Intelligent document chunking using Recursive Character Text Splitter
- ⚡ LangChain LCEL-based processing pipeline
- 💬 Interactive Streamlit chat interface
- 📚 Multi-document knowledge base
- 🔄 Persistent vector database
- ⚙️ Modular and scalable project architecture

---

# 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| LLM | Google Gemini API |
| Framework | LangChain, LCEL |
| Embeddings | Hugging Face (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB |
| UI | Streamlit |
| PDF Processing | PyPDF |
| Environment | python-dotenv |

---

# 📂 Project Structure

```text
AI-Knowledge-Assistant/
│
├── .streamlit/
│   └── config.toml
│
├── assets/
│
├── chroma_db/
│
├── data/
│
├── utils/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── helper.py
│   ├── loader.py
│   ├── prompt.py
│   ├── rag_chain.py
│   ├── splitter.py
│   └── vector_store.py
│
├── .env.example
├── .gitignore
├── app.py
├── config.py
├── README.md
└── requirements.txt
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/PavanKumar1119/ai-knowledge-assistant.git

cd ai-knowledge-assistant
```

---

## 2️⃣ Create Virtual Environment

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Configure Environment Variables

Create a `.env` file in the project root.

```env
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
```

---

## 5️⃣ Run the Application

```bash
streamlit run app.py
```

Open your browser and visit:

```
http://localhost:8501
```

---

# 🧠 How It Works

1. Upload one or more PDF documents.
2. Documents are loaded using LangChain document loaders.
3. Documents are split into smaller chunks.
4. Each chunk is converted into vector embeddings.
5. Embeddings are stored in ChromaDB.
6. User submits a natural language query.
7. Relevant document chunks are retrieved using semantic similarity search.
8. Retrieved context is passed to Google Gemini.
9. Gemini generates an accurate response grounded in the retrieved context.

---

# 🔄 System Architecture

```text
                 User
                  │
                  ▼
        Streamlit Web Interface
                  │
                  ▼
          PDF Document Upload
                  │
                  ▼
          LangChain Loader
                  │
                  ▼
     Recursive Text Splitter
                  │
                  ▼
     Hugging Face Embeddings
                  │
                  ▼
            ChromaDB
                  │
                  ▼
            Retriever
                  │
                  ▼
        LCEL RAG Pipeline
                  │
                  ▼
        Google Gemini API
                  │
                  ▼
      Context-Aware Response
```

---

# 📸 Application Preview

### Home Page

Upload one or multiple PDF documents to build a searchable knowledge base.

### Knowledge Base Creation

Documents are automatically processed, chunked, embedded, and stored in ChromaDB.

### Conversational Question Answering

Ask natural language questions and receive context-aware responses generated using Google Gemini.

---

# 💡 Skills Demonstrated

- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- LangChain
- LangChain Expression Language (LCEL)
- Google Gemini API Integration
- Semantic Search
- Hugging Face Embeddings
- ChromaDB Vector Database
- Streamlit Application Development
- Document Processing
- Prompt Design
- Vector Similarity Search

---

# 🎯 Learning Outcomes

- Built an end-to-end RAG application using modern GenAI technologies.
- Implemented semantic search using vector embeddings.
- Integrated Google Gemini API for context-aware responses.
- Developed a modular, scalable application following clean project architecture.
- Gained practical experience with LangChain LCEL and vector databases.

---

# 🔮 Future Enhancements

- Conversation memory
- Streaming responses
- Source citations
- Support for DOCX and TXT files
- Authentication and user management
- Docker containerization
- Cloud deployment

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Pattina Bhaskara Mutyala Pavan Kumar**

- GitHub: https://github.com/PavanKumar1119
- LinkedIn: https://www.linkedin.com/in/pavankumar1119

---

⭐ If you found this project useful, consider giving it a star on GitHub.
