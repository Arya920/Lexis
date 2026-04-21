# Lexis — Knowledge Assistant

Lexis is a modern, production-grade AI-powered chatbot and knowledge assistant. It supports document ingestion, retrieval-augmented generation (RAG), web search, and direct LLM chat, with a beautiful React-based frontend and a robust Flask backend. This README provides a comprehensive overview for developers and AI tools to understand, extend, or modify the project.

---

## Table of Contents
- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Directory Structure](#directory-structure)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Backend Services](#backend-services)
  - [app.py (Flask App)](#apppy-flask-app)
  - [services/](#services)
- [Frontend](#frontend)
- [API Endpoints](#api-endpoints)
- [Adding New Features](#adding-new-features)
- [Dependencies](#dependencies)
- [License](#license)

---

## Features
- **Conversational AI Chatbot** (LLM-based, supports direct, RAG, and web search modes)
- **Document Upload & Ingestion** (PDFs, chunking, vector storage)
- **Hybrid Retrieval** (Semantic + Keyword/BM25 + Reranking)
- **Web Search Integration** (Tavily API)
- **Session Management** (multi-session chat, local storage)
- **Rich UI** (React 18, dark/light mode, agentic features)
- **Extensible Prompt Templates**
- **Database Connector (MySQL)** (for future data/analytics features)

---

## Architecture Overview

```mermaid
graph TD;
  User[User (Browser)] -->|React UI| FlaskApp[Flask Backend]
  FlaskApp -->|/chat| LLM[LLM (via LangChain)]
  FlaskApp -->|/upload| Ingestion[PDF Ingestion]
  FlaskApp -->|/files| FileList[File Listing]
  FlaskApp -->|/remove-file| FileRemove[File Removal]
  FlaskApp -->|/chat (RAG)| Retrieval[Hybrid Retrieval]
  FlaskApp -->|/chat (Web)| WebSearch[Tavily API]
  FlaskApp -->|VectorStore| FAISS[FAISS Vector DB]
  FlaskApp -->|Prompts| Prompts[Prompt Templates]
```

---

## Directory Structure

```
Lexis/
├── app.py                # Main Flask app
├── requirements.txt      # Python dependencies
├── config/
│   └── settings.py       # All config variables
├── data/
│   ├── uploads/          # Uploaded PDFs
│   └── vectordb/         # FAISS index, chunk store
├── services/             # All backend logic
│   ├── db_connector.py   # MySQL DB connector
│   ├── embeddings.py     # Embedding/reranker models
│   ├── generation.py     # LLM answer generation
│   ├── ingestion.py      # PDF chunking & storage
│   ├── prompts.py        # Prompt templates
│   ├── retrieval.py      # Hybrid retrieval logic
│   └── tavily_search.py  # Web search API
├── static/
│   ├── app.js            # React frontend (single file)
│   └── style.css         # App styles
├── templates/
│   └── index4.html       # Main HTML (mounts React)
└── ...
```

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Lexis
   ```
2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables:**
   - Create a `.env` file in the root directory with the following keys:
     - `GROQ_API_KEY` (for LLM)
     - `TAVILY_API_KEY` (for web search)
   - Example:
     ```env
     GROQ_API_KEY=your_groq_api_key
     TAVILY_API_KEY=your_tavily_api_key
     ```
4. **Run the app:**
   ```bash
   python app.py
   ```
5. **Access the app:**
   - Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Configuration

All configuration is in `config/settings.py`:
- **Model names** (embedding, reranker, LLM)
- **Chunking parameters** (min/max sentences, chars, percentile)
- **Hybrid retrieval weights**
- **Directory paths** (uploads, vector DB)

---

## Backend Services

### app.py (Flask App)
- **Routes:**
  - `/` — Main UI (renders `index4.html`)
  - `/files` — List uploaded files
  - `/upload` — Upload and ingest PDF
  - `/remove-file` — Remove a file and update vector DB
  - `/chat` — Main chat endpoint (supports direct, RAG, and web search modes)
- **Features:**
  - Handles all API logic for chat, file management, and document ingestion
  - Integrates with all services in `services/`
  - Uses singleton pattern for generator and web search
  - Rebuilds FAISS index on file removal

### services/

#### db_connector.py
- **Purpose:** MySQL database connector (for future analytics/data features)
- **Key Methods:**
  - `connect_mysql(host, user, password, database)` — Connects to MySQL
  - `is_connected()` — Checks connection
  - `get_tables()` — Lists tables
  - `run_query(query)` — Runs SQL query (returns results or error)

#### embeddings.py
- **Purpose:** Loads embedding and reranker models (singleton)
- **Key Functions:**
  - `get_embedding_model()` — Loads HuggingFace embedding model
  - `get_reranker_model()` — Loads CrossEncoder reranker
- **Config:** Model names from `settings.py`

#### generation.py
- **Purpose:** Handles all LLM-based answer generation
- **Class:** `AnswerGenerator`
  - `generate_rag(query, context)` — RAG mode (context + query)
  - `generate_web(query, context)` — Web search mode
  - `generate_direct(query)` — Direct LLM mode
- **Uses:** Prompt templates from `prompts.py`

#### ingestion.py
- **Purpose:** PDF ingestion, chunking, and storage
- **Key Functions:**
  - `process_pdf(file_path)` — Loads PDF, splits into chunks, stores in FAISS and chunk store
  - `semantic_chunk_text(text, embedding_model)` — Industry-grade chunking (paragraph/sentence/word fallback, overlap)
  - `_save_chunk_store(chunks)` — Saves all chunks to JSON
  - `_build_documents(chunk_entries)` — Converts chunk entries to LangChain Documents

#### prompts.py
- **Purpose:** Centralized prompt templates for all modes
- **Class:** `PromptTemplates`
  - `rag_prompt(context, query)` — For RAG mode (context-restricted, conversational)
  - `web_prompt(context, query)` — For web search (concise, factual)
  - `direct_prompt(query)` — For direct LLM (no context restriction)

#### retrieval.py
- **Purpose:** Hybrid retrieval (semantic + keyword + rerank)
- **Key Classes/Functions:**
  - `BM25Index` — Keyword/BM25 index
  - `retrieve_docs(query, k)` —
    - Expands query
    - Loads FAISS and chunk store
    - Gets semantic and keyword scores
    - Combines with weights
    - Reranks with CrossEncoder
    - Returns top-k LangChain Documents (with all scores in metadata)

#### tavily_search.py
- **Purpose:** Web search using Tavily API
- **Class:** `TavilySearch`
  - `search(query)` — Returns dict with `answer` and `source` (URL)
  - Requires `TAVILY_API_KEY` in `.env`

---

## Frontend
- **Single-page React app** (in `static/app.js`)
- **Features:**
  - Multi-session chat (local storage)
  - File upload, removal, and listing
  - Mode toggles (RAG, web, direct, agentic)
  - Agent autocomplete and suggestions
  - Modern, responsive UI (see `static/style.css`)
  - All API calls to Flask endpoints
- **HTML entry:** `templates/index4.html` (mounts React root)

---

## API Endpoints

| Endpoint         | Method | Description                       |
|------------------|--------|-----------------------------------|
| `/`              | GET    | Main UI                           |
| `/files`         | GET    | List uploaded files               |
| `/upload`        | POST   | Upload and ingest PDF             |
| `/remove-file`   | POST   | Remove file and update vector DB  |
| `/chat`          | POST   | Main chat (direct/RAG/web search) |

---

## Adding New Features

- **To add a new retrieval mode:**
  - Add logic in `app.py` `/chat` endpoint
  - Add new prompt template in `prompts.py`
  - Update frontend mode toggles if needed
- **To support new file types:**
  - Extend `ingestion.py` to handle new formats
  - Update frontend file upload logic
- **To add new agents:**
  - Add agent definition in `static/app.js` (AGENTS array)
  - Add backend logic if needed
- **To add analytics/data features:**
  - Use `db_connector.py` to connect and query MySQL

---

## Dependencies

All dependencies are listed in `requirements.txt`. Key packages:
- Flask, Jinja2, Werkzeug
- LangChain, langchain_community, langchain_core
- sentence-transformers, cross-encoder
- FAISS
- Tavily
- React (via CDN), Babel

---

## License

MIT License

---

## Credits
- UI/UX: Inspired by modern chat UIs (ChatGPT, Claude)
- Backend: Modular, extensible, production-ready
- Authors: Arya Chakraborty

---

*This README is designed to be fully self-sufficient for AI tools and developers to understand, extend, or generate new features/code for Lexis.*
