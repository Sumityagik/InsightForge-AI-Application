# 🧠 InsightForge AI Application

>An Agentic AI web application system for intelligent information extraction using LangChain, Groq, KeyBERT and spaCy.

> An AI-powered research document analysis notebook that performs **Text Summarization**, **Keyword Extraction**, and **Named Entity Recognition (NER)** using modern NLP techniques and LangChain tools.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Agentic_AI-green)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-Embeddings-orange)
![KeyBERT](https://img.shields.io/badge/KeyBERT-Keyword_Extraction-yellow)
![spaCy](https://img.shields.io/badge/spaCy-NER-brightgreen)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-red)

---

# 📖 About the Project

InsightForge AI application is an intelligent flask based web application developed as part of my Agentic AI learning journey.

The software combines multiple Natural Language Processing, Web application designs, Agentic AI techniques into a modular pipeline capable of understanding research papers and large text documents.

Instead of using individual NLP models separately, the project follows an agent-inspired architecture where specialized tools work together to generate meaningful insights from text.

---

# ✨ Features

- 📄 Automatic Text Summarization
- 🔑 Keyword Extraction using KeyBERT
- 👤 Named Entity Recognition (NER)
- 🧠 Sentence Embeddings using Sentence Transformers
- ⚡ Modular AI Tool Design
- 🤖 LangChain Tool Integration
- 📊 Structured AI Analysis Report
- 🚀 Optimized Model Loading with Caching

---

# 🏗️ Project Workflow

```Application Home  page
      |
  Registered User
      |      \
      |If yes \ If no
      |       register 
    Login    /
      |    /
 Start Chating
      |
User Input
      │
      ▼
Text Preprocessing
      │
      ▼
Embedding Model
(Sentence Transformers)
      │
      ├───────────────┐
      ▼               ▼
Keyword Tool      NER Tool
(KeyBERT)          (spaCy)
      │               │
      └──────┬────────┘
             ▼
 Summarization Tool
(BART Transformer)
             │
             ▼
Structured AI Report
```

---

# 🛠️ Technologies Used

- Python
- Google Colab
- LangChain
- Sentence Transformers
- Hugging Face Transformers
- KeyBERT
- spaCy
- PyTorch
- NumPy
- HTML5
- CSS3
- Javascript
- Bootstrap
- SQLite
- Git/Github

---

# 📂 Project Structure
InsightForge-AI/
├── app.py                      # App factory, blueprint registration
├── requirements.txt
├── .env / .env.example
├── .gitignore
│
├── api/
│   ├── __init__.py
│   ├── auth_routes.py
│   ├── chat_routes.py
│   ├── document_routes.py
│   ├── analysis_routes.py
│   └── system_routes.py
│
├── services/
│   ├── chat_service.py         # conversation/message CRUD + orchestration
│   ├── ai_service.py           # Groq/LangChain calls, prompt assembly
│   ├── document_service.py     # extraction dispatch per file type
│   ├── upload_service.py       # validation, secure storage
│   ├── rag_service.py          # chunk, embed, retrieve
│   └── speech_service.py       # placeholder hooks for server-side STT later
│
├── src/
│   ├── config.py
│   ├── db.py                   # SQLAlchemy() instance, init_app
│   ├── registry.py             # singleton model loader (spaCy, KeyBERT, SentenceTransformer)
│   ├── orchestrator.py         # ties NLP pipeline + LLM into one pipeline call
│   ├── llm.py                  # LangChain-Groq client wrapper
│   ├── prompts.py              # prompt templates
│   ├── document_loader.py      # PyMuPDF / python-docx / pandas / OCR extraction
│   ├── embeddings.py           # SentenceTransformer wrapper
│   ├── vector_store.py         # FAISS index management (per-document)
│   └── nlp_tools.py            # summarizer, KeyBERT keyword extraction, spaCy NER
│
├── models/
│   ├── __init__.py
│   ├── user.py
│   ├── conversation.py
│   ├── message.py
│   ├── document.py
│   └── chunk.py
│
├── utils/
│   ├── decorators.py           # @login_required, @handle_errors
│   ├── validators.py           # file type/size validation
│   └── responses.py            # standard success/error JSON builders
│
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── chat.html
│   ├── about.html
│   ├── login.html
│   └── register.html
│
├── static/
│   ├── css/ (main.css, chat.css, auth.css)
│   ├── js/ (chat.js, auth.js, upload.js, voice.js, markdown.js, three-bg.js)
│   ├── images/
│   └── models/
│
├── uploads/                    # gitignored, user_id-scoped subfolders
├── instance/                   # sqlite db lives here
└── tests/
    ├── test_auth.py
    ├── test_chat.py
    ├── test_documents.py
    ├── test_nlp.py
    └── test_rag.py

# 📊 Example Output

```text
==================================================
🧠 InsightForge AI Analysis Report
==================================================

📄 SUMMARY
----------------------------
...

🔑 KEYWORDS
----------------------------
Google DeepMind
Gemini
Deep Learning
Transformer

👤 NAMED ENTITIES
----------------------------
Google DeepMind (Organization)
Demis Hassabis (Person)
London (Location)
```

---

# 📦 Requirements

```
langchain
langchain-core
sentence-transformers
transformers
torch
keybert
spacy
numpy
pandas
```

Download the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

---

# 🧠 What I Learned

This project helped me understand both NLP fundamentals and how to structure AI systems using reusable components.

Key concepts I learned include:

- Sentence Embeddings
- Semantic Similarity
- Keyword Extraction
- Named Entity Recognition
- Transformer-based Summarization
- LangChain Tools
- Prompt Engineering
- AI Model Registry Pattern
- Modular Software Design
- Model Caching
- AI Workflow Orchestration

Beyond implementing individual NLP models, I learned how to design a modular AI pipeline where specialized tools collaborate to analyze text efficiently.

---

# 📈 Future Improvements

- Flask Web Application
- ChatGPT-style User Interface
- PDF Upload Support
- DOCX Upload Support
- Research Paper Analyzer
- Multi-document Analysis
- AI Chat Interface
- Export Analysis as PDF
- REST API Support
- Docker Deployment

---

# 🎯 Project Objective

The objective of this project is to explore how multiple NLP techniques can be integrated into a unified AI workflow capable of analyzing unstructured text efficiently.

Rather than focusing on a single NLP task, InsightForge AI demonstrates how modular AI components can work together to generate richer insights from textual data.

---
## 🚀 Future Development

This notebook serves as the foundational prototype of **InsightForge AI**. The next phase of the project is to transform it into a real-world AI web application using **Flask**, featuring an interactive ChatGPT-inspired interface, document upload support (PDF/DOCX), REST APIs, and a modular backend architecture for seamless user interaction.

## Important Note: This is not a complete project ,There are some updations need to implement. Please don't clone it .

# 👨‍💻 Author

**Sumit Yagik**

B.Tech Computer Science & Engineering

AI & Machine Learning Enthusiast

Passionate about NLP, Generative AI, and Agentic AI Systems.

---

⭐ If you found this project interesting, consider giving it a star!
