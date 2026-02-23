# DocQuery - Intelligent Document Q&A System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green.svg)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-orange.svg)](https://openai.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade Retrieval-Augmented Generation (RAG) system that enables intelligent question-answering over your documents using semantic search and AI-powered generation.

## 🎯 Features

- **Multi-Format Support**: Process PDF, DOCX, PPTX, XLSX, TXT, and Markdown files
- **Advanced Document Processing**: 
  - Table extraction with HTML structure preservation
  - Image extraction from PDFs (base64 encoding)
  - Title-based intelligent chunking
- **Semantic Search**: Vector similarity search using OpenAI embeddings
- **Context-Aware Answers**: GPT-powered responses with source citations
- **Multi-User Workspaces**: Isolated sessions for different users
- **Real-time Streaming**: Stream LLM responses for better UX
- **Document Management**: Upload, delete, and filter documents via web UI

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## 🏗️ Architecture

```
┌─────────────────┐
│  User Question  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│         Document Processing Layer            │
│  ┌──────────────────────────────────────┐   │
│  │ Unstructured Library (hi_res)        │   │
│  │ - PDF: Tables + Images extraction    │   │
│  │ - DOCX/PPTX/XLSX: Structure parsing  │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│            Chunking Layer                    │
│  - Title-based chunking                      │
│  - Max 3000 chars/chunk                      │
│  - 500 char overlap                          │
│  - Metadata enrichment                       │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│          Embedding Layer                     │
│  OpenAI text-embedding-3-small               │
│  - 512 dimensions                            │
│  - Deduplication via content hashing         │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│         Vector Store (ChromaDB)              │
│  - Cosine similarity                         │
│  - Persistent storage                        │
│  - Multi-user collections                    │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│          Retrieval Layer                     │
│  - Top-K similarity search (K=5)             │
│  - Threshold filtering (0.30)                │
│  - Optional document filtering               │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│         Generation Layer                     │
│  GPT-4o-mini with RAG prompting              │
│  - Context injection                         │
│  - Source attribution                        │
│  - Streaming responses                       │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Answer + Sources│
└─────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- System dependencies (for PDF/image processing):
  ```bash
  # macOS
  brew install poppler tesseract libmagic
  
  # Ubuntu/Debian
  sudo apt-get install poppler-utils tesseract-ocr libmagic1
  
  # Windows
  # Download and install from official sources
  ```

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Jeel3011/DocQuery.git
   cd DocQuery
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

5. **Run the application**
   ```bash
   streamlit run chat.py
   ```

   The app will be available at `http://localhost:8501`

```
````md
## 📁 Project Structure

```
DocQuery/
├── chat.py                    # Streamlit web interface
├── src/
│   ├── components/
│   │   ├── config.py          # Configuration management
│   │   ├── data_ingestion.py  # Document processing & chunking
│   │   ├── embeddings.py      # Vector embedding management
│   │   ├── retrieval.py       # Semantic search
│   │   └── genration.py       # LLM answer generation
│   ├── logger.py              # Logging configuration
│   └── utils.py               # Helper functions
├── pipline.py                 # End-to-end RAG pipeline
├── requirements.txt           # Python dependencies
├── .env                       # API keys (not in repo)
├── .gitignore
└── README.md
```

## 🔍 How It Works

### 1. Document Processing

Documents are processed using the Unstructured library with advanced extraction:

- **PDF**: High-resolution strategy (`hi_res`) extracts text, tables (as HTML), and images (base64)
- **DOCX/PPTX**: Structure-aware parsing preserves formatting
- **XLSX**: Table detection and cell content extraction
- **TXT/MD**: Plain text processing

### 2. Intelligent Chunking

Title-based chunking creates semantic units:

```python
# Example chunk metadata
{
    "chunk_type": "text",
    "filename": "document.pdf",
    "page_number": 5,
    "chunk_index": 12,
    "content_hash": "a3b2c1...",  # SHA-256 for deduplication
    "chunk_id": "document.pdf::a3b2c1..."
}
```

### 3. Vector Storage

- Embeddings are generated using OpenAI's `text-embedding-3-small`
- Stored in ChromaDB with cosine similarity indexing
- Deduplication prevents redundant storage
- Multi-user isolation via collection namespacing

### 4. Retrieval

- Query is embedded using same model
- Top-K (default 5) most similar chunks retrieved
- Optional filtering by document name
- Results include metadata for source attribution

### 5. Answer Generation

- Retrieved chunks form context
- GPT-4o-mini generates answer with:
  - Strict grounding in provided context
  - Source citations in `[Source: filename, Page: X]` format
  - Fallback response if no relevant info found


## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt  # If available

# Run tests
python -m pytest tests/

# Format code
black src/
```

## 📊 Performance & Costs

**Typical Query Performance**:
- Document processing: ~2-5 seconds (one-time)
- Embedding generation: ~100ms per 1000 tokens
- Retrieval: <100ms (ChromaDB search)
- Answer generation: 500-2000ms (streaming)
- **Total**: ~1-3 seconds per query

**Estimated Costs** (OpenAI API):
- Embeddings: $0.02 per 1M tokens (`text-embedding-3-small`)
- LLM: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens (`gpt-4o-mini`)
- **Per Query**: ~$0.001-0.005

## 🔒 Security & Privacy

- API keys stored in `.env` (gitignored)
- Documents stored locally in `./docs` and `./vector_db`
- No data sent to third parties except OpenAI API
- Multi-user workspaces are isolated but stored on same disk

## 🗺️ Roadmap

- [ ] Add reranker for improved retrieval quality
- [ ] Multi-language support
- [ ] Voice query interface
- [ ] Export conversations to PDF/DOCX
- [ ] Integration with Google Drive / Dropbox
- [ ] Evaluation metrics (RAGAS framework)
- [ ] Support for web scraping (URLs as input)
- [ ] Fine-tuned embedding models for domain-specific docs

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) - LLM orchestration framework
- [Unstructured](https://unstructured.io/) - Document processing library
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [OpenAI](https://openai.com/) - Embeddings and LLM API
- [Streamlit](https://streamlit.io/) - Web interface framework

## 👤 Author

**Jeel Thummar**

- GitHub: [@Jeel3011](https://github.com/Jeel3011)
- Project: [DocQuery](https://github.com/Jeel3011/DocQuery)

## 📞 Support

For issues and questions:
- Open an [issue](https://github.com/Jeel3011/DocQuery/issues)
- Check existing [discussions](https://github.com/Jeel3011/DocQuery/discussions)

---

**Built using Python, LangChain, and OpenAI**
