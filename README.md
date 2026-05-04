# course_rag

#  NIT Trichy CSE Course Assistant
### A local, private RAG (Retrieval-Augmented Generation) system for CS students

> Ask questions about your course materials — cryptography, OS, databases, networks, algorithms, and semester guides — and get answers grounded in your actual lecture notes and textbooks. Runs entirely on your machine. No internet. No API keys. No cost.

---

##  Project Structure

```
~/course_rag/
│
├── rag_pipeline.py          # Core RAG engine (hierarchical chunking + retrieval)
├── app.py                   # Gradio web UI
├── subjects.yaml            # Master config — controls all subjects
│
├── course_materials/        # Drop your PDFs here
│   ├── crypto/              # Cryptography PDFs
│   ├── os/                  # Operating Systems PDFs
│   ├── db/                  # Databases PDFs
│   ├── networks/            # Computer Networks PDFs
│   ├── algo/                # Algorithms PDFs
│   └── guide/               # Semester guide PDFs (sem3.pdf, sem4.pdf, ...)
│
├── chroma_db/               # Auto-created — child chunk vectors (ChromaDB)
└── docstore/                # Auto-created — parent chunk text (JSON)
    ├── cryptography.json
    ├── operating_systems.json
    └── ...
```

---

## ⚙️ How It Works

### The RAG Pipeline

```
PDF files
   │
   ▼  PyMuPDF loader
Raw text pages (with metadata: filename, page, subject)
   │
   ▼  Hierarchical chunking
   │
   ├── Parent chunks  (large, e.g. 600 tokens)  → saved to docstore/
   │       │
   │       └── Child chunks  (small, e.g. 200 tokens)  → embedded → ChromaDB
   │
   ▼  At query time:
Student question
   │
   ▼  nomic-embed-text  →  768-dim vector
   │
   ▼  ChromaDB cosine similarity search  →  top child chunks
   │
   ▼  Fetch parent chunks from docstore (via doc_id in child metadata)
   │
   ▼  Mistral 7B (Ollama)  →  grounded answer + source citations
```

### Why Hierarchical Chunking?

| Problem | Solution |
|---|---|
| Small chunks = precise search but missing context | Child chunks used for search (precise) |
| Large chunks = good context but poor matching | Parent chunks sent to LLM (full context) |
| Best of both worlds | Search small, read large |

---

## 🚀 Setup (WSL / Ubuntu)

### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull mistral
ollama pull nomic-embed-text
```

### 2. Create project + virtual environment

```bash
mkdir -p ~/course_rag/course_materials/{crypto,os,db,networks,algo,guide}
mkdir -p ~/course_rag/{chroma_db,docstore}
cd ~/course_rag
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python packages

```bash
pip install \
  langchain \
  langchain-community \
  langchain-chroma \
  langchain-ollama \
  langchain-core \
  langchain-text-splitters \
  pymupdf \
  gradio \
  pyyaml
```

### 4. Add your PDFs

```bash
# Copy PDFs from Windows into WSL
cp /mnt/c/Users/YourName/Downloads/crypto_slides.pdf ~/course_rag/course_materials/crypto/
cp /mnt/c/Users/YourName/Downloads/ostep.pdf        ~/course_rag/course_materials/os/

# Semester guide PDFs should be named to include sem3/sem4/sem5/sem6
# e.g. sem5_guide.pdf, NIT_Trichy_Sem6.pdf — as long as "sem5" appears in the name
```

### 5. Run

```bash
# Build knowledge base + CLI chat
python rag_pipeline.py

# OR launch web UI (after DB is built)
python app.py
# Open http://localhost:7860 in your Windows browser
```

---

## 💬 CLI Commands

| Command | What it does |
|---|---|
| `<any question>` | Ask anything — auto-routes to right subject |
| `rebuild` | Re-ingest ALL subjects from scratch |
| `rebuild cryptography` | Re-ingest only one subject |
| `debug <question>` | Show raw parent chunks retrieved (useful for tuning) |
| `quit` | Exit |

---

## 📝 subjects.yaml — Adding a New Subject

Open `subjects.yaml` and add an entry:

```yaml
subjects:
  your_new_subject:
    folder: your_folder_name          # subfolder inside course_materials/
    chunk_size: 500                   # parent chunk size in tokens
    chunk_overlap: 75                 # overlap between parent chunks
    keywords: [keyword1, keyword2]    # used by router to detect subject
```

Then create the folder and drop PDFs in:

```bash
mkdir ~/course_rag/course_materials/your_folder_name
cp /path/to/your.pdf ~/course_rag/course_materials/your_folder_name/
python rag_pipeline.py  # or: type "rebuild your_new_subject" in CLI
```

**No code changes needed.**

---

## 🔧 Chunk Size Guide

| Subject type | chunk_size | chunk_overlap | Why |
|---|---|---|---|
| Math-heavy (Crypto) | 600 | 100 | Proofs need more context |
| Concept-heavy (OS, Networks) | 500 | 75 | Standard prose |
| Algorithm-heavy | 550 | 90 | Keep pseudocode intact |
| Semester guides | 650 | 100 | Dense, multi-topic pages |

Child chunk size is automatically computed as `chunk_size ÷ 3`.

---

## 🐛 Troubleshooting

**`ollama: command not found`**
```bash
# Restart WSL or re-source your shell
source ~/.bashrc
```

**`No PDFs found`**
```bash
# Verify PDFs are inside WSL, not on /mnt/c/ at query time
ls ~/course_rag/course_materials/crypto/
```

**`chromadb` import error**
```bash
pip install langchain-chroma --upgrade
```

**`create_retrieval_chain` not found**
```bash
# Must import from langchain_core, not langchain_classic
from langchain_core.chains import create_retrieval_chain
```

**Semester filter returns 0 results**
- Check that your guide PDF filenames contain `sem3`, `sem4`, `sem5`, or `sem6`
- The filter matches substrings, so `NIT_Sem5_Guide.pdf` works fine

**Slow responses**
```bash
# Check if GPU is available in WSL
nvidia-smi
# If yes, Ollama uses it automatically — responses will be 5-10x faster

# Or switch to a faster model
ollama pull phi3:mini   # 4GB RAM, 2-3x faster than Mistral
# Then change model="phi3:mini" in rag_pipeline.py line ~200
```

---

## 📚 Recommended Free Textbooks

### Cryptography
- *Understanding Cryptography* — Paar & Pelzl (springer.com)
- *A Graduate Course in Applied Cryptography* — Boneh & Shoup (crypto.stanford.edu)
- NIST FIPS 197 (AES), FIPS 186 (DSA) — csrc.nist.gov

### Operating Systems
- *Operating Systems: Three Easy Pieces* (OSTEP) — ostep.org ⭐ every chapter free
- MIT 6.828 lecture notes — pdos.csail.mit.edu

### Databases
- *Database Management Systems* — Ramakrishnan & Gehrke (library)
- PostgreSQL docs — postgresql.org/docs

### Networks
- *Computer Networks* — Tanenbaum (library)
- IETF RFCs — rfc-editor.org (TCP, UDP, HTTP, TLS — all free)

### Algorithms
- *Introduction to Algorithms* (CLRS) — library
- MIT 6.006 lecture notes — ocw.mit.edu

---

## 🧠 Tech Stack

| Component | Tool | Why |
|---|---|---|
| LLM | Mistral 7B via Ollama | Free, local, strong STEM reasoning |
| Embeddings | nomic-embed-text via Ollama | 768-dim, better than MiniLM for technical text |
| Vector DB | ChromaDB (HNSW cosine) | Fast, local, no server needed |
| Parent store | JSONDocstore (custom) | Persists across restarts unlike InMemoryStore |
| PDF loading | PyMuPDF | Best math/symbol extraction |
| Chunking | Hierarchical (custom) | Precise search + full context for LLM |
| Pipeline | LangChain Core | Retrieval chain wiring |
| UI | Gradio | 5-line chat interface |

---

## 💡 Tips

- **Always keep your own lecture slides as the primary source** — the model answers in terms of what was actually taught in your course
- **Name guide PDFs clearly** — include `sem5`, `sem6` etc. in the filename for semester filtering to work
- **Use `debug <question>`** before reporting wrong answers — it shows exactly which chunks were retrieved, making it easy to spot gaps in your materials
- **Run `rebuild <subject>`** whenever you add new PDFs to a subject — only that subject gets re-processed, not everything

---

*Built for NIT Trichy CSE — runs fully offline on WSL2 + Ollama*

Copyright & Licensing
© 2026 Arpita. All rights reserved.

This software is provided for academic use by students of NIT Trichy. Unauthorized distribution or commercial use is prohibited.
