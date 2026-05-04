import yaml
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

BASE_DIR  = Path.home() / "course_rag"
DB_DIR    = BASE_DIR / "chroma_db"
CFG_FILE  = BASE_DIR / "subjects.yaml"
MATERIALS = BASE_DIR / "course_materials"

PROMPT = """You are a CS course assistant for NIT Trichy CSE students.

Answer ONLY from the context below.

If asked about a professor or faculty member (like "Brindha mam", "Nithya mam",
"Sridevi mam", etc.), extract everything the context says about that person —
what course they teach, their teaching style, recommended resources, and any
specific advice given about their course.

If the question is about subjects, syllabus, PES/OES, or course structure,
explain clearly what each option contains and what topics are covered.

Always mention the subject, source file, and page.

If not in context, say:
"This isn't in the loaded course materials."

Context: {context}
Question: {input}
Answer:"""

PROFESSOR_NAMES = [
    "brindha", "nithya", "sridevi", "rajeshwari", "sivasankar",
    "oswald", "mary saira bhanu", "mam", "sir", "professor",
    "who teaches", "who handles", "handled by"
]

# Maps keywords in a question → which guide PDF filename to prefer
# Edit these to match your actual filenames in course_materials/guide/
SEM_HINTS = {
    "sem3": "sem3", "third semester": "sem3", "semester 3": "sem3",
    "sem4": "sem4", "fourth semester": "sem4", "semester 4": "sem4",
    "sem5": "sem5", "fifth semester": "sem5",  "semester 5": "sem5",
    "sem6": "sem6", "sixth semester": "sem6",  "semester 6": "sem6",
    # course → semester mapping so "Who teaches ARVR" hits sem5
    "arvr": "sem5", "augmented reality": "sem5", "virtual reality": "sem5",
    "cloud computing": "sem5", "ethical hacking": "sem5",
    "computer architecture": "sem5", "dbms": "sem5",
    "computer networks": "sem5", "aiml": "sem5",
    "compiler design": "sem6", "cryptography": "sem6",
    "embedded systems": "sem6", "web technology": "sem6",
    "data analytics": "sem6", "professional ethics": "sem6",
    "automata": "sem4", "flat": "sem4", "toc": "sem4",
    "daa": "sem4", "operating systems": "sem4", "adsa": "sem4",
    "design thinking": "sem4", "software engineering": "sem4",
    "dsa": "sem3", "data structures": "sem3", "computer organization": "sem3",
    "digital systems": "sem3", "combinatorics": "sem3", "graph theory": "sem3",
}

# ── Load config ───────────────────────────────────────────────────────────────
def load_config():
    with open(CFG_FILE) as f:
        return yaml.safe_load(f)["subjects"]

def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")

# ── Ingest one subject ────────────────────────────────────────────────────────
def ingest_subject(name, cfg, client):
    folder = MATERIALS / cfg["folder"]
    pdfs   = list(folder.glob("*.pdf"))
    if not pdfs:
        print(f"  [{name}] No PDFs found in {folder}, skipping.")
        return 0

    docs = []
    for pdf in pdfs:
        print(f"  [{name}] Loading {pdf.name}")
        pages = PyMuPDFLoader(str(pdf)).load()
        for p in pages:
            p.metadata.update({"subject": name, "filename": pdf.name})
        docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = cfg["chunk_size"],
        chunk_overlap = cfg["chunk_overlap"],
        separators    = ["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for c in chunks:
        c.metadata["subject"] = name

    Chroma.from_documents(
        documents           = chunks,
        embedding           = get_embeddings(),
        persist_directory   = str(DB_DIR),
        collection_name     = name,
        collection_metadata = {"hnsw:space": "cosine"},  # ← HNSW index, O(log n) search
    )
    print(f"  [{name}] {len(docs)} pages → {len(chunks)} chunks stored.")
    return len(chunks)

def build_all():
    config = load_config()
    total  = 0
    for name, cfg in config.items():
        total += ingest_subject(name, cfg, None)
    print(f"\nDone. Total chunks stored: {total}")

def load_collection(name):
    return Chroma(
        collection_name    = name,
        persist_directory  = str(DB_DIR),
        embedding_function = get_embeddings(),
    )

# ── Smart router ──────────────────────────────────────────────────────────────
def detect_subjects(question: str, config: dict) -> list[str]:
    q      = question.lower()
    scores = {}
    for name, cfg in config.items():
        score = sum(1 for kw in cfg["keywords"] if kw in q)
        if score > 0:
            scores[name] = score
    if not scores:
        return list(config.keys())
    top_score = max(scores.values())
    return [k for k, v in scores.items() if v == top_score]

def get_k(question: str, subject: str) -> int:
    q = question.lower()
    if any(name in q for name in PROFESSOR_NAMES):
        return 8
    if subject == "guide":
        return 6
    return 4

# ── NEW: detect which sem PDF to filter on for guide queries ──────────────────
def detect_sem_filter(question: str) -> str | None:
    """
    Returns a partial filename string if the question maps to a specific
    semester, e.g. 'sem5' → only chunks from files whose name contains 'sem5'.
    Returns None if no semester can be inferred (search all guide chunks).
    """
    q = question.lower()
    for phrase, sem in SEM_HINTS.items():
        if phrase in q:
            return sem
    return None

# ── Ask across one or more subjects ──────────────────────────────────────────
def ask(question: str, config: dict, force_subject: str = None) -> str:
    subjects = [force_subject] if force_subject else detect_subjects(question, config)
    print(f"Searching: {subjects}")

    llm    = OllamaLLM(model="mistral", temperature=0.1)
    prompt = PromptTemplate(
        input_variables=["context", "input"],
        template=PROMPT,
    )

    all_results = []
    for subj in subjects:
        try:
            vs = load_collection(subj)
            k  = get_k(question, subj)

            # ── For guide, try to filter to the right semester PDF first ──────
            retriever = None
            if subj == "guide":
                sem = detect_sem_filter(question)
                if sem:
                    print(f"  [{subj}] Filtering to files containing '{sem}'")
                    try:
                        retriever = vs.as_retriever(
                            search_kwargs={
                                "k": k,
                                "filter": {"filename": {"$contains": sem}}
                            }
                        )
                        # Quick test — if no docs match filter, fall back
                        test = retriever.invoke(question)
                        if not test:
                            print(f"  [{subj}] Filter returned 0 results, falling back to full search")
                            retriever = None
                    except Exception:
                        retriever = None

            if retriever is None:
                retriever = vs.as_retriever(search_kwargs={"k": k})

            combine_docs_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain    = create_retrieval_chain(retriever, combine_docs_chain)
            result = retrieval_chain.invoke({"input": question})
            all_results.append((subj, result))
        except Exception as e:
            print(f"  Warning: could not query {subj}: {e}")

    if not all_results:
        return "No results found across any subject."

    if len(all_results) == 1:
        subj, result = all_results[0]
        return format_answer(subj, result)

    combined = f"Found relevant content in {len(all_results)} subjects:\n\n"
    for subj, result in all_results:
        combined += f"--- {subj.upper()} ---\n"
        combined += format_answer(subj, result) + "\n\n"
    return combined

def format_answer(subject: str, result: dict) -> str:
    answer  = result["answer"]
    sources = []
    seen    = set()
    for doc in result["context"]:
        tag = f"{doc.metadata.get('filename','?')} p.{doc.metadata.get('page','?')}"
        if tag not in seen:
            sources.append(tag)
            seen.add(tag)
    out = answer
    if sources:
        out += "\n\nSources: " + " | ".join(sources)
    return out

def show_chunks(docs):
    out = "\n Retrieved Context:\n\n"
    for i, d in enumerate(docs):
        out += f"[Chunk {i+1} | {d.metadata.get('subject')}]\n"
        out += d.page_content[:200] + "\n\n"
    return out

# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config   = load_config()
    db_ready = DB_DIR.exists() and any(DB_DIR.iterdir())

    if not db_ready:
        print("Building knowledge base for all subjects...\n")
        build_all()

    print(f"\nLoaded subjects: {list(config.keys())}")
    print("Type 'quit' to exit, 'rebuild' to re-ingest, or 'debug <query>' to see raw chunks.\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit"):
            break
        if q.lower() == "rebuild":
            build_all()
            continue

        if q.lower().startswith("debug "):
            real_q   = q[6:].strip()
            subjects = detect_subjects(real_q, config)
            sem      = detect_sem_filter(real_q)
            print(f"Router picked: {subjects}")
            print(f"Sem filter: {sem if sem else 'none (searching all)'}")
            for subj in subjects:
                vs     = load_collection(subj)
                k      = get_k(real_q, subj)
                chunks = vs.similarity_search(real_q, k=k)
                print(f"\n[{subj}] top {k} chunks:")
                for i, c in enumerate(chunks):
                    print(f"  Chunk {i+1} ({c.metadata.get('filename')} p.{c.metadata.get('page')}):")
                    print(f"  {c.page_content[:300]}\n")
            continue

        if q:
            print("\n" + ask(q, config) + "\n")