import re
import json
import torch
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd


# ================================================================
# 1️⃣ Load and Preprocess Policy Text
# ================================================================

SOURCE_PATH = "data/processed/policy_chunks.txt"
print(f"📄 Loading policy text from: {SOURCE_PATH}")

text = Path(SOURCE_PATH).read_text(encoding="utf8", errors="ignore")
text = re.sub(r'\r\n?', '\n', text)
text = re.sub(r'\n{3,}', '\n\n', text)
text = text.strip()

print(f"📄 File length: {len(text):,} characters")


# ================================================================
# 2️⃣ Hybrid Chunking (Section-aware + Token Overlap)
# ================================================================

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

CHUNK_SIZE_TOKENS = 300
OVERLAP_RATIO = 0.2
STEP = int(CHUNK_SIZE_TOKENS * (1 - OVERLAP_RATIO))

heading_pattern = re.compile(
    r'^(Section|Annexure|Appendix)\s+([0-9A-Za-z.\-]+)\.?\s*(.*)$',
    re.MULTILINE
)

headings = []
for m in heading_pattern.finditer(text):
    headings.append({
        "start": m.start(),
        "section_id": m.group(2),
        "section_title": m.group(3).strip()
    })

if not headings:
    raise ValueError("❌ No section headings detected.")

sections = []
for i, h in enumerate(headings):
    start = h["start"]
    end = headings[i + 1]["start"] if i + 1 < len(headings) else len(text)
    sections.append({
        "section_id": h["section_id"],
        "section_title": h["section_title"],
        "text": text[start:end].strip(),
        "prev_section_id": headings[i - 1]["section_id"] if i > 0 else None,
        "next_section_id": headings[i + 1]["section_id"] if i + 1 < len(headings) else None
    })

final_chunks = []
for sec in sections:
    tokens = enc.encode(sec["text"])
    subchunk_id = 1
    for start in range(0, len(tokens), STEP):
        sub_tokens = tokens[start:start + CHUNK_SIZE_TOKENS]
        final_chunks.append({
            "section_id": sec["section_id"],
            "section_title": sec["section_title"],
            "prev_section_id": sec["prev_section_id"],
            "next_section_id": sec["next_section_id"],
            "subchunk_id": subchunk_id,
            "token_count": len(sub_tokens),
            "text": enc.decode(sub_tokens).strip()
        })
        subchunk_id += 1

print(f"✅ Hybrid chunking complete → {len(final_chunks)} chunks")

# ================================================================
# 3️⃣ Chunk Inspection (Saved for Transparency)
# ================================================================

df_chunks = pd.DataFrame([
    {
        "Section_ID": c["section_id"],
        "Section_Title": c["section_title"],
        "Subchunk_ID": c["subchunk_id"],
        "Tokens": c["token_count"],
        "Preview": c["text"][:200].replace("\n", " ")
    }
    for c in final_chunks
])

Path("outputs").mkdir(exist_ok=True)
df_chunks.to_csv("outputs/chunk_summary.csv", index=False)
print("💾 Chunk summary saved to outputs/chunk_summary.csv")

# ================================================================
# 4️⃣ Embeddings + FAISS Index
# ================================================================

from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

texts = [c["text"] for c in final_chunks]
embeddings = []

for i in range(0, len(texts), 32):
    embeddings.append(
        embed_model.encode(texts[i:i + 32], convert_to_numpy=True)
    )

embeddings = np.vstack(embeddings).astype("float32")
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

print(f"📊 FAISS index built with {index.ntotal} vectors")

# ================================================================
# 5️⃣ Reranker + LLM Setup
# ================================================================

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline
)

# --- Reranker ---
reranker_name = "BAAI/bge-reranker-base"
rerank_tokenizer = AutoTokenizer.from_pretrained(reranker_name)
reranker = AutoModelForSequenceClassification.from_pretrained(reranker_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
reranker.to(device).eval()

def rerank(query, docs, top_k=5):
    pairs = ([query] * len(docs), docs)
    inputs = rerank_tokenizer(
        *pairs,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        scores = reranker(**inputs).logits.squeeze(-1).cpu().numpy()

    idxs = scores.argsort()[::-1][:top_k]
    return idxs

# --- LLM ---
llm_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer_llm = AutoTokenizer.from_pretrained(llm_name)
model_llm = AutoModelForCausalLM.from_pretrained(
    llm_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

generator = pipeline(
    "text-generation",
    model=model_llm,
    tokenizer=tokenizer_llm
)

# ================================================================
# 6️⃣ LangGraph Agentic RAG Pipeline
# ================================================================

from langgraph.graph import StateGraph
from typing import TypedDict, List

class RAGState(TypedDict):
    query: str
    retrieved: List[dict]
    reranked: List[dict]
    answer: str

def retrieve_node(state: RAGState):
    q_emb = embed_model.encode([state["query"]], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    _, idxs = index.search(q_emb, 20)
    return {"retrieved": [final_chunks[i] for i in idxs[0]]}

def rerank_node(state: RAGState):
    docs = [d["text"] for d in state["retrieved"]]
    idxs = rerank(state["query"], docs)
    return {"reranked": [state["retrieved"][i] for i in idxs]}

def answer_node(state: RAGState):
    context = "\n\n".join(d["text"] for d in state["reranked"])
    prompt = f"""
You are a professional insurance policy analyst.

Use ONLY the information provided in the CONTEXT.
Do NOT add or infer information.
If the answer is not found, say:
"Not specified in the provided policy text."

CONTEXT:
{context}

QUESTION:
{state["query"]}

ANSWER:
"""
    output = generator(prompt, max_new_tokens=1000, temperature=0.0, do_sample=False)
    return {"answer": output[0]["generated_text"]}

workflow = StateGraph(RAGState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rerank", rerank_node)
workflow.add_node("answer", answer_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "answer")

def get_rag_app():
    return workflow.compile()


if __name__ == "__main__":
    print("✅ Agentic LangGraph RAG pipeline ready")

__all__ = ["app"]


