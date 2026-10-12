"""Retrieve, then generate: the question-answering pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from .index import Embedder, EmbedderError, VectorIndex
from .ingest import Chunk
from .llm import LLM, LLMError

SYSTEM_PROMPT = """You are RxReveal, an assistant that answers questions about a set of \
medical documents.

Rules:
- Answer ONLY from the numbered context passages. Do not use outside knowledge.
- Cite the passages you used with their numbers in square brackets, for example [1] or [2][4].
- If the passages do not contain the answer, say: "I could not find this in the documents."
- When the question asks about a group (for example "patients with tuberculosis"), \
summarise across all relevant passages and give counts where you can.
- Keep answers short and clear. Use bullet points for lists.
- Do not give a diagnosis or treatment advice for the person asking. You only explain \
what the documents say."""

NOT_FOUND = "I could not find this in the documents."


@dataclass
class Source:
    number: int
    chunk: Chunk
    score: float


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    mode: str = "llm"  # "llm", "retrieval_only", "not_found" or "error"
    error: str | None = None


def build_prompt(question: str, sources: list[Source], history: list[tuple[str, str]] | None = None) -> str:
    context = "\n\n".join(
        f"[{s.number}] (source: {s.chunk.source}, page {s.chunk.page})\n{s.chunk.text}" for s in sources
    )
    parts = []
    if history:
        recent = history[-3:]
        convo = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in recent)
        parts.append(f"Earlier conversation (for follow-up questions only):\n{convo}")
    parts.append(f"Context passages:\n{context}")
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


def retrieve(
    question: str, index: VectorIndex, embedder: Embedder, top_k: int, min_score: float
) -> list[Source]:
    hits = index.search(question, embedder, top_k=top_k)
    kept = [(c, s) for c, s in hits if s >= min_score]
    return [Source(number=i, chunk=c, score=s) for i, (c, s) in enumerate(kept, start=1)]


def answer_question(
    question: str,
    index: VectorIndex,
    embedder: Embedder,
    llm: LLM | None,
    top_k: int = 5,
    min_score: float = 0.2,
    history: list[tuple[str, str]] | None = None,
) -> Answer:
    try:
        sources = retrieve(question, index, embedder, top_k, min_score)
    except EmbedderError as exc:
        return Answer(text="Search is not available right now.", mode="error", error=str(exc))
    if not sources:
        return Answer(text=NOT_FOUND, sources=[], mode="not_found")

    if llm is None:
        return Answer(
            text="No Gemini API key is set, so here are the most relevant passages from your documents.",
            sources=sources,
            mode="retrieval_only",
        )

    try:
        text = llm.generate(SYSTEM_PROMPT, build_prompt(question, sources, history))
    except LLMError as exc:
        return Answer(
            text="I found relevant passages but could not generate an answer.",
            sources=sources,
            mode="retrieval_only",
            error=str(exc),
        )
    return Answer(text=text, sources=sources, mode="llm")
