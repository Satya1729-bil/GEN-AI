"""Embeddings and the FAISS vector index.

Two embedding backends:
- "gemini" (default): Google's Gemini embedding API. Free tier, needs GEMINI_API_KEY,
  no PyTorch, works on any laptop and keeps the Docker image small.
- "local": sentence-transformers on CPU. Works offline, but needs PyTorch
  (install requirements-local.txt). On Windows, PyTorch also needs the
  Microsoft Visual C++ Redistributable.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal, Protocol

import faiss
import numpy as np

from .ingest import Chunk

Kind = Literal["document", "query"]


class EmbedderError(RuntimeError):
    """Raised with a message that is safe to show to the user."""


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str], kind: Kind = "document") -> np.ndarray: ...


class SentenceEmbedder:
    """Local embedding model (runs on CPU, no API key needed, needs PyTorch)."""

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbedderError(
                "The local embedding backend needs PyTorch. Run "
                "'pip install -r requirements-local.txt', or set RX_EMBED_BACKEND=gemini."
            ) from exc
        except OSError as exc:  # Windows DLL errors from PyTorch
            raise EmbedderError(
                "PyTorch could not load on this computer (usually the Microsoft Visual C++ "
                "Redistributable is missing). Set RX_EMBED_BACKEND=gemini to use the Gemini API instead."
            ) from exc

        self.name = f"local:{model_name}"
        self._model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: list[str], kind: Kind = "document") -> np.ndarray:
        vectors = self._model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype="float32")


class GeminiEmbedder:
    """Gemini embedding API. Batches requests and retries when the free-tier limit is hit."""

    def __init__(
        self,
        api_key: str | None,
        model: str = "gemini-embedding-2",
        dimension: int = 768,
        batch_size: int = 32,
        max_retries: int = 5,
        client=None,
        sleep=time.sleep,
    ):
        if client is None:
            if not api_key:
                raise EmbedderError(
                    "GEMINI_API_KEY is not set. Search uses the Gemini embedding API, so add "
                    "your free key to the .env file."
                )
            from google import genai

            client = genai.Client(api_key=api_key)
        self._client = client
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._sleep = sleep
        self.name = f"gemini:{model}:{dimension}"
        # gemini-embedding-001 takes a task type; newer models take the task inside the text.
        self._uses_task_type = "embedding-001" in model

    def _format(self, text: str, kind: Kind) -> str:
        if self._uses_task_type:
            return text
        if kind == "query":
            return f"task: search result | query: {text}"
        return f"title: none | text: {text}"

    def _request(self, batch: list[str], kind: Kind):
        from google.genai import types

        if self._uses_task_type:
            contents = batch
            config = types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT",
                output_dimensionality=self.dimension,
            )
        else:
            # One Content per text, so the API returns one embedding per text.
            contents = [types.Content(parts=[types.Part(text=t)]) for t in batch]
            config = types.EmbedContentConfig(output_dimensionality=self.dimension)
        return self._client.models.embed_content(model=self.model, contents=contents, config=config)

    def _embed_batch(self, batch: list[str], kind: Kind) -> list[list[float]]:
        for attempt in range(self.max_retries + 1):
            try:
                response = self._request(batch, kind)
                vectors = [e.values for e in response.embeddings]
                if len(vectors) != len(batch):
                    raise EmbedderError(
                        f"Gemini returned {len(vectors)} embeddings for {len(batch)} texts."
                    )
                return vectors
            except EmbedderError:
                raise
            except Exception as exc:
                message = str(exc)
                rate_limited = "429" in message or "RESOURCE_EXHAUSTED" in message
                server_error = any(code in message for code in ("500", "503", "UNAVAILABLE"))
                if (rate_limited or server_error) and attempt < self.max_retries:
                    self._sleep(min(2 ** attempt * 2, 60))
                    continue
                if rate_limited:
                    raise EmbedderError(
                        "The Gemini free-tier limit was reached while indexing. Wait a minute and try again."
                    ) from exc
                if "API key" in message or "401" in message or "403" in message:
                    raise EmbedderError("The Gemini API key was rejected. Check GEMINI_API_KEY.") from exc
                if "404" in message or "not found" in message.lower():
                    raise EmbedderError(
                        f"The embedding model '{self.model}' was not found. "
                        "Set RX_GEMINI_EMBED_MODEL to a current model."
                    ) from exc
                raise EmbedderError(f"Gemini embedding request failed: {message[:300]}") from exc
        raise EmbedderError("Gemini embedding request failed after several retries.")

    def encode(self, texts: list[str], kind: Kind = "document") -> np.ndarray:
        formatted = [self._format(t, kind) for t in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(formatted), self.batch_size):
            vectors.extend(self._embed_batch(formatted[start : start + self.batch_size], kind))
        return np.asarray(vectors, dtype="float32")


def make_embedder(settings) -> Embedder:
    """Create the embedder chosen by RX_EMBED_BACKEND."""
    backend = settings.embed_backend.lower()
    if backend == "local":
        return SentenceEmbedder(settings.embed_model)
    if backend == "gemini":
        return GeminiEmbedder(
            api_key=settings.gemini_api_key,
            model=settings.gemini_embed_model,
            dimension=settings.embed_dimension,
        )
    raise EmbedderError(f"Unknown RX_EMBED_BACKEND '{settings.embed_backend}'. Use 'gemini' or 'local'.")


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class VectorIndex:
    """Cosine-similarity search over chunks (inner product on unit vectors)."""

    INDEX_FILE = "faiss.index"
    CHUNKS_FILE = "chunks.json"
    META_FILE = "meta.json"

    def __init__(self, index: faiss.Index, chunks: list[Chunk], meta: dict):
        self.index = index
        self.chunks = chunks
        self.meta = meta

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> "VectorIndex":
        if not chunks:
            raise ValueError("No text chunks to index. Check that the PDFs contain selectable text.")
        vectors = _normalize(embedder.encode([c.text for c in chunks], kind="document"))
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        meta = {
            "embed_model": embedder.name,
            "dimension": int(vectors.shape[1]),
            "chunks": len(chunks),
            "sources": sorted({c.source for c in chunks}),
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return cls(index, chunks, meta)

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / self.INDEX_FILE))
        (folder / self.CHUNKS_FILE).write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False), encoding="utf-8"
        )
        (folder / self.META_FILE).write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    @classmethod
    def exists(cls, folder: Path) -> bool:
        return all((folder / f).exists() for f in (cls.INDEX_FILE, cls.CHUNKS_FILE, cls.META_FILE))

    @classmethod
    def load(cls, folder: Path) -> "VectorIndex":
        index = faiss.read_index(str(folder / cls.INDEX_FILE))
        chunks = [
            Chunk.from_dict(d)
            for d in json.loads((folder / cls.CHUNKS_FILE).read_text(encoding="utf-8"))
        ]
        meta = json.loads((folder / cls.META_FILE).read_text(encoding="utf-8"))
        return cls(index, chunks, meta)

    def matches(self, embedder: Embedder) -> bool:
        """True when this index was built with the same embedder (same vector space)."""
        return self.meta.get("embed_model") == embedder.name

    def search(self, query: str, embedder: Embedder, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if self.index.ntotal == 0 or not query.strip():
            return []
        vector = _normalize(embedder.encode([query], kind="query"))
        k = min(top_k, self.index.ntotal)
        scores, ids = self.index.search(vector, k)
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
