from types import SimpleNamespace

import numpy as np
import pytest

from rxreveal.index import EmbedderError, GeminiEmbedder, VectorIndex
from rxreveal.ingest import chunk_pages
from rxreveal.rag import answer_question


class FakeModels:
    """Stands in for client.models; records calls and returns one vector per input."""

    def __init__(self, fail_times=0, error="429 RESOURCE_EXHAUSTED"):
        self.calls = []
        self.fail_times = fail_times
        self.error = error

    def embed_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        if self.fail_times:
            self.fail_times -= 1
            raise RuntimeError(self.error)
        out = []
        for item in contents:
            text = item if isinstance(item, str) else item.parts[0].text
            vec = np.zeros(8)
            vec[len(text) % 8] = 1.0
            out.append(SimpleNamespace(values=vec.tolist()))
        return SimpleNamespace(embeddings=out)


def make(models, **kw):
    return GeminiEmbedder(api_key=None, client=SimpleNamespace(models=models), sleep=lambda s: None, **kw)


def test_requires_api_key_without_client():
    with pytest.raises(EmbedderError, match="GEMINI_API_KEY"):
        GeminiEmbedder(api_key=None)


def test_documents_and_queries_use_task_prefixes():
    models = FakeModels()
    emb = make(models)
    emb.encode(["fever"], kind="document")
    emb.encode(["fever"], kind="query")
    doc_text = models.calls[0][1][0].parts[0].text
    query_text = models.calls[1][1][0].parts[0].text
    assert doc_text == "title: none | text: fever"
    assert query_text == "task: search result | query: fever"
    assert models.calls[0][2].output_dimensionality == 768


def test_embedding_001_uses_task_type():
    models = FakeModels()
    emb = make(models, model="gemini-embedding-001")
    emb.encode(["fever"], kind="query")
    _, contents, config = models.calls[0]
    assert contents == ["fever"]
    assert config.task_type == "RETRIEVAL_QUERY"


def test_batches_large_inputs():
    models = FakeModels()
    vectors = make(models, batch_size=10).encode([f"text {i}" for i in range(25)])
    assert vectors.shape == (25, 8)
    assert len(models.calls) == 3


def test_retries_on_rate_limit_then_succeeds():
    models = FakeModels(fail_times=2)
    vectors = make(models).encode(["a", "b"])
    assert vectors.shape == (2, 8)
    assert len(models.calls) == 3


def test_gives_clear_error_when_rate_limit_persists():
    models = FakeModels(fail_times=99)
    with pytest.raises(EmbedderError, match="free-tier limit"):
        make(models, max_retries=2).encode(["a"])


def test_bad_key_is_reported_without_retries():
    models = FakeModels(fail_times=99, error="400 API key not valid")
    with pytest.raises(EmbedderError, match="API key"):
        make(models).encode(["a"])
    assert len(models.calls) == 1


def test_index_records_embedder_and_detects_mismatch(sample_pages, embedder):
    index = VectorIndex.build(chunk_pages(sample_pages, "r.pdf"), make(FakeModels()))
    assert index.meta["embed_model"] == "gemini:gemini-embedding-2:768"
    assert index.matches(make(FakeModels()))
    assert not index.matches(embedder)


def test_search_error_becomes_a_friendly_answer(sample_pages, embedder):
    index = VectorIndex.build(chunk_pages(sample_pages, "r.pdf"), embedder)

    class Broken:
        name = embedder.name

        def encode(self, texts, kind="document"):
            raise EmbedderError("limit reached")

    result = answer_question("asthma", index, Broken(), llm=None, min_score=0.0)
    assert result.mode == "error"
    assert result.error == "limit reached"
