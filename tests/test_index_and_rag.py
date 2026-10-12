from rxreveal.index import VectorIndex
from rxreveal.ingest import chunk_pages
from rxreveal.llm import LLMError
from rxreveal.rag import NOT_FOUND, answer_question, build_prompt, retrieve


def make_index(sample_pages, embedder):
    return VectorIndex.build(chunk_pages(sample_pages, source="records.pdf"), embedder)


def test_search_finds_the_right_patient(sample_pages, embedder):
    index = make_index(sample_pages, embedder)
    hits = index.search("tuberculosis cough fever", embedder, top_k=1)
    assert "Tuberculosis" in hits[0][0].text


def test_save_and_load_round_trip(tmp_path, sample_pages, embedder):
    index = make_index(sample_pages, embedder)
    index.save(tmp_path)
    assert VectorIndex.exists(tmp_path)
    loaded = VectorIndex.load(tmp_path)
    assert loaded.meta["chunks"] == len(index.chunks)
    assert loaded.search("asthma inhaler", embedder, top_k=1)[0][0].text.startswith("Patient 3")


def test_answer_uses_llm_with_numbered_context(sample_pages, embedder, fake_llm):
    index = make_index(sample_pages, embedder)
    result = answer_question("symptoms of tuberculosis", index, embedder, fake_llm, top_k=2, min_score=0.0)
    assert result.mode == "llm"
    assert result.text == fake_llm.reply
    _, prompt = fake_llm.calls[0]
    assert "[1]" in prompt and "Question: symptoms of tuberculosis" in prompt


def test_without_api_key_returns_passages_only(sample_pages, embedder):
    index = make_index(sample_pages, embedder)
    result = answer_question("asthma", index, embedder, llm=None, min_score=0.0)
    assert result.mode == "retrieval_only"
    assert result.sources


def test_no_relevant_passages_returns_not_found(sample_pages, embedder, fake_llm):
    index = make_index(sample_pages, embedder)
    result = answer_question("quantum chromodynamics", index, embedder, fake_llm, min_score=0.5)
    assert result.mode == "not_found"
    assert result.text == NOT_FOUND
    assert not fake_llm.calls


def test_llm_error_falls_back_to_passages(sample_pages, embedder):
    class FailingLLM:
        def generate(self, system, prompt):
            raise LLMError("limit reached")

    index = make_index(sample_pages, embedder)
    result = answer_question("diabetes", index, embedder, FailingLLM(), min_score=0.0)
    assert result.mode == "retrieval_only"
    assert result.error == "limit reached"


def test_prompt_includes_recent_history_only(sample_pages, embedder):
    index = make_index(sample_pages, embedder)
    sources = retrieve("diabetes", index, embedder, top_k=1, min_score=0.0)
    history = [(f"q{i}", f"a{i}") for i in range(5)]
    prompt = build_prompt("and the treatment?", sources, history)
    assert "q4" in prompt and "q2" in prompt and "q1" not in prompt
