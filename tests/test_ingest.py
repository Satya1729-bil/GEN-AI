from rxreveal.ingest import chunk_pages


def test_splits_one_chunk_per_patient(sample_pages):
    chunks = chunk_pages(sample_pages, source="records.pdf")
    patient_chunks = [c for c in chunks if c.text.startswith("Patient")]
    assert len(patient_chunks) == 3
    assert all(c.source == "records.pdf" for c in chunks)


def test_record_that_crosses_a_page_stays_together(sample_pages):
    chunks = chunk_pages(sample_pages, source="records.pdf")
    patient2 = next(c for c in chunks if c.text.startswith("Patient 2"))
    assert "Diabetes" in patient2.text and "Insulin" in patient2.text
    assert patient2.page == 1


def test_page_numbers_follow_the_record_start(sample_pages):
    chunks = chunk_pages(sample_pages, source="records.pdf")
    patient3 = next(c for c in chunks if c.text.startswith("Patient 3"))
    assert patient3.page == 2


def test_documents_without_patient_headings_are_chunked_by_size():
    text = "\n\n".join(f"Paragraph {i}. " + "Clinical guidance text. " * 20 for i in range(10))
    chunks = chunk_pages([(1, text)], source="guide.pdf", size=500, overlap=50)
    assert len(chunks) > 3
    assert all(len(c.text) <= 520 for c in chunks)


def test_chunk_ids_are_unique(sample_pages):
    chunks = chunk_pages(sample_pages, source="records.pdf")
    assert len({c.id for c in chunks}) == len(chunks)
