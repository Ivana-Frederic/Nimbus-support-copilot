from app.rag.chunking import chunk_markdown


def test_splits_on_headings():
    md = "# Heading One\nSome text here.\n\n# Heading Two\nOther text here."
    chunks = chunk_markdown(md, max_chars=1000, overlap_chars=50)
    assert len(chunks) == 2
    assert chunks[0].heading == "Heading One"
    assert "Some text here" in chunks[0].text
    assert chunks[1].heading == "Heading Two"
    assert "Other text here" in chunks[1].text


def test_preserves_preamble_without_heading():
    md = "Intro text before any heading.\n\n# Heading\nBody."
    chunks = chunk_markdown(md)
    assert chunks[0].heading == ""
    assert "Intro text" in chunks[0].text


def test_long_section_is_split_with_overlap():
    body = "word " * 400  # ~2000 chars, well over max_chars
    md = f"# Big Section\n{body}"
    chunks = chunk_markdown(md, max_chars=500, overlap_chars=50)
    assert len(chunks) > 1
    for c in chunks:
        assert c.heading == "Big Section"
        assert len(c.text) <= 500 + len("Big Section") + 2


def test_empty_sections_are_skipped():
    md = "# Empty\n\n# Filled\nSome content."
    chunks = chunk_markdown(md)
    headings = [c.heading for c in chunks]
    assert "Filled" in headings
    assert all(c.text.strip() for c in chunks)


def test_empty_document_yields_no_chunks():
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n  ") == []
