"""Tests for doc ingest pipeline."""

from unittest.mock import patch

import pytest

from src.ingest.pipeline import chunk_text, clean_text, run_ingest


class TestCleanText:
    def test_strips_html_tags(self):
        result = clean_text("<p>Hello <b>world</b></p>")
        assert result == "Hello world"

    def test_normalizes_whitespace(self):
        result = clean_text("hello   world\n\nfoo   bar")
        assert result == "hello world foo bar"

    def test_strips_leading_trailing(self):
        result = clean_text("  hello  ")
        assert result == "hello"

    def test_html_and_whitespace_combined(self):
        result = clean_text("<div>  hello   <span>world</span>  </div>")
        assert result == "hello world"

    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""


class TestChunkText:
    def test_single_chunk_short_text(self):
        text = " ".join(f"word{i}" for i in range(10))
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_splits_with_correct_size(self):
        text = " ".join(f"w{i}" for i in range(100))
        chunks = chunk_text(text, chunk_size=40, overlap=10)
        first_chunk_words = chunks[0].split()
        assert len(first_chunk_words) == 40

    def test_overlap_between_chunks(self):
        text = " ".join(f"w{i}" for i in range(100))
        chunks = chunk_text(text, chunk_size=40, overlap=10)
        first_words = chunks[0].split()
        second_words = chunks[1].split()
        assert first_words[-10:] == second_words[:10]

    def test_empty_text(self):
        assert chunk_text("") == []

    def test_last_chunk_smaller(self):
        text = " ".join(f"w{i}" for i in range(50))
        chunks = chunk_text(text, chunk_size=40, overlap=10)
        assert len(chunks) == 2
        last_words = chunks[-1].split()
        assert len(last_words) <= 40


class TestRunIngest:
    @patch("src.ingest.pipeline.index_doc")
    @patch("src.ingest.pipeline.embed", return_value=[0.1, 0.2, 0.3])
    @patch("src.ingest.pipeline.load_config")
    def test_local_files_indexed(self, mock_config, mock_embed, mock_index, tmp_path):
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        (doc_dir / "file1.txt").write_text("Hello world content here.")
        (doc_dir / "file2.md").write_text("More content in markdown.")

        mock_config.return_value = {
            "elasticsearch": {"index": "test_index"},
            "doc_sources": [{"type": "local", "path": str(doc_dir)}],
        }

        result = run_ingest("dummy.yaml")
        assert result["documents_processed"] == 2
        assert result["chunks_indexed"] >= 2
        assert mock_embed.call_count >= 2
        assert mock_index.call_count >= 2

    @patch("src.ingest.pipeline.index_doc")
    @patch("src.ingest.pipeline.embed", return_value=[0.1])
    @patch("src.ingest.pipeline.load_config")
    def test_returns_accurate_counts(self, mock_config, mock_embed, mock_index, tmp_path):
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        (doc_dir / "single.txt").write_text("short text")

        mock_config.return_value = {
            "elasticsearch": {"index": "idx"},
            "doc_sources": [{"type": "local", "path": str(doc_dir)}],
        }

        result = run_ingest("dummy.yaml")
        assert result["documents_processed"] == 1
        assert result["chunks_indexed"] == 1

    @patch("src.ingest.pipeline.load_config")
    def test_missing_path_raises(self, mock_config):
        mock_config.return_value = {
            "elasticsearch": {"index": "idx"},
            "doc_sources": [{"type": "local", "path": "/nonexistent/path"}],
        }
        with pytest.raises(FileNotFoundError):
            run_ingest("dummy.yaml")

    @patch("src.ingest.pipeline.load_config")
    def test_s3_raises_not_implemented(self, mock_config):
        mock_config.return_value = {
            "elasticsearch": {"index": "idx"},
            "doc_sources": [{"type": "s3", "bucket": "my-bucket"}],
        }
        with pytest.raises(NotImplementedError):
            run_ingest("dummy.yaml")

    @patch("src.ingest.pipeline.index_doc")
    @patch("src.ingest.pipeline.embed", return_value=[0.5])
    @patch("src.ingest.pipeline.load_config")
    def test_recursive_file_discovery(self, mock_config, mock_embed, mock_index, tmp_path):
        doc_dir = tmp_path / "docs"
        sub_dir = doc_dir / "sub"
        sub_dir.mkdir(parents=True)
        (doc_dir / "top.txt").write_text("top level")
        (sub_dir / "nested.md").write_text("nested content")

        mock_config.return_value = {
            "elasticsearch": {"index": "idx"},
            "doc_sources": [{"type": "local", "path": str(doc_dir)}],
        }

        result = run_ingest("dummy.yaml")
        assert result["documents_processed"] == 2

    @patch("src.ingest.pipeline.load_config")
    def test_unknown_source_type_raises_value_error(self, mock_config):
        mock_config.return_value = {
            "elasticsearch": {"index": "idx"},
            "doc_sources": [{"type": "unknown"}],
        }
        with pytest.raises(ValueError, match="Unknown source type"):
            run_ingest("dummy.yaml")

    @patch("src.ingest.pipeline.index_doc")
    @patch("src.ingest.pipeline.embed", return_value=[0.1])
    @patch("src.ingest.pipeline.load_config")
    def test_empty_directory_produces_zero(self, mock_config, mock_embed, mock_index, tmp_path):
        doc_dir = tmp_path / "empty_docs"
        doc_dir.mkdir()

        mock_config.return_value = {
            "elasticsearch": {"index": "idx"},
            "doc_sources": [{"type": "local", "path": str(doc_dir)}],
        }

        result = run_ingest("dummy.yaml")
        assert result["documents_processed"] == 0
        assert result["chunks_indexed"] == 0

    @patch("src.ingest.pipeline.index_doc")
    @patch("src.ingest.pipeline.embed", return_value=[0.1])
    @patch("src.ingest.pipeline.load_config")
    def test_index_doc_receives_content_and_embedding(
        self, mock_config, mock_embed, mock_index, tmp_path
    ):
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        (doc_dir / "file.txt").write_text("test content")

        mock_config.return_value = {
            "elasticsearch": {"index": "my_index"},
            "doc_sources": [{"type": "local", "path": str(doc_dir)}],
        }

        run_ingest("dummy.yaml")
        _, kwargs = mock_index.call_args_list[0]
        if not kwargs:
            args = mock_index.call_args_list[0][0]
            assert args[0] == "my_index"
            body = args[2]
            assert "content" in body
            assert "embedding" in body
