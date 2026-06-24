"""Unit tests for GGUF path resolution and parser wiring (no model load)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from voice_control.config import ParserConfig
from voice_control.lfm_g1_gguf import LFMG1GGUFParser, resolve_gguf_path


def test_resolve_gguf_path_file(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"gguf")
    assert resolve_gguf_path(str(gguf)) == str(gguf.resolve())


def test_resolve_gguf_path_directory(tmp_path):
    gguf = tmp_path / "lfm_g1.gguf"
    gguf.write_bytes(b"gguf")
    assert resolve_gguf_path(str(tmp_path)) == str(gguf.resolve())


def test_resolve_gguf_path_missing_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no .gguf"):
        resolve_gguf_path(str(empty))


def test_gguf_parser_generate_and_map():
    end = "<|" + "redacted_tool_call_end_kimi" + "|>"
    raw = f'<|tool_call_start|>[stop(reason="user_request")]{end}'

    mock_llm = MagicMock()
    mock_llm.create_completion.return_value = {"choices": [{"text": raw}]}
    mock_tok = MagicMock()
    mock_tok.apply_chat_template.return_value = "prompt"

    parser = LFMG1GGUFParser(ParserConfig(lfm_gguf_path="/fake/model.gguf"))
    parser._llm = mock_llm
    parser._tokenizer = mock_tok

    results = parser.parse_plan("stop now")
    assert len(results) == 1
    assert results[0].ok is True
    assert results[0].command is not None
    assert results[0].command.tool == "stop"


def test_build_parser_registers_gguf_backend():
    from voice_control.parsers import build_parser

    with patch.object(LFMG1GGUFParser, "__init__", return_value=None):
        p = build_parser(ParserConfig(backend="lfm_g1_gguf"))
        assert p is not None
