"""Tests for src.config.loader.load_config."""

import textwrap

import pytest

from src.config.loader import load_config

VALID_YAML = textwrap.dedent("""\
    use_case: "test chatbot"
    risk_tolerance:
      deploy_threshold: 0.10
      warn_threshold: 0.25
    evaluation:
      num_prompts: 10
      prompt_categories: [factual_recall]
    doc_sources:
      - type: local
        path: ./docs/
    model:
      provider: bedrock
      model_id: "anthropic.claude-3-sonnet-20240229-v1:0"
    elasticsearch:
      host: "http://localhost:9200"
      index: "trusted_docs"
""")


def _write(tmp_path, content, name="config.yaml"):
    p = tmp_path / name
    p.write_text(content)
    return str(p)


def test_valid_yaml_returns_flattened_thresholds(tmp_path):
    path = _write(tmp_path, VALID_YAML)
    cfg = load_config(path)

    assert cfg["use_case"] == "test chatbot"
    assert cfg["thresholds"] == {"deploy": 0.10, "warn": 0.25, "reject": 0.3}
    assert "risk_tolerance" not in cfg
    assert cfg["model"]["provider"] == "bedrock"
    assert cfg["elasticsearch"]["index"] == "trusted_docs"
    assert isinstance(cfg["doc_sources"], list)


def test_missing_use_case_raises_value_error(tmp_path):
    yaml_content = VALID_YAML.replace('use_case: "test chatbot"\n', "")
    path = _write(tmp_path, yaml_content)

    with pytest.raises(ValueError, match="use_case"):
        load_config(path)


def test_missing_deploy_threshold_raises_value_error(tmp_path):
    yaml_content = VALID_YAML.replace("  deploy_threshold: 0.10\n", "")
    path = _write(tmp_path, yaml_content)

    with pytest.raises(ValueError, match=r"risk_tolerance\.deploy_threshold"):
        load_config(path)


def test_nonexistent_file_raises_file_not_found_error():
    with pytest.raises(FileNotFoundError):
        load_config("/no/such/file.yaml")


def test_none_path_defaults_to_dot_llm_reliability(tmp_path, monkeypatch):
    p = tmp_path / ".llm-reliability.yaml"
    p.write_text(VALID_YAML)
    monkeypatch.chdir(tmp_path)

    cfg = load_config(None)
    assert cfg["use_case"] == "test chatbot"


def test_extra_fields_are_preserved(tmp_path):
    extra = VALID_YAML + 'custom_key: "hello"\n'
    path = _write(tmp_path, extra)

    cfg = load_config(path)
    assert cfg["custom_key"] == "hello"


def test_missing_model_provider_raises_value_error(tmp_path):
    yaml_content = VALID_YAML.replace("  provider: bedrock\n", "")
    path = _write(tmp_path, yaml_content)

    with pytest.raises(ValueError, match=r"model\.provider"):
        load_config(path)


def test_missing_model_id_raises_value_error(tmp_path):
    yaml_content = VALID_YAML.replace('  model_id: "anthropic.claude-3-sonnet-20240229-v1:0"\n', "")
    path = _write(tmp_path, yaml_content)

    with pytest.raises(ValueError, match=r"model\.model_id"):
        load_config(path)


def test_missing_elasticsearch_host_raises_value_error(tmp_path):
    yaml_content = VALID_YAML.replace('  host: "http://localhost:9200"\n', "")
    path = _write(tmp_path, yaml_content)

    with pytest.raises(ValueError, match=r"elasticsearch\.host"):
        load_config(path)


def test_missing_elasticsearch_index_raises_value_error(tmp_path):
    yaml_content = VALID_YAML.replace('  index: "trusted_docs"\n', "")
    path = _write(tmp_path, yaml_content)

    with pytest.raises(ValueError, match=r"elasticsearch\.index"):
        load_config(path)


def test_missing_doc_sources_raises_value_error(tmp_path):
    yaml_content = VALID_YAML.replace("doc_sources:\n  - type: local\n    path: ./docs/\n", "")
    path = _write(tmp_path, yaml_content)

    with pytest.raises(ValueError, match="doc_sources"):
        load_config(path)


def test_missing_warn_threshold_raises_value_error(tmp_path):
    yaml_content = VALID_YAML.replace("  warn_threshold: 0.25\n", "")
    path = _write(tmp_path, yaml_content)

    with pytest.raises(ValueError, match=r"risk_tolerance\.warn_threshold"):
        load_config(path)


def test_empty_yaml_file_raises_error(tmp_path):
    path = _write(tmp_path, "")

    with pytest.raises((ValueError, TypeError)):
        load_config(path)
