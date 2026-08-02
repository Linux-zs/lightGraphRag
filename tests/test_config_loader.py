from pathlib import Path

from src.config_loader import load_config


def test_default_config_does_not_embed_local_docs_path():
    config = load_config()

    assert config["paths"]["docs_dir"] == "./docs"
    assert "D:/codes" not in config["paths"]["docs_dir"]


def test_local_config_override_is_merged(tmp_path, monkeypatch):
    default = tmp_path / "default.yaml"
    local = tmp_path / "local.yaml"
    default.write_text(
        """
paths:
  data_dir: ./data
  docs_dir: ./docs
siliconflow:
  embed_dim: 1024
""".strip(),
        encoding="utf-8",
    )
    local.write_text(
        """
paths:
  docs_dir: D:/local/docs
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.config_loader._DEFAULT_CONFIG_PATH", default)
    monkeypatch.setenv("LIGHTGRAPHRAG_CONFIG_LOCAL_PATH", str(local))

    config = load_config()

    assert config["paths"]["data_dir"] == "./data"
    assert config["paths"]["docs_dir"] == "D:/local/docs"
    assert config["siliconflow"]["embed_dim"] == 1024


def test_env_config_path_skips_default_local_override(tmp_path, monkeypatch):
    custom = tmp_path / "custom.yaml"
    local = tmp_path / "local.yaml"
    custom.write_text("paths:\n  docs_dir: ./custom-docs\n", encoding="utf-8")
    local.write_text("paths:\n  docs_dir: ./local-docs\n", encoding="utf-8")
    monkeypatch.setenv("LIGHTGRAPHRAG_CONFIG_PATH", str(custom))
    monkeypatch.setenv("LIGHTGRAPHRAG_CONFIG_LOCAL_PATH", str(local))

    config = load_config(Path("ignored.yaml"))

    assert config["paths"]["docs_dir"] == "./custom-docs"
