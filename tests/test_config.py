from pathlib import Path
from src.abm_geometry.config import Config, load_config


def test_config_defaults():
    cfg = Config()
    assert cfg.H == 30
    assert cfg.T == 50
    assert cfg.tau == 0.4


def test_load_config_from_yaml(tmp_path):
    yaml_content = "H: 10\nW: 10\nT: 5\n"
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_content)
    cfg = load_config(p)
    assert cfg.H == 10
    assert cfg.T == 5
    assert cfg.tau == 0.4  # default preserved
