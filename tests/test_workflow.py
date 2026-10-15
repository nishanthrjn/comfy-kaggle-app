from pathlib import Path

import pytest

from comfy_kaggle_app.comfy_client import load_json, prepare_workflow

ROOT = Path(__file__).resolve().parents[1]


def test_workflow_injection():
    source = load_json(ROOT / "workflows/text_to_image.json")
    mapping = load_json(ROOT / "workflows/node_map.json")
    result = prepare_workflow(
        source, mapping, prompt="fox", negative_prompt="blur", seed=99, checkpoint="x.safetensors"
    )
    assert result[mapping["positive"]]["inputs"]["text"] == "fox"
    assert result[mapping["negative"]]["inputs"]["text"] == "blur"
    assert result[mapping["sampler"]]["inputs"]["seed"] == 99
    assert result[mapping["checkpoint"]]["inputs"]["ckpt_name"] == "x.safetensors"
    assert source[mapping["sampler"]]["inputs"]["seed"] == 42


def test_invalid_mapping():
    graph = load_json(ROOT / "workflows/text_to_image.json")
    mapping = load_json(ROOT / "workflows/node_map.json")
    mapping["positive"] = "missing"
    with pytest.raises(ValueError, match="Invalid workflow mapping"):
        prepare_workflow(
            graph, mapping, prompt="fox", negative_prompt="blur", seed=1, checkpoint="x"
        )
