import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import httpx


class ComfyExecutionError(Exception):
    """Invalid or unsuccessful ComfyUI workflow execution."""


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return data


def prepare_workflow(
    template: dict[str, Any],
    mapping: dict[str, str],
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    checkpoint: str,
) -> dict[str, Any]:
    graph = copy.deepcopy(template)
    expected = {
        "positive": ("CLIPTextEncode", "text", prompt),
        "negative": ("CLIPTextEncode", "text", negative_prompt),
        "sampler": ("KSampler", "seed", seed),
        "checkpoint": ("CheckpointLoaderSimple", "ckpt_name", checkpoint),
    }
    for role, (kind, field, value) in expected.items():
        node_id = mapping.get(role)
        node = graph.get(node_id) if node_id else None
        if not isinstance(node, dict) or node.get("class_type") != kind:
            raise ValueError(f"Invalid workflow mapping for {role}: {node_id}")
        inputs = node.get("inputs")
        if not isinstance(inputs, dict) or field not in inputs:
            raise ValueError(f"Missing input {field} for {role}: {node_id}")
        inputs[field] = value
    return graph


class ComfyClient:
    def __init__(self, base_url: str, *, timeout_seconds: int = 420) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def generate(self, graph: dict[str, Any], *, output_node: str) -> bytes:
        async with httpx.AsyncClient(timeout=30) as client:
            queued = await client.post(f"{self.base_url}/prompt", json={"prompt": graph})
            queued.raise_for_status()
            payload = queued.json()
            if payload.get("error") or payload.get("node_errors") or not payload.get("prompt_id"):
                raise ComfyExecutionError(f"ComfyUI rejected prompt: {str(payload)[:800]}")
            prompt_id = payload["prompt_id"]
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self.timeout_seconds
            while loop.time() < deadline:
                response = await client.get(f"{self.base_url}/history/{prompt_id}")
                response.raise_for_status()
                result = response.json().get(prompt_id)
                if result is not None:
                    status = result.get("status", {})
                    if status.get("status_str") != "success":
                        raise ComfyExecutionError(f"Workflow status: {str(status)[:800]}")
                    images = result.get("outputs", {}).get(output_node, {}).get("images", [])
                    if not images:
                        raise ComfyExecutionError(
                            "Successful workflow returned no SaveImage output"
                        )
                    selected = images[0]
                    image_response = await client.get(
                        f"{self.base_url}/view",
                        params={
                            "filename": selected["filename"],
                            "subfolder": selected.get("subfolder", ""),
                            "type": selected.get("type", "output"),
                        },
                    )
                    image_response.raise_for_status()
                    data = image_response.content
                    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                        raise ComfyExecutionError("Received image without expected PNG signature")
                    return data
                await asyncio.sleep(2)
        raise TimeoutError(f"ComfyUI prompt {prompt_id} timed out")
