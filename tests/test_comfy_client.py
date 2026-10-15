import json

import httpx
import pytest

from comfy_kaggle_app.comfy_client import ComfyClient, ComfyExecutionError


@pytest.mark.asyncio
async def test_generate_retrieves_png(monkeypatch):
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        if request.url.path == "/prompt":
            payload = json.loads(request.content)
            assert payload["prompt"]["3"]["inputs"]["seed"] == 42
            return httpx.Response(200, json={"prompt_id": "p1", "number": 1, "node_errors": {}})
        if request.url.path == "/history/p1":
            return httpx.Response(
                200,
                json={
                    "p1": {
                        "status": {"status_str": "success"},
                        "outputs": {
                            "9": {
                                "images": [
                                    {"filename": "img.png", "subfolder": "", "type": "output"}
                                ]
                            }
                        },
                    }
                },
            )
        if request.url.path == "/view":
            return httpx.Response(200, content=b"\x89PNG\r\n\x1a\nTEST")
        return httpx.Response(404)

    real_client = httpx.AsyncClient

    def fake_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake_client)

    result = await ComfyClient("http://127.0.0.1:8188").generate(
        {"3": {"inputs": {"seed": 42}}}, output_node="9"
    )
    assert result.startswith(b"\x89PNG\r\n\x1a\n")
    assert seen == [("POST", "/prompt"), ("GET", "/history/p1"), ("GET", "/view")]


@pytest.mark.asyncio
async def test_generate_rejects_node_errors(monkeypatch):
    def handler(request):
        return httpx.Response(
            200, json={"error": {"type": "invalid_prompt"}, "node_errors": {"3": {"errors": []}}}
        )

    real_client = httpx.AsyncClient

    def fake_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake_client)
    with pytest.raises(ComfyExecutionError, match="rejected prompt"):
        await ComfyClient("http://127.0.0.1:8188").generate({}, output_node="9")
