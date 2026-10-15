import base64
import logging
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .comfy_client import ComfyClient, ComfyExecutionError, load_json, prepare_workflow

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    comfy_mock: bool = True
    comfy_base_url: str = "http://127.0.0.1:8188"
    comfy_checkpoint: str = "v1-5-pruned-emaonly.safetensors"
    comfy_timeout_seconds: int = 420
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    negative_prompt: str = "blurry, distorted, watermark"
    seed: int = Field(default=42, ge=0, le=2**64 - 1)


app = FastAPI(title="ComfyUI Kaggle Integration", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate")
async def generate(request: GenerateRequest) -> dict[str, str | None]:
    settings = Settings()
    if settings.comfy_mock:
        return {"status": "mock", "image_base64": None}

    try:
        template = load_json(ROOT / "workflows" / "text_to_image.json")
        mapping = load_json(ROOT / "workflows" / "node_map.json")
        graph = prepare_workflow(
            template,
            mapping,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
            checkpoint=settings.comfy_checkpoint,
        )
        client = ComfyClient(
            settings.comfy_base_url,
            timeout_seconds=settings.comfy_timeout_seconds,
        )
        image = await client.generate(graph, output_node=mapping["output"])
    except (OSError, ValueError, KeyError, ComfyExecutionError, TimeoutError, httpx.HTTPError):
        logger.exception("Generation failed")
        raise HTTPException(
            status_code=502, detail="Generation failed; inspect server logs"
        ) from None
    return {"status": "completed", "image_base64": base64.b64encode(image).decode("ascii")}
