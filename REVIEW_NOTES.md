# Review notes (static review only)

- No tests, GPU executions, Kaggle sessions or model downloads were run during review.
- API-format SD 1.5 workflow appears internally consistent, but must be accepted by ComfyUI `/prompt` at runtime.
- Existing ComfyUI commit pin was not confirmed against remote Git; clone/checkout may fail.
- Kaggle GPU quotas, accelerator availability, network access and model licensing are account/session dependent.
- `uv pip install -r ComfyUI/requirements.txt` may override Kaggle CUDA/PyTorch; verify CUDA availability after installation before continuing.
- Mock test added for accepted prompt/image retrieval and rejected graph. Run manually.
- A working SD 1.5 pipeline does not automatically support SDXL, FLUX or Qwen-Image. Each needs an appropriate API workflow and model artifacts.
- The app uses synchronous wait for image in an async API call; this is for one demonstration only. Production needs persistent jobs and bounded concurrency.
- Notebook CPU preparation may be lost when changing Kaggle accelerators; rerun only required preparation if runtime resets.

## Manual validation (Windows)

```powershell
uv sync --locked
uv run pytest -v
uv run ruff check .
```

Do not start GPU inference until local tests pass, model input exists and ComfyUI starts successfully.
