import modal
import subprocess

APP_NAME = "nemotron-llama-cpp"
CACHE_DIR = "/root/.cache/llama.cpp"
VOLUME_NAME = "llama-cache"

MODEL_REPO = "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF"
MODEL_FILE = "NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf"

app = modal.App(APP_NAME)

model_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(
        "ghcr.io/ggml-org/llama.cpp:server-cuda", add_python="3.12"
    )
    .pip_install("starlette", "httpx", "uvicorn")
)


@app.function(
    image=modal.Image.debian_slim(python_version="3.12").pip_install(
        "huggingface_hub[hf_transfer]"
    ),
    volumes={CACHE_DIR: model_vol},
    timeout=600,
)
def download_model():
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=CACHE_DIR,
    )
    model_vol.commit()
    print(f"downloaded {MODEL_FILE} to {CACHE_DIR}")


@app.cls(
    image=image,
    volumes={CACHE_DIR: model_vol},
    gpu="A10G",
    timeout=600,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=100)
class LlamaServer:
    @modal.enter()
    def start(self):
        model_path = f"{CACHE_DIR}/{MODEL_FILE}"
        cmd = [
            "llama-server",
            "--model", model_path,
            "--host", "0.0.0.0",
            "--port", "8080",
            "--n-gpu-layers", "-1",
            "--ctx-size", "8192",
            "--parallel", "4",
        ]
        self.proc = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.proc.terminate()

    @modal.asgi_app()
    def serve(self):
        from starlette.requests import Request
        from starlette.responses import StreamingResponse, Response
        from starlette.applications import Starlette
        from starlette.routing import Route
        import httpx

        LLAMA_URL = "http://127.0.0.1:8080"

        async def proxy(request: Request):
            path = request.url.path
            async with httpx.AsyncClient() as client:
                resp = await client.request(
                    method=request.method,
                    url=f"{LLAMA_URL}{path}",
                    headers=dict(request.headers),
                    content=await request.body(),
                    timeout=120.0,
                )
                if "text/event-stream" in resp.headers.get("content-type", ""):
                    async def stream():
                        async with httpx.AsyncClient() as sc:
                            async with sc.stream(
                                method=request.method,
                                url=f"{LLAMA_URL}{path}",
                                headers=dict(request.headers),
                                content=await request.body(),
                                timeout=120.0,
                            ) as sr:
                                async for chunk in sr.aiter_bytes():
                                    yield chunk
                    return StreamingResponse(stream(), media_type="text/event-stream")
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    headers=dict(resp.headers),
                )

        async def health(request: Request):
            return Response("ok")

        return Starlette(
            routes=[
                Route("/health", health),
                Route("/{path:path}", proxy, methods=["GET", "POST"]),
            ]
        )


@app.local_entrypoint()
def main(download: bool = False):
    if download:
        download_model.remote()
    else:
        print("deploy with: modal deploy modal_llama.py")
        print("download model first: modal run modal_llama.py --download")
