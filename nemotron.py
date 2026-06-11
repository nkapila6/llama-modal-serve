import os

import modal

APP_NAME = "nemotron-llama-cpp"
CACHE_DIR = "/root/.cache/llama.cpp"
VOLUME_NAME = "llama-cache"

MODEL_REPO = "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF"
MODEL_FILE = "NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf"

app = modal.App(APP_NAME)

model_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-runtime-ubuntu22.04", add_python="3.12"
    )
    .apt_install("libgomp1")
    .pip_install(
        "llama-cpp-python[server]",
        extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/cu124",
    )
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
    secrets=[modal.Secret.from_name("llama-api-key")],
)
@modal.concurrent(max_inputs=100)
class LlamaServer:
    @modal.asgi_app()
    def serve(self):
        from llama_cpp.server.app import create_app
        from llama_cpp.server.settings import ModelSettings, ServerSettings

        return create_app(
            server_settings=ServerSettings(api_key=os.environ["API_KEY"]),
            model_settings=[
                ModelSettings(
                    model=f"{CACHE_DIR}/{MODEL_FILE}",
                    n_gpu_layers=-1,
                    n_ctx=8192,
                )
            ],
        )


@app.local_entrypoint()
def main(download: bool = False):
    if download:
        download_model.remote()
    else:
        print("deploy with: modal deploy main.py")
        print("download model first: modal run main.py --download")
