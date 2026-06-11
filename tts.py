import io
import os
import struct

import modal

APP_NAME = "qwen-tts"
CACHE_DIR = "/root/.cache/qwen-tts"
VOLUME_NAME = "qwen-tts-cache"
MODEL_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

app = modal.App(APP_NAME)

model_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-runtime-ubuntu22.04", add_python="3.12"
    )
    .apt_install("libgomp1", "libsndfile1", "sox")
    .pip_install(
        "faster-qwen3-tts",
        "torch>=2.5.1",
        "numpy",
        "fastapi",
    )
)


@app.function(
    image=modal.Image.debian_slim(python_version="3.12").pip_install(
        "huggingface_hub[hf_transfer]", "transformers"
    ),
    volumes={CACHE_DIR: model_vol},
    timeout=900,
)
def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=f"{CACHE_DIR}/{MODEL_REPO}",
    )
    model_vol.commit()
    print(f"downloaded {MODEL_REPO} to {CACHE_DIR}")


def _wav_bytes(pcm, sample_rate: int) -> bytes:
    import numpy as np

    raw = np.clip(pcm * 32768, -32768, 32767).astype(np.int16).tobytes()
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(raw)))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate,
              sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(raw)))
    buf.write(raw)
    return buf.getvalue()


@app.cls(
    image=image,
    volumes={CACHE_DIR: model_vol},
    gpu="A10G",
    timeout=600,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("qwen-tts-api-key")],
)
class TTSServer:
    @modal.enter()
    def load_model(self):
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        self.model = FasterQwen3TTS.from_pretrained(
            f"{CACHE_DIR}/{MODEL_REPO}",
            device="cuda",
            dtype=torch.bfloat16,
        )
        self.sample_rate = self.model.sample_rate

    @modal.asgi_app()
    def serve(self):
        import numpy as np
        from fastapi import Depends, FastAPI, HTTPException
        from fastapi.responses import Response, StreamingResponse
        from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
        from pydantic import BaseModel

        web_app = FastAPI(title="Qwen3 TTS VoiceDesign")
        security = HTTPBearer()
        api_key = os.environ["API_KEY"]

        async def verify_key(creds: HTTPAuthorizationCredentials = Depends(security)):
            if creds.credentials != api_key:
                raise HTTPException(401, "invalid api key")

        class TTSRequest(BaseModel):
            text: str
            instruct: str = "A calm, clear female voice speaking naturally."
            language: str = "English"

        @web_app.get("/health")
        async def health():
            return {"status": "ok"}

        @web_app.post("/v1/audio/speech", dependencies=[Depends(verify_key)])
        async def generate_speech(req: TTSRequest):
            if not req.text.strip():
                raise HTTPException(400, "'text' is empty")

            audio_arrays, sr = self.model.generate_voice_design(
                text=req.text,
                instruct=req.instruct,
                language=req.language,
            )

            if not audio_arrays:
                raise HTTPException(500, "no audio generated")

            audio = audio_arrays[0]
            wav = _wav_bytes(audio, sr)

            return Response(content=wav, media_type="audio/wav")

        def _wav_header(sample_rate: int) -> bytes:
            hdr = io.BytesIO()
            hdr.write(b"RIFF")
            hdr.write(struct.pack("<I", 0xFFFFFFFF))
            hdr.write(b"WAVE")
            hdr.write(b"fmt ")
            hdr.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate,
                      sample_rate * 2, 2, 16))
            hdr.write(b"data")
            hdr.write(struct.pack("<I", 0xFFFFFFFF))
            return hdr.getvalue()

        def _pcm16(pcm):
            return np.clip(pcm * 32768, -32768, 32767).astype(np.int16).tobytes()

        @web_app.post("/v1/audio/speech/stream", dependencies=[Depends(verify_key)])
        async def generate_speech_stream(req: TTSRequest):
            if not req.text.strip():
                raise HTTPException(400, "'text' is empty")

            def audio_stream():
                yield _wav_header(self.sample_rate)
                for audio, sr, _ in self.model.generate_voice_design_streaming(
                    text=req.text,
                    instruct=req.instruct,
                    language=req.language,
                ):
                    yield _pcm16(audio)

            return StreamingResponse(audio_stream(), media_type="audio/wav")

        return web_app


@app.local_entrypoint()
def main(download: bool = False):
    if download:
        download_model.remote()
    else:
        print("deploy with: modal deploy qwen_tts.py")
        print("download model first: modal run qwen_tts.py --download")
