# llama-modal-serve

GPU model serving on [Modal](https://modal.com). Two apps:

- **nemotron.py** -- NVIDIA Nemotron-3-Nano-4B (text generation, OpenAI-compatible API via llama-cpp-python)
- **tts.py** -- Qwen3-TTS-12Hz-1.7B-VoiceDesign (text-to-speech with natural language voice descriptions)

## Setup

```bash
pip install uv
uv sync
```

You'll need a [Modal](https://modal.com) account and the CLI authenticated (`uv run modal token set`).

## Nemotron (text generation)

Serves a quantized Nemotron-3-Nano-4B GGUF model via llama-cpp-python's OpenAI-compatible server on an A10G GPU.

```bash
# create the api key secret
modal secret create llama-api-key API_KEY=your-key-here

# download the model to a modal volume
uv run modal run nemotron.py --download

# deploy
uv run modal deploy nemotron.py
```

### Usage

```bash
curl https://<your-url>/v1/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your-key-here" \
     -d '{
       "model": "NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf",
       "messages": [{"role": "user", "content": "Hello!"}]
     }'
```

Other endpoints: `/v1/models`, `/v1/completions`, `/docs`

## Qwen TTS (voice design)

Serves Qwen3-TTS-12Hz-1.7B-VoiceDesign via [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) on an A10G GPU. Describe any voice in natural language and generate speech.

```bash
# create the api key secret
modal secret create qwen-tts-api-key API_KEY=your-key-here

# download the model to a modal volume
uv run modal run tts.py --download

# deploy
uv run modal deploy tts.py
```

### Usage

```bash
curl -X POST https://<your-url>/v1/audio/speech \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your-key-here" \
     -d '{
       "text": "Welcome to the show, everyone.",
       "instruct": "Warm, confident male narrator with a slight British accent.",
       "language": "English"
     }' --output speech.wav
```

Other endpoints: `/health`, `/docs`
