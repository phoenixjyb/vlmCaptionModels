# VLM Caption Models

This directory contains the external caption models setup for the VLM Photo Engine.

## Setup

This directory was created with the setup script and contains:

- `.venv/` - Python virtual environment with model dependencies
- `inference.py` - Caption generation script  
- `requirements.txt` - Model dependencies
- `models/` - Optional: downloaded model cache
- `.cache/` - Optional: transformers cache directory

## Configuration

To use this external setup, set the environment variable:
```bash
export CAPTION_EXTERNAL_DIR=H:\wSpace\vlm-photo-engine\vlmCaptionModels
export CAPTION_PROVIDER=qwen3-vl
export QWEN2VL_MODEL_NAME=H:\wSpace\vlm-photo-engine\vlmCaptionModels\models\qwen3-vl-8b-instruct
```

Or in your .env file:
```
CAPTION_EXTERNAL_DIR=H:\wSpace\vlm-photo-engine\vlmCaptionModels
CAPTION_PROVIDER=qwen3-vl
QWEN2VL_MODEL_NAME=H:\wSpace\vlm-photo-engine\vlmCaptionModels\models\qwen3-vl-8b-instruct
```

## Supported Providers

- `qwen3-vl` - Qwen3-VL models (recommended for the RTX 3090 quality path)
- `qwen2.5-vl` - Qwen2.5-VL compatibility path
- `llava-next` - LLaVA-NeXT models  
- `blip2` - BLIP2 baseline models

## Manual Setup

If you need to manually install additional dependencies:

```bash
cd H:\wSpace\vlm-photo-engine\vlmCaptionModels
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install <additional-packages>
```

## Testing

Test the setup with:
```bash
cd H:\wSpace\vlm-photo-engine\vlmCaptionModels
.venv/bin/python inference.py --provider qwen3-vl --model models/qwen3-vl-8b-instruct --image /path/to/test/image.jpg
```

## Model Storage

Models will be automatically downloaded to:
- `models/` directory (if created)
- `.cache/` directory (transformers default)
- Or system-wide cache directory

Large models (7B+) require significant disk space and memory.

The preferred Windows model location is
`models/qwen3-vl-8b-instruct`. Keep the full checkpoint on Windows and load it
with 4-bit quantization for the 24 GB RTX 3090. The HTTP service accepts an
optional `prompt` form field on `POST /caption`; `QWEN2VL_PROMPT` remains the
service-level fallback.
