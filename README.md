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
export CAPTION_PROVIDER=qwen2.5-vl
```

Or in your .env file:
```
CAPTION_EXTERNAL_DIR=H:\wSpace\vlm-photo-engine\vlmCaptionModels
CAPTION_PROVIDER=qwen2.5-vl
```

## Supported Providers

- `qwen2-vl` - Qwen2.5-VL models (recommended for latest performance)
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
.venv/bin/python inference.py --provider qwen2.5-vl --model auto --image /path/to/test/image.jpg
```

## Model Storage

Models will be automatically downloaded to:
- `models/` directory (if created)
- `.cache/` directory (transformers default)
- Or system-wide cache directory

Large models (7B+) require significant disk space and memory.
