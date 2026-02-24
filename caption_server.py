#!/usr/bin/env python3
"""
Caption Models HTTP Server

Provides HTTP endpoints for image captioning using BLIP2 or Qwen-VL models.
Designed to replace direct model loading in Main API service.

Usage:
    python caption_server.py --port 8002 --provider blip2
    python caption_server.py --port 8002 --provider qwen3-vl

Endpoints:
    POST /caption - Generate caption for uploaded image
    GET /health - Service health check
    GET /model-info - Current model information
"""

import argparse
import logging
import time
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os

# Runtime configuration for VRAM usage control
_vram_mode = os.getenv("CAPTION_VRAM_MODE", "balanced").lower()  # balanced|full|auto|cpu
try:
    _max_gpu_gb = int(os.getenv("CAPTION_MAX_GPU_GB", "12"))
except Exception:
    _max_gpu_gb = 12

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model cache
_model_cache = {}
_gpu_device = None
_active_provider = os.getenv("CAPTION_HTTP_PROVIDER", "qwen3-vl").lower()
_active_model_name = os.getenv("CAPTION_HTTP_MODEL", "auto")

app = FastAPI(title="Caption Models Service", version="1.0.0")

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def normalize_provider(provider: str) -> str:
    p = (provider or "qwen3-vl").strip().lower()
    if p in ("qwen", "qwen2-vl", "qwen2.5-vl", "qwen2vl"):
        return "qwen2.5-vl"
    if p in ("qwen3", "qwen3-vl", "qwen3vl"):
        return "qwen3-vl"
    if p == "blip2":
        return "blip2"
    raise ValueError(f"Unsupported provider: {provider}")


def _is_qwen_provider(provider: str) -> bool:
    return provider in ("qwen2.5-vl", "qwen3-vl")


def _provider_cache_key(provider: str) -> str:
    if _is_qwen_provider(provider):
        return "qwen-vl"
    return provider


def resolve_qwen_model_name(model_name: str) -> str:
    if model_name and model_name not in ("auto", "default"):
        return model_name
    env_model = os.getenv("QWEN2VL_MODEL_NAME", "").strip()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_qwen3_8b = os.path.join(script_dir, "models", "qwen3-vl-8b-instruct")
    local_7b = os.path.join(script_dir, "models", "qwen2.5-vl-7b-instruct")
    local_3b = os.path.join(script_dir, "models", "qwen2.5-vl-3b-instruct")
    if env_model:
        return env_model
    if os.path.isdir(local_qwen3_8b):
        return local_qwen3_8b
    if os.path.isdir(local_7b):
        return local_7b
    if os.path.isdir(local_3b):
        return local_3b
    return "Qwen/Qwen3-VL-8B-Instruct"

def configure_gpu():
    """Configure GPU device to use RTX 3090 exclusively."""
    global _gpu_device
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"Found {gpu_count} CUDA devices")
            
            # Find RTX 3090
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                logger.info(f"PyTorch cuda:{i} -> {gpu_name}")
                if "RTX 3090" in gpu_name or "GeForce RTX 3090" in gpu_name:
                    _gpu_device = i
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
                    logger.info(f"✅ Configured to use RTX 3090 (PyTorch cuda:{i})")
                    return i
            
            logger.warning("⚠️ RTX 3090 not found, using default GPU")
            _gpu_device = 0
            return 0
        else:
            logger.warning("⚠️ CUDA not available, using CPU")
            return None
    except Exception as e:
        logger.error(f"❌ GPU configuration failed: {e}")
        return None

def load_blip2_model():
    """Load BLIP2 model on RTX 3090."""
    if "blip2" in _model_cache:
        return _model_cache["blip2"]
    
    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        import torch
        
        # Use local model path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "models", "blip2-opt-2.7b")
        
        logger.info(f"Loading BLIP2 model from: {model_path}")
        start_time = time.time()
        
        processor = Blip2Processor.from_pretrained(model_path)

        # Choose VRAM placement strategy based on environment
        chosen_mode = _vram_mode if _gpu_device is not None else "cpu"
        use_accelerate = False
        try:
            import accelerate  # noqa: F401
            use_accelerate = True
        except Exception:
            use_accelerate = False

        if chosen_mode in ("balanced", "auto") and use_accelerate and _gpu_device is not None:
            offload_dir = os.path.join(model_path, "offload")
            try:
                os.makedirs(offload_dir, exist_ok=True)
            except Exception:
                offload_dir = None
            max_memory = {
                f"cuda:{_gpu_device}": f"{_max_gpu_gb}GiB",
                "cpu": f"{max(_max_gpu_gb * 4, 16)}GiB",
            }
            logger.info(
                f"Using balanced VRAM mode with max GPU {max_memory[f'cuda:{_gpu_device}']} (offload_folder={offload_dir})"
            )
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory=max_memory,
                offload_folder=offload_dir,
            )
        elif chosen_mode == "full" and _gpu_device is not None:
            logger.info(f"Using full-GPU mode on cuda:{_gpu_device}")
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            ).to(f"cuda:{_gpu_device}")
        elif _gpu_device is not None:
            logger.info("Using default auto device_map placement")
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            logger.info("GPU not available; loading on CPU")
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
            )
        
        load_time = time.time() - start_time
        logger.info(f"✅ BLIP2 model loaded in {load_time:.1f}s")
        
        _model_cache["blip2"] = {
            "model": model,
            "processor": processor,
            "model_name": "blip2-opt-2.7b",
            "load_time": load_time,
            "device": f"cuda:{_gpu_device}" if _gpu_device is not None else "cpu",
            "vram_mode": chosen_mode,
            "max_gpu_gb": _max_gpu_gb,
        }
        
        return _model_cache["blip2"]
        
    except Exception as e:
        logger.error(f"❌ Failed to load BLIP2 model: {e}")
        raise


def load_qwen2vl_model():
    """Load Qwen VL model on RTX 3090."""
    cache_key = "qwen-vl"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    try:
        import torch
        from transformers import AutoProcessor, AutoConfig
        import transformers as _tf

        model_name = resolve_qwen_model_name(_active_model_name)
        logger.info(f"Loading Qwen VL model from: {model_name}")
        start_time = time.time()

        model_type = None
        try:
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            model_type = getattr(cfg, "model_type", None)
        except Exception as e:
            logger.warning(f"Could not resolve model_type from config: {e}")

        class_candidates = []
        if model_type == "qwen3_vl":
            class_candidates = [
                "Qwen3VLForConditionalGeneration",
                "Qwen2_5_VLForConditionalGeneration",
                "Qwen2VLForConditionalGeneration",
            ]
        elif model_type == "qwen2_5_vl":
            class_candidates = [
                "Qwen2_5_VLForConditionalGeneration",
                "Qwen3VLForConditionalGeneration",
                "Qwen2VLForConditionalGeneration",
            ]
        else:
            class_candidates = [
                "Qwen3VLForConditionalGeneration",
                "Qwen2_5_VLForConditionalGeneration",
                "Qwen2VLForConditionalGeneration",
            ]

        _QwenModel = None
        qwen_class = None
        for class_name in class_candidates:
            if hasattr(_tf, class_name):
                _QwenModel = getattr(_tf, class_name)
                qwen_class = class_name
                break
        if _QwenModel is None:
            raise RuntimeError("No compatible Qwen VL model class found in transformers")
        logger.info(f"Using transformer class: {qwen_class} (model_type={model_type})")

        use_4bit = os.getenv("QWEN2VL_LOAD_IN_4BIT", "true").lower() in ("1", "true", "yes")
        quant_type = os.getenv("QWEN2VL_4BIT_QUANT_TYPE", "nf4").strip() or "nf4"

        quantization_config = None
        if use_4bit and _gpu_device is not None:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=quant_type,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                logger.info(f"Using Qwen 4-bit quantization (type={quant_type})")
            except Exception as e:
                logger.warning(f"4-bit requested but unavailable: {e}. Falling back to bf16.")

        if quantization_config is not None:
            model = _QwenModel.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=f"cuda:{_gpu_device}" if _gpu_device is not None else "cpu",
                trust_remote_code=True,
            )
        else:
            model = _QwenModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if _gpu_device is not None else torch.float32,
                device_map=f"cuda:{_gpu_device}" if _gpu_device is not None else "cpu",
                trust_remote_code=True,
            )

        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        load_time = time.time() - start_time
        logger.info(f"✅ Qwen VL model loaded in {load_time:.1f}s ({qwen_class})")

        _model_cache[cache_key] = {
            "model": model,
            "processor": processor,
            "model_name": model_name,
            "load_time": load_time,
            "device": f"cuda:{_gpu_device}" if _gpu_device is not None else "cpu",
            "model_class": qwen_class,
            "load_in_4bit": bool(quantization_config is not None),
            "quant_type": quant_type if quantization_config is not None else None,
        }
        return _model_cache[cache_key]
    except Exception as e:
        logger.error(f"❌ Failed to load Qwen VL model: {e}")
        raise


def process_vision_info(messages):
    image_inputs = []
    video_inputs = []
    for message in messages:
        if isinstance(message.get("content"), list):
            for content in message["content"]:
                if content.get("type") == "image":
                    image_inputs.append(content["image"])
                elif content.get("type") == "video":
                    video_inputs.append(content["video"])
    return image_inputs, video_inputs


def generate_blip2_caption(image: Image.Image) -> str:
    """Generate caption using BLIP2 model."""
    try:
        import torch
        
        model_info = load_blip2_model()
        model = model_info["model"]
        processor = model_info["processor"]
        
        # Process image
        inputs = processor(image, return_tensors="pt")
        if _gpu_device is not None:
            inputs = {k: v.to(f"cuda:{_gpu_device}") for k, v in inputs.items()}
        
        # Generate caption
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
        
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # Clean up BLIP2 prefix if present
        if caption.lower().startswith("a picture of"):
            caption = caption[12:].strip()
        elif caption.lower().startswith("this is"):
            caption = caption[7:].strip()
        
        return caption
        
    except Exception as e:
        logger.error(f"❌ Caption generation failed: {e}")
        raise


def generate_qwen2vl_caption(image: Image.Image) -> str:
    """Generate caption using Qwen2.5-VL."""
    try:
        import torch

        model_info = load_qwen2vl_model()
        model = model_info["model"]
        processor = model_info["processor"]

        prompt = os.getenv("QWEN2VL_PROMPT", "Describe this image in detail.")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        proc_kwargs = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt",
        }
        if image_inputs:
            proc_kwargs["images"] = image_inputs
        if video_inputs:
            proc_kwargs["videos"] = video_inputs

        inputs = processor(**proc_kwargs)
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=200)

        if hasattr(generated_ids, "ndim") and generated_ids.ndim == 1:
            generated_ids = generated_ids.unsqueeze(0)

        trimmed = []
        try:
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids):
                start = int(len(in_ids))
                trimmed.append(out_ids[start:] if int(len(out_ids)) > start else out_ids)
        except Exception:
            trimmed = generated_ids

        decoded = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        caption = (decoded[0] if decoded else "").strip()
        if caption:
            return caption
        fallback = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return (fallback[0] if fallback else "").strip()
    except Exception as e:
        logger.error(f"❌ Qwen caption generation failed: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup."""
    global _active_provider
    try:
        _active_provider = normalize_provider(_active_provider)
    except Exception:
        logger.warning(f"Invalid provider '{_active_provider}', falling back to blip2")
        _active_provider = "blip2"
    logger.info("🚀 Starting Caption Models Service...")
    logger.info(f"Provider: {_active_provider} | Model: {_active_model_name}")

    # Configure GPU
    configure_gpu()

    # Pre-load selected model
    try:
        if _is_qwen_provider(_active_provider):
            load_qwen2vl_model()
        else:
            load_blip2_model()
        logger.info("✅ Caption Models Service ready")
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}")
        # Don't fail startup, allow lazy loading

@app.post("/caption")
async def generate_caption_endpoint(file: UploadFile = File(...)):
    """Generate caption for uploaded image."""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        start_time = time.time()
        if _is_qwen_provider(_active_provider):
            caption = generate_qwen2vl_caption(image)
            model_name = _model_cache.get("qwen-vl", {}).get("model_name", resolve_qwen_model_name(_active_model_name))
        else:
            caption = generate_blip2_caption(image)
            model_name = "blip2-opt-2.7b"
        generation_time = time.time() - start_time
        
        return {
            "caption": caption,
            "model": model_name,
            "provider": _active_provider,
            "generation_time_seconds": round(generation_time, 2),
            "device": f"cuda:{_gpu_device}" if _gpu_device is not None else "cpu"
        }
        
    except Exception as e:
        logger.error(f"Caption generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Service health check."""
    try:
        import torch
        
        gpu_available = torch.cuda.is_available()
        gpu_device = None
        gpu_memory = None
        
        if gpu_available and _gpu_device is not None:
            gpu_device = torch.cuda.get_device_name(_gpu_device)
            props = torch.cuda.get_device_properties(_gpu_device)
            gpu_memory = f"{props.total_memory / 1024**3:.1f} GB"
        
        model_loaded = _provider_cache_key(_active_provider) in _model_cache
        
        return {
            "status": "healthy",
            "active_provider": _active_provider,
            "models_loaded": list(_model_cache.keys()),
            "gpu_available": gpu_available,
            "gpu_device": gpu_device,
            "gpu_memory": gpu_memory,
            "current_device": f"cuda:{_gpu_device}" if _gpu_device is not None else "cpu",
            "model_cache_ready": model_loaded
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/model-info")
async def model_info():
    """Get current model information."""
    cache_key = _provider_cache_key(_active_provider)
    if cache_key not in _model_cache:
        return {"status": "not_loaded", "provider": _active_provider, "message": f"{_active_provider} model not loaded"}

    model_info = _model_cache[cache_key]
    return {
        "model_name": model_info["model_name"],
        "device": model_info["device"],
        "load_time": model_info["load_time"],
        "provider": _active_provider,
        "vram_mode": model_info.get("vram_mode"),
        "max_gpu_gb": model_info.get("max_gpu_gb"),
        "model_class": model_info.get("model_class"),
        "load_in_4bit": model_info.get("load_in_4bit"),
        "quant_type": model_info.get("quant_type"),
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Caption Models Service",
        "version": "1.0.0",
        "active_provider": _active_provider,
        "endpoints": [
            "POST /caption - Generate caption for image",
            "GET /health - Service health check",
            "GET /model-info - Model information"
        ]
    }

def main():
    global _active_provider, _active_model_name
    parser = argparse.ArgumentParser(description="Caption Models HTTP Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind to")
    parser.add_argument(
        "--provider",
        default="qwen3-vl",
        choices=["blip2", "qwen2.5-vl", "qwen2-vl", "qwen3-vl", "qwen3", "qwen"],
        help="Caption provider",
    )
    parser.add_argument("--model", default="auto", help="Model name or 'auto'")
    
    args = parser.parse_args()
    _active_provider = normalize_provider(args.provider)
    _active_model_name = args.model
    os.environ["CAPTION_HTTP_PROVIDER"] = _active_provider
    os.environ["CAPTION_HTTP_MODEL"] = _active_model_name

    logger.info(f"Starting Caption Models Service on {args.host}:{args.port}")
    logger.info(f"Provider: {_active_provider}")
    logger.info(f"Model: {_active_model_name}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()
