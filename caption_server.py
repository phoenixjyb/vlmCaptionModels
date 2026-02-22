#!/usr/bin/env python3
"""
Caption Models HTTP Server

Provides HTTP endpoints for image captioning using BLIP2 model.
Designed to replace direct model loading in Main API service.

Usage:
    python caption_server.py --port 8002 --provider blip2

Endpoints:
    POST /caption - Generate caption for uploaded image
    GET /health - Service health check
    GET /model-info - Current model information
"""

import argparse
import json
import logging
import time
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

app = FastAPI(title="Caption Models Service", version="1.0.0")

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup."""
    logger.info("🚀 Starting Caption Models Service...")
    
    # Configure GPU
    configure_gpu()
    
    # Pre-load BLIP2 model
    try:
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
        caption = generate_blip2_caption(image)
        generation_time = time.time() - start_time
        
        return {
            "caption": caption,
            "model": "blip2-opt-2.7b",
            "provider": "blip2",
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
        
        model_loaded = "blip2" in _model_cache
        
        return {
            "status": "healthy",
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
    if "blip2" not in _model_cache:
        return {"status": "not_loaded", "message": "BLIP2 model not loaded"}
    
    model_info = _model_cache["blip2"]
    return {
        "model_name": model_info["model_name"],
        "device": model_info["device"],
        "load_time": model_info["load_time"],
        "provider": "blip2",
        "vram_mode": model_info.get("vram_mode"),
        "max_gpu_gb": model_info.get("max_gpu_gb"),
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Caption Models Service",
        "version": "1.0.0",
        "endpoints": [
            "POST /caption - Generate caption for image",
            "GET /health - Service health check",
            "GET /model-info - Model information"
        ]
    }

def main():
    parser = argparse.ArgumentParser(description="Caption Models HTTP Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind to")
    parser.add_argument("--provider", default="blip2", choices=["blip2"], 
                        help="Caption provider (only BLIP2 supported for now)")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Caption Models Service on {args.host}:{args.port}")
    logger.info(f"Provider: {args.provider}")
    
    uvicorn.run(
        "caption_server:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()
