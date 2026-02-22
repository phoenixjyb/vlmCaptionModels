#!/usr/bin/env python3
"""
Caption Models Inference Script

This script should be placed in your external caption models directory 
(e.g., ../vlmCaptionModels/inference.py) with a virtual environment containing
the necessary dependencies for your chosen caption models.

Directory structure should be:
vlmCaptionModels/
├── .venv/                    # Virtual environment with transformers, torch, etc.
├── inference.py              # This script
├── requirements.txt          # Optional: model dependencies
├── models/                   # Optional: downloaded models cache
└── .cache/                   # Optional: transformers cache

The main process will call this script via subprocess with:
python inference.py --provider qwen2.5-vl --model auto --image /path/to/image.png

Expected JSON output format:
{
    "caption": "A detailed description of the image",
    "model": "qwen2.5-vl-7b-instruct",
    "provider": "qwen2.5-vl"
}
"""

import argparse
import json
import sys
import logging
import os
from pathlib import Path
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GPU Configuration - Force RTX 3090 usage
def configure_gpu():
    """Configure GPU device to use RTX 3090 exclusively with validation."""
    try:
        import torch
        if torch.cuda.is_available():
            # Get GPU information
            gpu_count = torch.cuda.device_count()
            logger.info(f"Found {gpu_count} CUDA devices")
            
            # Find RTX 3090 - CRITICAL: Use PyTorch device indices, not nvidia-smi indices
            rtx3090_device = None
            device_mapping = {}
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                device_mapping[i] = gpu_name
                logger.info(f"PyTorch cuda:{i} -> {gpu_name}")
                if "RTX 3090" in gpu_name or "GeForce RTX 3090" in gpu_name:
                    rtx3090_device = i
                    break
            
            if rtx3090_device is not None:
                # VALIDATION: Double-check the device is actually RTX 3090
                selected_gpu_name = torch.cuda.get_device_name(rtx3090_device)
                if "RTX 3090" not in selected_gpu_name and "GeForce RTX 3090" not in selected_gpu_name:
                    logger.error(f"❌ CRITICAL: Device cuda:{rtx3090_device} is not RTX 3090! Found: {selected_gpu_name}")
                    logger.error("Device mapping changed! Manual intervention required.")
                    raise RuntimeError(f"Expected RTX 3090 at cuda:{rtx3090_device}, but found: {selected_gpu_name}")
                
                # Set RTX 3090 as the default device
                torch.cuda.set_device(rtx3090_device)
                # Clear any cached memory
                torch.cuda.empty_cache()
                
                logger.info(f"✅ Configured to use RTX 3090 (PyTorch cuda:{rtx3090_device})")
                logger.info(f"✅ Validated GPU name: {selected_gpu_name}")
                
                # Log all device mappings for future reference
                logger.info("📊 Complete device mapping:")
                for device_id, gpu_name in device_mapping.items():
                    indicator = "🎯" if device_id == rtx3090_device else "⚪"
                    logger.info(f"  {indicator} cuda:{device_id} = {gpu_name}")
                
                return rtx3090_device
            else:
                logger.error("❌ RTX 3090 not found in any PyTorch device!")
                logger.error("Available devices:")
                for device_id, gpu_name in device_mapping.items():
                    logger.error(f"  cuda:{device_id} = {gpu_name}")
                logger.warning("⚠️ Falling back to first available GPU")
                torch.cuda.set_device(0)
                return 0
        else:
            logger.warning("⚠️ CUDA not available, using CPU")
            return None
    except ImportError:
        logger.warning("⚠️ PyTorch not available for GPU configuration")
        return None
    except Exception as e:
        logger.error(f"❌ GPU configuration failed: {e}")
        raise

# Configure GPU on module import
_gpu_device = configure_gpu()

def validate_rtx3090_usage():
    """Runtime validation that we're still using RTX 3090."""
    try:
        import torch
        if torch.cuda.is_available() and _gpu_device is not None:
            current_device = torch.cuda.current_device()
            current_gpu_name = torch.cuda.get_device_name(current_device)
            
            # Validate current device is RTX 3090
            if "RTX 3090" not in current_gpu_name and "GeForce RTX 3090" not in current_gpu_name:
                logger.error(f"❌ CRITICAL: Current device cuda:{current_device} is not RTX 3090!")
                logger.error(f"Expected: RTX 3090, Found: {current_gpu_name}")
                raise RuntimeError(f"GPU configuration changed! Using {current_gpu_name} instead of RTX 3090")
            
            # Validate it matches our detected device
            if current_device != _gpu_device:
                logger.warning(f"⚠️ Device mismatch: Expected cuda:{_gpu_device}, Currently using cuda:{current_device}")
                # Try to reset to correct device
                torch.cuda.set_device(_gpu_device)
                logger.info(f"🔧 Reset to RTX 3090 device: cuda:{_gpu_device}")
            
            logger.debug(f"✅ Validated: Using RTX 3090 at cuda:{current_device}")
            return True
        return False
    except Exception as e:
        logger.error(f"❌ RTX 3090 validation failed: {e}")
        raise

def load_qwen2vl_model(model_name: str):
    """Load Qwen2.5-VL model."""
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        import torch
        import os
        
        # Validate RTX 3090 usage before loading model
        validate_rtx3090_usage()
        
        if model_name == "auto" or model_name == "default":
            # Use absolute path to local model
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_name = os.path.join(script_dir, "models", "qwen2.5-vl-3b-instruct")
        
        logger.info(f"Loading Qwen2.5-VL model: {model_name}")
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=f"cuda:{_gpu_device}" if _gpu_device is not None else "cpu"
        )
        processor = AutoProcessor.from_pretrained(model_name)
        
        return model, processor, model_name
        
    except ImportError as e:
        logger.error("Failed to import Qwen2VL dependencies. Install with: pip install transformers torch qwen-vl-utils")
        raise
    except Exception as e:
        logger.error(f"Failed to load Qwen2.5-VL model: {e}")
        raise

def load_llava_next_model(model_name: str):
    """Load LLaVA-NeXT model."""
    try:
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        import torch
        
        # Validate RTX 3090 usage before loading model
        validate_rtx3090_usage()
        
        if model_name == "auto" or model_name == "default":
            model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
        
        logger.info(f"Loading LLaVA-NeXT model: {model_name}")
        
        processor = LlavaNextProcessor.from_pretrained(model_name)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=f"cuda:{_gpu_device}" if _gpu_device is not None else "auto"
        )
        
        return model, processor, model_name
        
    except ImportError as e:
        logger.error("Failed to import LLaVA-NeXT dependencies. Install with: pip install transformers torch")
        raise
    except Exception as e:
        logger.error(f"Failed to load LLaVA-NeXT model: {e}")
        raise

def load_blip2_model(model_name: str):
    """Load BLIP2 model."""
    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        import torch
        import os
        
        # Validate RTX 3090 usage before loading model
        validate_rtx3090_usage()
        
        if model_name == "auto" or model_name == "default":
            # Use absolute path to local model
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_name = os.path.join(script_dir, "models", "blip2-opt-2.7b")
        
        logger.info(f"Loading BLIP2 model: {model_name}")
        
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=f"cuda:{_gpu_device}" if _gpu_device is not None else "auto"
        )
        
        return model, processor, model_name
        
    except ImportError as e:
        logger.error("Failed to import BLIP2 dependencies. Install with: pip install transformers torch")
        raise
    except Exception as e:
        logger.error(f"Failed to load BLIP2 model: {e}")
        raise

def load_vitgpt2_model(model_name: str):
    """Load ViT-GPT2 image captioning model (lightweight)."""
    try:
        from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
        import torch
        import os

        # Validate RTX 3090 usage before loading model
        validate_rtx3090_usage()

        if model_name == "auto" or model_name == "default":
            model_name = "nlpconnect/vit-gpt2-image-captioning"

        logger.info(f"Loading ViT-GPT2 image captioning model: {model_name}")

        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Use the configured RTX 3090 device
        if _gpu_device is not None and torch.cuda.is_available():
            device = torch.device(f"cuda:{_gpu_device}")
            torch.cuda.set_device(_gpu_device)
        else:
            device = torch.device("cpu")
        model = model.to(device)

        return (model, {"fe": feature_extractor, "tok": tokenizer, "device": device}, model_name)
    except ImportError as e:
        logger.error("Failed to import ViT-GPT2 dependencies. Install with: pip install transformers torch pillow")
        raise
    except Exception as e:
        logger.error(f"Failed to load ViT-GPT2 model: {e}")
        raise

def generate_qwen2vl_caption(model, processor, image, model_name: str) -> str:
    """Generate caption using Qwen2.5-VL."""
    import torch
    
    # Prepare conversation message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in detail."}
            ]
        }
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Process inputs
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=200)
    
    # Trim input tokens and decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0].strip()

def generate_llava_next_caption(model, processor, image, model_name: str) -> str:
    """Generate caption using LLaVA-NeXT."""
    import torch
    
    prompt = "[INST] <image>\nDescribe this image in detail. [/INST]"
    
    inputs = processor(prompt, image, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    
    # Decode only the new tokens (after the input)
    response = processor.decode(output[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response.strip()

def generate_blip2_caption(model, processor, image, model_name: str) -> str:
    """Generate caption using BLIP2."""
    import torch
    
    inputs = processor(image, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200)
    
    caption = processor.decode(out[0], skip_special_tokens=True).strip()
    return caption

def generate_vitgpt2_caption(model, proc_bundle, image, model_name: str) -> str:
    """Generate caption using ViT-GPT2 (lightweight)."""
    import torch

    fe = proc_bundle["fe"]
    tok = proc_bundle["tok"]
    device = proc_bundle["device"]

    # Preprocess image
    pixel_values = fe(images=image, return_tensors="pt").pixel_values.to(device)

    # Generate ids
    with torch.no_grad():
        output_ids = model.generate(pixel_values, max_length=32, num_beams=4)

    # Decode
    caption = tok.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption

def process_vision_info(messages):
    """Process vision info for Qwen2VL."""
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

def generate_caption(provider: str, model_name: str, image_path: str) -> dict:
    """Generate caption for an image using the specified provider."""
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    logger.info(f"Loaded image: {image_path} ({image.size})")
    
    # Load model and generate caption based on provider
    if provider == "qwen2.5-vl":
        model, processor, actual_model_name = load_qwen2vl_model(model_name)
        caption = generate_qwen2vl_caption(model, processor, image, actual_model_name)
    elif provider == "llava-next":
        model, processor, actual_model_name = load_llava_next_model(model_name)
        caption = generate_llava_next_caption(model, processor, image, actual_model_name)
    elif provider == "blip2":
        model, processor, actual_model_name = load_blip2_model(model_name)
        caption = generate_blip2_caption(model, processor, image, actual_model_name)
    elif provider == "vitgpt2":
        model, proc_bundle, actual_model_name = load_vitgpt2_model(model_name)
        caption = generate_vitgpt2_caption(model, proc_bundle, image, actual_model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    logger.info(f"Generated caption: {caption[:100]}...")
    
    return {
        "caption": caption,
        "model": actual_model_name,
        "provider": provider
    }

def main():
    parser = argparse.ArgumentParser(description="Generate image captions using various VLM models")
    parser.add_argument("--provider", required=True, choices=["qwen2.5-vl", "llava-next", "blip2", "vitgpt2"],
                        help="Caption model provider to use")
    parser.add_argument("--model", default="auto", 
                        help="Model name or 'auto' for default")
    parser.add_argument("--image", required=True,
                        help="Path to input image")
    
    args = parser.parse_args()
    
    try:
        result = generate_caption(args.provider, args.model, args.image)
        print(json.dumps(result))
    except Exception as e:
        logger.error(f"Caption generation failed: {e}")
        error_result = {
            "error": str(e),
            "provider": args.provider,
            "model": args.model
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()
