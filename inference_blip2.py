#!/usr/bin/env python3
"""
BLIP2 Caption Inference Script (Alternative to Qwen2.5-VL)
Supports JSON communication for image captioning.
"""

import json
import sys
import base64
import io
import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration


class BLIP2Inference:
    def __init__(self, model_id: str = "Salesforce/blip2-opt-2.7b"):
        """Initialize BLIP2 model for inference."""
        self.model_id = model_id
        
        # Get GPU device from environment or default to RTX 3090 (cuda:1)
        gpu_device = os.environ.get('CAPTION_GPU_DEVICE', '1')
        
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_device}")
            torch.cuda.set_device(int(gpu_device))
            print(f"Using GPU device: cuda:{gpu_device} ({torch.cuda.get_device_name(int(gpu_device))})", file=sys.stderr)
        else:
            self.device = torch.device("cpu")
            print("CUDA not available, using CPU", file=sys.stderr)
            
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load the BLIP2 model."""
        try:
            print(f"Loading BLIP2 model on {self.device}...", file=sys.stderr)
            
            # Load processor
            self.processor = Blip2Processor.from_pretrained(self.model_id)
            
            # Load model with proper device mapping
            if self.device.type == "cuda":
                # Use specific GPU device for RTX 3090
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,  # Use float16 for better VRAM efficiency
                    device_map={"": self.device}  # Force to specific GPU
                )
            else:
                # CPU fallback
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32
                )
                self.model = self.model.to(self.device)
                
            # Set to evaluation mode
            self.model.eval()
            
            # Print VRAM usage if on CUDA
            if self.device.type == "cuda":
                gpu_id = self.device.index
                vram_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                vram_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                print(f"BLIP2 loaded successfully! VRAM usage: {vram_allocated:.1f}GB allocated, {vram_reserved:.1f}GB reserved", file=sys.stderr)
            else:
                print("BLIP2 model loaded successfully on CPU!", file=sys.stderr)
                
            return True
        except Exception as e:
            print(f"Error loading BLIP2 model: {e}", file=sys.stderr)
            return False
    
    def generate_caption(self, image_path: str, prompt: Optional[str] = None) -> str:
        """Generate caption for an image."""
        try:
            # If model failed to load, use stub mode
            if self.model is None:
                return "A photo (BLIP2 model not available)"
            
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            
            # Process image
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=100,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Decode caption
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
            return caption.strip()
            
        except Exception as e:
            print(f"Error generating caption: {e}", file=sys.stderr)
            return f"Error: {str(e)}"


def main():
    """Main function for JSON communication."""
    inference = BLIP2Inference()
    
    # Send ready signal
    print(json.dumps({"status": "loading"}), flush=True)
    
    # Load model
    if not inference.load_model():
        print(json.dumps({"status": "error", "message": "Failed to load BLIP2 model, using stub mode"}), flush=True)
        # Continue with stub mode
    else:
        # Send ready signal
        print(json.dumps({"status": "ready"}), flush=True)
    
    # Process requests
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
                
            try:
                request = json.loads(line)
                
                if request.get("action") == "caption":
                    image_path = request.get("image_path")
                    prompt = request.get("prompt")
                    
                    if not image_path or not Path(image_path).exists():
                        response = {
                            "status": "error",
                            "message": f"Image not found: {image_path}"
                        }
                    else:
                        caption = inference.generate_caption(image_path, prompt)
                        response = {
                            "status": "success",
                            "caption": caption
                        }
                
                elif request.get("action") == "health":
                    response = {
                        "status": "healthy",
                        "model": inference.model_id,
                        "device": inference.device
                    }
                
                elif request.get("action") == "exit":
                    response = {"status": "goodbye"}
                    print(json.dumps(response), flush=True)
                    break
                
                else:
                    response = {
                        "status": "error",
                        "message": f"Unknown action: {request.get('action')}"
                    }
                
                print(json.dumps(response), flush=True)
                
            except json.JSONDecodeError as e:
                error_response = {
                    "status": "error",
                    "message": f"Invalid JSON: {str(e)}"
                }
                print(json.dumps(error_response), flush=True)
            
    except KeyboardInterrupt:
        print(json.dumps({"status": "interrupted"}), flush=True)
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}), flush=True)


if __name__ == "__main__":
    main()
