import io
import os
import sys
import torch
import random
import logging
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
from typing import Optional, List

# Add vvembed and local paths for imports
current_dir = os.getcwd()
vvembed_path = os.path.join(current_dir, 'vvembed')
if vvembed_path not in sys.path:
    sys.path.insert(0, vvembed_path)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VibeVoiceServer")

# Import VibeVoice components
try:
    from vvembed.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vvembed.processor.vibevoice_processor import VibeVoiceProcessor
    from transformers import BitsAndBytesConfig
except ImportError as e:
    logger.error(f"Failed to import VibeVoice dependencies: {e}")
    sys.exit(1)

app = FastAPI(title="VibeVoice Q4 Server")

# Global model and processor
model = None
processor = None
model_lock = torch.cuda.is_available() and True # Use threading lock if needed

class TTSRequest(BaseModel):
    text: str
    ref_audio_path: Optional[str] = None
    seed: int = 42
    cfg_scale: float = 1.3
    diffusion_steps: int = 20
    temperature: float = 0.95
    top_p: float = 0.95
    use_sampling: bool = False

def load_vibevoice_q4(model_path: str):
    global model, processor
    
    logger.info(f"Loading pre-quantized VibeVoice Q4 from {model_path}")
    
    # BitsAndBytesConfig for loading pre-quantized weights
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )
    
    # Check for subfolder structure (often found in Q4 releases)
    actual_load_path = model_path
    sub_4bit = os.path.join(model_path, "4bit")
    if os.path.exists(sub_4bit):
        logger.info("Found 4bit subfolder, adjusting path...")
        # Note: Depending on how the model was saved, we might need to load from parent or subfolder
        # We'll try the directory containing config.json
        if not os.path.exists(os.path.join(model_path, "config.json")) and os.path.exists(os.path.join(sub_4bit, "config.json")):
            actual_load_path = sub_4bit

    # Load Model
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        actual_load_path,
        quantization_config=bnb_config,
        device_map="cuda", # Force CUDA for quantized models
        trust_remote_code=True,
        local_files_only=True
    )
    
    # Load Processor - Needs the Qwen tokenizer
    # We look for it in the standard ComfyUI location for this wrapper
    tokenizer_path = os.path.join(current_dir, "models", "vibevoice", "tokenizer")
    if not os.path.exists(tokenizer_path):
        logger.warning(f"Local tokenizer not found at {tokenizer_path}, will attempt HF fallback.")
        tokenizer_path = "Qwen/Qwen2.5-1.5B"
        
    processor = VibeVoiceProcessor.from_pretrained(
        actual_load_path,
        language_model_pretrained_name=tokenizer_path,
        trust_remote_code=True,
        local_files_only=True if os.path.exists(tokenizer_path) else False
    )
    
    logger.info("VibeVoice7bQ4 loaded successfully")

@app.on_event("startup")
async def startup_event():
    # Priority path for pre-quantized 7B model
    default_model_path = os.path.join(current_dir, "models", "vibevoice", "VibeVoice7bQ4")
    
    # Fallback to Large if 7bQ4 isn't found
    if not os.path.exists(default_model_path):
        default_model_path = os.path.join(current_dir, "models", "vibevoice", "VibeVoice-Large")

    if os.path.exists(default_model_path):
        try:
            load_vibevoice_q4(default_model_path)
        except Exception as e:
            logger.error(f"Failed to load model on startup: {e}")
    else:
        logger.warning(f"No VibeVoice model found at {default_model_path}. Please ensure models are in models/vibevoice/")

@app.post("/generate")
async def generate(request: TTSRequest):
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Set seeds
        torch.manual_seed(request.seed)
        np.random.seed(request.seed)
        
        # Format text
        formatted_text = f"Speaker 1: {request.text}"
        
        # Prepare voice samples
        voice_samples = []
        if request.ref_audio_path and os.path.exists(request.ref_audio_path):
            # Use audio processor to load
            wav = processor.audio_processor._load_audio_from_path(request.ref_audio_path)
            if processor.db_normalize:
                wav = processor.audio_normalizer(wav)
            voice_samples = [wav]
        else:
            # Create synthetic 1s sample as fallback (matching BaseVibeVoiceNode logic)
            t = np.linspace(0, 1.0, 24000, False)
            voice_sample = (0.6 * np.sin(2 * np.pi * 120 * t) * np.exp(-t * 0.3)).astype(np.float32)
            voice_samples = [voice_sample]

        # Process inputs
        inputs = processor(
            [formatted_text],
            voice_samples=[voice_samples],
            return_tensors="pt"
        )
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        model.set_ddpm_inference_steps(request.diffusion_steps)
        
        # Generate
        with torch.no_grad():
            output = model.generate(
                **inputs,
                tokenizer=processor.tokenizer,
                cfg_scale=request.cfg_scale,
                do_sample=request.use_sampling,
                temperature=request.temperature,
                top_p=request.top_p,
                max_new_tokens=None
            )
            
        if hasattr(output, 'speech_outputs') and output.speech_outputs:
            speech_tensors = output.speech_outputs
            audio_tensor = torch.cat(speech_tensors, dim=-1)
            
            # Convert to float32 for soundfile
            audio_np = audio_tensor.cpu().float().numpy()
            if audio_np.ndim > 1:
                audio_np = audio_np.squeeze()
            
            # Buffer output
            buffer = io.BytesIO()
            sf.write(buffer, audio_np, 24000, format='WAV')
            buffer.seek(0)
            
            return Response(content=buffer.read(), media_type="audio/wav")
        else:
            raise Exception("No audio generated")

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
