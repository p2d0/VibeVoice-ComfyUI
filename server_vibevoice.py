import io
import os
import sys
import torch
import random
import logging
import numpy as np
import soundfile as sf
import configparser
import re
from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Tuple, Any

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

# --- 1. Config Initialization ---
startup_config = configparser.ConfigParser()
startup_config.optionxform = str
startup_config.read('./custom_nodes/VibeVoice-ComfyUI/config.ini', encoding='utf-8')

class VoiceShuffler:
    def __init__(self):
        self._queue: List[str] = []

    def get_next_voice(self, config_ref_section) -> Tuple[str, str]:
        current_valid_suffixes = []
        for key in config_ref_section:
            if key.startswith('default_file'):
                suffix = key[len('default_file'):]
                current_valid_suffixes.append(suffix)
        if not current_valid_suffixes:
            return None, ""
        self._queue = [s for s in self._queue if s in current_valid_suffixes]
        if not self._queue:
            self._queue = list(current_valid_suffixes)
            random.shuffle(self._queue)
        active_suffix = self._queue.pop(0)
        ref_file = config_ref_section[f'default_file{active_suffix}']
        return ref_file, active_suffix

voice_shuffler = VoiceShuffler()

def get_resolved_param(cfg: configparser.ConfigParser, param_name: str, suffix: str, request_val: Any, default_fallback: Any) -> Any:
    if request_val is not None: return request_val
    if suffix:
        specific_key = f"{param_name}{suffix}"
        if cfg.has_section('Reference') and cfg.has_option('Reference', specific_key):
            val = cfg.get('Reference', specific_key)
            if val.lower() in ['true', 'false']: return cfg.getboolean('Reference', specific_key)
            try: return cfg.getfloat('Reference', specific_key)
            except: return val
    return cfg.get('Inference', param_name, fallback=default_fallback)

# Global model and processor
model = None
processor = None
model_lock = torch.cuda.is_available() and True # Use threading lock if needed

class TTSRequest(BaseModel):
    gen_text: str
    voices: Optional[List[str]] = None
    ref_file_path: Optional[str] = None
    seed: int = 42
    cfg_scale: Optional[float] = None
    cfg_rescale: Optional[float] = None
    diffusion_steps: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    use_sampling: Optional[bool] = None

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
        request_config = configparser.ConfigParser()
        request_config.optionxform = str
        request_config.read('config.ini', encoding='utf-8')

        # 1. Voice Selection
        active_suffix = ""
        use_ref_file = request.ref_file_path
        
        if not use_ref_file:
            if request.voices and len(request.voices) > 0:
                ref_section = request_config['Reference']
                valid_suffixes = [f"_{v}" if not v.startswith('_') else v for v in request.voices 
                                 if f"default_file{f'_{v}' if not v.startswith('_') else v}" in ref_section]
                if valid_suffixes:
                    active_suffix = random.choice(valid_suffixes)
                    use_ref_file = ref_section[f"default_file{active_suffix}"]
            
            if not use_ref_file:
                use_ref_file, active_suffix = voice_shuffler.get_next_voice(request_config['Reference'])

        # 2. Resolve Parameters
        cfg_scale = float(get_resolved_param(request_config, "cfg_scale", active_suffix, request.cfg_scale, 2.0))
        cfg_rescale = float(get_resolved_param(request_config, "cfg_rescale", active_suffix, request.cfg_rescale, 0.75))
        diffusion_steps = int(get_resolved_param(request_config, "diffusion_steps", active_suffix, request.diffusion_steps, 20))
        temperature = float(get_resolved_param(request_config, "temperature", active_suffix, request.temperature, 0.95))
        top_p = float(get_resolved_param(request_config, "top_p", active_suffix, request.top_p, 0.95))
        use_sampling = bool(get_resolved_param(request_config, "use_sampling", active_suffix, request.use_sampling, False))

        # Set seeds
        torch.manual_seed(request.seed)
        np.random.seed(request.seed)
        
        # 3. Prepare Audio Reference
        if use_ref_file and os.path.exists(use_ref_file):
            wav = processor.audio_processor._load_audio_from_path(use_ref_file)
        else:
            # Fallback synthetic noise
            wav = (0.01 * np.random.normal(0, 1, 24000)).astype(np.float32)
            
        if processor.db_normalize:
            wav = processor.audio_normalizer(wav)

        # 4. Prepare Model Inputs
        # VibeVoiceProcessor expects every line to have a speaker prefix.
        # We ensure each line starts with the [N]: format.
        # We add a speaker marker after every period to encourage sentence-by-sentence generation.
        processed_text = request.gen_text.replace('.', '.\n[1]: ')
        lines = [line.strip() for line in processed_text.split('\n') if line.strip()]
        formatted_lines = []
        for line in lines:
            # Check if line already has a valid marker ([N]: or Speaker N:)
            if re.match(r'^(?:Speaker\s+|\[)(\d+)(?:\]|)\s*:\s*(.*)$', line, re.IGNORECASE):
                formatted_lines.append(line)
            else:
                # Default to [1]: if no marker found
                formatted_lines.append(f"[1]: {line}")
        
        formatted_text = "\n".join(formatted_lines)
        print(formatted_text)
        
        inputs = processor(
            [formatted_text],
            voice_samples=[[wav]],
            return_tensors="pt"
        )

        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        model.set_ddpm_inference_steps(diffusion_steps)

        print(f"--- Request: {request.gen_text[:50]}... | Voice: {active_suffix} ---")
        print(f"--- Params: rescale={cfg_rescale} cfg={cfg_scale}, steps={diffusion_steps}, temp={temperature}, top_p={top_p}, sampling={use_sampling} ---")
        
        # Generate
        with torch.no_grad():
            output = model.generate(
                **inputs,
                tokenizer=processor.tokenizer,
                cfg_scale=cfg_scale,
                cfg_rescale=cfg_rescale,
                do_sample=use_sampling,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=None
            )
            
        if hasattr(output, 'speech_outputs') and output.speech_outputs:
            audio_tensor = torch.cat(output.speech_outputs, dim=-1)
            audio_np = audio_tensor.cpu().float().numpy().squeeze()
            
            # Apply normalization to the generated output to prevent loudness drift and clipping
            if processor.audio_normalizer:
                audio_np = processor.audio_normalizer(audio_np)
            
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
