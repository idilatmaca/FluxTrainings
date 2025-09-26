#!/usr/bin/env python3
# evaluate.py (simple, local LoRA path fix)
import os
from datetime import datetime
from typing import Optional

import torch
from diffusers import FluxPipeline


BASE_MODEL_ID = "black-forest-labs/FLUX.1-dev"
SUBJECT = "sks-idil person"

NEGATIVE_PROMPT = ""
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 3.5
HEIGHT = 1024
WIDTH = 768
SEED = 0

VARIATIONS = [
    ("baseline",      "natural headshot, neutral expression",                      ""),
    ("hair_red",      "natural headshot, neutral expression, deep red hair",       ""),
    ("hair_blonde",   "natural headshot, neutral expression, vibrant blonde hair", ""),
    ("hair_blue",     "natural headshot, neutral expression, blue hair",           ""),
    ("hair_pixie",    "natural headshot, neutral expression, short pixie-cut hair",""),
    ("glasses_on",    "natural headshot, neutral expression, wearing reading glasses", ""),
    ("glasses_off",   "natural headshot, neutral expression, without any glasses", ""),
    ("expr_smile",  "natural headshot, subtle smile, lips slightly parted, cheeks gently raised (AU12), eyes relaxed",
                    "exaggerated grin, extreme smile, cartoonish"),

    ("expr_laugh",  "natural headshot, soft laughter, teeth visible, cheeks raised, eye crinkles",
                    "grimace, scream, open-mouth wide, distorted"),

    ("expr_sad",    "natural headshot, subtle sadness, inner eyebrows raised and drawn together (AU1+4), mouth corners slightly downturned (AU15), gaze downward",
                    "smile, happy, grin, cheerful, laugh, excitement"),

    ("expr_angry",  "natural headshot, subtle anger, brows lowered and drawn together (AU4), upper eyelids lowered (AU7), lips pressed (AU24), jaw set",
                    "smile, cheerful, relaxed, surprise, wide eyes"),
    ("eyes_closed", "natural headshot, eyes shut, eyelids fully closed and touching, no iris visible, resting face",
                    "open eyes, visible pupils, visible iris, half-open eyes, squint"),
]


MODELS = [
    {
        "label": "idil_lora_v1",
        "lora_path": "/home/syntonym4090/Desktop/FluxTrainings/examples/research_projects/flux_lora_quantization/idil-train/pytorch_lora_weights.safetensors",
    },
    # {"label": "base", "lora_path": None},
]

OUTPUT_ROOT = "eval_outputs"

def set_deterministic(seed: int = 0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_pipe(lora_path: Optional[str]):
    pipe = FluxPipeline.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.bfloat16)
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    pipe.enable_model_cpu_offload()

    if lora_path:
        if os.path.isfile(lora_path):
            lora_dir = os.path.dirname(lora_path) or "."
            weight_name = os.path.basename(lora_path)
            pipe.load_lora_weights(lora_dir, weight_name=weight_name)
        else:
            pipe.load_lora_weights(lora_path)
    return pipe

def main():
    set_deterministic(SEED)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir = os.path.join(OUTPUT_ROOT, f"eval_{ts}")
    os.makedirs(session_dir, exist_ok=True)

    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(SEED)

    for m in MODELS:
        label, lora_path = m["label"], m["lora_path"]
        print(f"\n==> Model: {label} ({'base' if not lora_path else lora_path})")
        model_out = os.path.join(session_dir, label)
        os.makedirs(model_out, exist_ok=True)

        pipe = load_pipe(lora_path)

        for name, desc, extra_neg in VARIATIONS:
            prompt = f"{SUBJECT} {desc}".strip()
            neg = (NEGATIVE_PROMPT + " " + extra_neg).strip()

            print(f"   -> {name}: '{prompt}'")
            image = pipe(
                prompt=prompt,
                negative_prompt=neg,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                height=HEIGHT,
                width=WIDTH,
                generator=generator,
            ).images[0]

            out_path = os.path.join(model_out, f"{name}.png")
            image.save(out_path)
            print(f"      saved: {out_path}")

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nDone. All outputs: {session_dir}")

if __name__ == "__main__":
    main()
