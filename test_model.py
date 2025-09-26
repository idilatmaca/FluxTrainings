import torch
from diffusers import FluxPipeline
import os

base_model_id = "black-forest-labs/FLUX.1-dev"
lora_path = "/home/syntonym4090/Desktop/FluxTrainings/examples/research_projects/flux_lora_quantization/idil-train/pytorch_lora_weights.safetensors" #CHANGE THIS WITH YOUR MODEL PATH
output_folder = "style_comparison"
seed = 0

variations = [
		{"description": "", "filename": "person.png"},
    {"description": "with red hair", "filename": "hair_red.png"},
    {"description": "with vibrant blonde hair", "filename": "hair_blonde.png"},
    {"description": "with blue hair", "filename": "hair_blue.png"},
    {"description": "with pixie-cut hair", "filename": "pix_red.png"},
    {"description": "with reading glasses", "filename": "glasses.png"},
    
]

print("Loading pipeline and model...")
os.makedirs(output_folder, exist_ok=True)

pipe = FluxPipeline.from_pretrained(base_model_id, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
pipe.load_lora_weights(lora_path)

generator = torch.Generator(device="cuda").manual_seed(seed)

for variation in variations:
    description = variation["description"]
    filename = variation["filename"]
    prompt = f"sks-idil person {description}"
    image = pipe(prompt=prompt, generator=generator).images[0]

    output_path = os.path.join(output_folder, filename)
    image.save(output_path)
    print(f"--> Saved to {output_path}")

