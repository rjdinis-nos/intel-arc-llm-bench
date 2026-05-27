import os
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load variables from .env into os.environ
load_dotenv()
print(f"DEBUG: Active HF_TOKEN starts with: {os.getenv('HF_TOKEN', 'NOT FOUND')[:10]}...")

print("--- Initializing Hugging Face XPU Test ---")

# 1. Force the target device to your Intel Arc GPU
device = "xpu" if torch.xpu.is_available() else "cpu"
print(f"Targeting Hardware Backend: {device.upper()}")

# 2. Pick a lightweight, smart model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
#model_name = "Qwen/Qwen2.5-3B-Instruct"
#model_name = "google/gemma-2-2b-it"

print(f"Downloading/Loading tokenizer and model ({model_name})...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# We use torch_dtype=torch.bfloat16 to activate the fast Intel XMX matrix loops
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16
).to(device)

# 3. Formulate a prompt
prompt = "Explain why the sky is blue in one sentence."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# 4. Generate text on the GPU
print("\nGenerating response on Intel Arc hardware...")
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=50
)

# 5. Decode and display the result
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("\n🤖 Model Response:")
print(response)
