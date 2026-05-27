#import os
#import gradio as gr
#import torch
#from dotenv import load_dotenv
#from transformers import AutoModelForCausalLM, AutoTokenizer

# Load variables from .env into os.environ
#load_dotenv()
#print(f"DEBUG: Active HF_TOKEN starts with: {os.getenv('HF_TOKEN', 'NOT FOUND')[:10]}...")

import os
import logging
import sys
import gradio as gr
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Clear any existing system handlers
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

# 2. Build a dedicated Stream Handler that forces real-time flushing
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

# 3. Apply it to our app logger at the lowest level
logger = logging.getLogger("ARC_GPU_CHAT")
logger.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)

# 3. Load environment variables FIRST so we can check them
load_dotenv()

# 4. Read LOG_LEVEL from .env (fallback to INFO if it isn't set)
env_log_level = os.getenv("LOG_LEVEL", "INFO").upper()

if env_log_level == "DEBUG":
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

logger.info(f"Logging initialized at level: {env_log_level}")
logger.info(f"Active HF_TOKEN starts with: {os.getenv('HF_TOKEN', 'NOT FOUND')[:10]}...")


device = "xpu" if torch.xpu.is_available() else "cpu"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device)

def predict(message, history):
    # This forces stdout to bypass Gradio's worker thread routing entirely
    logger.debug(f"\n🚀 TRACE: Received new message: {message}")
    logger.debug(f"📦 TRACE: Current Raw History Structure: {history}")

    # Formulate chat layout using modern Gradio history format safely
    messages = []
    for msg in history:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            # If Gradio formats content as a dictionary instead of a string
            if isinstance(content, dict):
                content = content.get("text", "")
            elif isinstance(content, list):
                # Fallback if content is a structural media/text list sequence
                content = str(content[0] if content else "")
                
            messages.append({"role": msg["role"], "content": str(content)})
        else:
            # Safe fallback if history somehow uses old format [[user, bot]]
            messages.append({"role": "user", "content": str(msg[0])})
            messages.append({"role": "assistant", "content": str(msg[1])})
            
    messages.append({"role": "user", "content": str(message)})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=150)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    logger.debug(f"🤖 TRACE: Model Answer: {response}\n")
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

gr.ChatInterface(predict).launch(server_name="0.0.0.0")
