import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="cuda"
)

prompt = "Answer in one very short sentence: What is a corporation?"

messages = [
    {
        "role": "user",
        "content": prompt
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    text,
    return_tensors="pt"
).to("cuda")

# warm-up
with torch.no_grad():
    model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False
    )

torch.cuda.synchronize()

start = time.perf_counter()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False
    )

torch.cuda.synchronize()

end = time.perf_counter()

generated = outputs[0][inputs["input_ids"].shape[1]:]

answer = tokenizer.decode(
    generated,
    skip_special_tokens=True
)

elapsed_ms = (end - start) * 1000
tokens_generated = len(generated)

print("\nANSWER:")
print(answer)

print("\n=== LOCAL GENERATION ===")
print(f"Generated tokens: {tokens_generated}")
print(f"Total generation: {elapsed_ms:.2f} ms")
print(f"ms/token: {elapsed_ms / max(tokens_generated, 1):.2f}")