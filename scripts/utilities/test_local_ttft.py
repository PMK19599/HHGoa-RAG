import time
import threading
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="cuda"
)

messages = [
    {
        "role": "user",
        "content": "Answer in one very short sentence: What is a corporation?"
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

streamer = TextIteratorStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)

generation_kwargs = dict(
    **inputs,
    streamer=streamer,
    max_new_tokens=20,
    do_sample=False
)

start = time.perf_counter()

thread = threading.Thread(
    target=model.generate,
    kwargs=generation_kwargs
)

thread.start()

first_token_time = None
answer = ""

for text_piece in streamer:

    if text_piece:

        if first_token_time is None:
            torch.cuda.synchronize()
            first_token_time = time.perf_counter()

        answer += text_piece

thread.join()
torch.cuda.synchronize()

end = time.perf_counter()

print("\nANSWER:")
print(answer)

print("\n=== LOCAL STREAMING LATENCY ===")

if first_token_time:
    print(
        f"TTFT: {(first_token_time - start) * 1000:.2f} ms"
    )

print(
    f"Total: {(end - start) * 1000:.2f} ms"
)