import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL = "openai/gpt-oss-20b"

start = time.perf_counter()

stream = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": "Answer in one very short sentence: What is a corporation?"
        }
    ],
    reasoning_effort="low",
    include_reasoning=False,
    temperature=0,
    max_completion_tokens=30,
    stream=True,
)

first_content_time = None
answer = ""

for chunk in stream:

    if not chunk.choices:
        continue

    content = chunk.choices[0].delta.content

    if content:

        if first_content_time is None:
            first_content_time = time.perf_counter()

        answer += content

end = time.perf_counter()

print("\nANSWER:")
print(answer if answer else "[No content received]")

print("\n=== LATENCY ===")

if first_content_time is not None:
    print(
        f"TTFT: "
        f"{(first_content_time - start) * 1000:.2f} ms"
    )
else:
    print("TTFT: N/A")

print(
    f"Total completion: "
    f"{(end - start) * 1000:.2f} ms"
)