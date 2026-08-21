import os
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

api_key = os.getenv("SARVAM_API_KEY")
if not api_key:
    raise RuntimeError("SARVAM_API_KEY is not set")

client = SarvamAI(api_subscription_key=api_key)

print("Sending audio to Sarvam...")

with open("data/test_question.wav", "rb") as audio:
    response = client.speech_to_text.transcribe(
        file=audio,
        model="saaras:v3"
    )

print()
print("========================================")
print(" SARVAM STT RESULT")
print("========================================")
print(response)
