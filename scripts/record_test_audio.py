import sounddevice as sd
import soundfile as sf
from pathlib import Path

OUTPUT = Path("data/test_question.wav")
SAMPLE_RATE = 16000
DURATION = 6

print("========================================")
print(" HH Goa RAG - Microphone Test")
print("========================================")
print()
print("Speak your question after the countdown.")
print("Question example: What is a corporation?")
print()

for i in range(3, 0, -1):
    print(f"{i}...")
    sd.sleep(1000)

print("??? RECORDING...")
audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
)

sd.wait()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
sf.write(OUTPUT, audio, SAMPLE_RATE)

print("? Recording complete.")
print(f"Saved: {OUTPUT}")
print(f"Sample rate: {SAMPLE_RATE} Hz")
print(f"Duration: {DURATION} seconds")
