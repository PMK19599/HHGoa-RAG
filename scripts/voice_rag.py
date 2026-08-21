import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from sarvamai import SarvamAI


# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from rag_pipeline import run_query


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

AUDIO_PATH = PROJECT_ROOT / "data" / "test_question.wav"
STT_MODEL = "saaras:v3"


# ------------------------------------------------------------
# SARVAM STT
# ------------------------------------------------------------

def transcribe_audio(client, audio_path):
    start = time.perf_counter()

    with audio_path.open("rb") as audio:
        response = client.speech_to_text.transcribe(
            file=audio,
            model=STT_MODEL
        )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    return response, elapsed_ms


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("SARVAM_API_KEY")

    if not api_key:
        raise RuntimeError(
            "SARVAM_API_KEY is not set."
        )

    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_PATH}\n"
            "Run scripts\\record_test_audio.py first."
        )

    client = SarvamAI(
        api_subscription_key=api_key
    )

    print()
    print("=" * 60)
    print("              HH GOA VOICE RAG")
    print("=" * 60)

    print()
    print(f"Audio:     {AUDIO_PATH}")
    print(f"STT model: {STT_MODEL}")

    # --------------------------------------------------------
    # End-to-end timer starts BEFORE STT
    # --------------------------------------------------------

    overall_start = time.perf_counter()

    print()
    print("Transcribing audio...")

    response, stt_ms = transcribe_audio(
        client,
        AUDIO_PATH
    )

    transcript = response.transcript.strip()

    if not transcript:
        print()
        print("STT returned an empty transcript.")
        return

    print()
    print("=== TRANSCRIPT ===")
    print(transcript)

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    print()
    print("Running RAG...")

    rag_result = run_query(
        transcript,
        verbose=True
    )

    # --------------------------------------------------------
    # End-to-end latency
    # --------------------------------------------------------

    overall_ms = (
        time.perf_counter() - overall_start
    ) * 1000

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("              VOICE LATENCY")
    print("=" * 60)

    print(
        f"STT:                 {stt_ms:.2f} ms"
    )

    print(
        f"RAG total:           "
        f"{rag_result['total_ms']:.2f} ms"
    )

    print(
        f"End-to-end:          "
        f"{overall_ms:.2f} ms"
    )

    print(
        f"Decision:            "
        f"{rag_result['decision']}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
