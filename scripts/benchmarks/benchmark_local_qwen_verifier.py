import json
import time
import re
import numpy as np
import faiss
import torch

from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# CONFIG
# ============================================================

DATA = Path("data/sample_hin.jsonl")
INDEX_PATH = Path("data/bge_native.faiss")
META_PATH = Path("data/bge_native_metadata.json")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

NUM_QUERIES = 100

DENSE_K = 20
FINAL_K = 5
VERIFIER_TOP_K = 3

DENSE_THRESHOLD = 0.70
CROSS_THRESHOLD = 7.0

MAX_NEW_TOKENS = 8


# ============================================================
# LOAD
# ============================================================

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)

print("Loading cross-encoder...")
cross_model = CrossEncoder(CROSS_MODEL)

print("Loading FAISS index...")
index = faiss.read_index(str(INDEX_PATH))

print("Loading metadata...")
with META_PATH.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

with DATA.open("r", encoding="utf-8") as f:
    records = [
        json.loads(line)
        for line in f
    ][:NUM_QUERIES]

print(f"Index ready: {index.ntotal} passages")

print()
print("Loading Qwen locally...")

tokenizer = AutoTokenizer.from_pretrained(
    QWEN_MODEL,
    local_files_only=True
)

qwen = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL,
    local_files_only=True
)

qwen.eval()

device = next(qwen.parameters()).device

print(
    f"Qwen device: {device}"
)

print(
    f"Qwen parameters: "
    f"{sum(p.numel() for p in qwen.parameters()):,}"
)


# ============================================================
# WARMUP
# ============================================================

print()
print("Warming up models...")

embed_model.encode(
    ["warmup"],
    normalize_embeddings=True
)

cross_model.predict(
    [["warmup query", "warmup passage"]]
)


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(query):

    q_emb = embed_model.encode(
        [query],
        normalize_embeddings=True
    )

    q_emb = np.asarray(
        q_emb,
        dtype="float32"
    )

    dense_scores, indices = index.search(
        q_emb,
        DENSE_K
    )

    candidates = []

    for dense_score, idx in zip(
        dense_scores[0],
        indices[0]
    ):

        idx = int(idx)

        if idx < 0 or idx >= len(metadata):
            continue

        candidates.append({
            "dense": float(dense_score),
            "cross": 0.0,
            "text": metadata[idx]["text"]
        })

    if not candidates:
        return []

    pairs = [
        [query, candidate["text"]]
        for candidate in candidates
    ]

    scores = cross_model.predict(pairs)

    for candidate, score in zip(
        candidates,
        scores
    ):
        candidate["cross"] = float(score)

    candidates.sort(
        key=lambda x: x["cross"],
        reverse=True
    )

    return candidates[:FINAL_K]


# ============================================================
# PRODUCTION GUARDRAIL
# ============================================================

def production_answer(results):

    if not results:
        return False

    best = results[0]

    return (
        best["dense"] >= DENSE_THRESHOLD
        and best["cross"] >= CROSS_THRESHOLD
    )


# ============================================================
# QWEN VERIFIER
# ============================================================

SYSTEM_PROMPT = """
You are an evidence-support verifier.

Do NOT answer the question.

Determine whether the supplied evidence directly supports
the specific claim requested by the question.

Important:

- Relevance is not enough.
- Keyword overlap is not enough.
- A number in the evidence is not enough if it answers
  a different quantity.
- A topic mention is not enough.
- The evidence must support the requested relationship.
- If the question asks whether A causes B, evidence saying
  B causes A does NOT support it.
- If the evidence is incomplete, choose NOT_SUPPORTED.
- Do not use outside knowledge.

Return exactly one label:

SUPPORTED

or

NOT_SUPPORTED
""".strip()


def verify(query, evidence):

    user_prompt = f"""
Question:
{query}

Evidence:
{evidence}

Does the evidence directly support the specific information
requested by the question?

Return exactly one label:
SUPPORTED
or
NOT_SUPPORTED
""".strip()

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_prompt
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
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    start = time.perf_counter()

    with torch.inference_mode():

        output = qwen.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None
        )

    elapsed = (
        time.perf_counter() - start
    ) * 1000.0

    generated = output[
        0,
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip().upper()

    if "NOT_SUPPORTED" in response:
        decision = False
    elif "SUPPORTED" in response:
        decision = True
    else:
        # Fail closed.
        decision = False

    return decision, elapsed, response


# ============================================================
# EVALUATION
# ============================================================

print()
print("=" * 72)
print("          LOCAL QWEN EVIDENCE VERIFIER")
print("=" * 72)

print()
print(f"Queries:              {len(records)}")
print(f"Dense K:              {DENSE_K}")
print(f"Final K:              {FINAL_K}")
print(f"Verifier evidence K:  {VERIFIER_TOP_K}")
print(f"Dense threshold:      {DENSE_THRESHOLD}")
print(f"Cross threshold:      {CROSS_THRESHOLD}")
print(f"Verifier model:       {QWEN_MODEL}")

print()
print("Running evaluation...")

baseline_correct = 0
baseline_false_answers = 0
baseline_false_refusals = 0

verifier_correct = 0
verifier_false_answers = 0
verifier_false_refusals = 0

verifier_calls = 0
latencies = []

false_answer_catches = 0
true_answers_blocked = 0

interesting = []


for count, record in enumerate(
    records,
    start=1
):

    query = record["eng_query"]

    actual_answerable = bool(
        any(record["is_selected"])
    )

    results = retrieve(query)

    baseline = production_answer(results)

    # --------------------------------------------------------
    # Baseline metrics
    # --------------------------------------------------------

    if baseline == actual_answerable:
        baseline_correct += 1

    if baseline and not actual_answerable:
        baseline_false_answers += 1

    if not baseline and actual_answerable:
        baseline_false_refusals += 1

    # --------------------------------------------------------
    # Verifier only sees baseline ANSWER cases
    # --------------------------------------------------------

    final_answer = baseline

    if baseline and results:

        evidence_parts = []

        for i, result in enumerate(
            results[:VERIFIER_TOP_K],
            start=1
        ):

            evidence_parts.append(
                f"[Evidence {i}]\n"
                f"{result['text']}"
            )

        evidence = "\n\n".join(
            evidence_parts
        )

        try:

            decision, latency, raw = verify(
                query,
                evidence
            )

            verifier_calls += 1
            latencies.append(latency)

            final_answer = decision

            if (
                baseline
                and not actual_answerable
                and not decision
            ):
                false_answer_catches += 1

                interesting.append({
                    "type": "FALSE_ANSWER_CAUGHT",
                    "query": query,
                    "dense": results[0]["dense"],
                    "cross": results[0]["cross"],
                    "response": raw,
                    "evidence": results[0]["text"]
                })

            elif (
                baseline
                and actual_answerable
                and not decision
            ):
                true_answers_blocked += 1

                interesting.append({
                    "type": "TRUE_ANSWER_BLOCKED",
                    "query": query,
                    "dense": results[0]["dense"],
                    "cross": results[0]["cross"],
                    "response": raw,
                    "evidence": results[0]["text"]
                })

        except Exception as e:

            # Fail closed for experiment.
            final_answer = False

            verifier_calls += 1

            interesting.append({
                "type": "VERIFIER_ERROR",
                "query": query,
                "error": str(e)
            })

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    if final_answer == actual_answerable:
        verifier_correct += 1

    if final_answer and not actual_answerable:
        verifier_false_answers += 1

    if not final_answer and actual_answerable:
        verifier_false_refusals += 1

    if count % 10 == 0:
        print(
            f"Processed {count}/{len(records)}"
        )


# ============================================================
# METRICS
# ============================================================

baseline_accuracy = (
    baseline_correct / len(records)
)

verifier_accuracy = (
    verifier_correct / len(records)
)

answerable_count = sum(
    bool(any(r["is_selected"]))
    for r in records
)

unanswerable_count = (
    len(records) - answerable_count
)

baseline_answer_recall = (
    (
        answerable_count
        - baseline_false_refusals
    )
    / answerable_count
)

verifier_answer_recall = (
    (
        answerable_count
        - verifier_false_refusals
    )
    / answerable_count
)

baseline_refusal_precision = (
    (
        unanswerable_count
        - baseline_false_answers
    )
    / unanswerable_count
)

verifier_refusal_precision = (
    (
        unanswerable_count
        - verifier_false_answers
    )
    / unanswerable_count
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 72)
print("                  BASELINE VS QWEN")
print("=" * 72)

print()
print("FROZEN PRODUCTION BASELINE")

print(
    f"Correct answers:       "
    f"{answerable_count - baseline_false_refusals}"
)

print(
    f"Correct refusals:      "
    f"{unanswerable_count - baseline_false_answers}"
)

print(
    f"False answers:         "
    f"{baseline_false_answers}"
)

print(
    f"False refusals:        "
    f"{baseline_false_refusals}"
)

print(
    f"Accuracy:              "
    f"{baseline_accuracy:.3f}"
)

print(
    f"Answer recall:         "
    f"{baseline_answer_recall:.3f}"
)

print(
    f"Refusal precision:     "
    f"{baseline_refusal_precision:.3f}"
)


print()
print("LOCAL QWEN VERIFIER")

print(
    f"Correct answers:       "
    f"{answerable_count - verifier_false_refusals}"
)

print(
    f"Correct refusals:      "
    f"{unanswerable_count - verifier_false_answers}"
)

print(
    f"False answers:         "
    f"{verifier_false_answers}"
)

print(
    f"False refusals:        "
    f"{verifier_false_refusals}"
)

print(
    f"Accuracy:              "
    f"{verifier_accuracy:.3f}"
)

print(
    f"Answer recall:         "
    f"{verifier_answer_recall:.3f}"
)

print(
    f"Refusal precision:     "
    f"{verifier_refusal_precision:.3f}"
)


print()
print("VERIFIER EFFECT")

print(
    f"False answers caught:  "
    f"{false_answer_catches}"
)

print(
    f"True answers blocked:  "
    f"{true_answers_blocked}"
)

print(
    f"Verifier calls:        "
    f"{verifier_calls}"
)


if latencies:

    arr = np.asarray(
        latencies,
        dtype=np.float32
    )

    print()
    print("VERIFIER LATENCY")

    print(
        f"P50:                  "
        f"{np.percentile(arr, 50):.2f} ms"
    )

    print(
        f"P70:                  "
        f"{np.percentile(arr, 70):.2f} ms"
    )

    print(
        f"P100:                 "
        f"{np.max(arr):.2f} ms"
    )

    print(
        f"Mean:                  "
        f"{np.mean(arr):.2f} ms"
    )


# ============================================================
# VERDICT
# ============================================================

print()
print("=" * 72)
print("                         VERDICT")
print("=" * 72)

if (
    verifier_accuracy > baseline_accuracy
    and verifier_false_answers < baseline_false_answers
):

    print()
    print("PROMISING:")
    print(
        f"Accuracy {verifier_accuracy:.3f} "
        f"> baseline {baseline_accuracy:.3f}"
    )

    print(
        "The local Qwen verifier may be worth "
        "a second-stage optimization experiment."
    )

elif verifier_accuracy == baseline_accuracy:

    print()
    print("NO OVERALL ACCURACY IMPROVEMENT.")

    print(
        "Inspect false-answer catches versus "
        "true-answer blocks before considering integration."
    )

else:

    print()
    print("NO IMPROVEMENT.")

    print(
        "Keep the production pipeline frozen."
    )


print()
print(
    "Do NOT modify rag_pipeline.py "
    "based on this experiment alone."
)


# ============================================================
# INTERESTING CASES
# ============================================================

print()
print("=" * 72)
print("                    INTERESTING CASES")
print("=" * 72)

for i, case in enumerate(
    interesting,
    start=1
):

    print()
    print("-" * 72)

    print(
        f"CASE #{i}: {case['type']}"
    )

    print(
        f"Query: {case['query']}"
    )

    if "dense" in case:
        print(
            f"Dense: {case['dense']:.4f}"
        )

        print(
            f"Cross: {case['cross']:.4f}"
        )

        print(
            f"Qwen: {case['response']}"
        )

        print()
        print(
            "Evidence:"
        )

        print(
            case["evidence"][:600]
        )

    if "error" in case:
        print(
            f"Error: {case['error']}"
        )


print()
print("=" * 72)
print("             LOCAL QWEN VERIFIER COMPLETE")
print("=" * 72)
