# HHGoa-RAG

A guarded Retrieval-Augmented Generation (RAG) system built for the Hacker House Goa RAG challenge.

HHGoa-RAG is designed around a simple principle:

> **Relevant evidence is not automatically sufficient evidence.**

Instead of sending every retrieved passage directly to an LLM, the system uses dense retrieval, cross-encoder relevance scoring, a conservative guardrail, evidence selection, and grounded generation.

It can answer when the retrieved evidence passes the production criteria and refuse when the evidence is not strong enough.

---

## Core Pipeline

```text
User Query / Voice
       │
       ▼
   Query Input
       │
       ▼
BGE-small-en-v1.5
Dense Retrieval
       │
       ▼
Top-5 Candidates
       │
       ▼
MS-MARCO Cross-Encoder
Relevance Scoring
       │
       ▼
Production Guardrail
Dense ≥ 0.70
AND
Cross ≥ 7.0
       │
       ├──────────────► REFUSE
       │
       ▼
Top-3 Evidence
       │
       ▼
Qwen 3.6 27B
Grounded Generation
       │
       ▼
Final Answer