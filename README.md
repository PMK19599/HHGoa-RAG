## HHGoa-RAG



A guarded Retrieval-Augmented Generation (RAG) system built for the Hacker House Goa RAG challenge.



HHGoa-RAG combines dense retrieval, cross-encoder reranking, evidence filtering, and grounded generation to answer questions only when the retrieved evidence is strong enough.



## The Core Idea



A relevant passage is not always enough to answer a question.



HHGoa-RAG therefore treats these as separate stages:



Query → Retrieval → Reranking → Guardrail → Evidence → Generation



The system can either answer from retrieved evidence or refuse when the evidence is insufficient.



---



## Architecture



```text

User Query

&#x20;   │

&#x20;   ▼

BGE-small-en-v1.5

Dense Retrieval

&#x20;   │

&#x20;   ▼

Top-20 Candidates

&#x20;   │

&#x20;   ▼

MS-MARCO Cross-Encoder

Reranking

&#x20;   │

&#x20;   ▼

Top-5 Passages

&#x20;   │

&#x20;   ▼

Guardrail

Dense ≥ 0.70

AND

Cross ≥ 7.0

&#x20;   │

&#x20;   ├──────────────► REFUSE

&#x20;   │

&#x20;   ▼

Top-3 Evidence

&#x20;   │

&#x20;   ▼

GPT-OSS-20B

&#x20;   │

&#x20;   ▼

Final Grounded Answer

