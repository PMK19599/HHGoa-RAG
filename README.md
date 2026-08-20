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
    |
    v
BGE-small-en-v1.5
Dense Retrieval
    |
    v
Top-20 Candidates
    |
    v
MS-MARCO Cross-Encoder
Reranking
    |
    v
Top-5 Passages
    |
    v
Guardrail
Dense >= 0.70
AND
Cross >= 7.0
    |
    +------------------+
    |                  |
    v                  v
  REFUSE             ANSWER
                       |
                       v
                 Top-3 Evidence
                       |
                       v
                  GPT-OSS-20B
                       |
                       v
                Grounded Answer

