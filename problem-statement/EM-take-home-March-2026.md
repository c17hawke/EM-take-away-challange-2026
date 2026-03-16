# Enterprise Minds - AI Engineer Take-Home Challenge

## Agentic RAG Chatbot for Regulatory Documents

**Role:** AI Engineer – Legal & Regulatory LLM Solutions  
**Time Budget:** 2–3 days (~12–18 hours of focused work)  
**Submission:** Public GitHub repository — share the link with the hiring team  

> **Important:** A submission without a README architecture/workflow diagram will be considered incomplete, regardless of code quality. The repository must be self-contained so reviewers can run `docker-compose up --build` after populating `.env` from `.env.example`.
>
> **AI-Assisted Development Policy:** You are allowed (and encouraged) to use AI-based IDEs/tools (for example, GitHub Copilot, Cursor, or similar) while completing this challenge. However, you must be able to clearly explain your end-to-end workflow and justify major technical decisions (for example, model choice, chunking strategy, retrieval design, guardrails, and evaluation approach) in your README and F2F discussion.

---

## Problem Statement

Organisations operating AI systems today face a growing and fragmented regulatory landscape. Frameworks such as the NIST AI Risk Management Framework and legislation such as the EU AI Act impose specific obligations — around transparency, risk classification, human oversight, and documentation — that legal, compliance, and engineering teams must understand and act on quickly.

The challenge is that these documents are dense, multi-hundred-page artefacts written in technical and legal language. Finding a precise answer — for example, which NIST function governs a particular risk control, or which EU AI Act Article sets out transparency obligations for high-risk systems — requires navigating complex structure, cross-references, and evolving definitions.

A naive keyword search or a generic chatbot fails here. What is needed is a system that can retrieve grounded, traceable evidence from multiple source types, reason across documents, and surface citations that a compliance professional can actually verify. This challenge asks you to build exactly that.

---

## Objective

Build a working proof-of-concept Agentic RAG Chatbot capable of answering questions grounded in two real-world regulatory documents. The system must demonstrate clean agentic orchestration, context-aware reasoning, grounded and traceable responses, observability through tracing, and containerization. A basic Q&A chain is not sufficient.

---

## Required Source Documents

You must ingest **both** of the following sources. No substitutions are permitted.

| Type | Source | URL |
|---|---|---|
| PDF | NIST AI Risk Management Framework (NIST.AI.100-1) | https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf |
| Website | EU AI Act – Official Journal (OJ:L_202401689) | https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689 |

> Handling two different source types (PDF and live HTML) is intentional. It tests your ability to design a flexible, source-agnostic ingestion pipeline. How you parse, clean, and normalize each source is part of the evaluation.

---

## Mandatory Technology Stack

| Component | Required Tool(s) | Notes |
|---|---|---|
| Agentic RAG Orchestration | Google ADK | Primary agent framework |
| Data Ingestion & Chunking | LangChain | Use it for document ingestion, chunking, and metadata enrichment |
| Agentic Workflow Graph | LangGraph | Multi-step ingestion or retrieval workflow |
| Data Retrieval Interface | MCP Server | Expose retrieval as an MCP-compatible tool/endpoint |
| LLM Provider | Groq / Ollama / SambaNova Systems / OpenAI / Gemini | Your choice — justify it in the README |
| Tracing & Observability | LangSmith / Arize Phoenix OSS / Weights & Biases / other open-source platform | Choose at least one and instrument the full pipeline |
| Containerization | Docker + Docker Compose | Entire stack must run via `docker-compose up --build` |
| Version Control | Public GitHub Repository | Submit the repo link |

---

## Day 1 — Document Ingestion Pipeline

Design and implement a source-agnostic ingestion pipeline that processes both the PDF and the website, producing richly annotated, semantically meaningful chunks ready for embedding and retrieval.

### Task 1 — PDF Ingestion (NIST AI RMF)

- Load and parse the NIST AI RMF PDF using LangChain.
- Implement section-aware chunking that respects the document’s structural boundaries.
- Headings such as `GOVERN`, `MAP`, `MEASURE`, and `MANAGE` should not be split arbitrarily across chunks.
- Every chunk must carry metadata such as:
  - `source`
  - `section`
  - `page_number`
- Handle layout issues such as footnotes, headers, repeated page elements, and figure captions gracefully.
- Explain your parsing and chunking strategy in the README.
- *(Nice to have)* Preserve tabular data and diagrams where present, or note explicitly how they are handled.

### Task 2 — Website Ingestion (EU AI Act)

- Ingest the EU AI Act webpage using LangChain.
- Convert the webpage into clean text suitable for chunking and retrieval.
- Chunk by article boundary where possible.
- Each chunk should ideally map to one Article or a meaningful sub-section of an Article.
- Every chunk must carry metadata such as:
  - `source`
  - `article_id`
  - `article_title`
  - `chapter`
- Handle irrelevant HTML content such as navigation, cookie banners, boilerplate, and other non-document elements.

### Task 3 — Embedding & Vector Store

- Select an embedding model suitable for legal and regulatory language.
- Justify your embedding choice in the README.
- Store embeddings in a vector database running as a Docker service.
- Implement hybrid retrieval combining:
  - dense semantic retrieval
  - sparse / keyword-based retrieval
- Retrieved chunks must always include:
  - text
  - metadata
  - retrieval score

### Task 4 — LangGraph Ingestion Workflow

- Model the ingestion process as a LangGraph workflow with clearly named and independently testable nodes.
- A suggested shape is:

`Load -> Parse & Chunk -> Enrich Metadata -> Embed & Index`

- The workflow should support both source types through one coherent design, with branching only where necessary.
- Include robust logging and error handling for each node.

**Day 1 Deliverable:**  
An `ingestion/` module containing the workflow, source-specific parsing logic, chunking logic, and indexing pipeline.

---

## Day 2 — Agentic RAG with Google ADK, MCP Server, and Context Engineering

Build the core agent that answers regulatory questions. The agent must reason transparently, retrieve grounded evidence through an MCP server, manage context intelligently, and produce a structured response.

### Task 5 — MCP Server for Retrieval

- Expose your retrieval layer as a standalone MCP-compatible server.
- The MCP tool must accept a query string and return structured retrieval results.
- Each returned chunk must include:
  - `text`
  - `source`
  - `article_id` or `section`
  - `page_number` where applicable
  - `score`
- The MCP server must run as its own Docker service.
- Document the MCP interface clearly in the README.

### Task 6 — Context Engineering

Context engineering is the deliberate design of what enters the model’s context window at each step — not just retrieval, but how information is ranked, filtered, formatted, and carried across turns.

Implement the following:

- **Context budget**
  - Define a maximum token budget for retrieved context.
  - Trim, prioritise, or compress context to stay within that budget.

- **Relevance re-ranking**
  - Re-rank retrieved chunks before sending them to the model.
  - Explain your ranking strategy.

- **Structured prompt design**
  - Clearly separate:
    - system instructions
    - retrieved context
    - conversation history
    - current user query

- **Conversation memory management**
  - For multi-turn conversations, summarise or prune prior turns to prevent context overflow while retaining important facts.

- **Query decomposition**
  - For multi-part or cross-document questions, break the query into focused sub-queries, retrieve separately, and merge the context before synthesis.

### Task 7 — Google ADK Agent with Minimal Callback Guardrails

- Build the primary agent using Google ADK.
- Register the MCP retrieval server as a tool available to the agent.
- The agent should:
  - decompose complex queries where needed
  - retrieve supporting evidence
  - synthesise a grounded answer
  - produce structured output

Implement **minimal callback-based guardrails** in Google ADK at key points in the lifecycle:

- **Before-model callback**
  - Inspect the constructed prompt.
  - Detect prompt injection attempts or context overflows.
  - Block, sanitise, or adjust as needed.

- **Before-tool callback**
  - Validate the query being sent to the MCP retrieval tool.
  - Ensure the tool call is well-formed and relevant.

- **After-model callback**
  - Inspect the raw model response.
  - Flag hallucination signals such as fabricated references, unsupported claims, or invalid article numbers.

- All callback decisions must be logged.

> These callbacks are intended to be lightweight, minimal guardrails — not a full safety system. Keep them focused and proportional.

### Task 8 — Required Chatbot Output Format

Every chatbot response must follow this exact structure:

#### Reasoning / Thinking
A transparent account of how the agent interpreted the query, which sub-queries were generated, what context was retrieved, and how the answer was formed.

#### Answer
The final grounded response to the user’s question. It must not include claims unsupported by the retrieved context.

#### Citations
A structured list of the sources used to construct the answer. Each citation must include:
- document name
- article or section identifier
- page number where applicable
- cited snippet or referenced chunk

> Responses that do not follow the required `Reasoning / Thinking + Answer + Citation` format will be treated as non-compliant.

### Task 9 — Audit Trail

Alongside the user-facing output, every response must also generate a machine-readable audit trail, for example:

```json
{
  "query": "...",
  "sub_queries": ["...", "..."],
  "retrieved_chunks": [
    {
      "text": "...",
      "source": "...",
      "article": "...",
      "page": 12,
      "score": 0.91
    }
  ],
  "cited_sources": [
    "NIST AI RMF §GOVERN-1.1",
    "EU AI Act Art. 13"
  ],
  "grounding_score": 0.91,
  "hallucination_risk": false,
  "context_tokens_used": 1840,
  "guardrail_triggered": false,
  "llm_used": "your-chosen-model",
  "timestamp": "2026-03-14T10:00:00Z"
}
```

**Day 2 Deliverable:**  
An `agent/` module containing the Google ADK agent, MCP integration, context engineering logic, callback guardrails, output formatter, and audit trail generation.

---

## Day 3 — Tracing, Evaluation, Containerization, and Documentation

### Task 10 — Tracing & Observability

Instrument the full system using at least one tracing or observability platform.

This must include:
- ingestion flow
- retrieval flow
- agent reasoning steps
- callback decisions
- LLM calls

Each meaningful step should create a named trace/span.

At minimum, capture:
- latency per step
- token count per model call
- retrieval score and number of chunks returned
- guardrail trigger events
- hallucination flag
- grounding score

### Task 11 — Evaluation Suite

- Prepare at least **8 test questions** spanning both documents.
- Include at least **2 cross-document questions**.
- Evaluate the system on metrics such as:
  - faithfulness
  - answer relevance
  - context precision
- You may use RAGAS, an LLM judge, or a clearly explained custom approach.
- Output:
  - `evaluation/results.json`
  - `evaluation/report.md`
- The report must explain:
  - what the scores reveal
  - where the system struggles
  - what likely causes those failures

### Task 12 — Docker Compose Stack

All services must start with a single command:

```bash
docker-compose up --build
```

Suggested services:

| Service Name | Purpose |
|---|---|
| `vector-store` | Vector database |
| `ingestion` | One-shot ingestion runner |
| `mcp-server` | MCP retrieval endpoint |
| `agent-api` | Google ADK agent exposed through an API |
| `tracer` *(optional)* | Self-hosted tracing backend if applicable |

Requirements:
- Include `.env.example` with all required environment variables
- Add health checks where appropriate
- Ensure correct service dependencies and startup order

### Task 13 — GitHub Repository & README

The README is a core deliverable and must include:

- **Architecture / Workflow Diagram**
  - Mermaid or embedded image
  - Must show all components and data flows

- **Setup Instructions**
  - From clone to running system

- **Tech Stack Table**
  - Component, chosen tool, and justification

- **Context Engineering Design**
  - Context budget
  - reranking
  - memory handling
  - query decomposition

- **Guardrail Design**
  - Each callback
  - what it checks
  - where it runs
  - how it behaves

- **Sample Queries & Responses**
  - At least 3 sample interactions
  - Must show full `Reasoning / Thinking + Answer + Citation` format

- **Tracing Evidence**
  - Screenshots or trace links

- **Evaluation Results**
  - Summary of your evaluation suite

- **Known Limitations & Future Work**

**Day 3 Deliverable:**  
A fully instrumented, containerized system with evaluation outputs and complete documentation.

---

## Evaluation Rubric

| Dimension | Weight | What Assessors Look For |
|---|---:|---|
| Ingestion Quality | 15% | Source handling, chunking quality, metadata richness, workflow clarity |
| MCP Server Design | 10% | Clean tool interface, structured outputs, standalone service |
| Context Engineering | 15% | Context budgeting, reranking, memory strategy, query decomposition |
| Google ADK Agent | 15% | Multi-step reasoning, tool use, required output structure |
| Callback Guardrails | 10% | Placement, simplicity, usefulness, logging |
| Citation & Audit Trail | 10% | Traceable claims, complete audit object |
| Tracing & Observability | 10% | Named spans, key metrics, visible instrumentation |
| Containerization | 10% | Single-command startup, service health, reproducibility |
| Documentation & Diagram | 5% | Clear README and mandatory architecture diagram |

---

## Bonus Points

- Minimal Streamlit or Gradio UI as an additional Docker service
- Retrieval retry loop if confidence is low
- NER extraction for roles, obligations, or prohibited practices
- Dev Container support
- Basic CI workflow
- Strong cross-document reasoning with explicit source attribution

---

## Submission

Submit your **public GitHub repository link** to the hiring team.

Reviewers should be able to:

1. Clone the repository
2. Copy `.env.example` to `.env`
3. Fill in the required keys
4. Run:

```bash
docker-compose up --build
```

That single command must result in a working, queryable Agentic RAG Chatbot.

### Non-negotiables

1. README architecture/workflow diagram is present
2. Responses follow `Reasoning / Thinking -> Answer -> Citations`
3. `docker-compose up --build` works without manual setup steps
4. Both source documents are ingested
5. At least one tracing platform is instrumented, with evidence in the README
