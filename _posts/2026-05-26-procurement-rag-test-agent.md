---
layout: post
title: "Procurement RAG Test Agent"
date: 2026-05-26
categories: [AI, RAG, LLM, Python]
---

---
layout: post
title: "Procurement RAG Test Agent"
date: 2026-05-26
categories: [AI, RAG, LLM, Python]
---

## Procurement RAG Test Agent

This guide explains how to prepare a small proof-of-concept dataset for a procurement RAG test agent.

The goal is to prepare procurement and sustainment documents for AI-assisted search, so users can find source-based evidence from past documents.

### How to follow this process

1. **Set up the development environment**  
   Install Python and use an IDE such as PyCharm or VS Code.

2. **Create a project folder**  
   Organize the project with folders for source documents, extracted JSON, chunks, and final test files.

3. **Add source documents**  
   Place sample Word, PDF, Excel, text, Markdown, or CSV files in the dataset folder.

4. **Run the Python pipeline**  
   The pipeline scans the files, extracts text, and prepares the content for search.

5. **Create extracted JSON files**  
   The script extracts paragraphs, tables, PDF page text, and Excel rows into structured JSON blocks.

6. **Create chunks**  
   The extracted content is divided into smaller searchable chunks for RAG-style retrieval.

7. **Prepare embedding-ready output**  
   The pipeline creates `embedding_ready_chunks.jsonl` for future vector search or search indexing.

8. **Prepare Copilot-friendly files**  
   The chunks are converted into readable TXT files for validation testing.

### Guardrails

The test should focus on evidence retrieval, not answer invention and then AI can elaborate it.

A good response should:

- use only the provided source files
- return exact source text when possible
- include source file name and chunk ID
- show page or section when available
- say “No clear answer found” when evidence is not available
- keep the final decision with a human reviewer

### Main outputs

The process creates:

- structured JSON files
- searchable chunk files
- an embedding-ready JSONL file
- Copilot-friendly TXT files

### Purpose

This process supports a controlled RAG-style workflow where users can ask procurement or sustainment questions and retrieve relevant historical examples from source documents.