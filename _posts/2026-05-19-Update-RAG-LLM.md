---
layout: post
title: "Building a Simple RAG + LLM Workflow for Procurement Documents"
date: 2026-05-19
categories: [AI, RAG, LLM, Python]
---

# Building a Simple RAG + LLM Workflow for Procurement Documents

Recently, I worked on a small proof-of-concept project to explore how Retrieval-Augmented Generation (RAG) can help users search and reuse information from procurement and sustainment documents.

The goal was not to generate final procurement decisions automatically. Instead, the idea was to help users quickly find historical examples, related answers, and useful context from previous documents.

## Project Goal

The main objective was:

- Search a collection of procurement documents
- Extract meaningful text
- Chunk the content into smaller searchable pieces
- Prepare the data for use with an LLM or Copilot-style assistant
- Return relevant historical examples based on user questions

The dataset included:

- Word documents
- PDF files
- Excel sheets
- Mixed formatting and writing styles

## Step 1: Extracting Text

The first step was converting the source files into structured text.

I used Python libraries such as:

```python
python-docx
PyMuPDF (fitz)
pandas
````

Each document was converted into JSON format.

Example structure:

```json
{
  "document": "SBCA_Report.docx",
  "section": "Operational Considerations",
  "text": "The sustainment strategy should support..."
}
```

This made the documents easier to process and search later.

## Step 2: Chunking the Documents

Large documents are difficult for LLMs to process directly.

To solve this, the text was divided into smaller chunks.

Each chunk contained:

* Document name
* Section information
* Chunk text
* Metadata

Example:

```json
{
  "chunk_id": 15,
  "document": "SBCA_Report.docx",
  "text": "Lifecycle sustainment costs were evaluated..."
}
```

This is an important RAG step because retrieval works better with smaller focused chunks.

## Step 3: Preparing for Embeddings

After chunking, the data was prepared for embeddings.

The idea was:

1. Convert chunks into vector embeddings
2. Store embeddings in a searchable index
3. Compare user questions against embeddings
4. Return the most relevant chunks to the LLM

This allows the model to answer questions using real internal documents instead of general internet knowledge.

## Step 4: Testing with Copilot / LLM

The processed files were tested using:

* Copilot-style prompts
* SharePoint-hosted files
* Manual prompt testing

The questions were open-ended and often varied in wording.

Example questions:

* What sustainment risks were identified?
* What flexibility requirements existed in previous projects?
* What lessons learned were mentioned?

The system returned related historical examples from the source documents.

## Challenges

Some practical challenges included:

* Different document formats
* Inconsistent section structures
* OCR quality issues
* Large paragraphs
* Duplicate information
* Mixed writing styles

Another challenge was balancing preprocessing effort versus project complexity.

For smaller datasets, over-engineering can waste time.

## Lessons Learned

A few key lessons from this work:

* Simple preprocessing helps a lot
* Clean chunking improves retrieval quality
* Metadata becomes very important later
* Procurement documents often contain nuanced language
* Retrieval quality matters more than complex prompting

## Future Improvements

Possible future improvements include:

* Better semantic search
* Automatic metadata extraction
* Vector database integration
* Hybrid keyword + vector search
* Better chunk ranking
* Citation and source highlighting

## Final Thoughts

This project was a good hands-on exercise for understanding practical RAG workflows.

The most interesting part was seeing how historical procurement knowledge could become searchable and reusable using relatively simple tools and preprocessing steps.

Even with a small dataset, the workflow demonstrated how AI assistants can support research and decision-support tasks without replacing human expertise.


