# AI Knowledge Base — Week 2 Mini Project

A Retrieval-Augmented Generation (RAG) application built with Python, Streamlit, LangChain, ChromaDB, and the Gemini API.

## Features

- Upload one or more PDF research papers or company documents
- Extract PDF text and split it into overlapping chunks
- Create vector embeddings and store them in ChromaDB
- Ask questions through a chat interface
- Retrieve relevant passages and generate grounded answers
- Cite source filenames and page numbers; show retrieved excerpts
- Adjust chunk size and overlap
- Clear the current in-memory knowledge base

## Requirements

- Python 3.10+
- A Google Gemini API key
- Text-based PDFs. Scanned PDFs may require OCR before upload.

## Run locally

1. Open a terminal in this folder.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and add your API key as `GOOGLE_API_KEY=...`.
4. Start the app:

   ```bash
   streamlit run app.py
   ```

## GitHub submission

Create a public repository named `ai-knowledge-base-rag-week2` and upload the project files. Submit the repository URL on the internship page.

**Security:** Never commit `.env` or expose your API key. The `.gitignore` excludes `.env`.

## Limitations

- The knowledge base exists in the current app session; uploaded documents are not permanently stored by this starter version.
- Answers can still be incomplete or incorrect. Verify important claims against the cited PDF pages.
- API usage may be subject to provider quotas and charges.
