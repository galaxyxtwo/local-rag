# Local RAG Implementation with LangChain

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain, allowing users to query documents using local LLMs.
![supportAgentDiagram](https://github.com/user-attachments/assets/4704c072-9ac3-43ef-9dd4-b87c701984e2)


## Setup

1. Install dependencies:
```bash
pip install langchain chromadb llama-cpp-python sentence-transformers langchain-community

Download the Mistral model:

```bash
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf -O models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

Place your documents in the docs folder

Usage
Run the script:
```bash
python3 test_rag.py
```
Then enter your questions when prompted.

