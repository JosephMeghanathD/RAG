# 📘 Simple RAG vs Traditional SLM Demo (Mac M2 Optimized)

This project demonstrates how **Retrieval-Augmented Generation (RAG)** improves the accuracy of a **Small Language Model (SLM)** compared to using the model alone.

It uses:

* **Qwen2.5-1.5B-Instruct** (small, fast, great for Mac M2)
* **SentenceTransformers MiniLM** for embeddings
* A small **in-memory knowledge base** loaded from `knowledge.txt`

The script asks the same set of questions twice:

1. **Traditional Model:** No external context
2. **RAG Model:** Uses vector similarity to retrieve relevant info from the knowledge base

You will observe how RAG gives grounded, correct answers, while the traditional model guesses.

---

## 🚀 Features

* Runs efficiently on **Mac M2** using **Metal (MPS)** acceleration
* Demonstrates:

  * Embedding-based retrieval
  * Cosine similarity search
  * Prompt augmentation
  * Chat template usage for Qwen SLMs
* Fully self-contained — creates test data from `knowledge.txt`

---

## 📂 Project Structure

```
project/
│── script.py              # Main RAG vs Traditional test
│── knowledge.txt          # Your custom knowledge base text file
│── README.md              # This file
```

---

## 🛠️ Requirements

Install dependencies:

```bash
pip install torch sentence-transformers transformers
```

(For MPS: make sure you’re using Python ≥ 3.10 and PyTorch with MPS support.)

---

## 📄 How It Works

### 1. Load the SLM

The script loads **Qwen2.5-1.5B-Instruct** in half-precision (`float16`) for faster inference on M2.

### 2. Load Knowledge Base

`knowledge.txt` is split into line-level chunks and stored as an in-memory list.

### 3. Retrieve Context (RAG)

Query and documents are embedded using MiniLM, then cosine similarity is used to select the most relevant chunk.

### 4. Generate Responses

The model answers the same question:

* **Without context (traditional)**
* **With retrieved context (RAG)**

You can directly compare the two outputs.

---

## ▶️ Running the Test

Just run:

```bash
python rag.py
```

You’ll see output like:

* **Traditional Model:** vague or incorrect (because it doesn’t know your secret company data)
* **RAG Model:** accurate, grounded answers using retrieved text

---

## 📝 Notes

* Works offline — no API needed
* Ideal for understanding **why RAG is important**, even with small models
* Easy to extend for more advanced chunking, embeddings, or FAISS indexing

---

If you want, I can also generate:

✅ A more advanced README
✅ A diagram explaining the workflow
✅ A version with pip/conda setup instructions
Just ask!
