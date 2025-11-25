import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. SETUP & CONFIGURATION (Mac M2 Optimized)
# ==========================================

# Use 'mps' (Metal Performance Shaders) for Mac M2 GPU acceleration
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Running on device: {device.upper()}")

# MODEL CHOICE: Qwen2.5-1.5B-Instruct is small, smart, and fast for M2 Air
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading SLM (this may take a minute first time)...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,  # Half precision for speed on M2
    device_map=device
)

print("Loading Embedding Model...")
# Small, fast embedding model for retrieval
embedder = SentenceTransformer('all-MiniLM-L6-v2')


# ==========================================
# 2. CREATE IN-MEMORY "SECRET" DATA
# ==========================================
# This data represents internal knowledge the AI definitely was NOT trained on.
def load_and_chunk_file(filepath):
    """
    Reads a text file and splits it into chunks (lines).
    In production, you might use a RecursiveCharacterTextSplitter
    to handle large paragraphs better.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    # Split by newlines and remove empty strings
    # This creates a list where every line is a potential "fact" to retrieve
    chunks = [line.strip() for line in content.split('\n') if line.strip()]
    return chunks


knowledge_base = load_and_chunk_file("knowledge.txt")

print("Knowledge Base Indexed.")


# ==========================================
# 3. HELPER FUNCTIONS (Vector Logic)
# ==========================================

def get_relevant_context(query, documents, top_k=1):
    """
    Simple Vector Retrieval System:
    1. Embed the query.
    2. Embed the documents.
    3. Calculate Cosine Similarity.
    4. Return the most similar document.
    """
    query_embedding = embedder.encode([query])
    doc_embeddings = embedder.encode(documents)

    # Calculate Cosine Similarity (Dot product of normalized vectors)
    similarities = np.dot(query_embedding, doc_embeddings.T)

    # Get indices of top_k most similar docs
    top_indices = np.argsort(similarities[0])[-top_k:][::-1]

    return [documents[i] for i in top_indices]


def generate_response(prompt, model, tokenizer):
    """Generates text using the SLM."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
        {"role": "user", "content": prompt}
    ]

    # Apply chat template (critical for Instruct models)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=100,
        do_sample=False  # Greedy decoding for determinism
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# ==========================================
# 4. RUN THE TEST
# ==========================================

test_questions = [
    "Who is the current CEO of the company?",
    "What is the wifi password?",
    "What is Project Blue Sky?"
]

print("\n" + "=" * 50)
print("STARTING COMPARISON TEST")
print("=" * 50)

for question in test_questions:
    print(f"\nQUESTION: {question}")

    # --- SCENARIO A: TRADITIONAL LLM (No Context) ---
    # The model relies only on its training data (which doesn't know our fake company).
    base_response = generate_response(question, model, tokenizer)
    print(f"\n[Traditional Model]: {base_response}")

    # --- SCENARIO B: RAG (With Context) ---
    # 1. Retrieve relevant info from our knowledge_base
    retrieved_info = get_relevant_context(question, knowledge_base)
    context_str = "\n".join(retrieved_info)

    # 2. Augment the prompt
    rag_prompt = f"""
    Use the following context to answer the user's question.

    Context:
    {context_str}

    Question: 
    {question}
    """

    # 3. Generate
    rag_response = generate_response(rag_prompt, model, tokenizer)
    print(f"[RAG Model]       : {rag_response}")

print("\n" + "=" * 50)
print("TEST COMPLETE")
print("Notice how the Traditional Model guesses or gives generic answers,")
print("while the RAG Model answers correctly using the secret data.")
