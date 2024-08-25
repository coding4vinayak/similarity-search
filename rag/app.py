from flask import Flask, request, jsonify, render_template
import requests
import torch
import numpy as np
import faiss
import os
import json

app = Flask(__name__)

# LLaMA 3 API details
LLAMA_API_URL = "https://api.example.com/generate"
LLAMA_API_KEY = "your_llama_api_key"

# Initialize the Faiss index for vector search
dimension = 768  # Dimension of the embedding vector (assuming this matches the embedding size from your retrieval model)
index = faiss.IndexFlatL2(dimension)
doc_embeddings = []
doc_texts = []

# Load existing embeddings if available
embeddings_file = 'embeddings.json'
if os.path.exists(embeddings_file):
    with open(embeddings_file, 'r') as f:
        try:
            data = json.load(f)
            for item in data:
                doc_texts.append(item['text'])
                doc_embeddings.append(np.array(item['embedding']))
            index.add(np.array(doc_embeddings))
        except json.JSONDecodeError:
            pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add', methods=['POST'])
def add_data():
    data = request.form.get('text')
    if not data:
        return jsonify({'message': 'No text provided'}), 400

    # Here, assume you have a method to convert text into embeddings
    embedding = get_embedding(data)
    index.add(embedding)
    doc_texts.append(data)
    doc_embeddings.append(embedding.tolist())

    # Save to local file
    with open(embeddings_file, 'w') as f:
        json.dump([{'text': t, 'embedding': e} for t, e in zip(doc_texts, doc_embeddings)], f)

    return jsonify({'message': 'Data added successfully!'})

def get_embedding(text):
    # You should replace this with your actual method to get text embeddings
    # For instance, using LLaMA 3 or another pre-trained model
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('text')
    if not query:
        return jsonify({'message': 'No query provided'}), 400

    # Vectorize the query
    query_embedding = get_embedding(query)
    k = 5  # Number of top results to return
    distances, indices = index.search(query_embedding, k)

    # Collect the results
    results = [{'text': doc_texts[i], 'similarity': float(distances[0][j])} for j, i in enumerate(indices[0])]

    # Use the top result as context for LLaMA 3 generation
    if results:
        context = results[0]['text']
        generated_text = generate_with_llama(query, context)
        return jsonify({'query': query, 'results': results, 'generated_text': generated_text})
    else:
        return jsonify({'query': query, 'results': [], 'message': 'No relevant documents found.'})

def generate_with_llama(query, context):
    headers = {'Authorization': f'Bearer {LLAMA_API_KEY}', 'Content-Type': 'application/json'}
    payload = {
        "prompt": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:",
        "max_tokens": 150
    }
    response = requests.post(LLAMA_API_URL, headers=headers, json=payload)
    response_data = response.json()
    return response_data.get('text', '')

if __name__ == '__main__':
    app.run(debug=True)
