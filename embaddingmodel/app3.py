from flask import Flask, request, jsonify, render_template
import json
import os
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

app = Flask(__name__)

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Load embeddings from the JSON file
embeddings_file = 'embaddingmodel/data/embeddings.json'
if os.path.exists(embeddings_file):
    with open(embeddings_file, 'r') as f:
        try:
            embeddings = json.load(f)
        except json.JSONDecodeError:
            embeddings = {}
else:
    embeddings = {}

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/add', methods=['POST'])
def add_data():
    data = request.form.get('text')
    if not data:
        return jsonify({'message': 'No text provided'}), 400

    # Split the data into components, e.g., by line breaks or certain delimiters
    data_components = data.split("\n")

    # Vectorize each component separately
    for component in data_components:
        if component.strip():  # Ignore empty lines
            inputs = tokenizer(component, return_tensors="pt")
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().tolist()
            embeddings[component] = {'embedding': embedding}

    # Save the updated embeddings dictionary back to the JSON file
    with open(embeddings_file, 'w') as f:
        json.dump(embeddings, f)

    return jsonify({'message': 'Data added successfully!'})

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('text')
    if not query:
        return jsonify({'message': 'No query provided'}), 400

    # Vectorize the query
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()

    # Calculate similarities with stored embeddings
    results = []
    for text, data in embeddings.items():
        stored_embedding = np.array(data['embedding'])
        similarity = np.dot(query_embedding, stored_embedding.T) / (np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding))
        similarity = similarity.item()  # Convert tensor to scalar
        results.append((text, similarity))

    # Sort results by similarity
    results = sorted(results, key=lambda x: x[1], reverse=True)

    # Return the top results
    return jsonify({
        'query': query,
        'results': [{'text': text, 'similarity': similarity} for text, similarity in results[:5]]  # Return top 5 results
    })

if __name__ == '__main__':
    app.run(debug=True)
