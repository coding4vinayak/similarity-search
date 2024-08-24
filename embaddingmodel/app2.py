from flask import Flask, request, jsonify, render_template
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
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
    return render_template('index.html')

@app.route('/add', methods=['POST'])
def add_data():
    data = request.form.get('text')
    if not data:
        return jsonify({'message': 'No text provided'}), 400

    # Tokenize and encode the input text
    inputs = tokenizer(data, return_tensors="pt")
    outputs = model(**inputs)

    # Get the mean of the hidden states to use as the embedding
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().tolist()

    # Add the new embedding to the embeddings dictionary
    embeddings[data] = {'embedding': embedding}

    # Save the updated embeddings dictionary back to the JSON file
    with open(embeddings_file, 'w') as f:
        json.dump(embeddings, f)

    return jsonify({'message': 'Data added successfully!'})

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    if not query:
        return jsonify({'message': 'No query provided'}), 400

    # Tokenize and encode the input query
    query_inputs = tokenizer(query, return_tensors="pt")
    query_outputs = model(**query_inputs)

    # Get the mean of the hidden states to use as the query embedding
    query_embedding = query_outputs.last_hidden_state.mean(dim=1).detach().numpy()

    # Initialize variables to store the best match
    best_match = None
    best_similarity = -1  # Initialize with a low similarity score
    best_match_details = {}

    # Compare the query embedding with each stored embedding
    for text, entry in embeddings.items():
        if 'embedding' in entry:
            stored_embedding = entry['embedding']
        else:
            # Handle the case where 'embedding' might not be present
            continue
        
        # Convert stored_embedding from list to numpy array
        stored_embedding = np.array(stored_embedding)

        # Ensure that the shapes match
        if query_embedding.shape[1] != stored_embedding.shape[1]:
            continue
        
        similarity = cosine_similarity(query_embedding, stored_embedding).item()

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = text
            best_match_details = entry

    return jsonify({'best_match': best_match, 'similarity': best_similarity, 'details': best_match_details})

if __name__ == '__main__':
    app.run(debug=True)
