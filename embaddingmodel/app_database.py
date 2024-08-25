from flask import Flask, request, jsonify, render_template
import json
import sqlite3
import os
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

app = Flask(__name__)

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# SQLite database setup
DATABASE = 'embeddings.db'

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS embeddings (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          text TEXT NOT NULL,
                          embedding BLOB NOT NULL)''')
        conn.commit()

def save_embedding(text, embedding):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO embeddings (text, embedding) VALUES (?, ?)',
                       (text, json.dumps(embedding)))
        conn.commit()

def load_embeddings():
    embeddings = {}
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT text, embedding FROM embeddings')
        rows = cursor.fetchall()
        for text, embedding in rows:
            embeddings[text] = {'embedding': json.loads(embedding)}
    return embeddings

@app.route('/')
def index():
    return render_template('index_db.html')

@app.route('/add', methods=['POST'])
def add_data():
    data = request.form.get('text')
    if not data:
        return jsonify({'message': 'No text provided'}), 400

    # Split the data into components
    data_components = data.strip().split("\n")

    # Vectorize and store each component
    for component in data_components:
        if component.strip():  # Ignore empty lines
            key, text = component.split(":", 1)
            key = key.strip()
            text = text.strip()

            # Tokenize and get embeddings
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().tolist()

            # Save to SQLite
            save_embedding(text, embedding)

    return jsonify({'message': 'Data added successfully!'})

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('text')
    if not query:
        return jsonify({'message': 'No query provided'}), 400

    # Vectorize the query
    inputs = tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()

    # Load stored embeddings and calculate similarities
    embeddings = load_embeddings()
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
    if not os.path.exists(DATABASE):
        init_db()
    app.run(debug=True)
