from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import faiss
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import requests

app = Flask(__name__)

# Initialize the SQLite database
engine = create_engine('sqlite:///vectors.db')
Base = declarative_base()

# Define the table structure
class Vector(Base):
    __tablename__ = 'vectors'
    id = Column(Integer, primary_key=True)
    text = Column(String, nullable=False)
    vector = Column(LargeBinary, nullable=False)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Initialize Faiss index
dimension = 768  # Example: dimension of BERT embeddings
faiss_index = faiss.IndexFlatL2(dimension)

# Define base URL and API key for LLaMA API
LLAMA_API_BASE_URL = "//api.aimlapi.com/chat/completions"
LLAMA_API_KEY = "8c7b0bddeb924e31aa9b8960897012aa"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add', methods=['POST'])
def add_data():
    data = request.form.get('text')
    if not data:
        return jsonify({'message': 'No text provided'}), 400

    # Convert text to vector
    inputs = tokenizer(data, return_tensors="pt")
    outputs = model(**inputs)
    vector = outputs.last_hidden_state.mean(dim=1).detach().numpy()

    # Add vector to Faiss index
    faiss_index.add(vector)

    # Save vector and text to the database
    vector_binary = vector.tobytes()
    new_vector = Vector(text=data, vector=vector_binary)
    session.add(new_vector)
    session.commit()

    return jsonify({'message': 'Data added successfully!'})

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('text')
    if not query:
        return jsonify({'message': 'No query provided'}), 400

    # Convert query text to vector
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model(**inputs)
    query_vector = outputs.last_hidden_state.mean(dim=1).detach().numpy()

    # Search for similar vectors in Faiss index
    _, indices = faiss_index.search(query_vector, k=5)  # Top 5 similar results

    # Retrieve corresponding texts from the database
    similar_texts = []
    for idx in indices[0]:
        if idx >= 0:
            vector_entry = session.query(Vector).get(idx + 1)  # SQLAlchemy ID starts from 1
            similar_texts.append(vector_entry.text)

    # Use LLM (e.g., LLaMA) for context-aware generation
    context = " ".join(similar_texts)
    response = requests.post(f"{LLAMA_API_BASE_URL}/generate", 
                             json={
                                 'prompt': f"Given the context: {context}, and the query: {query}, provide a response."
                             },
                             headers={
                                 'Authorization': f"Bearer {LLAMA_API_KEY}"
                             })
    
    generated_response = response.json().get('text', '')

    return jsonify({
        'query': query,
        'context': similar_texts,
        'generated_response': generated_response
    })

if __name__ == '__main__':
    app.run(debug=True)
