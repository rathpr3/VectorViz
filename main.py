from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import json

app = FastAPI(title="Embedding Explorer", version="1.0.0")

# Load the sentence transformer model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class WordRequest(BaseModel):
    words: List[str]

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/embeddings")
async def get_embeddings(request: WordRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if len(request.words) < 2:
        raise HTTPException(status_code=400, detail="At least 2 words are required")
    
    # Clean and filter words
    words = [word.strip() for word in request.words if word.strip()]
    if len(words) < 2:
        raise HTTPException(status_code=400, detail="At least 2 valid words are required")
    
    try:
        # Generate embeddings
        embeddings = model.encode(words)
        
        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Apply PCA for 2D visualization
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Convert numpy arrays to lists for JSON serialization
        result = {
            "words": words,
            "embeddings": embeddings.tolist(),
            "similarity_matrix": similarity_matrix.tolist(),
            "embeddings_2d": embeddings_2d.tolist()
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing embeddings: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
