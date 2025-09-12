import requests
from typing import List
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import html
from pathlib import Path

load_dotenv()

# Initialize EmbeddingGemma model
embedding_model = None
cache_dir = Path("embeddings_cache")
cache_dir.mkdir(exist_ok=True)

def _initialize_embedding_model():
    global embedding_model
    if embedding_model is None:
        try:
            embedding_model = SentenceTransformer('google/embeddinggemma-300m')
        except Exception as e:
            print(f"Failed to load embedding model: {e}")
            embedding_model = None

def extract_text(html_content):
    # Remove scripts and styles
    html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    # Remove comments
    html_content = re.sub(r'<!--.*?-->', '', html_content, flags=re.DOTALL)
    # Remove tags
    text = re.sub(r'<[^>]+>', '', html_content)
    # Decode entities
    text = html.unescape(text)
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def search_web(query: str) -> str:
    """Search the web using Brave API and return a relevant summary using embeddings."""
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        return "Web search is not configured. Please set BRAVE_SEARCH_API_KEY in environment."
    
    _initialize_embedding_model()
    if embedding_model is None:
        return "Embedding model not available."
    
    try:
        # Brave search
        headers = {'X-Subscription-Token': api_key}
        params = {'q': query}
        response = requests.get('https://api.search.brave.com/res/v1/web/search', headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        results = data.get('web', {}).get('results', [])
        if not results:
            return "No search results found."
        
        # Get top 3 URLs
        top_urls = [r['url'] for r in results[:3]]
        
        all_chunks = []
        for url in top_urls:
            try:
                page_response = requests.get(url, timeout=10)
                page_response.raise_for_status()
                html_content = page_response.text
                text = extract_text(html_content)
                if text:
                    chunks = chunk_text(text)
                    all_chunks.extend(chunks)
            except Exception as e:
                continue  # Skip failed pages
        
        if not all_chunks:
            # Fallback to Brave descriptions
            summary = "Found search results: "
            for i, result in enumerate(results[:5], 1):
                desc = result.get('description', 'No description')[:200]
                summary += f"{i}. {desc}. "
            return summary
        
        # Embed chunks
        chunk_embeddings = embedding_model.encode(all_chunks)
        query_embedding = embedding_model.encode([query])[0]
        
        # Compute similarities
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        
        # Get top 5 chunks
        top_indices = np.argsort(similarities)[::-1][:5]
        top_chunks = [all_chunks[i] for i in top_indices]
        
        # Create summary
        summary = "Relevant information: " + " ".join(top_chunks[:3])[:1000] + "..."
        
        return summary
    
    except Exception as e:
        return f"Error performing web search: {str(e)}"

def search_web_schema():
    return {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for the given query and return a brief summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string"
                    }
                },
                "required": ["query"]
            }
        }
    }