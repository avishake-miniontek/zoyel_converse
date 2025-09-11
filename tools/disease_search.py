from typing import List, Dict, Any, Optional, Union
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import json
import logging
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Create database engine and session
engine = create_engine("sqlite:///voice_ai.db")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Initialize EmbeddingGemma model for native Gemma3 embeddings
embedding_model = None
disease_embeddings_cache = None
disease_data_cache = None
cache_dir = Path("embeddings_cache")
cache_dir.mkdir(exist_ok=True)

# Medical terms for relevance filtering
MEDICAL_TERMS = {
    'symptoms': {'pain', 'ache', 'fever', 'cough', 'rash', 'nausea', 'vomiting', 'diarrhea', 
                'headache', 'fatigue', 'weakness', 'swelling', 'inflammation', 'bleeding',
                'shortness', 'breathing', 'chest', 'stomach', 'abdominal', 'joint', 'muscle',
                'throat', 'sore', 'itchy', 'burning', 'tingling', 'numbness', 'dizzy',
                'blurred', 'vision', 'hearing', 'loss', 'discharge', 'spots', 'bumps', 'cramps',
                'bloating', 'constipated', 'tightness', 'tired', 'red', 'flaky', 'cracks'},
    'body_parts': {'head', 'neck', 'chest', 'back', 'arm', 'leg', 'hand', 'foot', 'eye',
                  'ear', 'nose', 'mouth', 'throat', 'stomach', 'abdomen', 'knee', 'elbow',
                  'shoulder', 'hip', 'ankle', 'wrist', 'finger', 'toe', 'skin', 'hair', 'elbows', 'knees'},
    'medical_conditions': {'infection', 'disease', 'syndrome', 'disorder', 'condition',
                          'illness', 'sickness', 'allergy', 'inflammation', 'cancer',
                          'tumor', 'diabetes', 'hypertension', 'asthma', 'pneumonia'}
}

def _initialize_embedding_model():
    """Initialize EmbeddingGemma model for native Gemma3 embeddings"""
    global embedding_model
    if embedding_model is None:
        try:
            logger.info("Loading EmbeddingGemma model (native Gemma3 embeddings)...")
            embedding_model = SentenceTransformer('google/embeddinggemma-300m')
            logger.info("EmbeddingGemma model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load EmbeddingGemma model: {e}")
            # Fallback to general model
            try:
                logger.info("Falling back to general sentence transformer model...")
                embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Fallback model loaded successfully")
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback model: {fallback_error}")
                raise
    return embedding_model

async def search_diseases(symptoms: str, max_results: int = 10) -> str:
    """
    Search for diseases using intelligent semantic matching with native Gemma3 embeddings.
    Simple interface: just pass symptoms as a string.

    Args:
        symptoms: User-provided symptoms as a single string
        max_results: Maximum number of results to return (default: 10)

    Returns:
        JSON string containing matched diseases or error message
    """
    try:
        logger.info(f"Disease search called with symptoms: '{symptoms}'")
        
        # Input validation
        if not symptoms or not isinstance(symptoms, str):
            return json.dumps({
                "success": False,
                "message": "Please provide symptoms as text",
                "matched_diseases": []
            })
        
        # Preprocess symptoms
        processed_symptoms = _preprocess_symptoms(symptoms)
        
        if len(processed_symptoms.strip()) < 3:
            return json.dumps({
                "success": False,
                "message": "Please provide more detailed symptoms",
                "matched_diseases": []
            })
        
        # Check if query is medically relevant
        if not _is_medical_query(processed_symptoms):
            return json.dumps({
                "success": False,
                "message": "I couldn't find any medical symptoms in your description. Please describe your health symptoms or conditions.",
                "matched_diseases": []
            })
        
        # Load or create disease embeddings
        if not _load_disease_embeddings():
            return json.dumps({
                "success": False,
                "message": "Error loading disease database",
                "matched_diseases": []
            })
        
        # Initialize embedding model
        model = _initialize_embedding_model()
        
        # Generate embedding for symptoms using EmbeddingGemma
        logger.info("Generating symptom embedding with EmbeddingGemma...")
        symptom_embedding = model.encode([processed_symptoms])
        
        # Calculate cosine similarities
        logger.info("Calculating similarities with disease database...")
        similarities = cosine_similarity(symptom_embedding, disease_embeddings_cache)[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:max_results * 2]  # Get more for filtering
        
        # Filter and format results
        matched_diseases = []
        for idx in top_indices:
            similarity_score = similarities[idx]
            
            # Only include results with reasonable similarity (threshold: 0.25)
            if similarity_score < 0.25:
                break
            
            disease_info = disease_data_cache[idx].copy()
            disease_info["match_score"] = round(float(similarity_score), 3)
            disease_info["match_type"] = "gemma3_semantic"
            disease_info["rank"] = len(matched_diseases) + 1
            
            matched_diseases.append(disease_info)
            
            if len(matched_diseases) >= max_results:
                break
        
        if not matched_diseases:
            return json.dumps({
                "success": False,
                "message": "No matching diseases found for the provided symptoms. Please try describing your symptoms differently or provide more specific details.",
                "matched_diseases": []
            })
        
        logger.info(f"Found {len(matched_diseases)} matching diseases")
        
        return json.dumps({
            "success": True,
            "message": f"Found {len(matched_diseases)} diseases matching your symptoms using Gemma3 semantic analysis",
            "search_method": "gemma3_semantic",
            "search_query": processed_symptoms,
            "matched_diseases": matched_diseases
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error in disease search: {e}", exc_info=True)
        return json.dumps({
            "success": False,
            "message": f"Error searching diseases: {str(e)}",
            "matched_diseases": []
        })

def _preprocess_symptoms(symptoms_text: str) -> str:
    """Preprocess and normalize symptom text"""
    # Basic cleaning
    text = symptoms_text.lower().strip()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Expand common medical abbreviations
    abbreviations = {
        'bp': 'blood pressure',
        'hr': 'heart rate',
        'temp': 'temperature',
        'resp': 'respiratory',
        'gi': 'gastrointestinal',
        'uti': 'urinary tract infection',
        'sob': 'shortness of breath',
        'loc': 'loss of consciousness'
    }
    
    for abbr, full in abbreviations.items():
        text = re.sub(r'\b' + abbr + r'\b', full, text)
    
    return text

def _is_medical_query(symptoms_text: str) -> bool:
    """Check if the input contains medical terms or symptoms"""
    text_lower = symptoms_text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    # Check for medical terms
    medical_word_count = 0
    total_meaningful_words = 0
    
    # Common stop words to ignore
    stop_words = {'i', 'have', 'am', 'is', 'the', 'a', 'an', 'and', 'or', 'but', 'my', 'me'}
    
    for word in words:
        if len(word) > 2 and word not in stop_words:
            total_meaningful_words += 1
            
            # Check if word is in any medical category
            for category_terms in MEDICAL_TERMS.values():
                if word in category_terms:
                    medical_word_count += 1
                    break
            
            # Check for partial matches with medical terms
            for category_terms in MEDICAL_TERMS.values():
                for term in category_terms:
                    if len(word) > 3 and (word in term or term in word):
                        medical_word_count += 0.5
                        break
    
    # Calculate medical relevance ratio
    if total_meaningful_words == 0:
        return False
    
    medical_ratio = medical_word_count / total_meaningful_words
    
    # Require at least 30% medical terms or at least 2 medical terms
    is_medical = medical_ratio >= 0.3 or medical_word_count >= 2
    
    logger.info(f"Medical relevance check: {medical_word_count}/{total_meaningful_words} = {medical_ratio:.2f}, is_medical: {is_medical}")
    return is_medical

def _load_disease_embeddings() -> bool:
    """Load cached disease embeddings or create them if they don't exist"""
    global disease_embeddings_cache, disease_data_cache
    
    embeddings_file = cache_dir / "disease_embeddings.pkl"
    data_file = cache_dir / "disease_data.pkl"
    
    if embeddings_file.exists() and data_file.exists():
        try:
            logger.info("Loading cached disease embeddings...")
            with open(embeddings_file, 'rb') as f:
                disease_embeddings_cache = pickle.load(f)
            with open(data_file, 'rb') as f:
                disease_data_cache = pickle.load(f)
            logger.info(f"Loaded {len(disease_data_cache)} cached disease embeddings")
            return True
        except Exception as e:
            logger.error(f"Failed to load cached embeddings: {e}")
    
    return _create_disease_embeddings()

def _create_disease_embeddings() -> bool:
    """Create embeddings for all diseases in the database"""
    global disease_embeddings_cache, disease_data_cache
    
    try:
        logger.info("Creating disease embeddings from database...")
        
        db = SessionLocal()
        try:
            # Get all active diseases
            query = text("""
                SELECT id, disease_name, icd11_code, snowmed_ct
                FROM disease_master 
                WHERE active_flag = 'Y'
                ORDER BY disease_name
            """)
            
            result = db.execute(query)
            diseases = result.fetchall()
            
            if not diseases:
                logger.error("No diseases found in database")
                return False
            
            # Initialize embedding model
            model = _initialize_embedding_model()
            
            # Prepare disease texts and data
            disease_texts = []
            disease_data = []
            
            for disease_row in diseases:
                disease_id, disease_name, icd11_code, snowmed_ct = disease_row
                
                # Create comprehensive text representation for better embeddings
                disease_text = disease_name.lower()
                
                # Add ICD code context if available
                if icd11_code:
                    disease_text += f" {icd11_code}"
                
                # Add SNOMED CT context if available
                if snowmed_ct:
                    disease_text += f" {snowmed_ct}"
                
                disease_texts.append(disease_text)
                disease_data.append({
                    'disease_id': disease_id,
                    'disease_name': disease_name,
                    'icd11_code': icd11_code or "",
                    'snowmed_ct': snowmed_ct or ""
                })
            
            # Generate embeddings in batches to avoid memory issues
            logger.info(f"Generating embeddings for {len(disease_texts)} diseases...")
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(disease_texts), batch_size):
                batch = disease_texts[i:i + batch_size]
                batch_embeddings = model.encode(batch, show_progress_bar=True)
                all_embeddings.append(batch_embeddings)
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(disease_texts) + batch_size - 1)//batch_size}")
            
            # Combine all embeddings
            disease_embeddings_cache = np.vstack(all_embeddings)
            disease_data_cache = disease_data
            
            # Cache the embeddings
            embeddings_file = cache_dir / "disease_embeddings.pkl"
            data_file = cache_dir / "disease_data.pkl"
            
            with open(embeddings_file, 'wb') as f:
                pickle.dump(disease_embeddings_cache, f)
            with open(data_file, 'wb') as f:
                pickle.dump(disease_data_cache, f)
            
            logger.info(f"Created and cached embeddings for {len(disease_data_cache)} diseases")
            return True
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error creating disease embeddings: {e}")
        return False

def search_diseases_schema():
    """Return the function schema for the search_diseases tool"""
    return {
        "type": "function",
        "function": {
            "name": "search_diseases",
            "description": "Search for diseases based on symptoms using intelligent Gemma3 semantic matching. Simply provide all symptoms as a single text string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symptoms": {
                        "type": "string",
                        "description": "All symptoms described by the user as a single text string (e.g., 'dry itchy patches on elbows and knees, worse in evenings')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10)",
                        "default": 10,
                    },
                },
                "required": ["symptoms"],
            },
            "examples": [
                {
                    "input": "I have dry, itchy patches on my elbows and behind my knees. The skin is flaky and sometimes cracks.",
                    "call": {
                        "name": "search_diseases",
                        "arguments": {"symptoms": "dry itchy patches on elbows and knees, flaky skin, cracking skin"},
                    },
                },
                {
                    "input": "I've been having stomach cramps, bloating, and irregular bowel movements for the past month.",
                    "call": {
                        "name": "search_diseases",
                        "arguments": {"symptoms": "stomach cramps, bloating, irregular bowel movements"},
                    },
                },
            ],
        },
    }