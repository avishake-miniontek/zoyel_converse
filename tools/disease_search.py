from typing import List, Dict, Any, Optional, Union
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import json
import logging
from openai import AsyncOpenAI
import os
import re
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Create database engine and session
engine = create_engine("sqlite:///voice_ai.db")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Initialize OpenAI client for medgemma model
try:
    openai_client = AsyncOpenAI(
        base_url=os.getenv("AI_BASE_URL"),
        api_key=os.getenv("AI_API_KEY")
    )
    logger.info("MedGemma OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    openai_client = None

async def search_diseases(symptoms: Union[str, List[str]], max_results: int = 10) -> str:
    """
    Search for diseases in the disease_master table based on user-provided symptoms.
    Uses a two-phase approach: intelligent keyword pre-filtering + MedGemma semantic matching.
    This prevents token overflow by only sending relevant diseases to the AI model.

    Args:
        symptoms: User-provided symptoms or conditions. Can be a string (e.g., "cough and sore throat") 
                 or a list of strings (e.g., ["cough", "sore throat"])
        max_results: Maximum number of results to return (default: 10)

    Returns:
        JSON string containing matched diseases or error message
    """
    try:
        db = SessionLocal()
        try:
            # Get all diseases from disease_master table
            query = text("""
                SELECT id, disease_name, icd11_code, snowmed_ct
                FROM disease_master 
                WHERE active_flag = 'Y'
                ORDER BY disease_name
            """)
            
            result = db.execute(query)
            all_diseases = result.fetchall()
            
            if not all_diseases:
                return json.dumps({
                    "success": False,
                    "message": "No diseases found in the database",
                    "matched_diseases": []
                })

            # Handle both string and list inputs for symptoms
            if isinstance(symptoms, list):
                symptoms_text = " ".join(str(symptom) for symptom in symptoms).strip()
            else:
                symptoms_text = str(symptoms).strip()
            
            if not symptoms_text or len(symptoms_text.strip()) < 2:
                return json.dumps({
                    "success": False,
                    "message": "Please provide more specific symptoms or conditions",
                    "matched_diseases": []
                })
            
            # Phase 1: Intelligent pre-filtering to reduce candidate set
            candidate_diseases = _intelligent_prefilter(all_diseases, symptoms_text)
            
            # Phase 2: Use MedGemma for semantic matching on filtered candidates
            if openai_client is not None and candidate_diseases:
                return await _medgemma_semantic_match(db, candidate_diseases, symptoms_text, max_results)
            else:
                # Fallback to enhanced keyword search on all diseases
                return _enhanced_keyword_search(db, all_diseases, symptoms_text, max_results)

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error searching diseases: {e}")
        return json.dumps({
            "success": False,
            "message": f"Error searching diseases: {str(e)}",
            "matched_diseases": []
        })


def _intelligent_prefilter(all_diseases, symptoms_text: str, max_candidates: int = 50) -> List:
    """
    Intelligently pre-filter diseases using enhanced keyword matching and medical synonyms.
    This reduces the candidate set to prevent token overflow in MedGemma.
    """
    try:
        # Clean and normalize symptoms
        symptoms_clean = symptoms_text.lower().strip()
        symptoms_words = re.findall(r'\b\w+\b', symptoms_clean)
        
        # Remove common stop words
        stop_words = {'and', 'or', 'the', 'a', 'an', 'i', 'have', 'am', 'is', 'with', 'my', 'me', 'like', 'think', 'feel'}
        symptoms_words = [word for word in symptoms_words if word not in stop_words and len(word) > 2]
        
        # Medical synonym expansion for better matching
        medical_synonyms = {
            'fever': ['pyrexia', 'temperature', 'hyperthermia', 'febrile'],
            'rash': ['eruption', 'dermatitis', 'exanthem', 'skin'],
            'throat': ['pharynx', 'pharyngeal', 'larynx', 'tonsil'],
            'sore': ['painful', 'tender', 'inflamed', 'ache'],
            'sandpaper': ['rough', 'papular', 'scaly', 'textured'],
            'cough': ['coughing', 'tussis', 'bronchial'],
            'headache': ['cephalgia', 'head', 'migraine'],
            'stomach': ['gastric', 'abdominal', 'belly', 'gastro'],
            'nausea': ['nauseous', 'sick', 'vomiting', 'emesis'],
            'diarrhea': ['loose', 'watery', 'bowel', 'stool'],
            'fatigue': ['tired', 'weakness', 'exhaustion', 'lethargy'],
            'joint': ['arthritis', 'articular', 'knee', 'elbow', 'wrist'],
            'muscle': ['muscular', 'myalgia', 'ache', 'pain'],
            'breathing': ['respiratory', 'dyspnea', 'shortness', 'breath'],
            'chest': ['thoracic', 'cardiac', 'heart', 'lung']
        }
        
        # Expand symptoms with synonyms
        expanded_symptoms = set(symptoms_words)
        for symptom in symptoms_words:
            if symptom in medical_synonyms:
                expanded_symptoms.update(medical_synonyms[symptom])
            # Also check if symptom is a synonym of any key
            for key, synonyms in medical_synonyms.items():
                if symptom in synonyms:
                    expanded_symptoms.add(key)
                    expanded_symptoms.update(synonyms)
        
        # Score diseases based on keyword matching
        scored_diseases = []
        for disease_row in all_diseases:
            disease_id, disease_name, icd11_code, snowmed_ct = disease_row
            disease_name_clean = disease_name.lower()
            disease_words = re.findall(r'\b\w+\b', disease_name_clean)
            
            score = 0
            matched_terms = []
            
            # Check for matches with expanded symptoms
            for symptom in expanded_symptoms:
                # Exact word match (highest score)
                if symptom in disease_words:
                    score += 20
                    matched_terms.append(symptom)
                # Partial match within disease name
                elif any(symptom in word or word in symptom for word in disease_words if len(symptom) > 3):
                    score += 10
                    matched_terms.append(symptom)
                # Substring match in full disease name
                elif symptom in disease_name_clean and len(symptom) > 3:
                    score += 5
                    matched_terms.append(symptom)
            
            # Bonus for multiple matches
            if len(matched_terms) > 1:
                score += len(matched_terms) * 3
            
            if score > 0:
                scored_diseases.append({
                    'disease_row': disease_row,
                    'score': score,
                    'matched_terms': matched_terms
                })
        
        # Sort by score and return top candidates
        scored_diseases.sort(key=lambda x: x['score'], reverse=True)
        return [item['disease_row'] for item in scored_diseases[:max_candidates]]
        
    except Exception as e:
        logger.error(f"Error in intelligent prefilter: {e}")
        # Return first N diseases as fallback
        return all_diseases[:max_candidates]


async def _medgemma_semantic_match(db, candidate_diseases, symptoms_text: str, max_results: int) -> str:
    """
    Use MedGemma-27B-IT model for semantic matching on pre-filtered candidate diseases.
    This prevents token overflow by working with a smaller, relevant disease set.
    """
    try:
        # Prepare disease data
        disease_data = {}
        disease_list = []
        
        for disease_row in candidate_diseases:
            disease_id, disease_name, icd11_code, snowmed_ct = disease_row
            disease_list.append(f"{disease_id}: {disease_name}")
            disease_data[disease_id] = {
                "disease_id": disease_id,
                "disease_name": disease_name,
                "icd11_code": icd11_code or "",
                "snowmed_ct": snowmed_ct or ""
            }
        
        # Create concise prompt for MedGemma
        prompt = f"""Medical symptom analysis task.

Symptoms: "{symptoms_text}"

Candidate diseases:
{chr(10).join(disease_list)}

Task: Rank diseases by likelihood based on symptoms. Return JSON array of disease IDs only.
Format: [id1, id2, id3]
Limit: {max_results} diseases

Response:"""

        # Call MedGemma with optimized parameters
        response = await openai_client.chat.completions.create(
            model=os.getenv("AI_MODEL_NAME"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150  # Reduced for efficiency
        )
        
        ai_response = response.choices[0].message.content.strip()
        logger.info(f"MedGemma semantic match for '{symptoms_text}': {ai_response}")
        
        # Parse AI response
        try:
            # Extract JSON array
            json_match = re.search(r'\[[\d,\s]+\]', ai_response)
            if json_match:
                disease_ids = json.loads(json_match.group())
            else:
                # Fallback: extract numbers
                disease_ids = [int(x) for x in re.findall(r'\b\d+\b', ai_response)]
            
            # Build results
            matched_diseases = []
            for i, disease_id in enumerate(disease_ids[:max_results]):
                if disease_id in disease_data:
                    disease_info = disease_data[disease_id].copy()
                    disease_info["match_score"] = round(1.0 - (i * 0.1), 2)
                    disease_info["match_type"] = "medgemma_semantic"
                    disease_info["rank"] = i + 1
                    matched_diseases.append(disease_info)
            
            if matched_diseases:
                return json.dumps({
                    "success": True,
                    "message": f"Found {len(matched_diseases)} diseases using AI semantic analysis",
                    "search_method": "medgemma_semantic",
                    "search_query": symptoms_text,
                    "candidates_analyzed": len(candidate_diseases),
                    "matched_diseases": matched_diseases
                }, indent=2)
            else:
                # Fallback to keyword search on candidates
                return _enhanced_keyword_search(db, candidate_diseases, symptoms_text, max_results)
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse MedGemma response: {e}")
            return _enhanced_keyword_search(db, candidate_diseases, symptoms_text, max_results)
        
    except Exception as e:
        logger.error(f"Error in MedGemma semantic match: {e}")
        return _enhanced_keyword_search(db, candidate_diseases, symptoms_text, max_results)


def _enhanced_keyword_search(db, diseases, symptoms_text: str, max_results: int) -> str:
    """
    Enhanced keyword-based search with medical synonyms and intelligent scoring.
    """
    try:
        import re
        
        # Clean and normalize the symptoms input
        symptoms_clean = symptoms_text.lower().strip()
        symptoms_words = re.findall(r'\b\w+\b', symptoms_clean)
        
        # Filter out common stop words that aren't medically relevant
        stop_words = {'and', 'or', 'the', 'a', 'an', 'i', 'have', 'am', 'is', 'with', 'my', 'me', 'like', 'think'}
        symptoms_words = [word for word in symptoms_words if word not in stop_words and len(word) > 2]
        
        if not symptoms_words:
            return json.dumps({
                "success": False,
                "message": "Please provide more specific symptoms or conditions",
                "matched_diseases": []
            })

        matched_diseases = []
        
        for disease_row in diseases:
            disease_id, disease_name, icd11_code, snowmed_ct = disease_row
            disease_name_clean = disease_name.lower()
            
            # Calculate match score with enhanced logic
            match_score = 0
            matched_terms = []
            
            # Check for exact word matches (high score)
            for symptom_word in symptoms_words:
                if symptom_word in disease_name_clean:
                    match_score += 15  # Increased weight for exact matches
                    matched_terms.append(symptom_word)
            
            # Check for partial matches and medical term variations
            for symptom_word in symptoms_words:
                if symptom_word not in matched_terms:
                    # Check if symptom word is contained in any word of the disease name
                    disease_words = re.findall(r'\b\w+\b', disease_name_clean)
                    for disease_word in disease_words:
                        if len(symptom_word) > 3:
                            if symptom_word in disease_word:
                                match_score += 8
                                if symptom_word not in matched_terms:
                                    matched_terms.append(symptom_word)
                            elif disease_word in symptom_word:
                                match_score += 5
                                if symptom_word not in matched_terms:
                                    matched_terms.append(symptom_word)
                    
                    # Medical synonym matching for common terms
                    medical_synonyms = {
                        'fever': ['pyrexia', 'temperature', 'hyperthermia'],
                        'rash': ['eruption', 'dermatitis', 'exanthem'],
                        'throat': ['pharynx', 'pharyngeal', 'larynx'],
                        'sore': ['painful', 'tender', 'inflamed'],
                        'sandpaper': ['rough', 'papular', 'scaly']
                    }
                    
                    for synonym_group in medical_synonyms.values():
                        if symptom_word in synonym_group:
                            for disease_word in disease_words:
                                if any(syn in disease_word for syn in synonym_group):
                                    match_score += 10
                                    if symptom_word not in matched_terms:
                                        matched_terms.append(symptom_word)

            # Add disease to results if it has a meaningful match score
            if match_score > 3:
                matched_diseases.append({
                    "disease_id": disease_id,
                    "disease_name": disease_name,
                    "icd11_code": icd11_code or "",
                    "snowmed_ct": snowmed_ct or "",
                    "match_score": round(match_score, 2),
                    "match_type": "enhanced_keyword",
                    "matched_terms": matched_terms
                })

        # Sort by match score (descending) and limit results
        matched_diseases.sort(key=lambda x: x["match_score"], reverse=True)
        matched_diseases = matched_diseases[:max_results]
        
        if not matched_diseases:
            return json.dumps({
                "success": False,
                "message": f"I couldn't find any diseases matching '{symptoms_text}' in our database. Could you provide more specific medical symptoms or try different terms?",
                "matched_diseases": []
            })

        return json.dumps({
            "success": True,
            "message": f"Found {len(matched_diseases)} potential diseases matching your symptoms",
            "search_method": "enhanced_keyword",
            "search_terms": symptoms_words,
            "matched_diseases": matched_diseases
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error in keyword search: {e}")
        return json.dumps({
            "success": False,
            "message": f"Error in keyword search: {str(e)}",
            "matched_diseases": []
        })


def search_diseases_schema():
    """Return the function schema for the search_diseases tool"""
    return {
        "type": "function",
        "function": {
            "name": "search_diseases",
            "description": "Search for diseases in the medical database based on user-provided symptoms or conditions. Uses intelligent matching to find relevant diseases from the disease_master table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symptoms": {
                        "type": "string",
                        "description": "User-provided symptoms or conditions (e.g., 'cough and sore throat', 'fever', 'headache')",
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
                    "input": "I have a cough and sore throat",
                    "call": {
                        "name": "search_diseases",
                        "arguments": {"symptoms": "cough and sore throat"},
                    },
                },
                {
                    "input": "I'm experiencing fever and headache",
                    "call": {
                        "name": "search_diseases",
                        "arguments": {"symptoms": "fever and headache"},
                    },
                },
            ],
        },
    }