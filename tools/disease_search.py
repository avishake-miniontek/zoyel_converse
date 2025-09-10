from typing import List, Dict, Any, Optional, Union
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import json
import logging
from openai import AsyncOpenAI
import os
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
    Uses MedGemma-27B-IT model for intelligent medical symptom understanding and disease matching.

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
                # If symptoms is a list, join them with spaces
                symptoms_text = " ".join(str(symptom) for symptom in symptoms).strip()
            else:
                # If symptoms is a string, use it directly
                symptoms_text = str(symptoms).strip()
            
            if not symptoms_text or len(symptoms_text.strip()) < 2:
                return json.dumps({
                    "success": False,
                    "message": "Please provide more specific symptoms or conditions",
                    "matched_diseases": []
                })
            
            # Use MedGemma model for intelligent disease matching
            if openai_client is not None:
                return await _medgemma_search(db, all_diseases, symptoms_text, max_results)
            else:
                # Fallback to enhanced keyword search
                return _keyword_search(db, all_diseases, symptoms_text, max_results)


        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error searching diseases: {e}")
        return json.dumps({
            "success": False,
            "message": f"Error searching diseases: {str(e)}",
            "matched_diseases": []
        })


async def _medgemma_search(db, all_diseases, symptoms_text: str, max_results: int) -> str:
    """
    Use MedGemma-27B-IT model to intelligently match symptoms to diseases.
    """
    try:
        # Prepare disease list for the AI model
        disease_list = []
        disease_data = {}
        
        for disease_row in all_diseases:
            disease_id, disease_name, icd11_code, snowmed_ct = disease_row
            disease_list.append(f"{disease_id}: {disease_name}")
            disease_data[disease_id] = {
                "disease_id": disease_id,
                "disease_name": disease_name,
                "icd11_code": icd11_code or "",
                "snowmed_ct": snowmed_ct or ""
            }
        
        # Create a prompt for the medical AI model
        prompt = f"""You are a medical AI assistant. Given the following symptoms: "{symptoms_text}"

Please analyze these symptoms and identify the most likely diseases from this list. Consider medical knowledge about symptom patterns, disease presentations, and differential diagnosis.

Available diseases (ID: NAME):
{chr(10).join(disease_list[:100])}  # Limit to first 100 for token efficiency

Instructions:
1. Analyze the symptoms medically
2. Match them to the most likely diseases from the list
3. Rank by likelihood (most likely first)
4. Return ONLY a JSON array of disease IDs in order of likelihood
5. Include maximum {max_results} diseases
6. Format: [disease_id1, disease_id2, disease_id3]

Example response: [13, 45, 78]

Response:"""

        # Call MedGemma model
        response = await openai_client.chat.completions.create(
            model=os.getenv("AI_MODEL_NAME"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temperature for consistent medical reasoning
            max_tokens=200
        )
        
        ai_response = response.choices[0].message.content.strip()
        logger.info(f"MedGemma response for '{symptoms_text}': {ai_response}")
        
        # Parse the AI response to extract disease IDs
        try:
            # Extract JSON array from response
            import re
            json_match = re.search(r'\[[\d,\s]+\]', ai_response)
            if json_match:
                disease_ids = json.loads(json_match.group())
            else:
                # Fallback: extract numbers from response
                disease_ids = [int(x) for x in re.findall(r'\b\d+\b', ai_response)]
            
            # Build matched diseases list
            matched_diseases = []
            for i, disease_id in enumerate(disease_ids[:max_results]):
                if disease_id in disease_data:
                    disease_info = disease_data[disease_id].copy()
                    disease_info["match_score"] = round(1.0 - (i * 0.1), 2)  # Decreasing score by rank
                    disease_info["match_type"] = "medgemma_ai"
                    disease_info["rank"] = i + 1
                    matched_diseases.append(disease_info)
            
            if not matched_diseases:
                return json.dumps({
                    "success": False,
                    "message": f"The medical AI couldn't find diseases matching '{symptoms_text}' in our database. Could you provide more specific symptoms?",
                    "matched_diseases": []
                })
            
            return json.dumps({
                "success": True,
                "message": f"Found {len(matched_diseases)} diseases matching your symptoms using medical AI analysis",
                "search_method": "medgemma_ai",
                "search_query": symptoms_text,
                "ai_reasoning": ai_response,
                "matched_diseases": matched_diseases
            }, indent=2)
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse MedGemma response: {e}")
            # Fallback to keyword search
            return _keyword_search(db, all_diseases, symptoms_text, max_results)
        
    except Exception as e:
        logger.error(f"Error in MedGemma search: {e}")
        # Fallback to keyword search
        return _keyword_search(db, all_diseases, symptoms_text, max_results)


def _keyword_search(db, all_diseases, symptoms_text: str, max_results: int) -> str:
    """
    Enhanced keyword-based search as fallback when semantic search fails.
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
        
        for disease_row in all_diseases:
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
                    "match_type": "keyword",
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
            "search_method": "keyword_enhanced",
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