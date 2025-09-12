from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import json
import logging

logger = logging.getLogger(__name__)

# Import the PatientData model from save_patient_data
from .save_patient_data import PatientData

# Create database engine and session
engine = create_engine("sqlite:///voice_ai.db")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def fetch_patient_data(session_id: str, fields: List[str] = None) -> str:
    """
    Fetch patient data from the database for a specific session.
    Returns data as JSON string to match the manual tool call parsing system.

    Args:
        session_id: Unique session identifier
        fields: Optional list of specific fields to retrieve. If None, returns all data.
                Available fields: age, gender, city, country, date, complaints, vitals,
                physical_examination, comorbidities, past_medical_history, current_medications,
                family_history, allergies, test_documents, test_results

    Returns:
        JSON string containing requested patient data or error message
    """
    try:
        db = SessionLocal()
        try:
            # Query patient data
            patient = (
                db.query(PatientData)
                .filter(PatientData.session_id == session_id)
                .first()
            )

            if not patient:
                return json.dumps({
                    "success": False,
                    "message": "No patient data found for this session",
                })

            # Build complete patient data dictionary in target JSON structure
            complete_data = {
                "session_id": patient.session_id,
                "age": {
                    "years": patient.age_years or 0,
                    "months": patient.age_months or 0,
                    "days": patient.age_days or 0,
                },
                "gender": patient.gender or "",
                "city": patient.city or "",
                "country": patient.country or "",
                "date": patient.date or "",
                "complaints": patient.complaints or [],
                "vitals": {
                    "weight_kg": patient.weight_kg or 0,
                    "height_cm": patient.height_cm or 0,
                    "bp_systolic": patient.bp_systolic or 0,
                    "bp_diastolic": patient.bp_diastolic or 0,
                    "temperature_c": patient.temperature_c or 0.0,
                    "heart_rate": patient.heart_rate or 0,
                    "respiration_rate": patient.respiration_rate or 0,
                    "spo2": patient.spo2 or 0,
                    "lmp": patient.lmp or "",
                },
                "physical_examination": patient.physical_examination or "",
                "comorbidities": patient.comorbidities or [],
                "past_medical_history": {
                    "past_illnesses": patient.past_illnesses or [],
                    "previous_procedures": patient.previous_procedures or [],
                },
                "current_medications": patient.current_medications or [],
                "family_history": patient.family_history or [],
                "allergies": {
                    "drug_allergies": patient.drug_allergies or [],
                    "food_allergies": patient.food_allergies or [],
                },
                "test_documents": patient.test_documents or [],
                "test_results": patient.test_results or [],
                "metadata": {
                    "created_at": patient.created_at.isoformat()
                    if patient.created_at
                    else None,
                    "updated_at": patient.updated_at.isoformat()
                    if patient.updated_at
                    else None,
                },
            }

            # If specific fields requested, filter the data
            if fields:
                filtered_data = {"session_id": patient.session_id}  # Always include session_id
                for field in fields:
                    if field in complete_data:
                        filtered_data[field] = complete_data[field]
                    else:
                        logger.warning(
                            f"Requested field '{field}' not found in patient data"
                        )

                return json.dumps({
                    "success": True,
                    "data": filtered_data
                }, indent=2)

            # Return all data
            return json.dumps({
                "success": True,
                "data": complete_data
            }, indent=2)

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error fetching patient data: {e}")
        return json.dumps({
            "success": False,
            "message": f"Error fetching patient data: {str(e)}"
        })


def fetch_patient_data_schema():
    """Return the function schema for the fetch_patient_data tool"""
    return {
        "type": "function",
        "function": {
            "name": "fetch_patient_data",
            "description": "Retrieve patient medical data from the database. Use this when the user asks to recall or confirm any of their previously provided medical information. Returns structured JSON with success status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Unique session identifier",
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of specific fields to retrieve. Available: age, gender, city, country, date, complaints, vitals, physical_examination, comorbidities, past_medical_history, current_medications, family_history, allergies, test_documents, test_results, metadata. If not specified, returns all data.",
                    },
                },
                "required": ["session_id"],
            },
            "examples": [
                {
                    "input": "Can you remind me what allergies I told you about?",
                    "call": {
                        "name": "fetch_patient_data",
                        "arguments": {"session_id": "abc123", "fields": ["allergies"]},
                    },
                },
                {
                    "input": "What's my medical history that we discussed?",
                    "call": {
                        "name": "fetch_patient_data",
                        "arguments": {
                            "session_id": "abc123",
                            "fields": [
                                "past_medical_history",
                                "comorbidities",
                                "family_history",
                            ],
                        },
                    },
                },
                {
                    "input": "Show me all my information",
                    "call": {
                        "name": "fetch_patient_data",
                        "arguments": {"session_id": "abc123"},
                    },
                },
            ],
        },
    }