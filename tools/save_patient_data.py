import json
from typing import Dict, List, Any
from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class PatientData(Base):
    __tablename__ = 'patient_data'
    session_id = Column(String, primary_key=True)
    data = Column(Text)

def save_patient_data(
    session_id: str,
    age: Dict[str, int],
    gender: str,
    city: str,
    country: str,
    date: str,
    complaints: List[Dict[str, str]],
    vitals: Dict[str, Any],
    physical_examination: str,
    comorbidities: List[str],
    past_history: Dict[str, List[str]],
    current_medications: List[str],
    family_history: List[str],
    allergies: Dict[str, List[str]],
    test_documents: List[str],
    test_results: List[str]
) -> str:
    """Save patient data to the database for the given session_id using SQLAlchemy."""
    patient_data = {
        "age": age,
        "gender": gender,
        "city": city,
        "country": country,
        "date": date,
        "complaints": complaints,
        "vitals": vitals,
        "physical_examination": physical_examination,
        "comorbidities": comorbidities,
        "past_history": past_history,
        "current_medications": current_medications,
        "family_history": family_history,
        "allergies": allergies,
        "test_documents": test_documents,
        "test_results": test_results
    }

    engine = create_engine('sqlite:///voice_ai.db')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        patient = session.query(PatientData).filter_by(session_id=session_id).first()
        if patient:
            patient.data = json.dumps(patient_data)
        else:
            patient = PatientData(session_id=session_id, data=json.dumps(patient_data))
            session.add(patient)
        session.commit()

    return f"Patient data saved successfully for session {session_id}."


def save_patient_data_schema():
    return {
        "name": "save_patient_data",
        "description": "Save the patient's complete medical information to the database for a given session. All fields are required; the AI should ask follow-up questions to gather any missing information before calling this tool. Lists should be empty if no information is available.",
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The unique session ID for the patient."
                },
                "age": {
                    "type": "object",
                    "properties": {
                        "years": {"type": "integer"},
                        "months": {"type": "integer"},
                        "days": {"type": "integer"}
                    },
                    "required": ["years", "months", "days"],
                    "description": "The patient's age in years, months, and days."
                },
                "gender": {
                    "type": "string",
                    "description": "The patient's gender (e.g., 'male', 'female')."
                },
                "city": {
                    "type": "string",
                    "description": "The patient's city."
                },
                "country": {
                    "type": "string",
                    "description": "The patient's country."
                },
                "date": {
                    "type": "string",
                    "description": "The date of the record in DD-MM-YYYY format."
                },
                "complaints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "symptom": {"type": "string"},
                            "severity": {"type": "string"},
                            "since": {"type": "string"}
                        },
                        "required": ["symptom", "severity", "since"]
                    },
                    "description": "List of patient's complaints, each with symptom, severity (low/medium/high), and since when."
                },
                "vitals": {
                    "type": "object",
                    "properties": {
                        "weight_kg": {"type": "number"},
                        "height_cm": {"type": "number"},
                        "bp_systolic": {"type": "integer"},
                        "bp_diastolic": {"type": "integer"},
                        "temperature_c": {"type": "number"},
                        "heart_rate": {"type": "integer"},
                        "respiration_rate": {"type": "integer"},
                        "spo2": {"type": "integer"},
                        "lmp": {"type": "string"}
                    },
                    "required": ["weight_kg", "height_cm", "bp_systolic", "bp_diastolic", "temperature_c", "heart_rate", "respiration_rate", "spo2", "lmp"],
                    "description": "Patient's vital signs."
                },
                "physical_examination": {
                    "type": "string",
                    "description": "Results of physical examination."
                },
                "comorbidities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of comorbidities (empty if none)."
                },
                "past_history": {
                    "type": "object",
                    "properties": {
                        "past_illnesses": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "previous_procedures": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["past_illnesses", "previous_procedures"],
                    "description": "Past illnesses and procedures (lists empty if none)."
                },
                "current_medications": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of current medications (empty if none)."
                },
                "family_history": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of family history items (empty if none)."
                },
                "allergies": {
                    "type": "object",
                    "properties": {
                        "drug_allergies": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "food_allergies": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["drug_allergies", "food_allergies"],
                    "description": "Allergies (lists empty if none)."
                },
                "test_documents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of test documents (empty if none)."
                },
                "test_results": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of test results (empty if none)."
                }
            },
            "required": [
                "session_id", "age", "gender", "city", "country", "date", "complaints",
                "vitals", "physical_examination", "comorbidities", "past_history",
                "current_medications", "family_history", "allergies", "test_documents",
                "test_results"
            ]
        },
        "examples": [
            {
                "input": "Save my medical info: I'm 24 years old, male, from Kolkata, India, etc.",
                "call": {
                    "name": "save_patient_data",
                    "arguments": {
                        "session_id": "test",
                        "age": {"years": 24, "months": 0, "days": 0},
                        "gender": "male",
                        "city": "kolkata",
                        "country": "india",
                        "date": "28-08-2025",
                        "complaints": [
                            {"symptom": "cough", "severity": "low", "since": "Monday"},
                            {"symptom": "nausea", "severity": "medium", "since": "26/08/2025"}
                        ],
                        "vitals": {
                            "weight_kg": 106,
                            "height_cm": 189,
                            "bp_systolic": 120,
                            "bp_diastolic": 80,
                            "temperature_c": 30,
                            "heart_rate": 80,
                            "respiration_rate": 22,
                            "spo2": 98,
                            "lmp": ""
                        },
                        "physical_examination": "a little chubby but lean",
                        "comorbidities": [],
                        "past_history": {"past_illnesses": [], "previous_procedures": []},
                        "current_medications": [],
                        "family_history": ["mother had a cerebral stroke in 2006"],
                        "allergies": {"drug_allergies": [], "food_allergies": []},
                        "test_documents": [],
                        "test_results": []
                    }
                }
            }
        ]
    }