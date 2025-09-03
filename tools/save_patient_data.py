from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json
import logging
from .get_time import get_time

logger = logging.getLogger(__name__)

Base = declarative_base()

class PatientData(Base):
    __tablename__ = "patient_data"

    session_id = Column(String, primary_key=True)
    
    # Age structure (keeping flat for easier queries while maintaining JSON support)
    age_years = Column(Integer, default=0)
    age_months = Column(Integer, default=0)
    age_days = Column(Integer, default=0)
    
    # Basic info
    gender = Column(String, default="")
    city = Column(String, default="")
    country = Column(String, default="")
    date = Column(String, default="")  # Format: DD-MM-YYYY to match your JSON
    
    # Complaints as JSON array
    complaints = Column(JSON, default=list)  # List of complaint objects
    
    # Vitals - using Float for temperature, Integer for others
    weight_kg = Column(Integer, default=0)
    height_cm = Column(Integer, default=0)
    bp_systolic = Column(Integer, default=0)
    bp_diastolic = Column(Integer, default=0)
    temperature_c = Column(Float, default=0.0)  # Changed to Float for decimal temperatures
    heart_rate = Column(Integer, default=0)
    respiration_rate = Column(Integer, default=0)
    spo2 = Column(Integer, default=0)
    lmp = Column(String, default="")  # Format: DD-MM-YYYY
    
    # Clinical findings - FIXED: Changed to Text instead of JSON for physical_examination
    physical_examination = Column(Text, default="")
    
    # Medical history as JSON arrays
    comorbidities = Column(JSON, default=list)  # List of strings
    
    # Past history - stored as JSON objects to support date information
    past_illnesses = Column(JSON, default=list)  # List of {"illness": str, "date": str}
    previous_procedures = Column(JSON, default=list)  # List of {"procedure": str, "date": str}
    
    # Current treatment
    current_medications = Column(JSON, default=list)  # List of strings
    
    # Family and allergies
    family_history = Column(JSON, default=list)  # List of strings
    drug_allergies = Column(JSON, default=list)  # List of strings
    food_allergies = Column(JSON, default=list)  # List of strings
    
    # Test data
    test_documents = Column(JSON, default=list)  # List of URLs/paths
    test_results = Column(JSON, default=list)  # List of test result objects
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Create database engine and session
engine = create_engine("sqlite:///voice_ai.db")
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def save_patient_data(
    session_id: str,
    # Age can be passed as nested dict or individual values
    age: Optional[Dict[str, int]] = None,
    age_years: int = 0,
    age_months: int = 0,
    age_days: int = 0,
    # Basic info
    gender: str = "",
    city: str = "",
    country: str = "",
    date: str = None,
    # Complaints
    complaints: List[Dict[str, str]] = None,
    # Vitals can be passed as nested dict or individual values
    vitals: Optional[Dict[str, Any]] = None,
    weight_kg: int = 0,
    height_cm: int = 0,
    bp_systolic: int = 0,
    bp_diastolic: int = 0,
    temperature_c: float = 0.0,  # Changed to float
    heart_rate: int = 0,
    respiration_rate: int = 0,
    spo2: int = 0,
    lmp: str = "",
    # Clinical findings - FIXED: Changed to handle both string and dict
    physical_examination: Any = "",
    comorbidities: List[str] = None,
    # Past history - can accept both formats
    past_history: Optional[Dict[str, List[Dict]]] = None,
    past_illnesses: List[Any] = None,  # Can be strings or objects
    previous_procedures: List[Any] = None,  # Can be strings or objects
    # Current treatment
    current_medications: List[str] = None,
    family_history: List[str] = None,
    # Allergies - can be passed as nested dict or individual lists
    allergies: Optional[Dict[str, List[str]]] = None,
    drug_allergies: List[str] = None,
    food_allergies: List[str] = None,
    # Test data
    test_documents: List[str] = None,
    test_results: List[Dict] = None,
) -> str:
    """
    Save patient data to the database. Creates or updates existing record.
    Supports both flat parameter style and nested JSON structure.

    Args:
        session_id: Unique session identifier
        age: Nested age object with years, months, days (optional)
        age_years, age_months, age_days: Individual age components
        gender: Patient gender
        city: Patient city
        country: Patient country
        date: Current date in DD-MM-YYYY format (auto-generated if None)
        complaints: List of complaint objects with symptom, severity, since
        vitals: Nested vitals object (optional)
        weight_kg, height_cm, etc.: Individual vital components
        physical_examination: Physical examination findings (string or dict)
        comorbidities: List of comorbidities
        past_history: Nested past history object (optional)
        past_illnesses: List of past illnesses (strings or objects)
        previous_procedures: List of previous procedures (strings or objects)
        current_medications: List of current medications
        family_history: List of family history items
        allergies: Nested allergies object (optional)
        drug_allergies, food_allergies: Individual allergy lists
        test_documents: List of test document URLs
        test_results: List of test result objects

    Returns:
        Success/error message
    """
    try:
        # Set date if not provided - convert to DD-MM-YYYY format
        if date is None:
            iso_date = get_time().split('T')[0]  # Get YYYY-MM-DD
            year, month, day = iso_date.split('-')
            date = f"{day}-{month}-{year}"  # Convert to DD-MM-YYYY

        # Handle nested age structure
        if age:
            age_years = age.get("years", age_years)
            age_months = age.get("months", age_months)
            age_days = age.get("days", age_days)

        # Handle nested vitals structure
        if vitals:
            weight_kg = vitals.get("weight_kg", weight_kg)
            height_cm = vitals.get("height_cm", height_cm)
            bp_systolic = vitals.get("bp_systolic", bp_systolic)
            bp_diastolic = vitals.get("bp_diastolic", bp_diastolic)
            temperature_c = vitals.get("temperature_c", temperature_c)
            heart_rate = vitals.get("heart_rate", heart_rate)
            respiration_rate = vitals.get("respiration_rate", respiration_rate)
            spo2 = vitals.get("spo2", spo2)
            lmp = vitals.get("lmp", lmp)

        # Handle nested past_history structure
        if past_history:
            past_illnesses = past_history.get("past_illnesses", past_illnesses)
            previous_procedures = past_history.get("previous_procedures", previous_procedures)

        # Handle nested allergies structure
        if allergies:
            # FIXED: Handle both 'drug_allergies'/'food_allergies' and 'drug'/'food' keys
            if "drug_allergies" in allergies:
                drug_allergies = allergies.get("drug_allergies", drug_allergies)
            elif "drug" in allergies:
                drug_allergies = allergies.get("drug", drug_allergies)
                
            if "food_allergies" in allergies:
                food_allergies = allergies.get("food_allergies", food_allergies)
            elif "food" in allergies:
                food_allergies = allergies.get("food", food_allergies)

        # FIXED: Handle physical_examination - convert dict to string if needed
        if isinstance(physical_examination, dict):
            # Convert dict to readable string format
            exam_parts = []
            for key, value in physical_examination.items():
                exam_parts.append(f"{key.title()}: {value}")
            physical_examination = "; ".join(exam_parts)
        elif not isinstance(physical_examination, str):
            physical_examination = str(physical_examination)

        # Initialize empty lists if None
        if complaints is None:
            complaints = []
        if comorbidities is None:
            comorbidities = []
        if past_illnesses is None:
            past_illnesses = []
        if previous_procedures is None:
            previous_procedures = []
        if current_medications is None:
            current_medications = []
        if family_history is None:
            family_history = []
        if drug_allergies is None:
            drug_allergies = []
        if food_allergies is None:
            food_allergies = []
        if test_documents is None:
            test_documents = []
        if test_results is None:
            test_results = []

        # Normalize past_illnesses and previous_procedures to object format
        normalized_past_illnesses = []
        for illness in past_illnesses:
            if isinstance(illness, str):
                normalized_past_illnesses.append({"illness": illness, "date": ""})
            elif isinstance(illness, dict):
                normalized_past_illnesses.append(illness)
        
        normalized_previous_procedures = []
        for procedure in previous_procedures:
            if isinstance(procedure, str):
                normalized_previous_procedures.append({"procedure": procedure, "date": ""})
            elif isinstance(procedure, dict):
                normalized_previous_procedures.append(procedure)

        # FIXED: Handle test_results - ensure proper structure
        normalized_test_results = []
        if test_results:
            for test in test_results:
                if isinstance(test, dict):
                    # Handle both 'parameters' and 'test_result_values' keys
                    if "parameters" in test:
                        # Convert 'parameters' to 'test_result_values' for consistency
                        normalized_test = {
                            "test_name": test.get("test_name", ""),
                            "test_result_values": test["parameters"]
                        }
                    elif "test_result_values" in test:
                        normalized_test = test
                    else:
                        # Default structure
                        normalized_test = {
                            "test_name": test.get("test_name", ""),
                            "test_result_values": []
                        }
                    normalized_test_results.append(normalized_test)

        db = SessionLocal()
        try:
            # Check if patient data already exists
            existing_patient = (
                db.query(PatientData)
                .filter(PatientData.session_id == session_id)
                .first()
            )

            if existing_patient:
                # Update existing record with non-zero/non-empty values only
                if age_years > 0:
                    existing_patient.age_years = age_years
                if age_months > 0:
                    existing_patient.age_months = age_months
                if age_days > 0:
                    existing_patient.age_days = age_days
                if gender:
                    existing_patient.gender = gender
                if city:
                    existing_patient.city = city
                if country:
                    existing_patient.country = country
                if date:
                    existing_patient.date = date
                
                # Update complaints (replace if provided)
                if complaints:
                    existing_patient.complaints = complaints
                
                # Update vitals (only non-zero values)
                if weight_kg > 0:
                    existing_patient.weight_kg = weight_kg
                if height_cm > 0:
                    existing_patient.height_cm = height_cm
                if bp_systolic > 0:
                    existing_patient.bp_systolic = bp_systolic
                if bp_diastolic > 0:
                    existing_patient.bp_diastolic = bp_diastolic
                if temperature_c > 0:
                    existing_patient.temperature_c = temperature_c
                if heart_rate > 0:
                    existing_patient.heart_rate = heart_rate
                if respiration_rate > 0:
                    existing_patient.respiration_rate = respiration_rate
                if spo2 > 0:
                    existing_patient.spo2 = spo2
                if lmp:
                    existing_patient.lmp = lmp
                
                # Update text fields
                if physical_examination:
                    existing_patient.physical_examination = physical_examination
                
                # Update lists (replace if provided)
                if comorbidities:
                    existing_patient.comorbidities = comorbidities
                if normalized_past_illnesses:
                    existing_patient.past_illnesses = normalized_past_illnesses
                if normalized_previous_procedures:
                    existing_patient.previous_procedures = normalized_previous_procedures
                if current_medications:
                    existing_patient.current_medications = current_medications
                if family_history:
                    existing_patient.family_history = family_history
                if drug_allergies:
                    existing_patient.drug_allergies = drug_allergies
                if food_allergies:
                    existing_patient.food_allergies = food_allergies
                if test_documents:
                    existing_patient.test_documents = test_documents
                if normalized_test_results:
                    existing_patient.test_results = normalized_test_results
                
                existing_patient.updated_at = datetime.utcnow()
                logger.info(f"Updated patient data for session: {session_id}")

            else:
                # Create new record
                new_patient = PatientData(
                    session_id=session_id,
                    age_years=age_years,
                    age_months=age_months,
                    age_days=age_days,
                    gender=gender,
                    city=city,
                    country=country,
                    date=date,
                    complaints=complaints,
                    weight_kg=weight_kg,
                    height_cm=height_cm,
                    bp_systolic=bp_systolic,
                    bp_diastolic=bp_diastolic,
                    temperature_c=temperature_c,
                    heart_rate=heart_rate,
                    respiration_rate=respiration_rate,
                    spo2=spo2,
                    lmp=lmp,
                    physical_examination=physical_examination,  # Now properly handled as string
                    comorbidities=comorbidities,
                    past_illnesses=normalized_past_illnesses,
                    previous_procedures=normalized_previous_procedures,
                    current_medications=current_medications,
                    family_history=family_history,
                    drug_allergies=drug_allergies,
                    food_allergies=food_allergies,
                    test_documents=test_documents,
                    test_results=normalized_test_results,  # Now properly structured
                )

                db.add(new_patient)
                logger.info(f"Created new patient data for session: {session_id}")

            db.commit()
            return "Patient data saved successfully"

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error saving patient data: {e}")
        return f"Error saving patient data: {str(e)}"

def save_patient_data_schema():
    """Return the function schema for the save_patient_data tool"""
    return {
        "type": "function",
        "function": {
            "name": "save_patient_data",
            "description": "Save comprehensive patient medical data to the database. Supports both nested JSON structure and flat parameters. Use this tool once you have collected patient information through conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Unique session identifier",
                    },
                    # Age - support both nested and flat
                    "age": {
                        "type": "object",
                        "description": "Nested age object",
                        "properties": {
                            "years": {"type": "integer", "default": 0},
                            "months": {"type": "integer", "default": 0},
                            "days": {"type": "integer", "default": 0}
                        }
                    },
                    "age_years": {"type": "integer", "description": "Age in years (alternative to nested age)", "default": 0},
                    "age_months": {"type": "integer", "description": "Additional months", "default": 0},
                    "age_days": {"type": "integer", "description": "Additional days", "default": 0},
                    
                    # Basic info
                    "gender": {"type": "string", "description": "Patient gender"},
                    "city": {"type": "string", "description": "Patient city"},
                    "country": {"type": "string", "description": "Patient country"},
                    "date": {
                        "type": "string",
                        "description": "Current date in DD-MM-YYYY format (auto-generated if not provided)",
                    },
                    
                    # Complaints
                    "complaints": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "symptom": {"type": "string"},
                                "severity": {"type": "string"},
                                "since": {"type": "string"},
                            },
                        },
                        "description": "List of patient complaints with symptom, severity, and duration",
                    },
                    
                    # Vitals - support both nested and flat
                    "vitals": {
                        "type": "object",
                        "description": "Nested vitals object",
                        "properties": {
                            "weight_kg": {"type": "integer"},
                            "height_cm": {"type": "integer"},
                            "bp_systolic": {"type": "integer"},
                            "bp_diastolic": {"type": "integer"},
                            "temperature_c": {"type": "number"},
                            "heart_rate": {"type": "integer"},
                            "respiration_rate": {"type": "integer"},
                            "spo2": {"type": "integer"},
                            "lmp": {"type": "string"}
                        }
                    },
                    "weight_kg": {"type": "integer", "description": "Weight in kg", "default": 0},
                    "height_cm": {"type": "integer", "description": "Height in cm", "default": 0},
                    "bp_systolic": {"type": "integer", "description": "Systolic BP", "default": 0},
                    "bp_diastolic": {"type": "integer", "description": "Diastolic BP", "default": 0},
                    "temperature_c": {"type": "number", "description": "Temperature in Celsius", "default": 0.0},
                    "heart_rate": {"type": "integer", "description": "Heart rate", "default": 0},
                    "respiration_rate": {"type": "integer", "description": "Respiration rate", "default": 0},
                    "spo2": {"type": "integer", "description": "Oxygen saturation", "default": 0},
                    "lmp": {"type": "string", "description": "Last menstrual period (DD-MM-YYYY)"},
                    
                    # Clinical findings
                    "physical_examination": {
                        "type": "string",
                        "description": "Physical examination findings as text",
                    },
                    
                    # Medical history
                    "comorbidities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of comorbidities",
                    },
                    
                    # Past history - support both nested and flat
                    "past_history": {
                        "type": "object",
                        "description": "Nested past history object",
                        "properties": {
                            "past_illnesses": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "illness": {"type": "string"},
                                        "date": {"type": "string"}
                                    }
                                }
                            },
                            "previous_procedures": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "procedure": {"type": "string"},
                                        "date": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "past_illnesses": {
                        "type": "array",
                        "items": {
                            "oneOf": [
                                {"type": "string"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "illness": {"type": "string"},
                                        "date": {"type": "string"}
                                    }
                                }
                            ]
                        },
                        "description": "List of past illnesses (strings or objects with date)",
                    },
                    "previous_procedures": {
                        "type": "array",
                        "items": {
                            "oneOf": [
                                {"type": "string"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "procedure": {"type": "string"},
                                        "date": {"type": "string"}
                                    }
                                }
                            ]
                        },
                        "description": "List of previous procedures (strings or objects with date)",
                    },
                    
                    # Current treatment
                    "current_medications": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of current medications",
                    },
                    "family_history": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of family history items",
                    },
                    
                    # Allergies - support both nested and flat
                    "allergies": {
                        "type": "object",
                        "description": "Nested allergies object",
                        "properties": {
                            "drug_allergies": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "food_allergies": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        }
                    },
                    "drug_allergies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of drug allergies",
                    },
                    "food_allergies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of food allergies",
                    },
                    
                    # Test data
                    "test_documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of test document URLs",
                    },
                    "test_results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "test_name": {"type": "string"},
                                "test_result_values": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "parameterName": {"type": "string"},
                                            "parameterValue": {"type": "string"},
                                            "parameterUnit": {"type": "string"},
                                            "parameterRefRange": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        },
                        "description": "List of structured test results",
                    },
                },
                "required": ["session_id"],
            }
        }
    }