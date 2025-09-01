from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, JSON
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
    age_years = Column(Integer)
    age_months = Column(Integer)
    age_days = Column(Integer)
    gender = Column(String)
    city = Column(String)
    country = Column(String)
    date = Column(String)
    complaints = Column(JSON)  # List of complaint objects
    weight_kg = Column(Integer)
    height_cm = Column(Integer)
    bp_systolic = Column(Integer)
    bp_diastolic = Column(Integer)
    temperature_c = Column(Integer)
    heart_rate = Column(Integer)
    respiration_rate = Column(Integer)
    spo2 = Column(Integer)
    lmp = Column(String)
    physical_examination = Column(Text)
    comorbidities = Column(JSON)  # List
    past_illnesses = Column(JSON)  # List
    previous_procedures = Column(JSON)  # List
    current_medications = Column(JSON)  # List
    family_history = Column(JSON)  # List
    drug_allergies = Column(JSON)  # List
    food_allergies = Column(JSON)  # List
    test_documents = Column(JSON)  # List
    test_results = Column(JSON)  # List
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Create database engine and session
engine = create_engine("sqlite:///voice_ai.db")
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def save_patient_data(
    session_id: str,
    age_years: int = 0,
    age_months: int = 0,
    age_days: int = 0,
    gender: str = "",
    city: str = "",
    country: str = "",
    date: str = None,  # Changed to None to allow automatic fetching
    complaints: List[Dict[str, str]] = None,
    weight_kg: int = 0,
    height_cm: int = 0,
    bp_systolic: int = 0,
    bp_diastolic: int = 0,
    temperature_c: int = 0,
    heart_rate: int = 0,
    respiration_rate: int = 0,
    spo2: int = 0,
    lmp: str = "",
    physical_examination: str = "",
    comorbidities: List[str] = None,
    past_illnesses: List[str] = None,
    previous_procedures: List[str] = None,
    current_medications: List[str] = None,
    family_history: List[str] = None,
    drug_allergies: List[str] = None,
    food_allergies: List[str] = None,
    test_documents: List[str] = None,
    test_results: List[str] = None,
) -> str:
    """
    Save patient data to the database. Creates or updates existing record. If date is not provided, uses get_time to set it.

    Args:
        session_id: Unique session identifier
        age_years: Patient age in years
        age_months: Additional months of age
        age_days: Additional days of age
        gender: Patient gender
        city: Patient city
        country: Patient country
        date: Current date (if None, fetched via get_time)
        complaints: List of complaint objects with symptom, severity, since
        weight_kg: Weight in kilograms
        height_cm: Height in centimeters
        bp_systolic: Systolic blood pressure
        bp_diastolic: Diastolic blood pressure
        temperature_c: Temperature in Celsius
        heart_rate: Heart rate
        respiration_rate: Respiration rate
        spo2: Oxygen saturation
        lmp: Last menstrual period
        physical_examination: Physical examination findings
        comorbidities: List of comorbidities
        past_illnesses: List of past illnesses
        previous_procedures: List of previous procedures
        current_medications: List of current medications
        family_history: List of family history items
        drug_allergies: List of drug allergies
        food_allergies: List of food allergies
        test_documents: List of test documents
        test_results: List of test results

    Returns:
        Success message
    """
    try:
        # Set date if not provided
        if date is None:
            date = get_time().split('T')[0]  # Extract YYYY-MM-DD from ISO 8601

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

        db = SessionLocal()
        try:
            # Check if patient data already exists
            existing_patient = (
                db.query(PatientData)
                .filter(PatientData.session_id == session_id)
                .first()
            )

            if existing_patient:
                # Update existing record
                existing_patient.age_years = age_years
                existing_patient.age_months = age_months
                existing_patient.age_days = age_days
                existing_patient.gender = gender
                existing_patient.city = city
                existing_patient.country = country
                existing_patient.date = date
                existing_patient.complaints = complaints
                existing_patient.weight_kg = weight_kg
                existing_patient.height_cm = height_cm
                existing_patient.bp_systolic = bp_systolic
                existing_patient.bp_diastolic = bp_diastolic
                existing_patient.temperature_c = temperature_c
                existing_patient.heart_rate = heart_rate
                existing_patient.respiration_rate = respiration_rate
                existing_patient.spo2 = spo2
                existing_patient.lmp = lmp
                existing_patient.physical_examination = physical_examination
                existing_patient.comorbidities = comorbidities
                existing_patient.past_illnesses = past_illnesses
                existing_patient.previous_procedures = previous_procedures
                existing_patient.current_medications = current_medications
                existing_patient.family_history = family_history
                existing_patient.drug_allergies = drug_allergies
                existing_patient.food_allergies = food_allergies
                existing_patient.test_documents = test_documents
                existing_patient.test_results = test_results
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
                    physical_examination=physical_examination,
                    comorbidities=comorbidities,
                    past_illnesses=past_illnesses,
                    previous_procedures=previous_procedures,
                    current_medications=current_medications,
                    family_history=family_history,
                    drug_allergies=drug_allergies,
                    food_allergies=food_allergies,
                    test_documents=test_documents,
                    test_results=test_results,
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
    return {
        "type": "function",
        "function": {
            "name": "save_patient_data",
            "description": "Save comprehensive patient medical data to the database. Use this tool once you have collected all necessary patient information through conversation. If date is not provided, it will be set automatically using get_time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Unique session identifier",
                    },
                    "age_years": {
                        "type": "integer",
                        "description": "Patient age in years",
                        "default": 0,
                    },
                    "age_months": {
                        "type": "integer",
                        "description": "Additional months of age",
                        "default": 0,
                    },
                    "age_days": {
                        "type": "integer",
                        "description": "Additional days of age",
                        "default": 0,
                    },
                    "gender": {"type": "string", "description": "Patient gender"},
                    "city": {"type": "string", "description": "Patient city"},
                    "country": {"type": "string", "description": "Patient country"},
                    "date": {
                        "type": "string",
                        "description": "Current date (automatically set if not provided)",
                        "default": None,
                    },
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
                    "weight_kg": {
                        "type": "integer",
                        "description": "Weight in kilograms",
                        "default": 0,
                    },
                    "height_cm": {
                        "type": "integer",
                        "description": "Height in centimeters",
                        "default": 0,
                    },
                    "bp_systolic": {
                        "type": "integer",
                        "description": "Systolic blood pressure",
                        "default": 0,
                    },
                    "bp_diastolic": {
                        "type": "integer",
                        "description": "Diastolic blood pressure",
                        "default": 0,
                    },
                    "temperature_c": {
                        "type": "integer",
                        "description": "Temperature in Celsius",
                        "default": 0,
                    },
                    "heart_rate": {
                        "type": "integer",
                        "description": "Heart rate",
                        "default": 0,
                    },
                    "respiration_rate": {
                        "type": "integer",
                        "description": "Respiration rate",
                        "default": 0,
                    },
                    "spo2": {
                        "type": "integer",
                        "description": "Oxygen saturation",
                        "default": 0,
                    },
                    "lmp": {"type": "string", "description": "Last menstrual period"},
                    "physical_examination": {
                        "type": "string",
                        "description": "Physical examination findings",
                    },
                    "comorbidities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of comorbidities",
                    },
                    "past_illnesses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of past illnesses",
                    },
                    "previous_procedures": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of previous medical procedures",
                    },
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
                    "test_documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of test documents",
                    },
                    "test_results": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of test results",
                    },
                },
                "required": ["session_id"],
            },
            "examples": [
                {
                    "input": "I've collected all the patient information",
                    "call": {
                        "name": "save_patient_data",
                        "arguments": {
                            "session_id": "abc123",
                            "age_years": 35,
                            "gender": "Female",
                            "city": "Mumbai",
                            "country": "India",
                            "complaints": [
                                {
                                    "symptom": "Headache",
                                    "severity": "Moderate",
                                    "since": "3 days",
                                }
                            ],
                            "weight_kg": 65,
                            "height_cm": 160,
                            "current_medications": ["Paracetamol 500mg"],
                        },
                    },
                }
            ],
        },
    }