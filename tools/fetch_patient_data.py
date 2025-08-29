import json
from typing import List, Dict, Any
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Assuming the Base and PatientData are defined in the same module or imported
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class PatientData(Base):
    __tablename__ = 'patient_data'
    session_id = Column(String, primary_key=True)
    data = Column(Text)

def fetch_patient_data(session_id: str, fields: List[str]) -> str:
    """Fetch and format specific patient data fields from the database for the given session_id using SQLAlchemy."""
    engine = create_engine('sqlite:///voice_ai.db')
    Session = sessionmaker(bind=engine)
    with Session() as session:
        patient = session.query(PatientData).filter_by(session_id=session_id).first()
        if not patient:
            return "No patient data found for this session."

        data: Dict[str, Any] = json.loads(patient.data)
        result_parts: List[str] = []

        for field in fields:
            if field not in data:
                result_parts.append(f"{field.capitalize()}: not available")
                continue
            value = data[field]

            if field == "age":
                v = value
                result_parts.append(f"Age: {v['years']} years, {v['months']} months, {v['days']} days")
            elif field in ["gender", "city", "country", "date"]:
                result_parts.append(f"{field.capitalize()}: {value}")
            elif field == "complaints":
                if not value:
                    result_parts.append("Complaints: none")
                else:
                    comp_str = "Complaints:\n" + "\n".join(
                        f"- {c['symptom']} ({c['severity']} severity) since {c['since']}" for c in value
                    )
                    result_parts.append(comp_str)
            elif field == "vitals":
                v = value
                vit_str = (
                    f"Vitals:\n"
                    f"- Weight: {v['weight_kg']} kg\n"
                    f"- Height: {v['height_cm']} cm\n"
                    f"- Blood pressure: {v['bp_systolic']}/{v['bp_diastolic']} mmHg\n"
                    f"- Temperature: {v['temperature_c']} °C\n"
                    f"- Heart rate: {v['heart_rate']} bpm\n"
                    f"- Respiration rate: {v['respiration_rate']} breaths/min\n"
                    f"- SpO2: {v['spo2']}%\n"
                    f"- LMP: {v['lmp'] or 'none'}"
                )
                result_parts.append(vit_str)
            elif field == "physical_examination":
                result_parts.append(f"Physical examination: {value or 'none'}")
            elif field in ["comorbidities", "current_medications", "test_documents", "test_results"]:
                result_parts.append(f"{field.capitalize()}: {', '.join(value) if value else 'none'}")
            elif field == "past_history":
                ph = value
                ill = ", ".join(ph['past_illnesses']) if ph['past_illnesses'] else "none"
                proc = ", ".join(ph['previous_procedures']) if ph['previous_procedures'] else "none"
                result_parts.append(f"Past history:\n- Illnesses: {ill}\n- Procedures: {proc}")
            elif field == "family_history":
                result_parts.append(f"Family history: {', '.join(value) if value else 'none'}")
            elif field == "allergies":
                al = value
                drug = ", ".join(al['drug_allergies']) if al['drug_allergies'] else "none"
                food = ", ".join(al['food_allergies']) if al['food_allergies'] else "none"
                result_parts.append(f"Allergies:\n- Drug allergies: {drug}\n- Food allergies: {food}")
            else:
                result_parts.append(f"{field.capitalize()}: {str(value)}")

        return "\n\n".join(result_parts)


def fetch_patient_data_schema():
    return {
        "name": "fetch_patient_data",
        "description": "Fetch specific patient data fields from the database for a session and return them in a formatted, natural language string. Specify the fields to retrieve (e.g., 'allergies', 'vitals'). The AI can then incorporate this directly into a response.",
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The unique session ID for the patient."
                },
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of fields to fetch (e.g., ['allergies', 'vitals', 'complaints'])."
                }
            },
            "required": ["session_id", "fields"]
        },
        "examples": [
            {
                "input": "What allergies do I have?",
                "call": {
                    "name": "fetch_patient_data",
                    "arguments": {"session_id": "test", "fields": ["allergies"]}
                }
            },
            {
                "input": "Tell me my vitals and complaints.",
                "call": {
                    "name": "fetch_patient_data",
                    "arguments": {"session_id": "test", "fields": ["vitals", "complaints"]}
                }
            },
            {
                "input": "What's my family history?",
                "call": {
                    "name": "fetch_patient_data",
                    "arguments": {"session_id": "test", "fields": ["family_history"]}
                }
            }
        ]
    }