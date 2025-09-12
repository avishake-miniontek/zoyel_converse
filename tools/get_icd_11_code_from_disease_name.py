import logging
from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()

class DiseaseMaster(Base):
    __tablename__ = 'disease_master'

    id = Column(Integer, primary_key=True)
    disease_name = Column(String, nullable=False)
    icd11_code = Column(String)
    org_id = Column(String)
    active_flag = Column(String)
    country_code = Column(String)
    snowmed_ct = Column(String, nullable=True)

# Create database engine and session
engine = create_engine("sqlite:///voice_ai.db")
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_icd_11_code_from_disease_name(disease_name: str) -> str:
    """
    Get the ICD-11 code for a given disease name from the disease_master database.
    Performs case-insensitive search.

    Args:
        disease_name: The name of the disease (case-insensitive)

    Returns:
        The ICD-11 code as a string, or empty string if not found
    """
    try:
        db = SessionLocal()
        try:
            # Perform case-insensitive search
            result = (
                db.query(DiseaseMaster.icd11_code)
                .filter(DiseaseMaster.disease_name.ilike(disease_name))
                .first()
            )

            if result and result.icd11_code:
                return result.icd11_code
            else:
                return ""

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error querying ICD-11 code for disease '{disease_name}': {e}")
        return ""

def get_icd_11_code_from_disease_name_schema():
    """Return the function schema for the get_icd_11_code_from_disease_name tool"""
    return {
        "type": "function",
        "function": {
            "name": "get_icd_11_code_from_disease_name",
            "description": "Get the ICD-11 code for a disease name from the disease_master database. Performs case-insensitive search on disease names like 'COMMON COLD', 'commOn Cold', or 'common cold'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "disease_name": {
                        "type": "string",
                        "description": "The name of the disease to search for (case-insensitive)",
                    },
                },
                "required": ["disease_name"],
            },
            "examples": [
                {
                    "input": "What is the ICD-11 code for common cold?",
                    "call": {"name": "get_icd_11_code_from_disease_name", "arguments": {"disease_name": "common cold"}}
                },
                {
                    "input": "Get ICD code for DIABETES MELLITUS",
                    "call": {"name": "get_icd_11_code_from_disease_name", "arguments": {"disease_name": "DIABETES MELLITUS"}}
                }
            ]
        }
    }
