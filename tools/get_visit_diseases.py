from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, Column, String, Integer, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

class VisitDiseaseMaster(Base):
    __tablename__ = "visit_disease_master"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)  # UUID for the session
    disease_id = Column(Integer, nullable=False)  # Foreign key to disease_master

# Create database engine and session
engine = create_engine("sqlite:///voice_ai.db")
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_visit_diseases(session_id: str) -> str:
    """
    Retrieve all diseases associated with a specific session from visit_disease_master.
    Joins with disease_master to get disease details.

    Args:
        session_id: Unique session identifier

    Returns:
        JSON string containing session diseases or error message
    """
    try:
        if not session_id:
            return json.dumps({
                "success": False,
                "message": "Session ID is required"
            })

        db = SessionLocal()
        try:
            # Query visit diseases with disease details
            query = text("""
                SELECT 
                    vdm.id as visit_id,
                    vdm.session_id,
                    vdm.disease_id,
                    dm.disease_name,
                    dm.icd11_code,
                    dm.snowmed_ct
                FROM visit_disease_master vdm
                JOIN disease_master dm ON vdm.disease_id = dm.id
                WHERE vdm.session_id = :session_id
                ORDER BY vdm.id
            """)
            
            result = db.execute(query, {"session_id": session_id})
            visit_diseases = result.fetchall()

            if not visit_diseases:
                return json.dumps({
                    "success": True,
                    "message": "No diseases found for this session",
                    "session_id": session_id,
                    "diseases": []
                })

            # Format the results
            diseases_list = []
            for row in visit_diseases:
                visit_id, sess_id, disease_id, disease_name, icd11_code, snowmed_ct = row
                diseases_list.append({
                    "visit_id": visit_id,
                    "disease_id": disease_id,
                    "disease_name": disease_name,
                    "icd11_code": icd11_code or "",
                    "snowmed_ct": snowmed_ct or ""
                })

            return json.dumps({
                "success": True,
                "message": f"Found {len(diseases_list)} diseases for session {session_id}",
                "session_id": session_id,
                "diseases": diseases_list,
                "count": len(diseases_list)
            }, indent=2)

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error retrieving visit diseases: {e}")
        return json.dumps({
            "success": False,
            "message": f"Error retrieving visit diseases: {str(e)}"
        })


def get_visit_diseases_schema():
    """Return the function schema for the get_visit_diseases tool"""
    return {
        "type": "function",
        "function": {
            "name": "get_visit_diseases",
            "description": "Retrieve all diseases associated with the current session from visit_disease_master table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Unique session identifier",
                    },
                },
                "required": ["session_id"],
            },
            "examples": [
                {
                    "input": "What diseases have I confirmed in this session?",
                    "call": {
                        "name": "get_visit_diseases",
                        "arguments": {"session_id": "session-uuid-123"},
                    },
                },
            ],
        },
    }
