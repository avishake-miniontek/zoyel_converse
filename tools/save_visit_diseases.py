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

def save_visit_diseases(session_id: str, disease_ids: List[int]) -> str:
    """
    Save confirmed diseases to the visit_disease_master table for the current session.
    This function stores the association between a session and the diseases the user confirmed.

    Args:
        session_id: Unique session identifier (UUID)
        disease_ids: List of disease IDs from disease_master table that user confirmed

    Returns:
        Success/error message as JSON string
    """
    try:
        if not session_id:
            return json.dumps({
                "success": False,
                "message": "Session ID is required"
            })

        if not disease_ids or not isinstance(disease_ids, list):
            return json.dumps({
                "success": False,
                "message": "Disease IDs must be provided as a list"
            })

        # Remove duplicates while preserving order
        disease_ids = list(dict.fromkeys(disease_ids))

        db = SessionLocal()
        try:
            # First, verify that all disease IDs exist in disease_master
            if len(disease_ids) == 1:
                # Handle single item case
                verification_query = text("""
                    SELECT id, disease_name 
                    FROM disease_master 
                    WHERE id = :disease_id AND active_flag = 'Y'
                """)
                result = db.execute(verification_query, {"disease_id": disease_ids[0]})
            else:
                # Handle multiple items case
                verification_query = text("""
                    SELECT id, disease_name 
                    FROM disease_master 
                    WHERE id IN :disease_ids AND active_flag = 'Y'
                """)
                result = db.execute(verification_query, {"disease_ids": tuple(disease_ids)})
            
            existing_diseases = result.fetchall()
            existing_ids = [row[0] for row in existing_diseases]
            existing_names = {row[0]: row[1] for row in existing_diseases}

            # Check for invalid disease IDs
            invalid_ids = [did for did in disease_ids if did not in existing_ids]
            if invalid_ids:
                return json.dumps({
                    "success": False,
                    "message": f"Invalid disease IDs found: {invalid_ids}. These diseases don't exist or are inactive.",
                    "invalid_ids": invalid_ids
                })

            # Check if diseases for this session already exist and remove them first
            # This allows for updating the disease list if user changes their mind
            delete_query = text("""
                DELETE FROM visit_disease_master 
                WHERE session_id = :session_id
            """)
            db.execute(delete_query, {"session_id": session_id})

            # Insert new disease associations
            inserted_count = 0
            for disease_id in disease_ids:
                new_visit_disease = VisitDiseaseMaster(
                    session_id=session_id,
                    disease_id=disease_id
                )
                db.add(new_visit_disease)
                inserted_count += 1

            db.commit()

            # Prepare response with saved diseases
            saved_diseases = [
                {
                    "disease_id": did,
                    "disease_name": existing_names.get(did, "Unknown")
                }
                for did in disease_ids
            ]

            return json.dumps({
                "success": True,
                "message": f"Successfully saved {inserted_count} diseases for session {session_id}",
                "session_id": session_id,
                "saved_diseases": saved_diseases,
                "count": inserted_count
            }, indent=2)

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error saving visit diseases: {e}")
        return json.dumps({
            "success": False,
            "message": f"Error saving visit diseases: {str(e)}"
        })


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


def save_visit_diseases_schema():
    """Return the function schema for the save_visit_diseases tool"""
    return {
        "type": "function",
        "function": {
            "name": "save_visit_diseases",
            "description": "Save confirmed diseases to the visit_disease_master table for the current session. Use this after the user has confirmed which diseases from the search results are correct.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Unique session identifier (UUID)",
                    },
                    "disease_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of disease IDs from disease_master table that the user confirmed",
                    },
                },
                "required": ["session_id", "disease_ids"],
            },
            "examples": [
                {
                    "input": "User confirms diseases with IDs 1, 556, and 557",
                    "call": {
                        "name": "save_visit_diseases",
                        "arguments": {
                            "session_id": "session-uuid-123",
                            "disease_ids": [1, 556, 557]
                        },
                    },
                },
            ],
        },
    }


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