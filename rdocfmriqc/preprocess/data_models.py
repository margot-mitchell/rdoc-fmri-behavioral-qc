"""
Data models for preprocessing.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class SubmissionData(BaseModel):
    """Model for submission data from Expfactory Deploy."""
    
    experiment_id: str = Field(..., description="ID of the experiment")
    experiment_name: str = Field(..., description="Name of the experiment")
    subject_id: str = Field(..., description="ID of the subject")
    session_id: str = Field(..., description="ID of the session")
    timestamp: str = Field(..., description="Timestamp of the submission")
    data: Dict[str, Any] = Field(..., description="Raw data from the experiment")
    
    class Config:
        arbitrary_types_allowed = True 