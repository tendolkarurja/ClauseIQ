from pydantic import BaseModel, Field
from typing import Optional

class Clause(BaseModel):
    """
    Represents a clause in a legal document.
    """
    title: str = Field(..., description="The title of the clause")
    text: str = Field(..., description="The text content of the clause")    
    section: Optional[str] = Field(None, description="The section of the contract where the clause is located (if seperately specified)")
    
    
class Contract(BaseModel):
    """
    Represents a legal contract consisting of multiple clauses.
    """
    title: str = Field(..., description="The title of the contract")
    metadata: Optional[dict] = Field(None, description="Additional metadata about the contract")
    clauses: list[Clause] = Field(..., description="A list of clauses in the contract")