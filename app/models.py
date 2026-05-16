from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    user_id : str
    message : str
    session_id : Optional[str] = None

class ChatResponse(BaseModel):
    reply : str
    memories_used : int
    latency_ms : float