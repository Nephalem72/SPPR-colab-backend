from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    legal_top_k: int = 5
    case_top_k: int = 5


class SimilarCasesRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = 5


class ChatContextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    legal_top_k: int = 5
    case_top_k: int = 5


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    legal_top_k: int = 5
    case_top_k: int = 5
