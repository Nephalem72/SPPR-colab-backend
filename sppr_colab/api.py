from __future__ import annotations

from fastapi import FastAPI

from .schemas import AnalyzeRequest, ChatContextRequest, ChatRequest, SimilarCasesRequest
from .service import analyze_text, answer_chat, build_chat_context, health_status, warmup
from .retrieval import search_similar_cases


app = FastAPI(title="SPPR Colab Backend")


@app.get("/health")
def health() -> dict:
    return health_status()


@app.get("/warmup")
def warmup_endpoint() -> dict:
    return warmup()


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    return analyze_text(request.text, legal_top_k=request.legal_top_k, case_top_k=request.case_top_k)


@app.post("/similar_cases")
def similar_cases(request: SimilarCasesRequest) -> dict:
    return {"query": request.query, "items": search_similar_cases(request.query, request.top_k)}


@app.post("/chat_context")
def chat_context(request: ChatContextRequest) -> dict:
    return build_chat_context(request.text, legal_top_k=request.legal_top_k, case_top_k=request.case_top_k)


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    return answer_chat(request.text, request.question, legal_top_k=request.legal_top_k, case_top_k=request.case_top_k)
