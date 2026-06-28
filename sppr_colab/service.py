from __future__ import annotations

from typing import Any

from .extraction import extract_case_facts, normalize_text
from .llm import generate_chat_answer
from .models import predict_role_ml
from .retrieval import fetch_full_cases, health_status, search_laws, search_similar_cases, warmup


def build_query(text: str, facts: dict[str, Any]) -> str:
    return " ".join(
        [
            text[:3000],
            " ".join(facts["articles_to_check"]),
            " ".join(item["label"] for item in facts["roles"]),
            " ".join(item["label"] for item in facts["punishments"]),
            "соучастие назначение наказания",
        ]
    )


def analyze_text(text: str, legal_top_k: int = 5, case_top_k: int = 5) -> dict[str, Any]:
    normalized = normalize_text(text)
    facts = extract_case_facts(normalized)
    query = build_query(normalized, facts)
    similar_cases = search_similar_cases(query, case_top_k)
    full_cases = fetch_full_cases(tuple(item["case_number"] for item in similar_cases))
    for item in similar_cases:
        item["full_case"] = full_cases.get(item["case_number"], {})
        item["full_case_found"] = bool(item["full_case"])
    return {
        "text": normalized,
        "facts": facts,
        "ml_role_model": predict_role_ml(normalized),
        "similar_cases": similar_cases,
        "legal_sources": search_laws(query, legal_top_k),
    }


def build_chat_context(text: str, legal_top_k: int = 5, case_top_k: int = 5) -> dict[str, Any]:
    payload = analyze_text(text, legal_top_k=legal_top_k, case_top_k=case_top_k)
    return {
        "case_text": payload["text"],
        "facts": payload["facts"],
        "ml_role_model": payload["ml_role_model"],
        "legal_sources": payload["legal_sources"],
        "similar_cases": [
            {
                "case_number": item["case_number"],
                "role": item["role_label"],
                "punishment": item["punishment_label"],
                "score": item["score"],
                "fragment": item["fragment"][:700],
                "context": item["context"][:500],
            }
            for item in payload["similar_cases"]
        ],
    }


def answer_chat(text: str, question: str, legal_top_k: int = 5, case_top_k: int = 5) -> dict[str, Any]:
    context = build_chat_context(text, legal_top_k=legal_top_k, case_top_k=case_top_k)
    response = generate_chat_answer(context, question)
    return {"context": context, **response}


__all__ = ["analyze_text", "build_chat_context", "answer_chat", "health_status", "warmup"]
