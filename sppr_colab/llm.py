from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import settings


SYSTEM_PROMPT = (
    "Ты юридический ассистент СППР. "
    "Отвечай на русском языке, сдержанно и по источникам. "
    "Не выдавай предположения за установленный факт. "
    "Если материалов недостаточно, прямо скажи об этом."
)


@lru_cache(maxsize=1)
def load_llm() -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(settings.llm_model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(settings.llm_model_id, trust_remote_code=True)
    model.eval()
    return tokenizer, model


def _context_to_text(context: dict[str, Any]) -> str:
    legal = context.get("legal_sources", [])
    cases = context.get("similar_cases", [])
    lines = [
        "Материалы текущего дела:",
        context.get("case_text", ""),
        "",
        "Выделенные признаки:",
        str(context.get("facts", {})),
        "",
        "Правовые материалы:",
    ]
    for idx, item in enumerate(legal, 1):
        lines.append(f"{idx}. {item.get('source', '')}: {item.get('text', '')[:700]}")
    lines.append("")
    lines.append("Похожие дела:")
    for idx, item in enumerate(cases, 1):
        lines.append(
            f"{idx}. Дело {item.get('case_number', '')}; роль {item.get('role', '')}; "
            f"наказание {item.get('punishment', '')}; score {item.get('score', '')}; "
            f"фрагмент {item.get('fragment', '')[:500]}"
        )
    return "\n".join(lines)


def generate_chat_answer(context: dict[str, Any], question: str) -> dict[str, Any]:
    tokenizer, model = load_llm()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": _context_to_text(context)},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=settings.llm_max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer_tokens = generated[0][inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    return {"model_id": settings.llm_model_id, "answer": answer}
