from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    drive_root: Path = Path(os.getenv("SPPR_DRIVE_ROOT", "/content/drive/MyDrive/SPPR"))
    data_dir: Path = Path(os.getenv("SPPR_DATA_DIR", "/content/drive/MyDrive/SPPR/data"))
    host: str = os.getenv("SPPR_HOST", "0.0.0.0")
    port: int = int(os.getenv("SPPR_PORT", "8000"))
    legacy_case_encoder: str = os.getenv(
        "SPPR_LEGACY_CASE_ENCODER",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    llm_model_id: str = os.getenv("SPPR_LLM_MODEL_ID", "Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct")
    llm_max_new_tokens: int = int(os.getenv("SPPR_LLM_MAX_NEW_TOKENS", "320"))

    @property
    def laws_path(self) -> Path:
        return self.data_dir / "laws.parquet"

    @property
    def final_dataset_path(self) -> Path:
        return self.data_dir / "final_roles_punishments_v3.parquet"

    @property
    def cases_path(self) -> Path:
        return self.data_dir / "cases_with_id.parquet"

    @property
    def role_model_path(self) -> Path:
        return self.data_dir / "role_model.pkl"

    @property
    def role_vectorizer_path(self) -> Path:
        return self.data_dir / "vectorizer.pkl"

    @property
    def embeddings_path(self) -> Path:
        return self.data_dir / "embeddings.pkl"

    @property
    def faiss_index_path(self) -> Path:
        return self.data_dir / "faiss_index.bin"


settings = Settings()
