# SPPR Colab Backend

Отдельный backend для тестов `RAG + LLM` в Google Colab.

Ожидаемая структура Google Drive:

```text
MyDrive/
  SPPR/
    data/
      laws.parquet
      final_roles_punishments_v3.parquet
      cases_with_id.parquet
      role_model.pkl
      vectorizer.pkl
      embeddings.pkl
      faiss_index.bin
    repo/
      SPPR-colab-backend/
```

Быстрый старт в Colab:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content
!git clone <YOUR_REPO_URL> SPPR-colab-backend
%cd /content/SPPR-colab-backend
!pip install -r requirements.txt
!python app_fastapi.py
```

По умолчанию backend ищет данные в:

```text
/content/drive/MyDrive/SPPR/data
```

Переменные окружения:

- `SPPR_DRIVE_ROOT` — корень папки проекта на Google Drive, по умолчанию `/content/drive/MyDrive/SPPR`
- `SPPR_DATA_DIR` — папка с parquet/моделями, по умолчанию `${SPPR_DRIVE_ROOT}/data`
- `SPPR_HOST` — host FastAPI, по умолчанию `0.0.0.0`
- `SPPR_PORT` — порт FastAPI, по умолчанию `8000`

Эндпоинты:

- `GET /health`
- `GET /warmup`
- `POST /analyze`
- `POST /similar_cases`
- `POST /chat_context`
- `POST /chat`
