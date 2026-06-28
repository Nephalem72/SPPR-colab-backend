# SPPR Colab Backend

Отдельный backend для тестов `RAG + LLM` в Google Colab.

Ожидаемая структура Google Drive:

```text
/content/drive/MyDrive/
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
!git clone https://github.com/Nephalem72/SPPR-colab-backend.git
%cd /content/SPPR-colab-backend
!pip install -r requirements.txt

import subprocess
import time

api_process = subprocess.Popen(["python", "app_fastapi.py"])
time.sleep(5)
print("FastAPI PID:", api_process.pid)
```

Пользовательский интерфейс запускается после FastAPI:

```python
!python app_gradio.py
```

В Colab Gradio напечатает публичную ссылку вида `https://....gradio.live`. FastAPI остаётся доступен только внутри Colab, а пользователь работает через внешний Gradio UI.

По умолчанию backend ищет данные в:

```text
/content/drive/MyDrive/SPPR/data
```

Переменные окружения:

- `SPPR_DRIVE_ROOT` — корень папки проекта на Google Drive, по умолчанию `/content/drive/MyDrive/SPPR`
- `SPPR_DATA_DIR` — папка с parquet/моделями, по умолчанию `${SPPR_DRIVE_ROOT}/data`
- `SPPR_HOST` — host FastAPI, по умолчанию `0.0.0.0`
- `SPPR_PORT` — порт FastAPI, по умолчанию `8000`
- `SPPR_LLM_MODEL_ID` — Hugging Face ID модели; это основной переключатель модели
- `SPPR_LLM_BACKEND` — backend генерации, сейчас `transformers`
- `SPPR_LLM_LOAD_IN_4BIT` — `true` для 4-bit загрузки на GPU Colab
- `SPPR_LLM_MAX_INPUT_TOKENS` — предел входного контекста, по умолчанию `6144`
- `SPPR_LLM_MAX_NEW_TOKENS` — предел ответа, по умолчанию `768`
- `SPPR_LLM_MAX_CONTINUATIONS` — сколько раз автоматически продолжать ответ при упоре в лимит, по умолчанию `2`
- `SPPR_RAG_PROFILE` — `fast`, `balanced` или `broad`
- `SPPR_DATABASE_URL` — подключение к БД; по умолчанию SQLite в `/content/drive/MyDrive/SPPR/data/sppr_history.db`
- `SPPR_ALLOW_USER_REGISTRATION` — разрешить создание пользователей через API, по умолчанию `true`
- `SPPR_REGISTRATION_SECRET` — необязательный секрет для `POST /users` в заголовке `X-Registration-Secret`
- `SPPR_API_URL` — внутренний адрес FastAPI для UI, по умолчанию `http://127.0.0.1:8000`
- `SPPR_UI_PORT` — порт Gradio, по умолчанию `7860`
- `SPPR_UI_SHARE` — создавать публичную Gradio-ссылку, по умолчанию `true`
- `SPPR_UI_USERNAME` и `SPPR_UI_PASSWORD` — необязательная защита внешнего UI паролем

Пример настройки эксперимента до импорта backend:

```python
import os

os.environ["SPPR_LLM_MODEL_ID"] = "Qwen/Qwen2.5-14B-Instruct"
os.environ["SPPR_LLM_LOAD_IN_4BIT"] = "true"
os.environ["SPPR_LLM_MAX_CONTINUATIONS"] = "2"
os.environ["SPPR_RAG_PROFILE"] = "balanced"
os.environ["SPPR_DATABASE_URL"] = "sqlite:////content/drive/MyDrive/SPPR/data/sppr_history.db"
```

Для замены модели достаточно поменять `SPPR_LLM_MODEL_ID` и перезапустить runtime/server. Индексы, RAG и API при этом не меняются. Профили позволяют отдельно сравнивать скорость и полноту retrieval.

Эндпоинты:

- `GET /health`
- `GET /warmup`
- `GET /config`
- `POST /analyze`
- `POST /similar_cases`
- `GET /case?case_number=...` — полный текст найденного дела
- `POST /chat_context`
- `POST /chat`

Пользователи и история:

- `POST /users` — создать пользователя и один раз получить API-токен
- `GET /me` — проверить текущего пользователя
- `POST /me/token` — заменить API-токен
- `DELETE /me` — удалить пользователя со всеми диалогами
- `POST /conversations` — создать диалог с текстом дела
- `GET /conversations` — список собственных диалогов
- `GET /conversations/{id}` — переписка и сохраненные источники
- `POST /conversations/{id}/messages` — вопрос с автоматическим RAG и историей
- `DELETE /conversations/{id}` — удалить диалог и его сообщения

Защищенные запросы передают заголовок:

```text
Authorization: Bearer <api_token>
```

Токен генерируется случайно и возвращается только при создании пользователя; в БД хранится SHA-256-хеш. Пользователь видит только свои диалоги. Для одного процесса Colab достаточно SQLite на Google Drive. Для нескольких серверов или постоянной эксплуатации задайте PostgreSQL, например `SPPR_DATABASE_URL=postgresql+psycopg://user:password@host/dbname`.

При публикации FastAPI через tunnel обязательно задайте `SPPR_REGISTRATION_SECRET` либо отключите открытую регистрацию. SQLite на Google Drive рассчитан на один процесс Colab; для реальной многопользовательской эксплуатации нужен PostgreSQL, резервное копирование и полноценная аутентификация. Тексты дел и переписка содержат чувствительные данные, поэтому доступ к Drive/БД должен быть ограничен.

Внешний Gradio UI предоставляет создание/вход в профиль, сохраненный список диалогов, чат, источники ответа и чтение полного текста похожего дела. Ключ пользователя сохраняется в локальном хранилище браузера. Для закрытого теста задайте `SPPR_UI_USERNAME` и `SPPR_UI_PASSWORD` до запуска.

`POST /chat` возвращает ответ, список использованных источников и метрики `retrieval_seconds`, `generation_seconds`, `total_seconds`, `input_tokens`, `output_tokens`. История диалога передаётся полем `history`, а полный собранный контекст можно включить через `return_context: true`.

Повторяемый прогон одной модели на фиксированных запросах:

```bash
python scripts/benchmark_api.py --profile balanced
```

Отчёт сохраняется в `eval/results/<model>__<profile>.json`. Для сравнения меняется `SPPR_LLM_MODEL_ID`, сервер перезапускается и запускается тот же benchmark. Помимо времени сохраняются ответы, источники и проверка вымышленных идентификаторов ссылок.

Пример запроса:

```json
{
  "text": "Описание обстоятельств дела...",
  "question": "Как может быть квалифицирована роль лица?",
  "history": [],
  "rag_profile": "balanced",
  "return_context": false
}
```

Проверка наличия файлов в Colab:

```python
from pathlib import Path

data_dir = Path("/content/drive/MyDrive/SPPR/data")
required = [
    "laws.parquet",
    "final_roles_punishments_v3.parquet",
    "cases_with_id.parquet",
    "role_model.pkl",
    "vectorizer.pkl",
    "embeddings.pkl",
    "faiss_index.bin",
]

for name in required:
    path = data_dir / name
    print(name, path.exists(), path.stat().st_size if path.exists() else "MISSING")
```
