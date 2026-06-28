# Файлы для Google Drive

Положи в Google Drive папку:

```text
MyDrive/SPPR/
```

Внутри нее создай папку:

```text
MyDrive/SPPR/data/
```

И загрузи туда файлы из локальной машины:

## Обязательные файлы

1. `D:\Notebooks\sppr\laws.parquet`
   -> `MyDrive/SPPR/data/laws.parquet`

2. `D:\Notebooks\sppr\final_roles_punishments_v3.parquet`
   -> `MyDrive/SPPR/data/final_roles_punishments_v3.parquet`

3. `D:\Notebooks\sppr\cases_with_id.parquet`
   -> `MyDrive/SPPR/data/cases_with_id.parquet`

4. `D:\Notebooks\sppr\role_model.pkl`
   -> `MyDrive/SPPR/data/role_model.pkl`

5. `D:\Notebooks\sppr\vectorizer.pkl`
   -> `MyDrive/SPPR/data/vectorizer.pkl`

6. `D:\Notebooks\sppr\embeddings.pkl`
   -> `MyDrive/SPPR/data/embeddings.pkl`

7. `D:\Notebooks\sppr\faiss_index.bin`
   -> `MyDrive/SPPR/data/faiss_index.bin`

## Итоговая структура

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
```

## Что не нужно загружать в Drive для первого запуска

- промежуточные parquet
- ноутбуки из локального проекта
- локальный Gradio UI
- старые временные артефакты
