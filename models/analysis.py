# Fichier: backend/models/analysis.py
from pydantic import BaseModel
from typing import List, Any, Dict, Optional

class FilePreviewResponse(BaseModel):
    filename: str
    total_rows: int
    total_columns: int
    columns_list: List[str]
    empty_columns: List[str]
    preview_data: List[Dict[str, Any]]
    partially_empty_columns: List[Dict[str, Any]] = []
    file_size_kb: float