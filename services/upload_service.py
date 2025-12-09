from fastapi import UploadFile
from utils.file_parsers import FileParser
from models.analysis import FilePreviewResponse

class UploadService:
    async def process_upload_preview(self, file: UploadFile) -> FilePreviewResponse:
        # 1. Parsing
        df = await FileParser.parse_file(file)
        
        # 2. Stats rapides
        stats = FileParser.get_file_stats(df, file.filename)
        
        # 3. Calcul taille fichier (approx)
        file.file.seek(0, 2)
        size = file.file.tell()
        stats.file_size_kb = round(size / 1024, 2)
        
        return stats

upload_service = UploadService()