import pandas as pd
import numpy as np
from fastapi import UploadFile, HTTPException
from io import BytesIO
from typing import List, Any, Optional
from models.analysis import FilePreviewResponse 

class FileParser:
    @staticmethod
    async def parse_file(file: UploadFile) -> pd.DataFrame:
        contents = await file.read()
        file.file.seek(0)
        
        filename = file.filename.lower()
        
        try:
            if filename.endswith(('.xlsx', '.xls')):
                xl = pd.ExcelFile(BytesIO(contents))
                sheet_names = xl.sheet_names
                
                target_sheet = sheet_names[0]
                for sheet in sheet_names:
                    if "group" not in sheet.lower() and "uuid" not in sheet.lower():
                        target_sheet = sheet
                        break
                
                df = pd.read_excel(BytesIO(contents), sheet_name=target_sheet)
                
            elif filename.endswith('.csv'):
                try:
                    df = pd.read_csv(BytesIO(contents), encoding='utf-8')
                    if len(df.columns) <= 1:
                        df = pd.read_csv(BytesIO(contents), sep=';', encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(BytesIO(contents), sep=';', encoding='latin1')
            else:
                raise HTTPException(400, "Format non supporté")
                
            # --- CORRECTION MAJEURE ICI ---
            
            # 1. Remplacer les chaînes vides "" et les espaces " " par NaN
            # Le regex ^\s*$ cible tout ce qui est vide ou ne contient que des espaces
            df = df.replace(r'^\s*$', np.nan, regex=True)
            
            # 2. Remplacer les "None" texte ou "NaN" texte par de vrais NaN
            df = df.replace(['None', 'none', 'NaN', 'nan', 'NULL', 'null'], np.nan)

            # 3. (Optionnel pour le JSON de retour) Convertir les NaN finaux en None pour l'API
            # Mais on garde df avec des NaN pour les calculs internes
            
            return df
            
        except Exception as e:
            raise HTTPException(400, f"Erreur lecture fichier: {str(e)}")

    @staticmethod
    def get_file_stats(
        df: pd.DataFrame, 
        filename: str, 
        file_id: str, 
        raw_file_id: Optional[str] = None
    ) -> FilePreviewResponse:
        
        # On recalcule les totaux sur le DF nettoyé
        total_rows = len(df)
        empty_cols = []
        partial_cols = []
        
        for col in df.columns:
            # On s'assure que missing_count compte bien les NaN
            missing_count = df[col].isna().sum() 
            
            if missing_count == total_rows:
                empty_cols.append(col)
            elif missing_count > 0:
                percentage = round((missing_count / total_rows) * 100, 1)
                partial_cols.append({
                    "name": col,
                    "count": int(missing_count),
                    "percentage": percentage
                })
        
        partial_cols.sort(key=lambda x: x['percentage'], reverse=True)
        
        # Pour la preview JSON, on remplace les NaN par None (car JSON n'aime pas NaN)
        preview_df = df.head(5).replace({np.nan: None})
        preview = preview_df.to_dict(orient='records')
        
        final_raw_id = raw_file_id if raw_file_id else file_id

        return FilePreviewResponse(
            file_id=file_id,
            raw_file_id=final_raw_id,
            filename=filename,
            total_rows=total_rows,
            total_columns=len(df.columns),
            columns_list=list(df.columns),
            empty_columns=empty_cols,
            partially_empty_columns=partial_cols,
            preview_data=preview,
            file_size_kb=0.0 
        )