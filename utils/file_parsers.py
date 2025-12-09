# Fichier: backend/utils/file_parsers.py
import pandas as pd
import numpy as np
from fastapi import UploadFile, HTTPException
from io import BytesIO
from typing import List, Any 

# AJOUTE CETTE LIGNE D'IMPORT IMPORTANTE 👇
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
                
            df = df.replace({np.nan: None})
            
            return df
            
        except Exception as e:
            raise HTTPException(400, f"Erreur lecture fichier: {str(e)}")

    @staticmethod
    def get_file_stats(df: pd.DataFrame, filename: str) -> FilePreviewResponse:
        """Génère les stats immédiates"""
        
        # Identifier les colonnes entièrement vides
        empty_cols = [col for col in df.columns if df[col].isnull().all()]
        
        # Conversion safe pour le JSON
        preview = df.head(5).replace({np.nan: None}).to_dict(orient='records')
        
        return FilePreviewResponse(
            filename=filename,
            total_rows=len(df),
            total_columns=len(df.columns),
            columns_list=list(df.columns),
            empty_columns=empty_cols,
            preview_data=preview,
            file_size_kb=0.0 # Calculé dans le service
        )
    
    @staticmethod
    def get_file_stats(df: pd.DataFrame, filename: str) -> FilePreviewResponse:
        """Génère les stats immédiates incluant les données partielles"""
        
        total_rows = len(df)
        empty_cols = []
        partial_cols = []
        
        # Analyse de chaque colonne
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            
            if missing_count == total_rows:
                # 100% vide
                empty_cols.append(col)
            elif missing_count > 0:
                # Partiellement vide
                percentage = round((missing_count / total_rows) * 100, 1)
                partial_cols.append({
                    "name": col,
                    "count": int(missing_count),
                    "percentage": percentage
                })
        
        # On trie les partiels du plus vide au moins vide pour l'affichage
        partial_cols.sort(key=lambda x: x['percentage'], reverse=True)
        
        # Conversion safe pour le JSON
        preview = df.head(5).replace({np.nan: None}).to_dict(orient='records')
        
        return FilePreviewResponse(
            filename=filename,
            total_rows=total_rows,
            total_columns=len(df.columns),
            columns_list=list(df.columns),
            empty_columns=empty_cols,
            partially_empty_columns=partial_cols, # <--- Ajouté ici
            preview_data=preview,
            file_size_kb=0.0 
        )