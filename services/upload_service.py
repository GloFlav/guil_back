import shutil
import os
import uuid
import pandas as pd
import numpy as np # Indispensable pour les NaN
from fastapi import UploadFile
from utils.file_parsers import FileParser
from models.analysis import FilePreviewResponse
from config.settings import settings
from services.smart_cleaner import smart_cleaner 

class UploadService:
    async def process_upload_preview(self, file: UploadFile) -> FilePreviewResponse:
        # 1. Gestion des IDs et Chemins
        unique_id = str(uuid.uuid4())
        ext = file.filename.split('.')[-1].lower()
        
        raw_filename = f"raw_{unique_id}.{ext}"   # Fichier Original (Backup)
        clean_filename = f"clean_{unique_id}.{ext}" # Fichier de Travail
        
        raw_path = os.path.join(settings.excel_output_dir, raw_filename)
        clean_path = os.path.join(settings.excel_output_dir, clean_filename)
        
        # 2. Sauvegarde du fichier BRUT (Intouchable)
        with open(raw_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 3. Lecture pour traitement
        if ext == 'csv':
            try:
                # engine='python' est plus robuste pour les CSV mal formés
                df = pd.read_csv(raw_path, engine='python')
            except:
                df = pd.read_csv(raw_path, sep=';', encoding='latin1', engine='python')
        else:
            df = pd.read_excel(raw_path)

        # ==============================================================================
        # 4. PHASE 1 : LE "KARCHER" (Nettoyage Technique Radical)
        # ==============================================================================
        
        cols_before = set(df.columns)

        # Remplacement PUISSANT : Espaces, Tabulations, NBSP (\u00A0), Sauts de ligne
        # [\s\u00A0]+ matche n'importe quel espace blanc normal ou insécable
        df = df.replace(r'^[\s\u00A0]*$', np.nan, regex=True)
        
        # Remplacement des littéraux
        df = df.replace(['nan', 'NaN', 'None', 'null', 'NULL', 'NA'], np.nan)

        # Suppression des 100% vides
        df = df.dropna(axis=1, how='all')
        
        removed_radical = list(cols_before - set(df.columns))

        # ==============================================================================
        # 5. PHASE 2 : LE SMART CLEANING (Intelligence >90%)
        # ==============================================================================
        
        # On passe le DataFrame déjà propre au SmartCleaner
        # aggressive=True permet de supprimer aussi les colonnes inutiles (ex: 99% vide)
        df_clean, logs_smart = smart_cleaner.apply_smart_cleaning(df, aggressive=True)
        
        # On calcule quelles colonnes le SmartCleaner a ajouté à la suppression
        removed_smart = list(set(df.columns) - set(df_clean.columns))
        
        # Liste totale des suppressions pour l'affichage Front
        all_removed_cols = removed_radical + removed_smart
        
        # ==============================================================================

        # 6. Sauvegarde du fichier CLEAN FINAL
        if ext == 'csv':
            df_clean.to_csv(clean_path, index=False)
        else:
            # Correction pour Excel (suppression timezones)
            for col in df_clean.select_dtypes(include=['datetimetz']).columns:
                df_clean[col] = df_clean[col].dt.tz_localize(None)
            df_clean.to_excel(clean_path, index=False)
            
        # 7. Génération des Stats pour le Front
        stats = FileParser.get_file_stats(
            df=df_clean, 
            filename=file.filename, 
            file_id=clean_filename, 
            raw_file_id=raw_filename
        )
        
        # On injecte la liste complète des colonnes tuées
        stats.removed_empty_columns = all_removed_cols 
        stats.file_size_kb = round(os.path.getsize(clean_path) / 1024, 2)
        
        return stats

upload_service = UploadService()