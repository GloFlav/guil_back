import pandas as pd
import numpy as np
import os
import uuid
import re
from config.settings import settings
from services.smart_cleaner import smart_cleaner

class CleaningService:
    
    def normalize_column_name(self, name: str) -> str:
        s = str(name).lower()
        s = s.replace('é', 'e').replace('è', 'e').replace('à', 'a')
        # On garde chiffres et lettres
        s = re.sub(r'[^a-z0-9]+', '_', s)
        return s.strip('_')

    def auto_clean_file(self, file_path: str, output_format: str = "xlsx", remove_sparse: bool = False) -> dict:
        logs = []
        
        # 1. CHARGEMENT
        if file_path.endswith('.csv'):
            try:
                # On tente de lire avec le moteur python qui est plus permissif
                df = pd.read_csv(file_path, engine='python')
            except:
                df = pd.read_csv(file_path, sep=';', encoding='latin1', engine='python')
        else:
            df = pd.read_excel(file_path)
        
        initial_cols = len(df.columns)

        # =========================================================
        # 2. NETTOYAGE PRÉLIMINAIRE (C'est ce qui manquait !)
        # =========================================================
        
        # A. Transformer les chaînes vides "" et les espaces "   " en NaN
        df = df.replace(r'^\s*$', np.nan, regex=True)
        
        # B. Transformer les littéraux de vide (souvent présents dans les exports CSV)
        df = df.replace(['nan', 'NaN', 'None', 'null', 'NULL', 'NA'], np.nan)
        
        # =========================================================

        # 3. SUPPRESSION 100% VIDE
        # Maintenant que les espaces sont des NaN, dropna va fonctionner
        empty_cols = [col for col in df.columns if df[col].isnull().all()]
        df = df.drop(columns=empty_cols)
        
        if empty_cols: 
            logs.append(f"Suppression de {len(empty_cols)} colonnes entièrement vides.")

        # 4. NETTOYAGE INTELLIGENT >90% (Si activé)
        if remove_sparse:
            df, smart_logs = smart_cleaner.apply_smart_cleaning(df)
            logs.extend(smart_logs)

        # 5. SUPPRESSION DES DOUBLONS (Lignes)
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            logs.append(f"Suppression de {initial_rows - len(df)} doublons.")

        # 6. STANDARDISATION DES HEADERS
        rename_map = {col: self.normalize_column_name(col) for col in df.columns}
        df = df.rename(columns=rename_map)
        
        # 7. TYPAGE OPTIMISÉ
        df = df.convert_dtypes()

        # 8. SAUVEGARDE
        output_filename = f"cleaned_{uuid.uuid4()}.{output_format}"
        output_path = os.path.join(settings.excel_output_dir, output_filename)
        
        if output_format == "csv":
            df.to_csv(output_path, index=False)
        else:
            # Fix Excel dates
            for col in df.select_dtypes(include=['datetimetz']).columns:
                df[col] = df[col].dt.tz_localize(None)
            df.to_excel(output_path, index=False)
            
        return {
            "path": output_path,
            "removed_total": (initial_cols - len(df.columns)),
            "details": logs
        }

cleaning_service = CleaningService()