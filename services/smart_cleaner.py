import pandas as pd
import numpy as np
import re

class SmartCleaner:
    
    # Mots-clés critiques (Identifiants, Statuts, ML, Temps, Géo)
    CRITICAL_KEYWORDS = [
        'id', 'uuid', 'code', 'ref', 'index', 'pk', 'fk',
        'status', 'etat', 'state', 'flag', 'validation',
        'target', 'cible', 'fraud', 'churn', 'result', 'sortie', 'output',
        'date', 'time', 'start', 'end', 'debut', 'fin',
        'gps', 'lat', 'lon', 'altitude', 'precision'
    ]

    def analyze_column(self, series: pd.Series, col_name: str, threshold: float = 0.9) -> dict:
        # Nettoyage local
        series = series.replace(r'^\s*$', np.nan, regex=True)
        missing_rate = series.isnull().mean()
        
        # 1. Si 100% vide -> Poubelle immédiate (sauf si on veut absolument garder la structure)
        if missing_rate == 1.0:
            return {'action': 'drop_empty', 'reason': '100% Vide'}

        # 2. Si peu de vide -> Garder
        if missing_rate < threshold:
            return {'action': 'keep', 'reason': 'Données suffisantes'}

        # --- INTELLIGENCE (>90% vide) ---
        
        clean_name = str(col_name).lower()
        
        # Règle A : Mots-clés critiques (AVEC REGEX STRICTE)
        # On cherche le mot clé entouré de séparateurs (début, fin, _, espace, point)
        # Exemple : match 'user_id', 'id', 'id-client' MAIS PAS 'amidy' ou 'olona'
        for keyword in self.CRITICAL_KEYWORDS:
            # Pattern : (Debut ou non-lettre) + mot + (Fin ou non-lettre)
            pattern = r'(?:^|[^a-z0-9])' + re.escape(keyword) + r'(?:$|[^a-z0-9])'
            
            if re.search(pattern, clean_name):
                return {'action': 'keep', 'reason': f"Mot-clé critique '{keyword}'"}

        # Règle B : Valeurs uniques (Flags)
        unique_values = series.dropna().unique()
        if len(unique_values) == 1:
            val = str(unique_values[0])
            if val in ['1', '1.0', 'True', 'true', 'Oui', 'Yes', 'Eny']:
                return {'action': 'keep', 'reason': 'Flag binaire potentiel'}
            return {'action': 'drop_sparse', 'reason': 'Valeur unique constante'}

        return {'action': 'drop_sparse', 'reason': f"Trop vide ({missing_rate:.1%})"}

    def apply_smart_cleaning(self, df: pd.DataFrame, aggressive: bool = False) -> tuple[pd.DataFrame, list]:
        # Nettoyage préliminaire global (Espaces insécables inclus \xa0)
        df = df.replace(r'^\s*$', np.nan, regex=True)
        df = df.replace(['nan', 'NaN', 'None', 'null'], np.nan)
        
        cols_to_drop = []
        logs = []
        
        initial_cols = df.columns
        
        for col in initial_cols:
            decision = self.analyze_column(df[col], col)
            action = decision['action']
            
            if action == 'drop_empty':
                cols_to_drop.append(col)
                logs.append(f"{col}")
            
            elif action == 'drop_sparse':
                if aggressive:
                    cols_to_drop.append(col)
                    logs.append(f"{col}")
        
        df_clean = df.drop(columns=cols_to_drop)
        return df_clean, logs

smart_cleaner = SmartCleaner()