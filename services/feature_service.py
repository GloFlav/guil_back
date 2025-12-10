import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, Any, List

class FeatureService:
    
    # ----------------------------------------------------
    # 1. NETTOYAGE STRUCTUREL & LOGIQUE (Fusion/Suppression de redondance)
    # ----------------------------------------------------

    def group_and_merge_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Détecte et agrège les colonnes binaires groupées (select multiple)."""
        df_copy = df.copy()
        flag_cols = [col for col in df_copy.columns if df_copy[col].nunique(dropna=True) <= 2 and col.count('_') >= 4]
        groups = {}
        
        # Création des groupes
        for col in flag_cols:
            root_elements = col.split('_')
            root = '_'.join(root_elements[:4]) 
            if root not in groups:
                groups[root] = []
            groups[root].append(col)

        cols_to_drop = []
        
        for root, group in groups.items():
            if len(group) > 1:
                # Conversion des flags en numérique (1/0)
                numeric_flags = df_copy[group].apply(lambda s: s.replace({'Eny': 1, 'Tsia': 0, 'oui': 1, 'non': 0}), errors='ignore').fillna(0)
                
                new_col_name = root + '_COUNT'
                df_copy[new_col_name] = numeric_flags.sum(axis=1)
                
                cols_to_drop.extend(group)

        df_copy = df_copy.drop(columns=cols_to_drop, errors='ignore')
        return df_copy

    def remove_high_correlation(self, df: pd.DataFrame, threshold: float = 0.99) -> pd.DataFrame:
        """Supprime une des deux colonnes si elles sont presque parfaitement corrélées."""
        numeric_df = df.select_dtypes(include=np.number).fillna(0)
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        cols_to_drop = [column for column in upper.columns if any(upper[column] >= threshold)]
                
        df = df.drop(columns=cols_to_drop, errors='ignore')
        return df
    
    # ----------------------------------------------------
    # 2. GESTION DES INCOHÉRENCES ET OUTLIERS (NOUVEAU)
    # ----------------------------------------------------

    def handle_inconsistencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Correction des erreurs basiques (âge irréaliste, etc.)."""
        
        # Heuristique pour les colonnes d'âge (contenant 'age' ou 'taona')
        age_cols = [col for col in df.columns if 'age' in col.lower() or 'taona' in col.lower()]
        
        for col in age_cols:
            if pd.api.types.is_numeric_dtype(df[col].dtype):
                # Remplacer les âges irréalistes (< 0 ou > 100) par NaN pour imputation future
                df[col] = np.where((df[col] < 0) | (df[col] > 100), np.nan, df[col])

        # Retourne le DataFrame avec les incohérences marquées (en NaN)
        return df
    
    def handle_outliers_iqr(self, df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
        """Détection et neutralisation des outliers via la méthode IQR."""
        
        for col in df.select_dtypes(include=np.number).columns:
            # On ne traite pas les colonnes binaires (0/1) ou de comptage faible
            if df[col].nunique() <= 10: 
                continue
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            # Neutralisation: remplacer l'outlier par la valeur limite (capping)
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            
        return df
    
    # ----------------------------------------------------
    # 3. ENCHAÎNEMENT ET ENCODAGE FINAL (SMART - pas d'explosion!)
    # ----------------------------------------------------

    def process_features(self, df: pd.DataFrame, target_variable: str) -> pd.DataFrame:
        
        # 1. Préparation et Simplification (Fusion, Corrélation)
        df = self.group_and_merge_features(df)
        df = self.remove_high_correlation(df, threshold=0.99)
        
        # 2. Correction des Données
        df = self.handle_inconsistencies(df)
        df = self.handle_outliers_iqr(df)

        # 3. Normalisation de la Casse et des Formats (Step 19)
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col].dtype):
                # Normalisation de la casse, suppression des espaces inutiles
                df[col] = df[col].astype(str).str.strip().str.lower().replace(r'\s+', ' ', regex=True)
            
            # Harmonisation des formats de date (convertir les formats string communs en datetime)
            if 'date' in col.lower() or 'fotoana' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
                except:
                    pass 

        # 4. Imputation et Encodage Final (SMART - pas d'explosion de colonnes!)
        cols_to_drop = []
        
        for col in df.columns:
            
            # Imputation des variables d'entrée (trous restants)
            if df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[col].dtype):
                    df[col] = df[col].fillna(df[col].median())
                elif pd.api.types.is_object_dtype(df[col].dtype) or pd.api.types.is_categorical_dtype(df[col].dtype):
                    df[col] = df[col].fillna('non_defini')

            # Encodage et Standardisation (Target/Features)
            if col == target_variable:
                # Encodage de la cible
                if df[col].dtype == object or pd.api.types.is_categorical_dtype(df[col].dtype):
                    le = LabelEncoder()
                    df[col] = df[col].astype(str)
                    df[col] = le.fit_transform(df[col])
                continue

            # 🚨 SMART ENCODING: Éviter l'explosion de colonnes avec get_dummies!
            # Au lieu de: df = pd.get_dummies(...) qui crée N nouvelles colonnes
            # On utilise: LabelEncoder() qui ne crée qu'une seule colonne
            elif pd.api.types.is_object_dtype(df[col].dtype):
                unique_count = df[col].nunique()
                
                if unique_count <= 5:
                    # Petites catégories: get_dummies OK (5 → 5 colonnes max)
                    df = pd.get_dummies(df, columns=[col], prefix=col, dummy_na=False)
                elif unique_count <= 50:
                    # Catégories moyennes: LabelEncoder (ne crée qu'1 colonne)
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                else:
                    # Trop de catégories (>50): supprimer (non-utile pour ML)
                    cols_to_drop.append(col)
            
            elif pd.api.types.is_numeric_dtype(df[col].dtype) and df[col].nunique() > 2:
                # Normalisation des numériques (MinMax scale)
                col_min = df[col].min()
                col_max = df[col].max()
                if col_max > col_min:
                    df[col] = (df[col] - col_min) / (col_max - col_min)
        
        # Supprimer les colonnes à haut cardinality
        df = df.drop(columns=cols_to_drop, errors='ignore')
                
        return df

feature_service = FeatureService()