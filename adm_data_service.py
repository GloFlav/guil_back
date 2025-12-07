import pandas as pd
import logging
from typing import List, Dict, Optional
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class AdministrativeDataService:
    """Service pour gérer les données administratives Madagascar"""
    
    def __init__(self, csv_path: str = "./data/mdg_adm3.csv"):
        self.csv_path = csv_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Charge le fichier CSV des données administratives"""
        try:
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(f"Fichier CSV non trouvé: {self.csv_path}")
            
            self.df = pd.read_csv(self.csv_path)
            logger.info(f"Données administratives chargées: {len(self.df)} lignes")
            
            # Afficher les colonnes disponibles
            logger.info(f"Colonnes disponibles: {list(self.df.columns)}")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du CSV: {e}")
            raise
    
    def get_adm1_regions(self) -> List[str]:
        """Retourne la liste des régions ADM1 uniques"""
        if self.df is None:
            return []
        return self.df['ADM1_EN'].unique().tolist()
    
    def get_adm2_districts(self, adm1: Optional[str] = None) -> List[Dict]:
        """Retourne les districts ADM2 avec leurs métadonnées"""
        if self.df is None:
            return []
        
        result = []
        if adm1:
            filtered = self.df[self.df['ADM1_EN'] == adm1]
        else:
            filtered = self.df
        
        # Récupérer les districts uniques
        for _, row in filtered.drop_duplicates(subset=['ADM2_PCODE']).iterrows():
            result.append({
                'pcode': row['ADM2_PCODE'],
                'name': row['ADM2_EN'],
                'type': row['ADM2_TYPE'],
                'adm1': row['ADM1_EN']
            })
        
        return result
    
    def get_adm3_locations(self, adm1: Optional[str] = None, 
                          adm2: Optional[str] = None) -> List[Dict]:
        """Retourne les localisations ADM3 avec leurs métadonnées"""
        if self.df is None:
            return []
        
        result = []
        filtered = self.df
        
        if adm1:
            filtered = filtered[filtered['ADM1_EN'] == adm1]
        
        if adm2:
            filtered = filtered[filtered['ADM2_EN'] == adm2]
        
        for _, row in filtered.iterrows():
            result.append({
                'pcode': row['ADM3_PCODE'],
                'name': row['ADM3_EN'],
                'adm2': row['ADM2_EN'],
                'adm1': row['ADM1_EN'],
                'adm2_type': row['ADM2_TYPE']
            })
        
        return result
    
    def search_locations_by_context(self, context: str, 
                                   num_locations: int = 5) -> List[Dict]:
        """
        Cherche les localisations pertinentes selon le contexte fourni
        par l'utilisateur (par ex: "zones rurales", "districts urbains")
        """
        if self.df is None:
            return []
        
        result = []
        context_lower = context.lower()
        
        # Logique simple basée sur mots-clés
        if any(word in context_lower for word in ['rural', 'campagne', 'village']):
            # Filtrer les zones moins densément peuplées (ex: arrondissements spécifiques)
            filtered = self.df[~self.df['ADM2_EN'].str.contains('1er|2e|3e|4e|5e|6e', regex=True)]
        elif any(word in context_lower for word in ['urbain', 'ville', 'centre']):
            # Filtrer les arrondissements centraux
            filtered = self.df[self.df['ADM2_EN'].str.contains('1er|2e|3e', regex=True)]
        else:
            filtered = self.df
        
        # Récupérer les localisations uniques
        unique_locations = filtered.drop_duplicates(subset=['ADM3_PCODE'])
        
        for _, row in unique_locations.head(num_locations).iterrows():
            result.append({
                'pcode': row['ADM3_PCODE'],
                'name': row['ADM3_EN'],
                'adm2': row['ADM2_EN'],
                'adm1': row['ADM1_EN'],
                'adm2_type': row['ADM2_TYPE']
            })
        
        return result
    
    def get_statistics(self) -> Dict:
        """Retourne des statistiques sur les données"""
        if self.df is None:
            return {}
        
        return {
            'total_locations_adm3': len(self.df['ADM3_PCODE'].unique()),
            'total_districts_adm2': len(self.df['ADM2_PCODE'].unique()),
            'total_regions_adm1': len(self.df['ADM1_EN'].unique()),
            'regions': self.df['ADM1_EN'].unique().tolist()
        }

# Instance globale du service
adm_service = AdministrativeDataService()