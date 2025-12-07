# backend/services/administrative_data_service.py
"""
Service de gestion des données administratives de Madagascar
Charge et gère les régions, districts et localités (ADM1, ADM2, ADM3)
"""

import logging
import os
from typing import List, Dict, Any, Optional
import pandas as pd
from config.settings import settings

logger = logging.getLogger(__name__)

class AdministrativeDataService:
    """Service pour gérer les données administratives de Madagascar"""
    
    def __init__(self):
        """Initialise le service en chargeant les données du CSV"""
        self.df: Optional[pd.DataFrame] = None
        self.adm1_regions: Dict[str, List[str]] = {}
        self.adm2_districts: Dict[str, List[Dict[str, str]]] = {}
        self.adm3_locations: Dict[str, List[Dict[str, str]]] = {}
        
        self._load_administrative_data()
    
    def _load_administrative_data(self):
        """Charge les données administratives depuis le CSV"""
        try:
            # Vérifier que le fichier existe
            if not os.path.exists(settings.mdg_data_file):
                logger.warning(f"Fichier de données administratives non trouvé: {settings.mdg_data_file}")
                logger.info("Les données administratives ne seront pas disponibles")
                return
            
            logger.info(f"Chargement des données administratives depuis {settings.mdg_data_file}")
            
            # Charger le CSV
            self.df = pd.read_csv(settings.mdg_data_file, encoding='utf-8')
            
            # Construire les index par niveau administratif
            self._index_adm_levels()
            
            logger.info(f"Données administratives chargées: {len(self.df)} lignes")
            logger.info(f"Régions (ADM1): {len(self.adm1_regions)}")
            logger.info(f"Districts (ADM2): {len(self.adm2_districts)}")
            logger.info(f"Localités (ADM3): {len(self.adm3_locations)}")
        
        except Exception as e:
            logger.warning(f"Erreur lors du chargement des données administratives: {e}")
            self.df = None
    
    def _index_adm_levels(self):
        """Indexe les données par niveau administratif"""
        # Vérifier que df n'est pas None ET qu'il n'est pas vide
        if self.df is None or self.df.empty:
            logger.warning("DataFrame vide, indexation impossible")
            return
        
        try:
            # Indexer ADM1 (régions)
            for _, row in self.df.iterrows():
                adm1 = row.get('ADM1_EN', '')
                adm2 = row.get('ADM2_EN', '')
                adm3 = row.get('ADM3_EN', '')
                pcode = row.get('ADM3_PCODE', '')
                
                if not adm1 or not adm2:
                    continue
                
                # ADM1 -> listes de ADM2
                if adm1 not in self.adm1_regions:
                    self.adm1_regions[adm1] = []
                if adm2 not in self.adm1_regions[adm1]:
                    self.adm1_regions[adm1].append(adm2)
                
                # ADM2 -> listes de ADM3 avec pcode
                if adm2 not in self.adm2_districts:
                    self.adm2_districts[adm2] = []
                
                adm3_entry = {"pcode": pcode, "name": adm3, "adm1": adm1}
                if adm3_entry not in self.adm2_districts[adm2]:
                    self.adm2_districts[adm2].append(adm3_entry)
                
                # ADM3 -> détails
                if adm3 not in self.adm3_locations:
                    self.adm3_locations[adm3] = {
                        "pcode": pcode,
                        "adm1": adm1,
                        "adm2": adm2,
                        "adm3": adm3
                    }
        
        except Exception as e:
            logger.error(f"Erreur lors de l'indexation: {e}", exc_info=True)
    
    def get_adm1_regions(self) -> List[str]:
        """Retourne la liste des régions (ADM1)"""
        return sorted(list(self.adm1_regions.keys()))
    
    def get_adm2_districts(self, adm1: str) -> List[str]:
        """Retourne les districts (ADM2) d'une région"""
        return sorted(self.adm1_regions.get(adm1, []))
    
    def get_adm3_locations(self, adm2: str) -> List[Dict[str, Any]]:
        """Retourne les localités (ADM3) d'un district"""
        locations = self.adm2_districts.get(adm2, [])
        return sorted(locations, key=lambda x: x.get("name", ""))
    
    def search_locations_by_context(self, geographic_zones: str, num_locations: int = 5) -> List[Dict[str, Any]]:
        """
        Recherche des localités basées sur le contexte géographique
        
        Args:
            geographic_zones: Zones mentionnées par l'utilisateur (ex: "Analamanga, Antananarivo")
            num_locations: Nombre de lieux à retourner
        
        Returns:
            Liste de localités sélectionnées
        """
        try:
            # Vérifier que le dataframe existe et n'est pas vide
            if self.df is None or self.df.empty:
                logger.warning("Données administratives non disponibles")
                # Retourner des lieux par défaut
                return self._get_default_locations(num_locations)
            
            logger.info(f"Recherche de {num_locations} lieux pour zones: {geographic_zones}")
            
            # Nettoyer les zones mentionnées
            zones_list = [z.strip() for z in geographic_zones.split(',') if z.strip()]
            
            # Filtrer le dataframe
            filtered_df = self.df.copy()
            
            # Si des zones spécifiques sont mentionnées, filtrer par ADM1 ou ADM2
            if zones_list and zones_list[0]:
                zone_filter = filtered_df['ADM1_EN'].isin(zones_list) | filtered_df['ADM2_EN'].isin(zones_list)
                filtered_df = filtered_df[zone_filter]
            
            # Si pas de résultats après filtrage, prendre tous les résultats
            if filtered_df.empty:
                filtered_df = self.df.copy()
            
            # Sélectionner les lieux uniques
            unique_locations = filtered_df.drop_duplicates(subset=['ADM3_EN']).head(num_locations)
            
            # Construire la liste de lieux
            locations = []
            for _, row in unique_locations.iterrows():
                location = {
                    "pcode": str(row.get('ADM3_PCODE', '')),
                    "name": str(row.get('ADM3_EN', '')),
                    "adm1": str(row.get('ADM1_EN', '')),
                    "adm2": str(row.get('ADM2_EN', '')),
                    "adm3": str(row.get('ADM3_EN', '')),
                    "latitude": None,
                    "longitude": None
                }
                locations.append(location)
            
            logger.info(f"Trouvé {len(locations)} lieux")
            return locations
        
        except Exception as e:
            logger.error(f"Erreur lors de la recherche de lieux: {e}", exc_info=True)
            return self._get_default_locations(num_locations)
    
    def _get_default_locations(self, num_locations: int) -> List[Dict[str, Any]]:
        """Retourne des lieux par défaut si les données ne sont pas disponibles"""
        default_locations = [
            {
                "pcode": "MG101001",
                "name": "Antananarivo",
                "adm1": "Analamanga",
                "adm2": "Antananarivo I",
                "adm3": "Antananarivo",
                "latitude": -18.8792,
                "longitude": 47.5079
            },
            {
                "pcode": "MG101002",
                "name": "Avaradrano",
                "adm1": "Analamanga",
                "adm2": "Antananarivo II",
                "adm3": "Avaradrano",
                "latitude": -19.0,
                "longitude": 47.5
            },
            {
                "pcode": "MG101003",
                "name": "Fianarantsoa",
                "adm1": "Fianarantsoa",
                "adm2": "Fianarantsoa II",
                "adm3": "Fianarantsoa",
                "latitude": -21.4557,
                "longitude": 47.2901
            },
            {
                "pcode": "MG101004",
                "name": "Antalaha",
                "adm1": "Sava",
                "adm2": "Antalaha",
                "adm3": "Antalaha",
                "latitude": -14.9,
                "longitude": 50.3
            },
            {
                "pcode": "MG101005",
                "name": "Toliara",
                "adm1": "Androy",
                "adm2": "Toliara II",
                "adm3": "Toliara",
                "latitude": -23.3648,
                "longitude": 43.6672
            }
        ]
        
        return default_locations[:num_locations]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques des données administratives"""
        if self.df is None or self.df.empty:
            return {
                "total_records": 0,
                "num_regions": 0,
                "num_districts": 0,
                "num_locations": 0,
                "status": "Données non disponibles"
            }
        
        return {
            "total_records": len(self.df),
            "num_regions": len(self.adm1_regions),
            "num_districts": len(self.adm2_districts),
            "num_locations": len(self.adm3_locations),
            "status": "Données disponibles"
        }

# Instance globale du service
adm_service = AdministrativeDataService()