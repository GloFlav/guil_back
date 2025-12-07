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
            {"pcode":"MG21201001","name":"Tanana Ambony","adm1":"Haute Matsiatra","adm2":"Fianarantsoa I","adm3":"Tanana Ambony","latitude":-21.4557,"longitude":47.2901},
            {"pcode":"MG21201002","name":"Tanana Ambany","adm1":"Haute Matsiatra","adm2":"Fianarantsoa I","adm3":"Tanana Ambany","latitude":-21.4557,"longitude":47.2901},
            {"pcode":"MG21201003","name":"Andrainjato Avaratra","adm1":"Haute Matsiatra","adm2":"Fianarantsoa I","adm3":"Andrainjato Avaratra","latitude":-21.4557,"longitude":47.2901},
            {"pcode":"MG21201004","name":"Andrainjato Sud","adm1":"Haute Matsiatra","adm2":"Fianarantsoa I","adm3":"Andrainjato Sud","latitude":-21.4557,"longitude":47.2901},
            {"pcode":"MG21201005","name":"Manolafaka","adm1":"Haute Matsiatra","adm2":"Fianarantsoa I","adm3":"Manolafaka","latitude":-21.4557,"longitude":47.2901},
            {"pcode":"MG21201006","name":"Lalazana","adm1":"Haute Matsiatra","adm2":"Fianarantsoa I","adm3":"Lalazana","latitude":-21.4557,"longitude":47.2901},
            {"pcode":"MG21201007","name":"Vatosola","adm1":"Haute Matsiatra","adm2":"Fianarantsoa I","adm3":"Vatosola","latitude":-21.4557,"longitude":47.2901},
            {"pcode":"MG21205010","name":"Ambalavao","adm1":"Haute Matsiatra","adm2":"Ambalavao","adm3":"Ambalavao","latitude":-21.8129,"longitude":46.9858},
            {"pcode":"MG21205030","name":"Manamisoa","adm1":"Haute Matsiatra","adm2":"Ambalavao","adm3":"Manamisoa","latitude":-21.8129,"longitude":46.9858},
            {"pcode":"MG21205050","name":"Iarintsena","adm1":"Haute Matsiatra","adm2":"Ambalavao","adm3":"Iarintsena","latitude":-21.8129,"longitude":46.9858},
            {"pcode":"MG21205071","name":"Ambohimandroso","adm1":"Haute Matsiatra","adm2":"Ambalavao","adm3":"Ambohimandroso","latitude":-21.8129,"longitude":46.9858},
            {"pcode":"MG21205072","name":"Andrainjato","adm1":"Haute Matsiatra","adm2":"Ambalavao","adm3":"Andrainjato","latitude":-21.8129,"longitude":46.9858},
            {"pcode":"MG21205090","name":"Anjoma","adm1":"Haute Matsiatra","adm2":"Ambalavao","adm3":"Anjoma","latitude":-21.8129,"longitude":46.9858},
            {"pcode":"MG21205110","name":"Kirano","adm1":"Haute Matsiatra","adm2":"Ambalavao","adm3":"Kirano","latitude":-21.8129,"longitude":46.9858},
            {"pcode":"MG21205130","name":"Sendrisoa","adm1":"Haute Matsiatra","adm2":"Ambalavao","adm3":"Sendrisoa","latitude":-21.8129,"longitude":46.9858},
            {"pcode":"MG21205150","name":"Besoa","adm1":"Haute Matsiatra","adm2":"Ambalavao","adm3":"Besoa","latitude":-21.8129,"longitude":46.9858},
            {"pcode":"MG21205170","name":"Mahazony","adm1":"Haute Matsiatra","adm2":"Ambalavao","adm3":"Mahazony","latitude":-21.8129,"longitude":46.9858},
            {"pcode":"MG21205190","name":"Ambinanindovoka","adm1":"Haute Matsiatra","adm2":"Ambalavao","adm3":"Ambinanindovoka","latitude":-21.8129,"longitude":46.9858},
            {"pcode":"MG21205211","name":"Ankaramena","adm1":"Haute Matsiatra","adm2":"Ambalavao","adm3":"Ankaramena","latitude":-21.8129,"longitude":46.9858},
            {"pcode":"MG21205212","name":"Ambinaniroa","adm1":"Haute Matsiatra","adm2":"Ambalavao","adm3":"Ambinaniroa","latitude":-21.8129,"longitude":46.9858},
            {"pcode":"MG21205230","name":"Ambohimahamasina","adm1":"Haute Matsiatra","adm2":"Ambalavao","adm3":"Ambohimahamasina","latitude":-21.8129,"longitude":46.9858},
            {"pcode":"MG21205250","name":"Miarinarivo","adm1":"Haute Matsiatra","adm2":"Ambalavao","adm3":"Miarinarivo","latitude":-21.8129,"longitude":46.9858},
            {"pcode":"MG21205270","name":"Vohitsaoka","adm1":"Haute Matsiatra","adm2":"Ambalavao","adm3":"Vohitsaoka","latitude":-21.8129,"longitude":46.9858},
            {"pcode":"MG21208050","name":"Ankerana","adm1":"Haute Matsiatra","adm2":"Ambohimahasoa","adm3":"Ankerana","latitude":-21.9319,"longitude":47.0319},
            {"pcode":"MG21208071","name":"Manandroy","adm1":"Haute Matsiatra","adm2":"Ambohimahasoa","adm3":"Manandroy","latitude":-21.9319,"longitude":47.0319},
            {"pcode":"MG21208072","name":"Ambalakindresy","adm1":"Haute Matsiatra","adm2":"Ambohimahasoa","adm3":"Ambalakindresy","latitude":-21.9319,"longitude":47.0319},
            {"pcode":"MG21208090","name":"Ankafina Tsarafidy","adm1":"Haute Matsiatra","adm2":"Ambohimahasoa","adm3":"Ankafina Tsarafidy","latitude":-21.9319,"longitude":47.0319},
            {"pcode":"MG21208110","name":"Sahave","adm1":"Haute Matsiatra","adm2":"Ambohimahasoa","adm3":"Sahave","latitude":-21.9319,"longitude":47.0319},
            {"pcode":"MG21208130","name":"Morafeno","adm1":"Haute Matsiatra","adm2":"Ambohimahasoa","adm3":"Morafeno","latitude":-21.9319,"longitude":47.0319},
            {"pcode":"MG21208150","name":"Ikalalao","adm1":"Haute Matsiatra","adm2":"Ambohimahasoa","adm3":"Ikalalao","latitude":-21.9319,"longitude":47.0319},
            {"pcode":"MG21208170","name":"Vohiposa","adm1":"Haute Matsiatra","adm2":"Ambohimahasoa","adm3":"Vohiposa","latitude":-21.9319,"longitude":47.0319},
            {"pcode":"MG21208190","name":"Ambatosoa","adm1":"Haute Matsiatra","adm2":"Ambohimahasoa","adm3":"Ambatosoa","latitude":-21.9319,"longitude":47.0319},
            {"pcode":"MG21208211","name":"Isaka","adm1":"Haute Matsiatra","adm2":"Ambohimahasoa","adm3":"Isaka","latitude":-21.9319,"longitude":47.0319},
            {"pcode":"MG21208212","name":"Vohitrarivo","adm1":"Haute Matsiatra","adm2":"Ambohimahasoa","adm3":"Vohitrarivo","latitude":-21.9319,"longitude":47.0319},
            {"pcode":"MG21208230","name":"Ambohinamboarina","adm1":"Haute Matsiatra","adm2":"Ambohimahasoa","adm3":"Ambohinamboarina","latitude":-21.9319,"longitude":47.0319},
            {"pcode":"MG21208251","name":"Sahatona","adm1":"Haute Matsiatra","adm2":"Ambohimahasoa","adm3":"Sahatona","latitude":-21.9319,"longitude":47.0319},
            {"pcode":"MG21208252","name":"Camp Robin","adm1":"Haute Matsiatra","adm2":"Ambohimahasoa","adm3":"Camp Robin","latitude":-21.9319,"longitude":47.0319},
            {"pcode":"MG21208270","name":"Befeta","adm1":"Haute Matsiatra","adm2":"Ambohimahasoa","adm3":"Befeta","latitude":-21.9319,"longitude":47.0319},
            {"pcode":"MG21208290","name":"Fiadanana","adm1":"Haute Matsiatra","adm2":"Ambohimahasoa","adm3":"Fiadanana","latitude":-21.9319,"longitude":47.0319},
            {"pcode":"MG21219010","name":"Ikalamavony","adm1":"Haute Matsiatra","adm2":"Ikalamavony","adm3":"Ikalamavony","latitude":-21.7419,"longitude":47.3667},
            {"pcode":"MG21219030","name":"Mangidy","adm1":"Haute Matsiatra","adm2":"Ikalamavony","adm3":"Mangidy","latitude":-21.7419,"longitude":47.3667},
            {"pcode":"MG21219050","name":"Solila","adm1":"Haute Matsiatra","adm2":"Ikalamavony","adm3":"Solila","latitude":-21.7419,"longitude":47.3667},
            {"pcode":"MG21219070","name":"Ambatomainty","adm1":"Haute Matsiatra","adm2":"Ikalamavony","adm3":"Ambatomainty","latitude":-21.7419,"longitude":47.3667},
            {"pcode":"MG21219090","name":"Fitampito","adm1":"Haute Matsiatra","adm2":"Ikalamavony","adm3":"Fitampito","latitude":-21.7419,"longitude":47.3667},
            {"pcode":"MG21219110","name":"Tanamarina Sakay","adm1":"Haute Matsiatra","adm2":"Ikalamavony","adm3":"Tanamarina Sakay","latitude":-21.7419,"longitude":47.3667},
            {"pcode":"MG21219130","name":"Tsitondroina","adm1":"Haute Matsiatra","adm2":"Ikalamavony","adm3":"Tsitondroina","latitude":-21.7419,"longitude":47.3667},
            {"pcode":"MG21219150","name":"Tanamarina Bekisopa","adm1":"Haute Matsiatra","adm2":"Ikalamavony","adm3":"Tanamarina Bekisopa","latitude":-21.7419,"longitude":47.3667},
            {"pcode":"MG21220011","name":"Andrainjato Centre","adm1":"Haute Matsiatra","adm2":"Lalangina","adm3":"Andrainjato Centre","latitude":-21.6456,"longitude":47.4233},
            {"pcode":"MG21220012","name":"Andrainjato Est","adm1":"Haute Matsiatra","adm2":"Lalangina","adm3":"Andrainjato Est","latitude":-21.6456,"longitude":47.4233},
            {"pcode":"MG21220030","name":"Ambalakely","adm1":"Haute Matsiatra","adm2":"Lalangina","adm3":"Ambalakely","latitude":-21.6456,"longitude":47.4233},
            {"pcode":"MG21220070","name":"Ivoamba","adm1":"Haute Matsiatra","adm2":"Lalangina","adm3":"Ivoamba","latitude":-21.6456,"longitude":47.4233},
            {"pcode":"MG21220090","name":"Taindambo","adm1":"Haute Matsiatra","adm2":"Lalangina","adm3":"Taindambo","latitude":-21.6456,"longitude":47.4233},
            {"pcode":"MG21220111","name":"Mahatsinjony","adm1":"Haute Matsiatra","adm2":"Lalangina","adm3":"Mahatsinjony","latitude":-21.6456,"longitude":47.4233},
            {"pcode":"MG21220112","name":"Sahambavy","adm1":"Haute Matsiatra","adm2":"Lalangina","adm3":"Sahambavy","latitude":-21.6456,"longitude":47.4233},
            {"pcode":"MG21220210","name":"Ambalamahasoa","adm1":"Haute Matsiatra","adm2":"Lalangina","adm3":"Ambalamahasoa","latitude":-21.6456,"longitude":47.4233},
            {"pcode":"MG21220250","name":"Alakamisy Ambohimaha","adm1":"Haute Matsiatra","adm2":"Lalangina","adm3":"Alakamisy Ambohimaha","latitude":-21.6456,"longitude":47.4233},
            {"pcode":"MG21220330","name":"Androy","adm1":"Haute Matsiatra","adm2":"Lalangina","adm3":"Androy","latitude":-21.6456,"longitude":47.4233},
            {"pcode":"MG21220350","name":"Alatsinainy Ialamarina","adm1":"Haute Matsiatra","adm2":"Lalangina","adm3":"Alatsinainy Ialamarina","latitude":-21.6456,"longitude":47.4233},
            {"pcode":"MG21220430","name":"Fandrandava","adm1":"Haute Matsiatra","adm2":"Lalangina","adm3":"Fandrandava","latitude":-21.6456,"longitude":47.4233},
            {"pcode":"MG21220630","name":"Ialananindro","adm1":"Haute Matsiatra","adm2":"Lalangina","adm3":"Ialananindro","latitude":-21.6456,"longitude":47.4233},
            {"pcode":"MG21224010","name":"Mahasoabe","adm1":"Haute Matsiatra","adm2":"Vohibato","adm3":"Mahasoabe","latitude":-21.5739,"longitude":47.2844},
            {"pcode":"MG21224030","name":"Vinanitelo","adm1":"Haute Matsiatra","adm2":"Vohibato","adm3":"Vinanitelo","latitude":-21.5739,"longitude":47.2844},
            {"pcode":"MG21224050","name":"Alakamisy Itenina","adm1":"Haute Matsiatra","adm2":"Vohibato","adm3":"Alakamisy Itenina","latitude":-21.5739,"longitude":47.2844},
            {"pcode":"MG21224070","name":"Talata Ampano","adm1":"Haute Matsiatra","adm2":"Vohibato","adm3":"Talata Ampano","latitude":-21.5739,"longitude":47.2844},
            {"pcode":"MG21224090","name":"Ihazoara","adm1":"Haute Matsiatra","adm2":"Vohibato","adm3":"Ihazoara","latitude":-21.5739,"longitude":47.2844},
            {"pcode":"MG21224110","name":"Ankaromalaza Mifanasoa","adm1":"Haute Matsiatra","adm2":"Vohibato","adm3":"Ankaromalaza Mifanasoa","latitude":-21.5739,"longitude":47.2844},
            {"pcode":"MG21224130","name":"Vohimarina","adm1":"Haute Matsiatra","adm2":"Vohibato","adm3":"Vohimarina","latitude":-21.5739,"longitude":47.2844},
            {"pcode":"MG21224150","name":"Maneva","adm1":"Haute Matsiatra","adm2":"Vohibato","adm3":"Maneva","latitude":-21.5739,"longitude":47.2844},
            {"pcode":"MG21224170","name":"Vohitrafeno","adm1":"Haute Matsiatra","adm2":"Vohibato","adm3":"Vohitrafeno","latitude":-21.5739,"longitude":47.2844},
            {"pcode":"MG21224190","name":"Andranovorivato","adm1":"Haute Matsiatra","adm2":"Vohibato","adm3":"Andranovorivato","latitude":-21.5739,"longitude":47.2844},
            {"pcode":"MG21224210","name":"Vohibato Ouest","adm1":"Haute Matsiatra","adm2":"Vohibato","adm3":"Vohibato Ouest","latitude":-21.5739,"longitude":47.2844},
            {"pcode":"MG21224230","name":"Andranomiditra","adm1":"Haute Matsiatra","adm2":"Vohibato","adm3":"Andranomiditra","latitude":-21.5739,"longitude":47.2844},
            {"pcode":"MG21224250","name":"Soaindrana","adm1":"Haute Matsiatra","adm2":"Vohibato","adm3":"Soaindrana","latitude":-21.5739,"longitude":47.2844},
            {"pcode":"MG21224270","name":"Mahaditra","adm1":"Haute Matsiatra","adm2":"Vohibato","adm3":"Mahaditra","latitude":-21.5739,"longitude":47.2844},
            {"pcode":"MG21225010","name":"Isorana","adm1":"Haute Matsiatra","adm2":"Isandra","adm3":"Isorana","latitude":-21.7234,"longitude":47.2157},
            {"pcode":"MG21225030","name":"Anjoma Itsara","adm1":"Haute Matsiatra","adm2":"Isandra","adm3":"Anjoma Itsara","latitude":-21.7234,"longitude":47.2157},
            {"pcode":"MG21225050","name":"Andoharanomaitso","adm1":"Haute Matsiatra","adm2":"Isandra","adm3":"Andoharanomaitso","latitude":-21.7234,"longitude":47.2157},
            {"pcode":"MG21225070","name":"Ambondrona","adm1":"Haute Matsiatra","adm2":"Isandra","adm3":"Ambondrona","latitude":-21.7234,"longitude":47.2157},
            {"pcode":"MG21225090","name":"Ambalamidera II","adm1":"Haute Matsiatra","adm2":"Isandra","adm3":"Ambalamidera II","latitude":-21.7234,"longitude":47.2157},
            {"pcode":"MG21225110","name":"Ankarinarivo Manirisoa","adm1":"Haute Matsiatra","adm2":"Isandra","adm3":"Ankarinarivo Manirisoa","latitude":-21.7234,"longitude":47.2157},
            {"pcode":"MG21225130","name":"Soatanana","adm1":"Haute Matsiatra","adm2":"Isandra","adm3":"Soatanana","latitude":-21.7234,"longitude":47.2157},
            {"pcode":"MG21225150","name":"Nasandratrony","adm1":"Haute Matsiatra","adm2":"Isandra","adm3":"Nasandratrony","latitude":-21.7234,"longitude":47.2157},
            {"pcode":"MG21225170","name":"Mahazoarivo","adm1":"Haute Matsiatra","adm2":"Isandra","adm3":"Mahazoarivo","latitude":-21.7234,"longitude":47.2157},
            {"pcode":"MG21225190","name":"Fanjakana","adm1":"Haute Matsiatra","adm2":"Isandra","adm3":"Fanjakana","latitude":-21.7234,"longitude":47.2157},
            {"pcode":"MG21225210","name":"Iavonomby Vohibola","adm1":"Haute Matsiatra","adm2":"Isandra","adm3":"Iavonomby Vohibola","latitude":-21.7234,"longitude":47.2157},
            {"pcode":"MG23209252","name":"Ambohitsara Est","adm1":"Vatovavy Fitovinany","adm2":"Mananjary","adm3":"Ambohitsara Est","latitude":-21.2256,"longitude":48.3089},
            {"pcode":"MG23209270","name":"Mahavoky Nord","adm1":"Vatovavy Fitovinany","adm2":"Mananjary","adm3":"Mahavoky Nord","latitude":-21.2256,"longitude":48.3089},
            {"pcode":"MG23209290","name":"Andonabe","adm1":"Vatovavy Fitovinany","adm2":"Mananjary","adm3":"Andonabe","latitude":-21.2256,"longitude":48.3089},
            {"pcode":"MG23209310","name":"Ambohinihaonana","adm1":"Vatovavy Fitovinany","adm2":"Mananjary","adm3":"Ambohinihaonana","latitude":-21.2256,"longitude":48.3089},
            {"pcode":"MG23209331","name":"Marofototra","adm1":"Vatovavy Fitovinany","adm2":"Mananjary","adm3":"Marofototra","latitude":-21.2256,"longitude":48.3089},
            {"pcode":"MG23209332","name":"Andranomavo","adm1":"Vatovavy Fitovinany","adm2":"Mananjary","adm3":"Andranomavo","latitude":-21.2256,"longitude":48.3089},
            {"pcode":"MG23209350","name":"Vatohandrina","adm1":"Vatovavy Fitovinany","adm2":"Mananjary","adm3":"Vatohandrina","latitude":-21.2256,"longitude":48.3089},
            {"pcode":"MG23209370","name":"Ambalahosy Nord","adm1":"Vatovavy Fitovinany","adm2":"Mananjary","adm3":"Ambalahosy Nord","latitude":-21.2256,"longitude":48.3089},
            {"pcode":"MG23209391","name":"Ambodinonoka","adm1":"Vatovavy Fitovinany","adm2":"Mananjary","adm3":"Ambodinonoka","latitude":-21.2256,"longitude":48.3089},
            {"pcode":"MG23209392","name":"Antaretra","adm1":"Vatovavy Fitovinany","adm2":"Mananjary","adm3":"Antaretra","latitude":-21.2256,"longitude":48.3089},
            {"pcode":"MG23209410","name":"Kianjavato","adm1":"Vatovavy Fitovinany","adm2":"Mananjary","adm3":"Kianjavato","latitude":-21.2256,"longitude":48.3089},
            {"pcode":"MG23209430","name":"Sandrohy","adm1":"Vatovavy Fitovinany","adm2":"Mananjary","adm3":"Sandrohy","latitude":-21.2256,"longitude":48.3089},
            {"pcode":"MG23209450","name":"Anosimparihy","adm1":"Vatovavy Fitovinany","adm2":"Mananjary","adm3":"Anosimparihy","latitude":-21.2256,"longitude":48.3089},
            {"pcode":"MG23209470","name":"Manakana Nord","adm1":"Vatovavy Fitovinany","adm2":"Mananjary","adm3":"Manakana Nord","latitude":-21.2256,"longitude":48.3089},
            {"pcode":"MG23209490","name":"Namorona","adm1":"Vatovavy Fitovinany","adm2":"Mananjary","adm3":"Namorona","latitude":-21.2256,"longitude":48.3089},
            {"pcode":"MG23210010","name":"Manakara","adm1":"Vatovavy Fitovinany","adm2":"Manakara Atsimo","adm3":"Manakara","latitude":-22.1482,"longitude":48.4533},
            {"pcode":"MG23210031","name":"Tataho","adm1":"Vatovavy Fitovinany","adm2":"Manakara Atsimo","adm3":"Tataho","latitude":-22.1482,"longitude":48.4533},
            {"pcode":"MG23210032","name":"Mangatsiotra","adm1":"Vatovavy Fitovinany","adm2":"Manakara Atsimo","adm3":"Mangatsiotra","latitude":-22.1482,"longitude":48.4533},
            {"pcode":"MG23210051","name":"Marofarihy","adm1":"Vatovavy Fitovinany","adm2":"Manakara Atsimo","adm3":"Marofarihy","latitude":-22.1482,"longitude":48.4533},
            {"pcode":"MG23210052","name":"Anosiala","adm1":"Vatovavy Fitovinany","adm2":"Manakara Atsimo","adm3":"Anosiala","latitude":-22.1482,"longitude":48.4533},
            {"pcode":"MG23210070","name":"Ambila","adm1":"Vatovavy Fitovinany","adm2":"Manakara Atsimo","adm3":"Ambila","latitude":-22.1482,"longitude":48.4533},
            {"pcode":"MG23210090","name":"Sorombo","adm1":"Vatovavy Fitovinany","adm2":"Manakara Atsimo","adm3":"Sorombo","latitude":-22.1482,"longitude":48.4533},
            {"pcode":"MG23210110","name":"Mizilo Gara","adm1":"Vatovavy Fitovinany","adm2":"Manakara Atsimo","adm3":"Mizilo Gara","latitude":-22.1482,"longitude":48.4533},
            {"pcode":"MG23210130","name":"Ambohitsara M","adm1":"Vatovavy Fitovinany","adm2":"Manakara Atsimo","adm3":"Ambohitsara M","latitude":-22.1482,"longitude":48.4533},
            {"pcode":"MG24216033","name":"Tolohomiady","adm1":"Ihorombe","adm2":"Ihosy","adm3":"Tolohomiady","latitude":-22.3956,"longitude":45.7642},
            {"pcode":"MG24216050","name":"Irina","adm1":"Ihorombe","adm2":"Ihosy","adm3":"Irina","latitude":-22.3956,"longitude":45.7642},
            {"pcode":"MG24216070","name":"Sahambano","adm1":"Ihorombe","adm2":"Ihosy","adm3":"Sahambano","latitude":-22.3956,"longitude":45.7642},
            {"pcode":"MG24216090","name":"Analaliry","adm1":"Ihorombe","adm2":"Ihosy","adm3":"Analaliry","latitude":-22.3956,"longitude":45.7642},
            {"pcode":"MG24216110","name":"Mahasoa","adm1":"Ihorombe","adm2":"Ihosy","adm3":"Mahasoa","latitude":-22.3956,"longitude":45.7642},
            {"pcode":"MG24216131","name":"Ambatolahy","adm1":"Ihorombe","adm2":"Ihosy","adm3":"Ambatolahy","latitude":-22.3956,"longitude":45.7642},
            {"pcode":"MG24216132","name":"Soamatasy","adm1":"Ihorombe","adm2":"Ihosy","adm3":"Soamatasy","latitude":-22.3956,"longitude":45.7642},
            {"pcode":"MG24216151","name":"Zazafotsy","adm1":"Ihorombe","adm2":"Ihosy","adm3":"Zazafotsy","latitude":-22.3956,"longitude":45.7642},
            {"pcode":"MG24216152","name":"Antsoha","adm1":"Ihorombe","adm2":"Ihosy","adm3":"Antsoha","latitude":-22.3956,"longitude":45.7642},
            {"pcode":"MG24216170","name":"Sakalalina","adm1":"Ihorombe","adm2":"Ihosy","adm3":"Sakalalina","latitude":-22.3956,"longitude":45.7642},
            {"pcode":"MG24216191","name":"Satrokala","adm1":"Ihorombe","adm2":"Ihosy","adm3":"Satrokala","latitude":-22.3956,"longitude":45.7642},
            {"pcode":"MG24216192","name":"Andiolava","adm1":"Ihorombe","adm2":"Ihosy","adm3":"Andiolava","latitude":-22.3956,"longitude":45.7642},
            {"pcode":"MG24216210","name":"Analavoka","adm1":"Ihorombe","adm2":"Ihosy","adm3":"Analavoka","latitude":-22.3956,"longitude":45.7642},
            {"pcode":"MG24216231","name":"Ranohira","adm1":"Ihorombe","adm2":"Ihosy","adm3":"Ranohira","latitude":-22.3956,"longitude":45.7642},
            {"pcode":"MG24216232","name":"Ilakaka","adm1":"Ihorombe","adm2":"Ihosy","adm3":"Ilakaka","latitude":-22.3956,"longitude":45.7642},
            {"pcode":"MG42410112","name":"Antsiatsiaka","adm1":"Sofia","adm2":"Mandritsara","adm3":"Antsiatsiaka","latitude":-14.3867,"longitude":48.7842},
            {"pcode":"MG42410130","name":"Amboaboa","adm1":"Sofia","adm2":"Mandritsara","adm3":"Amboaboa","latitude":-14.3867,"longitude":48.7842},
            {"pcode":"MG42410150","name":"Manampaneva","adm1":"Sofia","adm2":"Mandritsara","adm3":"Manampaneva","latitude":-14.3867,"longitude":48.7842},
            {"pcode":"MG42410171","name":"Ambarikorano","adm1":"Sofia","adm2":"Mandritsara","adm3":"Ambarikorano","latitude":-14.3867,"longitude":48.7842},
            {"pcode":"MG42410172","name":"Andratamarina","adm1":"Sofia","adm2":"Mandritsara","adm3":"Andratamarina","latitude":-14.3867,"longitude":48.7842},
            {"pcode":"MG42410191","name":"Ambaripaika","adm1":"Sofia","adm2":"Mandritsara","adm3":"Ambaripaika","latitude":-14.3867,"longitude":48.7842},
            {"pcode":"MG42410192","name":"Ambodiamontana Kianga","adm1":"Sofia","adm2":"Mandritsara","adm3":"Ambodiamontana Kianga","latitude":-14.3867,"longitude":48.7842},
            {"pcode":"MG42410211","name":"Ambalakirajy","adm1":"Sofia","adm2":"Mandritsara","adm3":"Ambalakirajy","latitude":-14.3867,"longitude":48.7842},
            {"pcode":"MG42410212","name":"Tsarajomoka","adm1":"Sofia","adm2":"Mandritsara","adm3":"Tsarajomoka","latitude":-14.3867,"longitude":48.7842},
            {"pcode":"MG42410231","name":"Antsatramidola","adm1":"Sofia","adm2":"Mandritsara","adm3":"Antsatramidola","latitude":-14.3867,"longitude":48.7842},
            {"pcode":"MG42410232","name":"Ankiakabe-Fonoko","adm1":"Sofia","adm2":"Mandritsara","adm3":"Ankiakabe-Fonoko","latitude":-14.3867,"longitude":48.7842},
            {"pcode":"MG42410250","name":"Marotandrano","adm1":"Sofia","adm2":"Mandritsara","adm3":"Marotandrano","latitude":-14.3867,"longitude":48.7842},
            {"pcode":"MG42410270","name":"Ambilombe","adm1":"Sofia","adm2":"Mandritsara","adm3":"Ambilombe","latitude":-14.3867,"longitude":48.7842},
            {"pcode":"MG42410290","name":"Andohajango","adm1":"Sofia","adm2":"Mandritsara","adm3":"Andohajango","latitude":-14.3867,"longitude":48.7842},
            {"pcode":"MG42410310","name":"Anjiabe","adm1":"Sofia","adm2":"Mandritsara","adm3":"Anjiabe","latitude":-14.3867,"longitude":48.7842},
            {"pcode":"MG42410330","name":"Ambohisoa","adm1":"Sofia","adm2":"Mandritsara","adm3":"Ambohisoa","latitude":-14.3867,"longitude":48.7842},
            {"pcode":"MG42410350","name":"Ampatakamaroreny","adm1":"Sofia","adm2":"Mandritsara","adm3":"Ampatakamaroreny","latitude":-14.3867,"longitude":48.7842},
            {"pcode":"MG42410371","name":"Ankiabe-Salohy","adm1":"Sofia","adm2":"Mandritsara","adm3":"Ankiabe-Salohy","latitude":-14.3867,"longitude":48.7842},
            {"pcode":"MG11101001","name":"1er Arrondissement","adm1":"Analamanga","adm2":"1er Arrondissement","adm3":"1er Arrondissement","latitude":-18.8792,"longitude":47.5079},
            {"pcode":"MG11101002","name":"2e Arrondissement","adm1":"Analamanga","adm2":"2e Arrondissement","adm3":"2e Arrondissement","latitude":-18.8792,"longitude":47.5079},
            {"pcode":"MG11101003","name":"3e Arrondissement","adm1":"Analamanga","adm2":"3e Arrondissement","adm3":"3e Arrondissement","latitude":-18.8792,"longitude":47.5079},
            {"pcode":"MG11101004","name":"4e Arrondissement","adm1":"Analamanga","adm2":"4e Arrondissement","adm3":"4e Arrondissement","latitude":-18.8792,"longitude":47.5079},
            {"pcode":"MG11101005","name":"5e Arrondissement","adm1":"Analamanga","adm2":"5e Arrondissement","adm3":"5e Arrondissement","latitude":-18.8792,"longitude":47.5079},
            {"pcode":"MG11101006","name":"6e Arrondissement","adm1":"Analamanga","adm2":"6e Arrondissement","adm3":"6e Arrondissement","latitude":-18.8792,"longitude":47.5079},
            {"pcode":"MG11102010","name":"Alasora","adm1":"Analamanga","adm2":"Antananarivo Avaradrano","adm3":"Alasora","latitude":-19.0,"longitude":47.5},
            {"pcode":"MG11102039","name":"Ankadikely Ilafy","adm1":"Analamanga","adm2":"Antananarivo Avaradrano","adm3":"Ankadikely Ilafy","latitude":-19.0,"longitude":47.5},
            {"pcode":"MG11102050","name":"Ambohimanambola","adm1":"Analamanga","adm2":"Antananarivo Avaradrano","adm3":"Ambohimanambola","latitude":-19.0,"longitude":47.5},
            {"pcode":"MG11102079","name":"Sabotsy Namehana","adm1":"Analamanga","adm2":"Antananarivo Avaradrano","adm3":"Sabotsy Namehana","latitude":-19.0,"longitude":47.5},
            {"pcode":"MG11102099","name":"Ambohimangakely","adm1":"Analamanga","adm2":"Antananarivo Avaradrano","adm3":"Ambohimangakely","latitude":-19.0,"longitude":47.5},
            {"pcode":"MG11102210","name":"Manandriana","adm1":"Analamanga","adm2":"Antananarivo Avaradrano","adm3":"Manandriana","latitude":-19.0,"longitude":47.5},
            {"pcode":"MG11102230","name":"Ambohimalaza Miray","adm1":"Analamanga","adm2":"Antananarivo Avaradrano","adm3":"Ambohimalaza Miray","latitude":-19.0,"longitude":47.5},
            {"pcode":"MG11102279","name":"Fiaferana","adm1":"Analamanga","adm2":"Antananarivo Avaradrano","adm3":"Fiaferana","latitude":-19.0,"longitude":47.5},
            {"pcode":"MG11102319","name":"Ambohimanga Rova","adm1":"Analamanga","adm2":"Antananarivo Avaradrano","adm3":"Ambohimanga Rova","latitude":-19.0,"longitude":47.5},
            {"pcode":"MG11102350","name":"Viliahazo","adm1":"Analamanga","adm2":"Antananarivo Avaradrano","adm3":"Viliahazo","latitude":-19.0,"longitude":47.5},
            {"pcode":"MG11102379","name":"Talata Volonondry","adm1":"Analamanga","adm2":"Antananarivo Avaradrano","adm3":"Talata Volonondry","latitude":-19.0,"longitude":47.5},
            {"pcode":"MG11102390","name":"Anjeva Gara","adm1":"Analamanga","adm2":"Antananarivo Avaradrano","adm3":"Anjeva Gara","latitude":-19.0,"longitude":47.5},
            {"pcode":"MG11102410","name":"Masindray","adm1":"Analamanga","adm2":"Antananarivo Avaradrano","adm3":"Masindray","latitude":-19.0,"longitude":47.5},
            {"pcode":"MG11102430","name":"Ankadinandriana","adm1":"Analamanga","adm2":"Antananarivo Avaradrano","adm3":"Ankadinandriana","latitude":-19.0,"longitude":47.5},
            {"pcode":"MG11103010","name":"Ambohidratrimo","adm1":"Analamanga","adm2":"Ambohidratrimo","adm3":"Ambohidratrimo","latitude":-18.9667,"longitude":47.65},
            {"pcode":"MG11103030","name":"Anosiala","adm1":"Analamanga","adm2":"Ambohidratrimo","adm3":"Anosiala","latitude":-18.9667,"longitude":47.65},
            {"pcode":"MG11103050","name":"Talatamaty","adm1":"Analamanga","adm2":"Ambohidratrimo","adm3":"Talatamaty","latitude":-18.9667,"longitude":47.65},
            {"pcode":"MG11103070","name":"Antehiroka","adm1":"Analamanga","adm2":"Ambohidratrimo","adm3":"Antehiroka","latitude":-18.9667,"longitude":47.65},
            {"pcode":"MG11103090","name":"Iarinarivo","adm1":"Analamanga","adm2":"Ambohidratrimo","adm3":"Iarinarivo","latitude":-18.9667,"longitude":47.65},
            {"pcode":"MG11103111","name":"Ivato Firaisana","adm1":"Analamanga","adm2":"Ambohidratrimo","adm3":"Ivato Firaisana","latitude":-18.9667,"longitude":47.65},
            {"pcode":"MG11103112","name":"Ivato Aeroport","adm1":"Analamanga","adm2":"Ambohidratrimo","adm3":"Ivato Aeroport","latitude":-18.9667,"longitude":47.65},
            {"pcode":"MG11103130","name":"Ambohitrimanjaka","adm1":"Analamanga","adm2":"Ambohidratrimo","adm3":"Ambohitrimanjaka","latitude":-18.9667,"longitude":47.65},
            {"pcode":"MG11103150","name":"Mahitsy","adm1":"Analamanga","adm2":"Ambohidratrimo","adm3":"Mahitsy","latitude":-18.9667,"longitude":47.65},
            {"pcode":"MG11103171","name":"Merimandroso","adm1":"Analamanga","adm2":"Ambohidratrimo","adm3":"Merimandroso","latitude":-18.9667,"longitude":47.65},
            {"pcode":"MG11103172","name":"Ambatolampy","adm1":"Analamanga","adm2":"Ambohidratrimo","adm3":"Ambatolampy","latitude":-18.9667,"longitude":47.65},
            {"pcode":"MG11103190","name":"Ampangabe","adm1":"Analamanga","adm2":"Ambohidratrimo","adm3":"Ampangabe","latitude":-18.9667,"longitude":47.65},
            {"pcode":"MG11103210","name":"Ampanotokana","adm1":"Analamanga","adm2":"Ambohidratrimo","adm3":"Ampanotokana","latitude":-18.9667,"longitude":47.65},
            {"pcode":"MG11103230","name":"Mananjara","adm1":"Analamanga","adm2":"Ambohidratrimo","adm3":"Mananjara","latitude":-18.9667,"longitude":47.65},
            {"pcode":"MG11103251","name":"Manjakavaradrano","adm1":"Analamanga","adm2":"Ambohidratrimo","adm3":"Manjakavaradrano","latitude":-18.9667,"longitude":47.65},
            {"pcode":"MG11103252","name":"Antsahafilo","adm1":"Analamanga","adm2":"Ambohidratrimo","adm3":"Antsahafilo","latitude":-18.9667,"longitude":47.65},
            {"pcode":"MG11103270","name":"Ambohimanjaka","adm1":"Analamanga","adm2":"Ambohidratrimo","adm3":"Ambohimanjaka","latitude":-18.9667,"longitude":47.65},
            {"pcode":"MG11103290","name":"Fiadanana","adm1":"Analamanga","adm2":"Ambohidratrimo","adm3":"Fiadanana","latitude":-18.9667,"longitude":47.65}
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