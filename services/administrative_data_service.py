"""
Service de gestion des données administratives de Madagascar
Charge et gère les régions, districts et localités (ADM1, ADM2, ADM3)
Basé sur le fichier mdg_adm3.csv
"""

import logging
import os
import random
import pandas as pd
from typing import List, Dict, Any, Optional
from config.settings import settings

logger = logging.getLogger(__name__)

class AdministrativeDataService:
    """Service pour gérer les données administratives de Madagascar"""
    
    # Dictionnaire de correspondance pour les noms courants
    ALIASES = {
        # Capitales et grandes villes
        "tana": ["antananarivo", "analamanga", "1er arrondissement"],
        "antananarivo": ["antananarivo", "analamanga"],
        "fianar": ["fianarantsoa", "haute matsiatra"],
        "fianarantsoa": ["fianarantsoa", "haute matsiatra"],
        "tamatave": ["toamasina", "atsinanana"],
        "toamasina": ["toamasina", "atsinanana"],
        "majunga": ["mahajanga", "boeny"],
        "mahajanga": ["mahajanga", "boeny"],
        "tulear": ["toliara", "atsimo andrefana"],
        "toliara": ["toliara", "atsimo andrefana", "toliary"],
        "antsiranana": ["antsiranana", "diana"],
        "diego": ["antsiranana", "diana", "diego"],
        "fort-dauphin": ["taolagnaro", "anosy"],
        "fort dauphin": ["taolagnaro", "anosy"],
        
        # Régions spécifiques
        "sava": ["sava", "sambava", "antalaha", "vohemar", "andapa"],
        "nosy be": ["nosy-be", "diana"],
        "sainte marie": ["sainte marie", "analanjirofo"],
        "ste marie": ["sainte marie", "analanjirofo"]
    }

    # Mots-clés indiquant une volonté explicite de couvrir tout le pays
    NATIONAL_KEYWORDS = [
        "partout", "national", "nationale", "toute l'ile", "toute l'île", 
        "tout madagascar", "ensemble de madagascar", "dans tout le pays"
    ]

    def __init__(self):
        """Initialise le service"""
        self.df: Optional[pd.DataFrame] = None
        self._load_administrative_data()
    
    def _load_administrative_data(self):
        """Charge les données administratives depuis le CSV"""
        try:
            if not os.path.exists(settings.mdg_data_file):
                logger.warning(f"Fichier CSV non trouvé: {settings.mdg_data_file}")
                return
            
            logger.info(f"Chargement des données depuis {settings.mdg_data_file}")
            
            # Chargement du CSV avec pandas
            self.df = pd.read_csv(settings.mdg_data_file, encoding='utf-8', dtype=str)
            
            # Nettoyage des noms (strip espaces)
            cols_to_clean = ['ADM1_EN', 'ADM2_EN', 'ADM3_EN']
            for col in cols_to_clean:
                if col in self.df.columns:
                    self.df[col] = self.df[col].str.strip()

            logger.info(f"Données chargées: {len(self.df)} localités disponibles")
        
        except Exception as e:
            logger.error(f"Erreur chargement CSV: {e}", exc_info=True)
            self.df = None

    def search_locations_by_context(self, geographic_zones: str, num_locations: int = 5) -> List[Dict[str, Any]]:
        """
        Recherche intelligente des localités.
        Règles :
        1. "Partout/National" -> Aléatoire sur tout Madagascar.
        2. "Tana/Sava/..." -> Filtrage spécifique.
        3. "Madagascar/Mada/Vide" -> Défaut sur HAUTE MATSIATRA.
        """
        try:
            if self.df is None or self.df.empty:
                return self._get_fallback_locations(num_locations)

            geo_input = geographic_zones.lower().strip() if geographic_zones else ""
            
            # --- RÈGLE 1 : Détection de la demande "Partout" ---
            is_explicitly_national = any(k in geo_input for k in self.NATIONAL_KEYWORDS)
            
            search_terms = []
            if geo_input and not is_explicitly_national:
                # Nettoyage et extraction des termes
                raw_words = [w.strip() for w in geo_input.replace(',', ' ').split() if len(w) > 3]
                
                for word in raw_words:
                    # On ignore "madagascar" ou "mada" sauf si c'est le seul mot (géré plus bas)
                    if word in ["madagascar", "mada"]:
                        continue
                        
                    if word in self.ALIASES:
                        search_terms.extend(self.ALIASES[word])
                    else:
                        search_terms.append(word)

            logger.info(f"Termes de recherche: {search_terms} (National explicite: {is_explicitly_national})")

            filtered_df = pd.DataFrame()

            # --- LOGIQUE DE FILTRAGE ---
            
            if is_explicitly_national:
                # Cas : "Partout à Madagascar" -> On prend tout le pays
                logger.info("Mode 'Partout' détecté : Sélection nationale aléatoire.")
                filtered_df = self.df

            elif search_terms:
                # Cas : Lieu spécifique demandé (ex: "Tana", "Boeny") -> On filtre
                pattern = '|'.join([f"(?i){term}" for term in search_terms])
                mask = (
                    self.df['ADM1_EN'].str.contains(pattern, na=False, regex=True) |
                    self.df['ADM2_EN'].str.contains(pattern, na=False, regex=True) |
                    self.df['ADM3_EN'].str.contains(pattern, na=False, regex=True)
                )
                filtered_df = self.df[mask]
                
                # Si le filtre ne donne rien, on fallback sur Haute Matsiatra ou tout le pays ?
                # Ici on fallback sur Haute Matsiatra par sécurité
                if filtered_df.empty:
                    logger.info("Aucun résultat pour les termes. Fallback -> Haute Matsiatra.")
                    mask_default = self.df['ADM1_EN'].str.contains("Haute Matsiatra", case=False, na=False)
                    filtered_df = self.df[mask_default]

            else:
                # Cas : Vide, "Madagascar" ou "Mada" -> DÉFAUT HAUTE MATSIATRA
                logger.info("Lieu générique ou non spécifié. Défaut -> Haute Matsiatra.")
                mask_default = self.df['ADM1_EN'].str.contains("Haute Matsiatra", case=False, na=False)
                filtered_df = self.df[mask_default]

            # --- SÉLECTION ALÉATOIRE ---
            
            count = min(num_locations, len(filtered_df))
            if count == 0:
                return self._get_fallback_locations(num_locations)
                
            selected_rows = filtered_df.sample(n=count)

            # Construction de la réponse
            locations = []
            for _, row in selected_rows.iterrows():
                loc = {
                    "pcode": row.get('ADM3_PCODE', ''),
                    "name": row.get('ADM3_EN', 'Inconnu'),
                    "adm1": row.get('ADM1_EN', ''),
                    "adm2": row.get('ADM2_EN', ''),
                    "adm3": row.get('ADM3_EN', ''),
                    "latitude": None,
                    "longitude": None
                }
                locations.append(loc)
            
            locations.sort(key=lambda x: x['name'])
            return locations

        except Exception as e:
            logger.error(f"Erreur recherche lieux: {e}", exc_info=True)
            return self._get_fallback_locations(num_locations)

    def _get_fallback_locations(self, num_locations: int) -> List[Dict[str, Any]]:
        """Fallback hardcodé en Haute Matsiatra si le CSV plante"""
        defaults = [
            {"pcode":"MG21201001","name":"Tanana Ambony","adm1":"Haute Matsiatra","adm2":"Fianarantsoa","adm3":"Tanana Ambony"},
            {"pcode":"MG21201002","name":"Tanana Ambany","adm1":"Haute Matsiatra","adm2":"Fianarantsoa","adm3":"Tanana Ambany"},
            {"pcode":"MG21201003","name":"Andrainjato","adm1":"Haute Matsiatra","adm2":"Fianarantsoa","adm3":"Andrainjato"},
            {"pcode":"MG21205010","name":"Ambalavao","adm1":"Haute Matsiatra","adm2":"Ambalavao","adm3":"Ambalavao"},
            {"pcode":"MG21225010","name":"Isorana","adm1":"Haute Matsiatra","adm2":"Isandra","adm3":"Isorana"}
        ]
        # Si on en veut plus, on répète ou on coupe
        if num_locations > len(defaults):
            return defaults
        random.shuffle(defaults)
        return defaults[:num_locations]

# Instance globale
adm_service = AdministrativeDataService()