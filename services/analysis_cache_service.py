# backend/services/analysis_cache_service.py
"""
Service de cache/stockage pour persister les résultats d'analyse
Permet de chaîner: File Structure → EDA → Full Analysis
"""

import json
import os
import pickle
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class AnalysisCacheService:
    """
    Gère le stockage des métadonnées d'analyse par fichier.
    Structure:
    {
      "file_id": {
        "file_structure": {...},      # Résultat file_structure_analysis_service
        "eda_results": {...},         # Résultat eda_service
        "feature_engineered": {...},  # DF après feature engineering
        "full_analysis": {...},       # Résultat final complet
        "context": {...},             # Contexte inféré
        "timestamps": {...}
      }
    }
    """
    
    def __init__(self, cache_dir: str = "./analysis_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.in_memory = {}  # Cache rapide en mémoire
        logger.info(f"AnalysisCacheService initialized: {cache_dir}")
    
    # ========================================================================
    # 1. ENREGISTREMENT FILE STRUCTURE
    # ========================================================================
    
    def save_file_structure(self, file_id: str, structure_data: Dict[str, Any]) -> bool:
        """
        Enregistre les résultats de file_structure_analysis_service
        
        Args:
            file_id: ID du fichier (clean_*.xlsx)
            structure_data: Résultat de analyze_file_structure()
        
        Returns:
            Success boolean
        """
        try:
            if file_id not in self.in_memory:
                self.in_memory[file_id] = {}
            
            self.in_memory[file_id]["file_structure"] = {
                "data": structure_data,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
            logger.info(f"✓ File structure enregistrée pour {file_id}")
            self._save_to_disk(file_id)
            return True
        
        except Exception as e:
            logger.error(f"Erreur save_file_structure: {e}")
            return False
    
    # ========================================================================
    # 2. ENREGISTREMENT EDA RÉSULTATS
    # ========================================================================
    
    def save_eda_results(self, file_id: str, eda_results: Dict[str, Any]) -> bool:
        """
        Enregistre les résultats complets d'EDA
        
        Args:
            file_id: ID du fichier
            eda_results: Résultat de eda_service.run_full_eda()
        
        Returns:
            Success boolean
        """
        try:
            if file_id not in self.in_memory:
                self.in_memory[file_id] = {}
            
            # Limitation du stockage (garder les données essentielles)
            eda_compact = {
                "metrics": eda_results.get("metrics", {}),
                "charts_data": eda_results.get("charts_data", {}),
                "ai_insights": eda_results.get("ai_insights", []),
                "auto_target": eda_results.get("auto_target"),
                "summary": eda_results.get("summary", {}),
                "timestamp": datetime.now().isoformat()
            }
            
            self.in_memory[file_id]["eda_results"] = eda_compact
            logger.info(f"✓ EDA enregistrée pour {file_id}")
            self._save_to_disk(file_id)
            return True
        
        except Exception as e:
            logger.error(f"Erreur save_eda_results: {e}")
            return False
    
    # ========================================================================
    # 3. ENREGISTREMENT CONTEXTE D'ANALYSE
    # ========================================================================
    
    def save_analysis_context(self, file_id: str, context: Dict[str, Any]) -> bool:
        """
        Enregistre le contexte inféré par context_analyst
        
        Args:
            file_id: ID du fichier
            context: Dict avec target_variable, focus_variables, analysis_type, etc.
        """
        try:
            if file_id not in self.in_memory:
                self.in_memory[file_id] = {}
            
            self.in_memory[file_id]["context"] = {
                "data": context,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✓ Contexte enregistré pour {file_id}: target={context.get('target_variable')}")
            self._save_to_disk(file_id)
            return True
        
        except Exception as e:
            logger.error(f"Erreur save_analysis_context: {e}")
            return False
    
    # ========================================================================
    # 4. ENREGISTREMENT DATAFRAME TRAITÉ
    # ========================================================================
    
    def save_processed_dataframe(self, file_id: str, df: pd.DataFrame, 
                                 stage: str = "feature_engineered") -> bool:
        """
        Enregistre le DataFrame à chaque étape du traitement
        
        Args:
            file_id: ID du fichier
            df: DataFrame traité
            stage: Étape ('cleaned', 'feature_engineered', 'final')
        """
        try:
            if file_id not in self.in_memory:
                self.in_memory[file_id] = {}
            
            # Ne stocker que les métadonnées du DF (pas le DF complet en mémoire)
            df_metadata = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "missing_values": df.isnull().sum().to_dict(),
                "numeric_cols": df.select_dtypes(include=['number']).columns.tolist(),
                "categorical_cols": df.select_dtypes(include=['object', 'category']).columns.tolist(),
                "timestamp": datetime.now().isoformat()
            }
            
            self.in_memory[file_id][f"df_{stage}"] = df_metadata
            logger.info(f"✓ DataFrame ({stage}) enregistré: {df.shape}")
            
            return True
        
        except Exception as e:
            logger.error(f"Erreur save_processed_dataframe: {e}")
            return False
    
    # ========================================================================
    # 5. ENREGISTREMENT RÉSULTAT COMPLET
    # ========================================================================
    
    def save_full_analysis(self, file_id: str, full_analysis: Dict[str, Any]) -> bool:
        """
        Enregistre le résultat final complet de l'analyse
        
        Args:
            file_id: ID du fichier
            full_analysis: Résultat final FullAnalysisResult-compatible
        """
        try:
            if file_id not in self.in_memory:
                self.in_memory[file_id] = {}
            
            self.in_memory[file_id]["full_analysis"] = {
                "data": full_analysis,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
            logger.info(f"✓ Analyse complète enregistrée pour {file_id}")
            self._save_to_disk(file_id)
            return True
        
        except Exception as e:
            logger.error(f"Erreur save_full_analysis: {e}")
            return False
    
    # ========================================================================
    # 6. RÉCUPÉRATION DE DONNÉES
    # ========================================================================
    
    def get_file_structure(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Récupère la structure du fichier"""
        data = self.in_memory.get(file_id, {})
        return data.get("file_structure", {}).get("data")
    
    def get_eda_results(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Récupère les résultats EDA"""
        data = self.in_memory.get(file_id, {})
        return data.get("eda_results")
    
    def get_analysis_context(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Récupère le contexte d'analyse"""
        data = self.in_memory.get(file_id, {})
        return data.get("context", {}).get("data")
    
    def get_full_analysis(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Récupère l'analyse complète"""
        data = self.in_memory.get(file_id, {})
        return data.get("full_analysis", {}).get("data")
    
    def get_analysis_status(self, file_id: str) -> Dict[str, Any]:
        """
        Retourne l'état d'avancement de l'analyse d'un fichier
        
        Returns:
            {
              "file_id": "...",
              "file_structure": "completed" | "pending" | "failed",
              "eda": "completed" | "pending" | "failed",
              "full_analysis": "completed" | "pending" | "failed",
              "timestamp": "...",
              "progress": 0-100
            }
        """
        data = self.in_memory.get(file_id, {})
        
        file_struct_status = "completed" if "file_structure" in data else "pending"
        eda_status = "completed" if "eda_results" in data else "pending"
        full_analysis_status = "completed" if "full_analysis" in data else "pending"
        
        # Calcul du progrès
        progress = 0
        if file_struct_status == "completed": progress += 30
        if eda_status == "completed": progress += 35
        if full_analysis_status == "completed": progress += 35
        
        return {
            "file_id": file_id,
            "file_structure": file_struct_status,
            "eda": eda_status,
            "full_analysis": full_analysis_status,
            "progress": progress,
            "timestamp": datetime.now().isoformat()
        }
    
    # ========================================================================
    # 7. NETTOYAGE & PERSISTENCE
    # ========================================================================
    
    def _save_to_disk(self, file_id: str):
        """Sauvegarde en JSON sur disque"""
        try:
            filepath = os.path.join(self.cache_dir, f"{file_id}_analysis.json")
            
            # Conversion des données sérialisables
            data_to_save = {}
            for key, value in self.in_memory[file_id].items():
                if isinstance(value, dict):
                    data_to_save[key] = value
                else:
                    data_to_save[key] = str(value)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"✓ Sauvegarde disque: {filepath}")
        
        except Exception as e:
            logger.warning(f"Erreur sauvegarde disque: {e}")
    
    def clear_cache(self, file_id: str) -> bool:
        """Supprime toute l'analyse d'un fichier"""
        try:
            if file_id in self.in_memory:
                del self.in_memory[file_id]
            
            filepath = os.path.join(self.cache_dir, f"{file_id}_analysis.json")
            if os.path.exists(filepath):
                os.remove(filepath)
            
            logger.info(f"✓ Cache supprimé pour {file_id}")
            return True
        
        except Exception as e:
            logger.error(f"Erreur clear_cache: {e}")
            return False
    
    def get_all_cached_files(self) -> List[str]:
        """Retourne la liste de tous les fichiers analysés"""
        return list(self.in_memory.keys())

# Instance globale
analysis_cache = AnalysisCacheService()