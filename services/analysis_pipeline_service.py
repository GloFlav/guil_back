# backend/services/analysis_pipeline_service.py
"""
Pipeline complet d'analyse de données
Chaîne: File Structure → EDA → Feature Engineering → Full Analysis
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional, Callable
import asyncio

from services.analysis_cache_service import analysis_cache
from services.file_structure_analysis_service import file_structure_analysis_service
from services.eda_service import eda_service
from services.feature_service import feature_service
from services.context_analyst import context_analyst
from config.settings import settings
import os

logger = logging.getLogger(__name__)

class AnalysisPipelineService:
    """
    Orchestration complète du pipeline d'analyse
    
    Flux:
    1. File Structure Analysis → métadonnées + TTS
    2. EDA → statistiques + visualisations
    3. Context Inference → détection target + type analyse
    4. Feature Engineering → nettoyage + encodage
    5. Full Analysis → résultats finaux
    """
    
    async def run_complete_analysis_pipeline(
        self, 
        file_id: str,
        file_path: str,
        user_prompt: str = "",
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Lance l'analyse complète d'un fichier
        
        Args:
            file_id: ID du fichier (clean_*.xlsx)
            file_path: Chemin complet du fichier
            user_prompt: Prompt utilisateur optionnel
            progress_callback: Fonction callback pour suivre la progression
        
        Returns:
            Dict avec toute l'analyse
        """
        
        logger.info(f"🚀 Démarrage pipeline complet pour {file_id}")
        
        try:
            # =================================================================
            # PHASE 1: FILE STRUCTURE ANALYSIS (Métadonnées + TTS)
            # =================================================================
            
            await self._progress(progress_callback, 5, "📊 Phase 1: Analyse structure...")
            
            # Charger le fichier
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
            
            logger.info(f"Fichier chargé: {df.shape}")
            
            # Analyse structure
            file_stats = {
                "file_id": file_id,
                "filename": os.path.basename(file_path),
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "empty_columns": [],
                "partially_empty_columns": []
            }
            
            file_structure_result = await file_structure_analysis_service.analyze_file_structure(
                file_path=file_path,
                df=df,
                file_stats=file_stats
            )
            
            if not file_structure_result.get("success"):
                raise Exception(f"File structure failed: {file_structure_result.get('error')}")
            
            # Cache le résultat
            analysis_cache.save_file_structure(file_id, file_structure_result)
            logger.info(f"✓ Phase 1 complétée")
            
            # =================================================================
            # PHASE 2: CONTEXT INFERENCE (Détecte target + type analyse)
            # =================================================================
            
            await self._progress(progress_callback, 20, "🎯 Phase 2: Inférence contexte...")
            
            # Préparer sample pour le context
            cols = df.columns.tolist()
            sample = df.head(5).copy()
            for c in sample.select_dtypes(include=['datetime64','datetimetz']):
                sample[c] = sample[c].dt.strftime('%Y-%m-%d')
            data_sample = sample.replace({pd.isna(sample): None}).to_dict('records')
            
            context = await context_analyst.infer_analysis_goal(user_prompt, cols, data_sample)
            logger.info(f"✓ Contexte inféré: target={context.get('target_variable')}")
            
            # Cache le contexte
            analysis_cache.save_analysis_context(file_id, context)
            
            # =================================================================
            # PHASE 3: FEATURE ENGINEERING
            # =================================================================
            
            await self._progress(progress_callback, 35, "⚙️ Phase 3: Feature Engineering...")
            
            df_processed = feature_service.process_features(
                df.copy(), 
                context.get("target_variable", "")
            )
            
            logger.info(f"✓ Features: {df.shape} → {df_processed.shape}")
            analysis_cache.save_processed_dataframe(file_id, df_processed, "feature_engineered")
            
            # =================================================================
            # PHASE 4: EDA (Exploration Data Analysis)
            # =================================================================
            
            await self._progress(progress_callback, 55, "📈 Phase 4: EDA et statistiques...")
            
            eda_results = await eda_service.run_full_eda(
                df_processed, 
                context, 
                user_prompt
            )
            
            logger.info(f"✓ EDA complétée")
            analysis_cache.save_eda_results(file_id, eda_results)
            
            # =================================================================
            # PHASE 5: FULL ANALYSIS (Synthèse finale)
            # =================================================================
            
            await self._progress(progress_callback, 80, "🎯 Phase 5: Synthèse finale...")
            
            full_analysis = {
                "file_id": file_id,
                "filename": file_stats["filename"],
                "analysis_type": context.get("analysis_type", "descriptive"),
                "target_variable": context.get("target_variable"),
                "summary": {
                    "total_rows_original": len(df),
                    "total_cols_original": len(df.columns),
                    "total_rows_final": len(df_processed),
                    "total_cols_final": len(df_processed.columns),
                    "numeric_cols": len(df_processed.select_dtypes(include=['number']).columns),
                    "categorical_cols": len(df_processed.select_dtypes(include=['object', 'category']).columns),
                    "missing_values": int(df_processed.isnull().sum().sum())
                },
                "structure": file_structure_result.get("structure", {}),
                "context": context,
                "eda": {
                    "univariate": eda_results.get("metrics", {}).get("univariate", {}),
                    "correlation": eda_results.get("metrics", {}).get("correlation", {}),
                    "clustering": eda_results.get("metrics", {}).get("clustering"),
                    "statistical_tests": eda_results.get("metrics", {}).get("tests", [])[:10],  # Top 10
                    "themes": eda_results.get("metrics", {}).get("themes", {}),
                    "distributions": eda_results.get("charts_data", {}).get("distributions", {}),
                    "pie_charts": eda_results.get("charts_data", {}).get("pies", []),
                    "scatter_plots": eda_results.get("charts_data", {}).get("scatters", [])
                },
                "insights": eda_results.get("ai_insights", []),
                "tts_text": file_structure_result.get("tts_text", ""),
                "ai_summary": file_structure_result.get("ai_summary", ""),
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            # Cache le résultat final
            analysis_cache.save_full_analysis(file_id, full_analysis)
            
            await self._progress(progress_callback, 100, "✅ Analyse complète!")
            logger.info(f"✅ Pipeline terminé pour {file_id}")
            
            return full_analysis
        
        except Exception as e:
            logger.error(f"❌ Erreur pipeline: {e}", exc_info=True)
            await self._progress(progress_callback, 0, f"❌ Erreur: {str(e)}")
            raise
    
    async def get_analysis_status(self, file_id: str) -> Dict[str, Any]:
        """
        Récupère l'état actuel de l'analyse
        
        Returns:
            {
              "file_id": "...",
              "file_structure": "completed|pending",
              "eda": "completed|pending",
              "full_analysis": "completed|pending",
              "progress": 0-100
            }
        """
        return analysis_cache.get_analysis_status(file_id)
    
    async def get_analysis_results(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Récupère les résultats complets d'une analyse"""
        return analysis_cache.get_full_analysis(file_id)
    
    def clear_analysis(self, file_id: str) -> bool:
        """Supprime une analyse du cache"""
        return analysis_cache.clear_cache(file_id)
    
    # ========================================================================
    # Utilitaires
    # ========================================================================
    
    async def _progress(
        self, 
        callback: Optional[Callable],
        percentage: int,
        message: str
    ):
        """Envoie une mise à jour de progression"""
        if callback:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message, percentage)
                else:
                    callback(message, percentage)
            except Exception as e:
                logger.warning(f"Erreur callback: {e}")

# Instance globale
analysis_pipeline = AnalysisPipelineService()