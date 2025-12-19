"""
🤖 SMART ANALYTICS ORCHESTRATOR V2 - ORCHESTRATEUR DE CONTINUITÉ EDA -> ML
"""

import logging
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)


class SmartAnalyticsOrchestrator:
    """
    🚀 ORCHESTRATEUR COMPLET SMART ANALYTICS
    Gère le pipeline: EDA → Feature Engineering → ML → Storytelling
    """
    
    def __init__(self):
        self.analyses_cache = {}
        self.status_cache = {}
        
        # Import des services
        from services.eda_service import eda_service
        from services.feature_forge_service import feature_forge_service
        from services.ml_pipeline_service import ml_pipeline_service
        from services.insight_storyteller_service import insight_storyteller_service
        from services.context_analyst import context_analyst
        
        self.eda_service = eda_service
        self.feature_forge = feature_forge_service
        self.ml_pipeline = ml_pipeline_service
        self.storyteller = insight_storyteller_service
        self.context_analyst = context_analyst
        
        logger.info("🚀 Smart Analytics Orchestrator initialisé")
    
    def _update_status(self, file_id: str, phase: str, message: str, percentage: int):
        """Met à jour le statut de l'analyse"""
        self.status_cache[file_id] = {
            "phase": phase,
            "message": message,
            "percentage": percentage,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _load_and_prepare_data(self, file_path: str) -> pd.DataFrame:
        """Charge et prépare les données"""
        try:
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.tsv'):
                df = pd.read_csv(file_path, sep='\t')
            else:
                raise ValueError(f"Format non supporté: {file_path}")
            
            logger.info(f"📂 Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement données: {e}")
            raise
    
    async def _infer_context(self, df: pd.DataFrame, user_prompt: str) -> Dict[str, Any]:
        """Infère le contexte d'analyse"""
        try:
            sample = df.head(5).copy()
            for col in sample.select_dtypes(include=['datetime64', 'datetimetz']):
                sample[col] = sample[col].dt.strftime('%Y-%m-%d')
            
            data_sample = sample.replace({np.nan: None}).to_dict('records')
            
            context = await self.context_analyst.infer_analysis_goal(
                user_prompt, df.columns.tolist(), data_sample
            )
            
            logger.info(f"🎯 Contexte inféré: {context.get('analysis_type', 'unknown')}")
            return context
            
        except Exception as e:
            logger.error(f"❌ Erreur inférence contexte: {e}")
            return {
                "analysis_type": "exploratory",
                "target_variable": "",
                "focus_variables": []
            }
    
    async def _select_best_clustering_from_eda(self, eda_results: Dict[str, Any]) -> Optional[str]:
        """
        🔍 SÉLECTIONNE LE MEILLEUR CLUSTERING PARMI LES OPTIONS EDA
        
        L'EDA suggère plusieurs segmentations (k=2, 3, 4, 5).
        Critères de sélection :
        1. Score de silhouette (le plus élevé)
        2. Équilibre des clusters (éviter les trop petits)
        3. Interprétabilité (clusters distincts)
        """
        if not eda_results or "metrics" not in eda_results:
            return None
        
        metrics = eda_results["metrics"]
        multi_clustering = metrics.get("multi_clustering", {})
        
        if not multi_clustering.get("clusterings"):
            return None
        
        clusterings = multi_clustering["clusterings"]
        best_key = None
        best_score = -1
        
        logger.info("🎯 Analyse des segmentations EDA pour sélection...")
        
        for key, clustering_data in clusterings.items():
            if not clustering_data:
                continue
            
            # Score de silhouette (critère principal)
            silhouette = clustering_data.get("silhouette_score")
            if silhouette is None:
                continue
            
            # Vérifier l'équilibre des clusters
            cluster_dist = clustering_data.get("cluster_distribution", [])
            if not cluster_dist:
                continue
            
            sizes = [cluster.get("count", 0) for cluster in cluster_dist]
            total = sum(sizes)
            
            if total == 0:
                continue
            
            # Calculer l'équilibre (ratio du plus petit au plus grand)
            min_size = min(sizes)
            max_size = max(sizes)
            balance_ratio = min_size / max_size if max_size > 0 else 0
            
            # Score composite
            # 70% silhouette + 30% équilibre
            composite_score = (silhouette * 0.7) + (balance_ratio * 0.3)
            
            logger.info(f"  📊 {key}: silhouette={silhouette:.3f}, balance={balance_ratio:.3f}, score={composite_score:.3f}")
            
            if composite_score > best_score:
                best_score = composite_score
                best_key = key
        
        if best_key:
            best_data = clusterings[best_key]
            n_clusters = best_data.get("n_clusters", 0)
            logger.info(f"✅ Meilleure segmentation: {best_key} avec {n_clusters} clusters (score: {best_score:.3f})")
            
            # Retourner les informations du meilleur clustering
            return {
                "clustering_key": best_key,
                "n_clusters": n_clusters,
                "silhouette_score": best_data.get("silhouette_score"),
                "balance_ratio": best_score,
                "method_used": best_data.get("method_used", "unknown")
            }
        
        return None
    
    async def run_complete_analysis(self, file_id: str, file_path: str, 
                                   user_prompt: str = "", 
                                   progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        🚀 EXÉCUTE LE PIPELINE COMPLET SMART ANALYTICS
        
        Phases:
        1. Chargement et préparation des données
        2. Analyse EDA exploratoire
        3. Sélection du meilleur clustering
        4. Feature Engineering
        5. Modélisation ML (avec segments EDA)
        6. Storytelling
        """
        
        logger.info("=" * 60)
        logger.info(f"🚀 SMART ANALYTICS - DÉMARRAGE POUR {file_id}")
        logger.info("=" * 60)
        
        results = {
            "success": True,
            "file_id": file_id,
            "phases_completed": [],
            "phases_skipped": [],
            "execution_time": None,
            "data": {},
            "summary": {},
            "continuity": {
                "status": "none",
                "message": "Aucune continuité établie"
            }
        }
        
        start_time = datetime.now()
        
        try:
            # 🔥 PHASE 1: Chargement et préparation
            if progress_callback:
                await progress_callback("Chargement des données...", 10)
            
            self._update_status(file_id, "loading", "Chargement des données", 10)
            
            df = await self._load_and_prepare_data(file_path)
            
            # 🔥 PHASE 2: Inférence du contexte
            if progress_callback:
                await progress_callback("Analyse du contexte...", 20)
            
            self._update_status(file_id, "context", "Inférence du contexte", 20)
            
            context = await self._infer_context(df, user_prompt)
            results["data"]["context"] = context
            results["phases_completed"].append("context_inference")
            
            # 🔥 PHASE 3: EDA Exploratoire
            if progress_callback:
                await progress_callback("Analyse exploratoire (EDA)...", 30)
            
            self._update_status(file_id, "eda", "Analyse exploratoire EDA", 30)
            
            eda_results = await self.eda_service.run_full_eda(
                df=df,
                file_structure={},
                context=context,
                user_prompt=user_prompt
            )
            
            results["data"]["eda"] = eda_results
            results["phases_completed"].append("eda")
            
            # 🔥 PHASE 4: SÉLECTION DU MEILLEUR CLUSTERING
            if progress_callback:
                await progress_callback("Sélection de la meilleure segmentation...", 40)
            
            self._update_status(file_id, "clustering_selection", "Sélection segmentation", 40)
            
            best_clustering = await self._select_best_clustering_from_eda(eda_results)
            
            if best_clustering:
                results["data"]["best_clustering"] = best_clustering
                results["continuity"]["status"] = "clustering_selected"
                results["continuity"]["message"] = f"Segmentation {best_clustering['clustering_key']} sélectionnée ({best_clustering['n_clusters']} clusters)"
                logger.info(f"🎯 Segmentation sélectionnée: {best_clustering['clustering_key']}")
            
            # 🔥 PHASE 5: Feature Engineering
            if progress_callback:
                await progress_callback("Feature Engineering...", 50)
            
            self._update_status(file_id, "feature_engineering", "Feature Engineering", 50)
            
            try:
                feature_results = await self.feature_forge.forge_features(
                    df=df,
                    context=context,
                    options={"include_eda_features": True}
                )
                results["data"]["feature_engineering"] = feature_results
                results["phases_completed"].append("feature_engineering")
            except Exception as e:
                logger.warning(f"⚠️ Feature Engineering échoué: {e}")
                results["phases_skipped"].append("feature_engineering")
            
            # 🔥 PHASE 6: Modélisation ML AVEC CONTINUITÉ
            if progress_callback:
                await progress_callback("Modélisation Machine Learning...", 70)
            
            self._update_status(file_id, "ml_pipeline", "Modélisation ML", 70)
            
            ml_context = context.copy()
            
            # Si on a une segmentation EDA, l'utiliser comme cible
            if best_clustering and "eda" in results["data"]:
                # Ajouter les segments au DataFrame pour le ML
                clustering_key = best_clustering["clustering_key"]
                eda_clusters = results["data"]["eda"]["metrics"]["multi_clustering"]["clusterings"]
                
                if clustering_key in eda_clusters:
                    clustering_data = eda_clusters[clustering_key]
                    scatter_points = clustering_data.get("scatter_points", [])
                    
                    if len(scatter_points) == len(df):
                        # Créer une colonne avec les clusters
                        clusters = [p["cluster"] for p in scatter_points]
                        df_ml = df.copy()
                        df_ml["_EDA_SEGMENT"] = clusters
                        
                        ml_context["target_variable"] = "_EDA_SEGMENT"
                        results["continuity"]["status"] = "full"
                        results["continuity"]["message"] = f"ML entraîné sur les segments EDA ({best_clustering['n_clusters']} clusters)"
                        
                        logger.info(f"🔄 ML utilisera les segments EDA comme cible")
                    else:
                        df_ml = df
                else:
                    df_ml = df
            else:
                df_ml = df
            
            # Exécuter le ML Pipeline
            try:
                ml_results = await self.ml_pipeline.run_ml_pipeline(
                    df=df_ml,
                    context=ml_context,
                    eda_results=eda_results,
                    options={
                        "tune_hyperparams": True,
                        "test_size": 0.2
                    }
                )
                
                results["data"]["ml_pipeline"] = ml_results
                results["phases_completed"].append("ml_pipeline")
                
                # Mettre à jour la continuité si le ML a utilisé les segments EDA
                if ml_results.get("eda_integration", {}).get("segments_used", False):
                    results["continuity"]["status"] = "ml_trained_on_eda"
                    results["continuity"]["message"] = "ML entraîné avec succès sur les segments EDA"
                
            except Exception as e:
                logger.warning(f"⚠️ ML Pipeline échoué: {e}")
                results["phases_skipped"].append("ml_pipeline")
                results["data"]["ml_pipeline"] = {
                    "success": False,
                    "error": str(e),
                    "ml_applicable": False
                }
            
            # 🔥 PHASE 7: Storytelling
            if progress_callback:
                await progress_callback("Génération du rapport...", 90)
            
            self._update_status(file_id, "storytelling", "Génération rapport", 90)
            
            try:
                storytelling_results = await self.storyteller.tell_the_story(
                    eda_results=eda_results,
                    ml_results=results["data"].get("ml_pipeline", {}),
                    feature_engineering=results["data"].get("feature_engineering", {}),
                    context=context,
                    options={"use_llm_enrichment": True}
                )
                
                results["data"]["storytelling"] = storytelling_results
                results["phases_completed"].append("storytelling")
            except Exception as e:
                logger.warning(f"⚠️ Storytelling échoué: {e}")
                results["phases_skipped"].append("storytelling")
            
            # 🔥 PHASE 8: Synthèse finale
            if progress_callback:
                await progress_callback("Finalisation...", 100)
            
            self._update_status(file_id, "complete", "Analyse terminée", 100)
            
            # Calcul du temps d'exécution
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            results["execution_time"] = execution_time
            
            # Résumé synthétique
            results["summary"] = {
                "total_phases": 7,
                "completed": len(results["phases_completed"]),
                "skipped": len(results["phases_skipped"]),
                "continuity_status": results["continuity"]["status"],
                "best_clustering": best_clustering.get("clustering_key") if best_clustering else None,
                "ml_applicable": results["data"].get("ml_pipeline", {}).get("ml_applicable", False),
                "execution_time_seconds": execution_time
            }
            
            # Mettre en cache
            self.analyses_cache[file_id] = results
            
            logger.info("=" * 60)
            logger.info(f"✅ SMART ANALYTICS TERMINÉ POUR {file_id}")
            logger.info(f"   Phases: {len(results['phases_completed'])}/{7}")
            logger.info(f"   Temps: {execution_time:.1f}s")
            logger.info(f"   Continuité: {results['continuity']['status']}")
            logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur Smart Analytics: {e}", exc_info=True)
            results["success"] = False
            results["error"] = str(e)
            results["execution_time"] = (datetime.now() - start_time).total_seconds()
            return results
    
    async def get_analysis_status(self, file_id: str) -> Dict[str, Any]:
        """Récupère le statut d'une analyse"""
        return self.status_cache.get(file_id, {
            "phase": "unknown",
            "message": "Analyse non trouvée",
            "percentage": 0,
            "timestamp": datetime.now().isoformat()
        })
    
    async def get_analysis_results(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Récupère les résultats d'une analyse"""
        return self.analyses_cache.get(file_id)
    
    def clear_cache(self, file_id: str):
        """Supprime les résultats en cache"""
        if file_id in self.analyses_cache:
            del self.analyses_cache[file_id]
        if file_id in self.status_cache:
            del self.status_cache[file_id]
        logger.info(f"🗑️ Cache vidé pour {file_id}")


# 🔥 INSTANCE GLOBALE - IMPORTÉE DANS MAIN.PY
smart_analytics_orchestrator = SmartAnalyticsOrchestrator()


async def analyze_file_complete(file_id: str, file_path: str, user_prompt: str = "") -> Dict[str, Any]:
    """
    🚀 FONCTION D'ANALYSE COMPLÈTE - POUR L'ENDPOINT API
    
    Wrapper autour de l'orchestrateur pour une utilisation simple
    """
    return await smart_analytics_orchestrator.run_complete_analysis(
        file_id=file_id,
        file_path=file_path,
        user_prompt=user_prompt
    )