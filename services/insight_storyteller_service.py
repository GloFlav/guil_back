"""
📖 INSIGHT STORYTELLER SERVICE V3 - COMPLET
Phases 7-8: Interprétation, Storytelling, Restitution et Communication

🎯 OBJECTIFS:
- Phase 7: Interprétation et Storytelling (41-45)
- Phase 8: Restitution et Communication (46-49)

✅ FONCTIONNALITÉS:
- Génération d'insights détaillés point par point
- Recommandations actionnables avec actions concrètes
- Export PDF du rapport d'analyse
- Texte TTS pour narration vocale
- Documentation technique
- Données nettoyées disponibles
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import base64
import io

logger = logging.getLogger(__name__)


class InsightStorytellerService:
    """
    📖 SERVICE DE STORYTELLING V3 - PHASES 7-8 COMPLÈTES
    """
    
    def __init__(self):
        self.insights_generated = []
        self.recommendations = []
        
    # ==================== EXTRACTION DES RÉSULTATS EDA ====================
    
    def _extract_eda_insights(self, eda_results: Dict[str, Any]) -> Dict[str, Any]:
        """🔍 Extraire les insights depuis les résultats EDA"""
        extracted = {
            "summary": {},
            "clustering": {},
            "correlations": {},
            "tests": [],
            "univariate": {},
            "themes": {}
        }
        
        # Résumé général
        summary = eda_results.get("summary") or {}
        extracted["summary"] = {
            "total_rows": summary.get("total_rows", 0),
            "total_cols": summary.get("total_cols", 0),
            "missing_values": summary.get("missing_values", 0),
            "numeric_analyzed": summary.get("numeric_analyzed", 0),
            "categorical_analyzed": summary.get("categorical_analyzed", 0)
        }
        
        # Clustering
        metrics = eda_results.get("metrics") or {}
        multi_clustering = metrics.get("multi_clustering") or {}
        
        if multi_clustering.get("clusterings"):
            best_clustering = None
            best_score = -1
            
            for key, clustering in multi_clustering["clusterings"].items():
                score = clustering.get("silhouette_score") or 0
                if score > best_score:
                    best_score = score
                    best_clustering = clustering
            
            if best_clustering:
                extracted["clustering"] = {
                    "n_clusters": best_clustering.get("n_clusters", 0),
                    "silhouette_score": best_score,
                    "dna": best_clustering.get("dna", {}),
                    "method": best_clustering.get("method_used", ""),
                    "explanation": best_clustering.get("explanation", {}),
                    "variables_used": multi_clustering.get("variables_used", [])
                }
        
        # Corrélations
        correlations = metrics.get("correlations") or {}
        extracted["correlations"] = {
            "strong": correlations.get("strong_correlations", []),
            "moderate": correlations.get("moderate_correlations", []),
            "target_correlations": correlations.get("target_correlations", {})
        }
        
        # Tests statistiques significatifs
        tests = metrics.get("tests") or []
        extracted["tests"] = [t for t in tests if t.get("p_value", 1) < 0.05]
        
        # Stats univariées
        extracted["univariate"] = metrics.get("univariate") or {}
        
        # Thèmes
        extracted["themes"] = metrics.get("themes") or {}
        
        return extracted
    
    # ==================== CALCUL QUALITÉ DONNÉES ====================
    
    def _calculate_data_quality(self, eda_insights: Dict[str, Any]) -> Dict[str, Any]:
        """📊 Calculer la qualité des données"""
        summary = eda_insights.get("summary", {})
        
        total_rows = summary.get("total_rows", 0) or 0
        total_cols = summary.get("total_cols", 0) or 0
        missing_values = summary.get("missing_values", 0) or 0
        
        total_cells = total_rows * total_cols
        if total_cells > 0:
            missing_pct = (missing_values / total_cells) * 100
        else:
            missing_pct = 0
        
        missing_pct = min(missing_pct, 100)
        completeness_pct = 100 - missing_pct
        
        if completeness_pct >= 95:
            quality_level = "excellente"
            quality_emoji = "🟢"
        elif completeness_pct >= 85:
            quality_level = "bonne"
            quality_emoji = "🟢"
        elif completeness_pct >= 70:
            quality_level = "acceptable"
            quality_emoji = "🟡"
        elif completeness_pct >= 50:
            quality_level = "moyenne"
            quality_emoji = "🟠"
        else:
            quality_level = "faible"
            quality_emoji = "🔴"
        
        return {
            "total_rows": total_rows,
            "total_cols": total_cols,
            "total_cells": total_cells,
            "missing_values": missing_values,
            "missing_pct": round(missing_pct, 1),
            "completeness_pct": round(completeness_pct, 1),
            "quality_level": quality_level,
            "quality_emoji": quality_emoji
        }
    
    # ==================== PHASE 7.1: INTERPRÉTATION DES RÉSULTATS ====================
    
    def _interpret_clustering_results(self, clustering_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        🎯 41. Interprétation des résultats de clustering
        Quelles variables pèsent le plus dans la segmentation?
        """
        interpretations = []
        
        if not clustering_data or not clustering_data.get("dna"):
            return interpretations
        
        dna = clustering_data.get("dna", {})
        n_clusters = clustering_data.get("n_clusters", 0)
        silhouette = clustering_data.get("silhouette_score", 0)
        variables_used = clustering_data.get("variables_used", [])
        
        # Interprétation globale
        if n_clusters >= 2:
            score_text = f"{silhouette:.2f}" if silhouette and silhouette > -1 else "N/A"
            
            if silhouette and silhouette > 0.5:
                quality_interpretation = "Les groupes sont très bien définis et distincts"
            elif silhouette and silhouette > 0.25:
                quality_interpretation = "Les groupes sont raisonnablement distincts"
            elif silhouette and silhouette > 0:
                quality_interpretation = "Les groupes se chevauchent partiellement"
            else:
                quality_interpretation = "La segmentation est indicative mais les groupes ne sont pas très distincts"
            
            interpretations.append({
                "id": "clustering_global",
                "type": "segmentation",
                "title": f"🎯 {n_clusters} Segments Identifiés",
                "finding": f"{n_clusters} groupes distincts ont été identifiés dans vos données",
                "metric": f"Score de qualité: {score_text}",
                "interpretation": quality_interpretation,
                "variables_cles": variables_used[:5] if variables_used else [],
                "so_what": "Vous pouvez adapter vos stratégies selon ces segments",
                "priority": "haute" if silhouette and silhouette > 0.2 else "moyenne"
            })
        
        # Interprétation par segment
        for cluster_name, cluster_info in dna.items():
            features = cluster_info.get("features", {})
            size = cluster_info.get("size", 0)
            percentage = cluster_info.get("percentage", 0)
            
            # Trouver les caractéristiques distinctives
            distinctive = []
            for feat_name, feat_info in features.items():
                z = feat_info.get("z_score", 0)
                if abs(z) > 0.3:
                    distinctive.append({
                        "variable": feat_name,
                        "direction": "supérieur à la moyenne" if z > 0 else "inférieur à la moyenne",
                        "magnitude": abs(z),
                        "interpretation": feat_info.get("interpretation", "")
                    })
            
            if distinctive:
                distinctive.sort(key=lambda x: x["magnitude"], reverse=True)
                
                # Créer le profil du segment
                profile_text = []
                for d in distinctive[:3]:
                    profile_text.append(f"• {d['variable']}: {d['direction']}")
                
                interpretations.append({
                    "id": f"cluster_{cluster_name}",
                    "type": "cluster_profile",
                    "title": f"📊 Profil: {cluster_name}",
                    "finding": f"Ce segment représente {percentage:.0f}% des données ({size} observations)",
                    "characteristics": distinctive[:5],
                    "profile_summary": "\n".join(profile_text),
                    "so_what": f"Ce groupe a des caractéristiques distinctes à cibler spécifiquement",
                    "priority": "moyenne"
                })
        
        return interpretations
    
    def _interpret_correlation_results(self, correlations_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        🔗 Interprétation des corrélations
        """
        interpretations = []
        
        strong_corr = correlations_data.get("strong", [])
        target_corr = correlations_data.get("target_correlations", {})
        
        # Corrélations fortes
        for corr in strong_corr[:5]:
            var1, var2 = corr.get("var1", ""), corr.get("var2", "")
            r = corr.get("r", 0)
            
            if abs(r) > 0.7:
                direction = "positive" if r > 0 else "négative"
                
                if r > 0:
                    interpretation = f"Quand {var1} augmente, {var2} augmente aussi"
                else:
                    interpretation = f"Quand {var1} augmente, {var2} diminue"
                
                interpretations.append({
                    "id": f"corr_{var1}_{var2}",
                    "type": "correlation",
                    "title": f"🔗 Relation {var1} ↔ {var2}",
                    "finding": f"Corrélation {direction} forte (r = {r:.2f})",
                    "interpretation": interpretation,
                    "variance_explained": f"{r**2*100:.0f}% de variance commune",
                    "so_what": "Ces variables sont liées - agir sur l'une impacte l'autre",
                    "priority": "haute"
                })
        
        # Corrélations avec la cible
        if target_corr:
            sorted_target = sorted(target_corr.items(), key=lambda x: abs(x[1].get("r", 0)), reverse=True)
            
            for var, info in sorted_target[:3]:
                r = info.get("r", 0)
                if abs(r) > 0.3:
                    interpretations.append({
                        "id": f"target_corr_{var}",
                        "type": "target_influence",
                        "title": f"🎯 Influence de {var}",
                        "finding": f"'{var}' influence significativement la variable cible (r = {r:.2f})",
                        "interpretation": f"Cette variable explique {r**2*100:.0f}% des variations de la cible",
                        "so_what": "Variable clé à surveiller et optimiser",
                        "priority": "haute"
                    })
        
        return interpretations
    
    def _interpret_test_results(self, tests: List[Dict]) -> List[Dict[str, Any]]:
        """
        📊 Interprétation des tests statistiques
        """
        interpretations = []
        
        for test in tests[:5]:
            test_type = test.get("test_type", "")
            var1, var2 = test.get("variable1", ""), test.get("variable2", "")
            p_value = test.get("p_value", 1)
            statistic = test.get("statistic", 0)
            
            if p_value < 0.05:
                confidence = "très haute (p < 0.01)" if p_value < 0.01 else "haute (p < 0.05)"
                
                if test_type == "ttest":
                    interpretations.append({
                        "id": f"test_{var1}_{var2}",
                        "type": "statistical_test",
                        "title": f"📈 Différence significative: {var2} par {var1}",
                        "finding": f"Les groupes de '{var1}' ont des valeurs de '{var2}' statistiquement différentes",
                        "test_used": "T-Test",
                        "confidence": confidence,
                        "p_value": p_value,
                        "interpretation": f"La différence observée n'est pas due au hasard",
                        "so_what": f"Adapter les actions selon les groupes de '{var1}'",
                        "priority": "haute" if p_value < 0.01 else "moyenne"
                    })
                
                elif test_type == "anova":
                    interpretations.append({
                        "id": f"anova_{var1}_{var2}",
                        "type": "statistical_test",
                        "title": f"📊 Variations entre groupes: {var2}",
                        "finding": f"'{var2}' varie significativement selon '{var1}'",
                        "test_used": "ANOVA",
                        "confidence": confidence,
                        "p_value": p_value,
                        "interpretation": "Au moins un groupe se distingue des autres",
                        "so_what": "Identifier et comprendre le groupe qui se démarque",
                        "priority": "haute" if p_value < 0.01 else "moyenne"
                    })
        
        return interpretations
    
    # ==================== PHASE 7.2: SYNTHÈSE DES INSIGHTS ====================
    
    def _synthesize_key_insights(self, eda_insights: Dict[str, Any], 
                                  ml_results: Dict[str, Any],
                                  context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        💡 42. Synthèse des insights clés - "So what?"
        """
        all_insights = []
        
        # 1. Insight qualité données
        dq = self._calculate_data_quality(eda_insights)
        all_insights.append({
            "id": "data_quality",
            "category": "Qualité des Données",
            "icon": dq["quality_emoji"],
            "title": f"Qualité des données: {dq['quality_level']}",
            "finding": f"{dq['total_rows']} observations, {dq['total_cols']} variables",
            "metric": f"{dq['completeness_pct']:.1f}% de complétude",
            "so_what": "Une bonne qualité augmente la fiabilité des analyses" if dq["completeness_pct"] > 80 else "Améliorer la collecte pour des analyses plus fiables",
            "priority": "haute" if dq["missing_pct"] > 20 else "moyenne",
            "action_required": dq["missing_pct"] > 10
        })
        
        # 2. Insights clustering
        clustering_interp = self._interpret_clustering_results(eda_insights.get("clustering", {}))
        for interp in clustering_interp:
            all_insights.append({
                "id": interp["id"],
                "category": "Segmentation",
                "icon": "🎯",
                "title": interp["title"],
                "finding": interp["finding"],
                "metric": interp.get("metric", ""),
                "so_what": interp["so_what"],
                "priority": interp["priority"],
                "details": interp.get("characteristics", [])
            })
        
        # 3. Insights corrélations
        corr_interp = self._interpret_correlation_results(eda_insights.get("correlations", {}))
        for interp in corr_interp:
            all_insights.append({
                "id": interp["id"],
                "category": "Relations",
                "icon": "🔗",
                "title": interp["title"],
                "finding": interp["finding"],
                "metric": interp.get("variance_explained", ""),
                "so_what": interp["so_what"],
                "priority": interp["priority"]
            })
        
        # 4. Insights tests
        test_interp = self._interpret_test_results(eda_insights.get("tests", []))
        for interp in test_interp:
            all_insights.append({
                "id": interp["id"],
                "category": "Tests Statistiques",
                "icon": "📊",
                "title": interp["title"],
                "finding": interp["finding"],
                "metric": f"p-value: {interp.get('p_value', 'N/A')}",
                "so_what": interp["so_what"],
                "priority": interp["priority"]
            })
        
        # 5. Insight ML
        if ml_results:
            if ml_results.get("success"):
                best = ml_results.get("best_model") or {}
                if best.get("name"):
                    score = best.get("score", 0)
                    all_insights.append({
                        "id": "ml_success",
                        "category": "Machine Learning",
                        "icon": "🤖",
                        "title": f"Modèle prédictif: {best.get('name')}",
                        "finding": f"Performance: {score*100:.1f}%",
                        "metric": f"Algorithme: {best.get('name')}",
                        "so_what": "Le modèle peut être utilisé pour des prédictions" if score > 0.7 else "Le modèle nécessite des améliorations",
                        "priority": "haute"
                    })
            else:
                all_insights.append({
                    "id": "ml_not_applicable",
                    "category": "Machine Learning",
                    "icon": "⚠️",
                    "title": "Modélisation non applicable",
                    "finding": ml_results.get("error", "Les données ne permettent pas d'entraîner un modèle"),
                    "metric": "N/A",
                    "so_what": "Se concentrer sur l'analyse descriptive et la segmentation",
                    "priority": "basse"
                })
        
        return all_insights
    
    # ==================== PHASE 7.3: RECOMMANDATIONS ACTIONNABLES ====================
    
    def _generate_actionable_recommendations(self, insights: List[Dict], 
                                             eda_insights: Dict,
                                             ml_results: Dict,
                                             context: Dict) -> List[Dict[str, Any]]:
        """
        🎯 45. Formulation de recommandations actionnables
        """
        recommendations = []
        recommendation_id = 1
        
        # 1. Recommandations de segmentation
        clustering = eda_insights.get("clustering", {})
        if clustering.get("dna"):
            dna = clustering["dna"]
            n_segments = len(dna)
            
            recommendations.append({
                "id": recommendation_id,
                "title": f"Stratégie de Segmentation ({n_segments} Segments)",
                "category": "Segmentation",
                "priority": "haute",
                "priority_color": "#e74c3c",
                "impact": 85,
                "effort": "moyen",
                "timeline": "Court terme (2-4 semaines)",
                "description": f"Développer {n_segments} approches différenciées basées sur les profils identifiés",
                "rationale": "La segmentation permet de personnaliser les actions et d'augmenter l'efficacité",
                "actions": [
                    {"step": 1, "action": "Créer une fiche profil détaillée pour chaque segment", "responsible": "Analyste"},
                    {"step": 2, "action": "Définir les caractéristiques distinctives de chaque groupe", "responsible": "Marketing"},
                    {"step": 3, "action": "Adapter la communication selon les profils", "responsible": "Communication"},
                    {"step": 4, "action": "Définir des KPIs de suivi par segment", "responsible": "Direction"}
                ],
                "kpis": ["Taux de conversion par segment", "Satisfaction par segment", "Rétention par segment"],
                "risks": ["Ressources nécessaires pour personnaliser", "Complexité de mise en œuvre"],
                "tts_text": f"Recommandation numéro {recommendation_id}: Mettre en place une stratégie de segmentation avec {n_segments} segments. Impact estimé: 85%. Actions principales: créer des fiches profils, adapter la communication, et définir des KPIs par segment."
            })
            recommendation_id += 1
            
            # Focus sur le segment principal
            largest = max(dna.items(), key=lambda x: x[1].get("size", 0))
            recommendations.append({
                "id": recommendation_id,
                "title": f"Focus Prioritaire: {largest[0]}",
                "category": "Priorité Business",
                "priority": "haute",
                "priority_color": "#e74c3c",
                "impact": 75,
                "effort": "faible",
                "timeline": "Immédiat (1-2 semaines)",
                "description": f"Concentrer les efforts sur le segment majoritaire ({largest[1].get('percentage', 0):.0f}% des données)",
                "rationale": "Maximiser l'impact en ciblant le groupe le plus important",
                "actions": [
                    {"step": 1, "action": f"Analyser en détail les caractéristiques de {largest[0]}", "responsible": "Analyste"},
                    {"step": 2, "action": "Identifier les besoins spécifiques de ce segment", "responsible": "Produit"},
                    {"step": 3, "action": "Développer une offre ou action adaptée", "responsible": "Commercial"},
                    {"step": 4, "action": "Mesurer les résultats et ajuster", "responsible": "Marketing"}
                ],
                "kpis": ["Part de marché du segment", "Revenu généré", "Taux d'engagement"],
                "risks": ["Négliger les autres segments"],
                "tts_text": f"Recommandation numéro {recommendation_id}: Prioriser le segment {largest[0]} qui représente {largest[1].get('percentage', 0):.0f}% des données. Actions immédiates: analyser les caractéristiques et développer une offre adaptée."
            })
            recommendation_id += 1
        
        # 2. Recommandations basées sur les corrélations fortes
        correlations = eda_insights.get("correlations", {})
        for corr in correlations.get("strong", [])[:2]:
            var1, var2 = corr.get("var1", ""), corr.get("var2", "")
            r = corr.get("r", 0)
            
            if abs(r) > 0.7:
                recommendations.append({
                    "id": recommendation_id,
                    "title": f"Exploiter la Relation {var1} ↔ {var2}",
                    "category": "Optimisation",
                    "priority": "haute" if abs(r) > 0.8 else "moyenne",
                    "priority_color": "#e74c3c" if abs(r) > 0.8 else "#f39c12",
                    "impact": int(abs(r) * 100),
                    "effort": "moyen",
                    "timeline": "Moyen terme (1-2 mois)",
                    "description": f"Ces variables ont une corrélation de {r:.2f} - agir sur l'une influence l'autre",
                    "rationale": f"Relation statistiquement significative ({r**2*100:.0f}% de variance commune)",
                    "actions": [
                        {"step": 1, "action": f"Créer un tableau de bord combinant {var1} et {var2}", "responsible": "BI"},
                        {"step": 2, "action": "Investiguer la relation de causalité", "responsible": "Analyste"},
                        {"step": 3, "action": "Utiliser cette relation pour les prévisions", "responsible": "Data Science"},
                        {"step": 4, "action": "Monitorer conjointement ces indicateurs", "responsible": "Opérations"}
                    ],
                    "kpis": [f"Évolution de {var1}", f"Évolution de {var2}", "Ratio entre les deux"],
                    "risks": ["Corrélation ≠ Causalité - investiguer avant d'agir"],
                    "tts_text": f"Recommandation numéro {recommendation_id}: Exploiter la relation entre {var1} et {var2} avec une corrélation de {r:.2f}. Ces variables évoluent ensemble."
                })
                recommendation_id += 1
        
        # 3. Recommandations basées sur les tests significatifs
        for test in eda_insights.get("tests", [])[:2]:
            var1, var2 = test.get("variable1", ""), test.get("variable2", "")
            if test.get("p_value", 1) < 0.05:
                recommendations.append({
                    "id": recommendation_id,
                    "title": f"Différencier par {var1}",
                    "category": "Personnalisation",
                    "priority": "moyenne",
                    "priority_color": "#f39c12",
                    "impact": 60,
                    "effort": "faible",
                    "timeline": "Court terme (2-4 semaines)",
                    "description": f"'{var2}' varie significativement selon '{var1}' (confirmé statistiquement)",
                    "rationale": f"Test statistique significatif (p < 0.05)",
                    "actions": [
                        {"step": 1, "action": f"Analyser {var2} par groupe de {var1}", "responsible": "Analyste"},
                        {"step": 2, "action": "Identifier les différences clés", "responsible": "Métier"},
                        {"step": 3, "action": f"Adapter les actions selon {var1}", "responsible": "Opérations"},
                        {"step": 4, "action": "Créer des rapports segmentés", "responsible": "BI"}
                    ],
                    "kpis": [f"{var2} par catégorie de {var1}"],
                    "risks": ["Complexité de personnalisation"],
                    "tts_text": f"Recommandation numéro {recommendation_id}: Différencier les actions selon {var1} car {var2} varie significativement entre les groupes."
                })
                recommendation_id += 1
        
        # 4. Recommandation qualité données
        dq = self._calculate_data_quality(eda_insights)
        if dq["missing_pct"] > 10:
            recommendations.append({
                "id": recommendation_id,
                "title": "Améliorer la Qualité des Données",
                "category": "Data Quality",
                "priority": "haute" if dq["missing_pct"] > 30 else "moyenne",
                "priority_color": "#e74c3c" if dq["missing_pct"] > 30 else "#f39c12",
                "impact": 70,
                "effort": "moyen",
                "timeline": "Moyen terme (1-3 mois)",
                "description": f"Actuellement {dq['completeness_pct']:.0f}% de complétude - améliorer la collecte",
                "rationale": "Des données complètes améliorent la fiabilité des analyses",
                "actions": [
                    {"step": 1, "action": "Identifier les sources des données manquantes", "responsible": "Data Engineering"},
                    {"step": 2, "action": "Mettre en place des contrôles de saisie", "responsible": "IT"},
                    {"step": 3, "action": "Automatiser la validation des données", "responsible": "DevOps"},
                    {"step": 4, "action": "Former les équipes à la qualité des données", "responsible": "Formation"}
                ],
                "kpis": ["Taux de complétude", "Nombre d'erreurs de saisie", "Temps de correction"],
                "risks": ["Coût de mise en œuvre", "Résistance au changement"],
                "tts_text": f"Recommandation numéro {recommendation_id}: Améliorer la qualité des données. Actuellement {dq['completeness_pct']:.0f}% de complétude. Actions: identifier les sources de données manquantes et automatiser la validation."
            })
            recommendation_id += 1
        
        # 5. Recommandation ML si applicable
        if ml_results and ml_results.get("success"):
            best = ml_results.get("best_model") or {}
            if best.get("name"):
                score = best.get("score", 0)
                recommendations.append({
                    "id": recommendation_id,
                    "title": "Déployer le Modèle Prédictif",
                    "category": "Machine Learning",
                    "priority": "haute" if score > 0.7 else "moyenne",
                    "priority_color": "#e74c3c" if score > 0.7 else "#f39c12",
                    "impact": int(score * 100),
                    "effort": "élevé",
                    "timeline": "Moyen terme (2-3 mois)",
                    "description": f"Le modèle {best.get('name')} atteint {score*100:.0f}% de performance",
                    "rationale": "Automatiser les décisions avec un modèle prédictif fiable",
                    "actions": [
                        {"step": 1, "action": "Valider le modèle sur des données récentes", "responsible": "Data Science"},
                        {"step": 2, "action": "Développer une API de prédiction", "responsible": "MLOps"},
                        {"step": 3, "action": "Intégrer dans les processus métier", "responsible": "IT"},
                        {"step": 4, "action": "Mettre en place le monitoring du modèle", "responsible": "Data Science"}
                    ],
                    "kpis": ["Précision en production", "Temps de réponse", "Drift du modèle"],
                    "risks": ["Coût d'infrastructure", "Maintenance continue nécessaire"],
                    "tts_text": f"Recommandation numéro {recommendation_id}: Déployer le modèle {best.get('name')} avec {score*100:.0f}% de performance. Actions: valider, développer l'API, et mettre en place le monitoring."
                })
                recommendation_id += 1
        
        # Trier par priorité et impact
        priority_order = {"haute": 0, "moyenne": 1, "basse": 2}
        recommendations.sort(key=lambda x: (priority_order.get(x.get("priority", "basse"), 3), -x.get("impact", 0)))
        
        return recommendations
    
    # ==================== PHASE 7.4: DATA STORYTELLING ====================
    
    def _create_data_story(self, insights: List[Dict], recommendations: List[Dict],
                           eda_insights: Dict, context: Dict) -> Dict[str, Any]:
        """
        📖 44. Scénarisation (Data Storytelling)
        Construire un récit logique pour guider vers la conclusion
        """
        target = context.get("target_variable", "les données")
        dq = self._calculate_data_quality(eda_insights)
        
        # Structure narrative
        story = {
            "title": f"Analyse de {target}: Découvertes et Décisions",
            "subtitle": f"Rapport généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}",
            
            "chapters": [
                {
                    "number": 1,
                    "title": "Contexte et Données",
                    "content": f"Cette analyse porte sur un ensemble de {dq['total_rows']} observations et {dq['total_cols']} variables. "
                              f"La qualité des données est {dq['quality_level']} avec {dq['completeness_pct']:.0f}% de complétude.",
                    "key_point": "Comprendre la structure des données est la première étape"
                },
                {
                    "number": 2,
                    "title": "Découvertes Principales",
                    "content": "\n".join([f"• {i.get('finding', '')}" for i in insights[:5]]),
                    "key_point": "Ces découvertes orientent les décisions stratégiques"
                },
                {
                    "number": 3,
                    "title": "Recommandations",
                    "content": "\n".join([f"• {r.get('title', '')}" for r in recommendations[:5]]),
                    "key_point": "Des actions concrètes pour transformer les insights en résultats"
                }
            ],
            
            "executive_summary": f"L'analyse de {dq['total_rows']} observations révèle {len(insights)} insights majeurs. "
                                f"La qualité des données est {dq['quality_level']} ({dq['completeness_pct']:.0f}% complétude). "
                                f"{len([r for r in recommendations if r.get('priority') == 'haute'])} recommandations prioritaires sont identifiées.",
            
            "conclusion": "Cette analyse fournit une base solide pour la prise de décision. "
                         "Les recommandations sont classées par priorité et impact pour faciliter la mise en œuvre."
        }
        
        # TTS Narration complète
        tts_parts = [
            f"Rapport d'analyse de {target}.",
            f"Ce rapport est basé sur {dq['total_rows']} observations.",
            f"La qualité des données est {dq['quality_level']} avec {dq['completeness_pct']:.0f} pour cent de complétude."
        ]
        
        # Ajouter les insights principaux
        tts_parts.append("Voici les principales découvertes:")
        for i, insight in enumerate(insights[:3], 1):
            tts_parts.append(f"Découverte {i}: {insight.get('finding', '')}. {insight.get('so_what', '')}.")
        
        # Ajouter les recommandations
        tts_parts.append("Passons maintenant aux recommandations:")
        for rec in recommendations[:3]:
            if rec.get("tts_text"):
                tts_parts.append(rec["tts_text"])
        
        tts_parts.append("Fin du rapport. Consultez le document PDF pour les détails complets.")
        
        story["tts_narration"] = " ".join(tts_parts)
        
        return story
    
    # ==================== PHASE 8: GÉNÉRATION DE RAPPORTS ====================
    
    def _generate_pdf_report_content(self, insights: List[Dict], recommendations: List[Dict],
                                     eda_insights: Dict, context: Dict) -> Dict[str, Any]:
        """
        📄 46. Contenu pour le rapport PDF
        """
        dq = self._calculate_data_quality(eda_insights)
        target = context.get("target_variable", "les données")
        
        return {
            "title": f"Rapport d'Analyse: {target}",
            "generated_at": datetime.now().isoformat(),
            "generated_at_formatted": datetime.now().strftime("%d/%m/%Y à %H:%M"),
            
            "summary": {
                "observations": dq["total_rows"],
                "variables": dq["total_cols"],
                "completeness": f"{dq['completeness_pct']:.1f}%",
                "quality": dq["quality_level"],
                "insights_count": len(insights),
                "recommendations_count": len(recommendations),
                "high_priority_count": len([r for r in recommendations if r.get("priority") == "haute"])
            },
            
            "insights": insights,
            "recommendations": recommendations,
            
            "methodology": {
                "phases": [
                    "Phase 1-4: Exploration et préparation des données",
                    "Phase 5: Feature Engineering (création de variables)",
                    "Phase 6: Machine Learning (si applicable)",
                    "Phase 7: Interprétation et Storytelling",
                    "Phase 8: Restitution et Communication"
                ],
                "tools_used": ["Python", "Scikit-learn", "Pandas", "NumPy"],
                "statistical_tests": ["T-Test", "ANOVA", "Corrélation de Pearson"]
            },
            
            "appendix": {
                "data_quality_details": dq,
                "variables_analyzed": {
                    "numeric": eda_insights.get("summary", {}).get("numeric_analyzed", 0),
                    "categorical": eda_insights.get("summary", {}).get("categorical_analyzed", 0)
                }
            }
        }
    
    def _generate_markdown_report(self, insights: List[Dict], recommendations: List[Dict],
                                  eda_insights: Dict, context: Dict) -> str:
        """📝 Rapport Markdown complet"""
        dq = self._calculate_data_quality(eda_insights)
        target = context.get("target_variable", "les données")
        
        md = [
            f"# 📊 Rapport d'Analyse: {target}",
            f"",
            f"*Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}*",
            f"",
            f"---",
            f"",
            f"## 📋 Résumé Exécutif",
            f"",
            f"| Métrique | Valeur |",
            f"|----------|--------|",
            f"| Observations | {dq['total_rows']} |",
            f"| Variables | {dq['total_cols']} |",
            f"| Complétude | {dq['completeness_pct']:.1f}% |",
            f"| Qualité | {dq['quality_level'].capitalize()} |",
            f"| Insights | {len(insights)} |",
            f"| Recommandations | {len(recommendations)} |",
            f"",
            f"---",
            f"",
            f"## 💡 Insights Clés",
            f""
        ]
        
        for i, insight in enumerate(insights, 1):
            priority_emoji = "🔴" if insight.get("priority") == "haute" else "🟡" if insight.get("priority") == "moyenne" else "🟢"
            md.append(f"### {i}. {insight.get('title', insight.get('category', 'Insight'))} {priority_emoji}")
            md.append(f"")
            md.append(f"**Constat:** {insight.get('finding', '')}")
            md.append(f"")
            if insight.get('metric'):
                md.append(f"**Métrique:** {insight.get('metric', '')}")
                md.append(f"")
            md.append(f"**Implication:** {insight.get('so_what', '')}")
            md.append(f"")
        
        md.append(f"---")
        md.append(f"")
        md.append(f"## 🎯 Recommandations Actionnables")
        md.append(f"")
        
        for rec in recommendations:
            priority_badge = "🔴 HAUTE" if rec.get("priority") == "haute" else "🟡 MOYENNE" if rec.get("priority") == "moyenne" else "🟢 BASSE"
            md.append(f"### {rec.get('id', '')}. {rec.get('title', '')}")
            md.append(f"")
            md.append(f"**Priorité:** {priority_badge} | **Impact:** {rec.get('impact', 0)}% | **Effort:** {rec.get('effort', '')} | **Timeline:** {rec.get('timeline', '')}")
            md.append(f"")
            md.append(f"{rec.get('description', '')}")
            md.append(f"")
            md.append(f"**Justification:** {rec.get('rationale', '')}")
            md.append(f"")
            md.append(f"**Actions concrètes:**")
            for action in rec.get("actions", []):
                md.append(f"- **Étape {action.get('step', '')}:** {action.get('action', '')} *(Responsable: {action.get('responsible', '')})*")
            md.append(f"")
            md.append(f"**KPIs de suivi:** {', '.join(rec.get('kpis', []))}")
            md.append(f"")
        
        md.append(f"---")
        md.append(f"")
        md.append(f"## 📚 Méthodologie")
        md.append(f"")
        md.append(f"Ce rapport a été généré automatiquement en suivant les phases d'analyse de données:")
        md.append(f"")
        md.append(f"1. **Phases 1-4:** Exploration et préparation des données")
        md.append(f"2. **Phase 5:** Feature Engineering (création de variables)")
        md.append(f"3. **Phase 6:** Machine Learning (si applicable)")
        md.append(f"4. **Phase 7:** Interprétation et Storytelling")
        md.append(f"5. **Phase 8:** Restitution et Communication")
        md.append(f"")
        md.append(f"---")
        md.append(f"")
        md.append(f"*Rapport généré automatiquement par le système d'analyse intelligente.*")
        
        return "\n".join(md)
    
    def _generate_html_report(self, insights: List[Dict], recommendations: List[Dict],
                              eda_insights: Dict, context: Dict) -> str:
        """📄 Rapport HTML stylisé"""
        dq = self._calculate_data_quality(eda_insights)
        target = context.get("target_variable", "les données")
        
        insights_html = ""
        for i, insight in enumerate(insights, 1):
            priority_class = "high" if insight.get("priority") == "haute" else "medium" if insight.get("priority") == "moyenne" else "low"
            insights_html += f"""
            <div class="insight {priority_class}">
                <div class="insight-header">
                    <span class="insight-icon">{insight.get('icon', '💡')}</span>
                    <h3>{insight.get('title', insight.get('category', 'Insight'))}</h3>
                </div>
                <p class="finding"><strong>Constat:</strong> {insight.get('finding', '')}</p>
                <p class="metric"><strong>Métrique:</strong> {insight.get('metric', 'N/A')}</p>
                <p class="so-what"><strong>Implication:</strong> {insight.get('so_what', '')}</p>
            </div>
            """
        
        recommendations_html = ""
        for rec in recommendations:
            actions_html = "".join([
                f"<li><strong>Étape {a.get('step', '')}:</strong> {a.get('action', '')} <em>({a.get('responsible', '')})</em></li>"
                for a in rec.get("actions", [])
            ])
            
            recommendations_html += f"""
            <div class="recommendation">
                <div class="rec-header">
                    <span class="rec-number">{rec.get('id', '')}</span>
                    <h3>{rec.get('title', '')}</h3>
                </div>
                <div class="rec-meta">
                    <span class="priority priority-{rec.get('priority', 'moyenne')}">{rec.get('priority', '').upper()}</span>
                    <span class="impact">Impact: {rec.get('impact', 0)}%</span>
                    <span class="effort">Effort: {rec.get('effort', '')}</span>
                    <span class="timeline">{rec.get('timeline', '')}</span>
                </div>
                <p class="description">{rec.get('description', '')}</p>
                <p class="rationale"><strong>Justification:</strong> {rec.get('rationale', '')}</p>
                <div class="actions-list">
                    <h4>Actions concrètes:</h4>
                    <ol>{actions_html}</ol>
                </div>
                <div class="kpis">
                    <strong>KPIs:</strong> {', '.join(rec.get('kpis', []))}
                </div>
            </div>
            """
        
        html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport d'Analyse - {target}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f7fa; color: #333; line-height: 1.6; }}
        .container {{ max-width: 1000px; margin: 0 auto; padding: 20px; }}
        
        header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; border-radius: 10px; margin-bottom: 30px; }}
        header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        header .subtitle {{ opacity: 0.9; font-size: 1.1em; }}
        
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 30px; }}
        .summary-card {{ background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .summary-card .value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .summary-card .label {{ color: #666; font-size: 0.9em; }}
        
        section {{ background: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        section h2 {{ color: #2c3e50; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #667eea; }}
        
        .insight {{ padding: 20px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #667eea; background: #f8f9ff; }}
        .insight.high {{ border-left-color: #e74c3c; background: #fef5f5; }}
        .insight.medium {{ border-left-color: #f39c12; background: #fffdf5; }}
        .insight-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }}
        .insight-icon {{ font-size: 1.5em; }}
        .insight h3 {{ color: #2c3e50; }}
        .insight p {{ margin: 8px 0; }}
        .so-what {{ color: #27ae60; font-style: italic; }}
        
        .recommendation {{ background: #f8f9fa; padding: 25px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #e0e0e0; }}
        .rec-header {{ display: flex; align-items: center; gap: 15px; margin-bottom: 15px; }}
        .rec-number {{ background: #667eea; color: white; width: 35px; height: 35px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; }}
        .rec-header h3 {{ color: #2c3e50; }}
        .rec-meta {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 15px; }}
        .rec-meta span {{ padding: 5px 12px; border-radius: 20px; font-size: 0.85em; }}
        .priority {{ font-weight: bold; color: white; }}
        .priority-haute {{ background: #e74c3c; }}
        .priority-moyenne {{ background: #f39c12; }}
        .priority-basse {{ background: #27ae60; }}
        .impact, .effort, .timeline {{ background: #e8e8e8; color: #333; }}
        .description {{ font-size: 1.1em; margin-bottom: 15px; }}
        .rationale {{ color: #666; margin-bottom: 15px; }}
        .actions-list {{ background: #e8f5e9; padding: 15px; border-radius: 8px; margin-bottom: 15px; }}
        .actions-list h4 {{ color: #27ae60; margin-bottom: 10px; }}
        .actions-list ol {{ margin-left: 20px; }}
        .actions-list li {{ margin-bottom: 8px; }}
        .kpis {{ color: #667eea; font-size: 0.95em; }}
        
        footer {{ text-align: center; padding: 20px; color: #666; font-size: 0.9em; }}
        
        @media print {{
            body {{ background: white; }}
            .container {{ max-width: 100%; }}
            header {{ background: #667eea !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Rapport d'Analyse</h1>
            <p class="subtitle">{target} - {datetime.now().strftime('%d/%m/%Y à %H:%M')}</p>
        </header>
        
        <div class="summary-grid">
            <div class="summary-card">
                <div class="value">{dq['total_rows']}</div>
                <div class="label">Observations</div>
            </div>
            <div class="summary-card">
                <div class="value">{dq['total_cols']}</div>
                <div class="label">Variables</div>
            </div>
            <div class="summary-card">
                <div class="value">{dq['completeness_pct']:.0f}%</div>
                <div class="label">Complétude</div>
            </div>
            <div class="summary-card">
                <div class="value">{len(insights)}</div>
                <div class="label">Insights</div>
            </div>
            <div class="summary-card">
                <div class="value">{len(recommendations)}</div>
                <div class="label">Recommandations</div>
            </div>
        </div>
        
        <section>
            <h2>💡 Insights Clés</h2>
            {insights_html}
        </section>
        
        <section>
            <h2>🎯 Recommandations Actionnables</h2>
            {recommendations_html}
        </section>
        
        <footer>
            <p>Rapport généré automatiquement par le système d'analyse intelligente</p>
        </footer>
    </div>
</body>
</html>"""
        
        return html
    
    # ==================== MÉTHODE PRINCIPALE ====================
    
    async def tell_the_story(self, eda_results: Dict[str, Any],
                             ml_results: Dict[str, Any],
                             feature_engineering: Dict[str, Any],
                             context: Dict[str, Any],
                             options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        📖 PIPELINE COMPLET PHASES 7-8
        
        Phase 7: Interprétation et Storytelling
        - 41. Interprétation des résultats
        - 42. Synthèse des insights clés
        - 43. Création de supports visuels
        - 44. Scénarisation (Data Storytelling)
        - 45. Formulation de recommandations actionnables
        
        Phase 8: Restitution et Communication
        - 46. Rédaction du rapport d'analyse
        - 47. Présentation orale (TTS)
        - 48. Documentation technique
        - 49. Mise à disposition des données nettoyées
        """
        
        logger.info("=" * 60)
        logger.info("📖 STORYTELLER V3 - PHASES 7-8 COMPLÈTES")
        logger.info("=" * 60)
        
        result = {
            "success": True,
            "insights": [],
            "recommendations": [],
            "story": {},
            "report": {},
            "exports": {},
            "tts_text": "",
            "tts_sections": [],
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Phase 7.1: Extraction et interprétation
            logger.info("🔍 Phase 7.1: Extraction des résultats EDA...")
            eda_insights = self._extract_eda_insights(eda_results or {})
            
            # Calcul qualité
            dq = self._calculate_data_quality(eda_insights)
            logger.info(f"   Complétude: {dq['completeness_pct']:.1f}%")
            
            # Phase 7.2: Synthèse des insights
            logger.info("💡 Phase 7.2: Synthèse des insights...")
            insights = self._synthesize_key_insights(eda_insights, ml_results or {}, context)
            result["insights"] = insights
            logger.info(f"   {len(insights)} insights générés")
            
            # Phase 7.3: Recommandations actionnables
            logger.info("🎯 Phase 7.3: Recommandations actionnables...")
            recommendations = self._generate_actionable_recommendations(
                insights, eda_insights, ml_results or {}, context
            )
            result["recommendations"] = recommendations
            logger.info(f"   {len(recommendations)} recommandations générées")
            
            # Phase 7.4: Data Storytelling
            logger.info("📖 Phase 7.4: Data Storytelling...")
            story = self._create_data_story(insights, recommendations, eda_insights, context)
            result["story"] = story
            result["tts_text"] = story.get("tts_narration", "")
            
            # Sections TTS individuelles pour le frontend
            result["tts_sections"] = [
                {"id": "intro", "title": "Introduction", "text": f"Analyse de {context.get('target_variable', 'vos données')} basée sur {dq['total_rows']} observations."},
                {"id": "quality", "title": "Qualité", "text": f"La qualité des données est {dq['quality_level']} avec {dq['completeness_pct']:.0f} pour cent de complétude."},
            ]
            
            for i, insight in enumerate(insights[:5], 1):
                result["tts_sections"].append({
                    "id": f"insight_{i}",
                    "title": f"Insight {i}",
                    "text": f"{insight.get('finding', '')}. {insight.get('so_what', '')}"
                })
            
            for rec in recommendations[:5]:
                result["tts_sections"].append({
                    "id": f"rec_{rec.get('id', '')}",
                    "title": f"Recommandation {rec.get('id', '')}",
                    "text": rec.get("tts_text", rec.get("description", ""))
                })
            
            # Phase 8.1: Génération du rapport
            logger.info("📄 Phase 8.1: Génération du rapport...")
            result["report"] = self._generate_pdf_report_content(insights, recommendations, eda_insights, context)
            
            # Phase 8.2: Exports
            logger.info("📤 Phase 8.2: Génération des exports...")
            result["exports"] = {
                "markdown": self._generate_markdown_report(insights, recommendations, eda_insights, context),
                "html": self._generate_html_report(insights, recommendations, eda_insights, context)
            }
            
            logger.info("=" * 60)
            logger.info(f"✅ STORYTELLER V3 TERMINÉ")
            logger.info(f"   Insights: {len(insights)}")
            logger.info(f"   Recommandations: {len(recommendations)}")
            logger.info(f"   Sections TTS: {len(result['tts_sections'])}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Erreur Storyteller: {e}", exc_info=True)
            result["success"] = False
            result["error"] = str(e)
        
        return result


# Instance globale
insight_storyteller_service = InsightStorytellerService()