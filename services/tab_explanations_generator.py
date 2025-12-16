"""
📝 GÉNÉRATEUR D'EXPLICATIONS PAR ONGLET - MULTI-LLM + ROBUSTE
Gère Claude, Gemini, GPT-4 avec fallback intelligent
Extraction JSON agressive pour éviter les parse errors
✅ FIX: Gestion de multi_clustering None
"""

import json
import logging
import os
import asyncio
from typing import Dict, Any, List, Optional
from anthropic import Anthropic

logger = logging.getLogger(__name__)


def _extract_json_robust(text: str) -> Optional[Dict]:
    """Extraction JSON AGRESSIVE - priorise le résultat valide"""
    
    if not text:
        return None
    
    # 1️⃣ NETTOYAGE INITIAL
    text = text.replace('```json', '').replace('```', '').strip()
    
    # 2️⃣ ESSAI DIRECT
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 3️⃣ CHERCHER LE JSON VALIDE LE PLUS LONG
    logger.warning(f"⚠️ JSON parse error, searching for valid JSON...")
    
    # Chercher tous les '{' et '}'
    start_indices = [i for i, c in enumerate(text) if c == '{']
    
    if not start_indices:
        logger.error("❌ Aucun '{' trouvé, retour fallback")
        return None
    
    # Pour chaque position de départ, chercher le JSON valide
    best_json = None
    best_length = 0
    
    for start in start_indices:
        # Chercher le '}' correspondant
        depth = 0
        for end in range(start, len(text)):
            if text[end] == '{':
                depth += 1
            elif text[end] == '}':
                depth -= 1
                if depth == 0:
                    # Trouvé un candidate JSON
                    candidate = text[start:end+1]
                    try:
                        parsed = json.loads(candidate)
                        if len(candidate) > best_length:
                            best_json = parsed
                            best_length = len(candidate)
                            logger.info(f"✅ JSON valide trouvé: {len(candidate)} chars")
                        break
                    except:
                        pass
    
    return best_json


class TabExplanationsGenerator:
    """Génère les explications par onglet avec Multi-LLM - AMÉLIORÉ POUR CLUSTERING"""

    @staticmethod
    def create_tab_explanation_tasks(eda_data: Dict[str, Any], 
                                     context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Crée les tâches avec CONTEXTE complet - INCLUT CLUSTERING DÉTAILLÉ"""
        
        tasks = []
        univariate = eda_data.get('univariate', {})
        multi_clustering = eda_data.get('multi_clustering') or {}  # ✅ FIX: Gérer None
        tests = eda_data.get('tests', [])
        correlations = eda_data.get('correlations', {})
        
        target = context.get('target_variable', 'Unknown')
        analysis_type = context.get('analysis_type', 'descriptive')
        focus_vars = context.get('focus_variables', [])
        
        n_num = len([s for s in univariate.values() if s.get('type') == 'numeric'])
        n_cat = len([s for s in univariate.values() if s.get('type') == 'categorical'])
        
        # ===== TASK 1: OVERVIEW =====
        overview_data = {
            "total_vars": len(univariate),
            "numeric": n_num,
            "categorical": n_cat,
            "target": target,
            "analysis_type": analysis_type,
            "rows": eda_data.get('rows', 0),
            "columns": eda_data.get('cols', 0),
            "focus_variables": focus_vars[:5]
        }
        
        tasks.append({
            "task_id": "overview",
            "prompt": f"""Analyse ce dataset et réponds UNIQUEMENT avec du JSON valide.

CONTEXTE: {analysis_type}
CIBLE: {target}
VARIABLES FOCUS: {', '.join(focus_vars[:3]) if focus_vars else 'Aucune'}

DONNÉES:
- {eda_data.get('rows', 0)} lignes, {eda_data.get('cols', 0)} colonnes
- {n_num} variables numériques, {n_cat} variables catégorielles
- Cible: {target}

Réponds AVEC CE FORMAT EXACT (rien d'autre):
{{
  "title": "Synthèse du Dataset",
  "summary": "2-3 phrases décrivant le dataset et son objectif",
  "recommendation": "Actions concrètes à prendre",
  "details": {{
    "points_cles": ["point1", "point2"],
    "complexite": "simple/moyenne/complexe"
  }}
}}""",
            "data": json.dumps(overview_data)
        })
        
        # ===== TASK 2: STATISTICS =====
        numeric_stats = [{"var": var, "cv": s.get('cv', 0), "skew": s.get('skew', 0)} 
                        for var, s in univariate.items() 
                        if s.get('type') == 'numeric'][:5]
        
        stats_summary = {
            "numeric": n_num,
            "categorical": n_cat,
            "high_variance": len([s for s in numeric_stats if s.get('cv', 0) > 1.0]),
            "high_skew": sum(1 for s in numeric_stats if abs(s.get('skew', 0)) > 1.0),
            "target": target,
            "missing_data": sum(1 for s in univariate.values() if s.get('missing_pct', 0) > 20)
        }
        
        tasks.append({
            "task_id": "stats",
            "prompt": f"""Analyse ces statistiques descriptives et réponds UNIQUEMENT avec du JSON valide:

DONNÉES: {json.dumps(stats_summary)}

POINTS IMPORTANTS:
- {stats_summary['high_variance']} variables à haute variance
- {stats_summary['high_skew']} variables très asymétriques
- {stats_summary['missing_data']} variables avec >20% de données manquantes

Réponds AVEC CE FORMAT:
{{
  "title": "Statistiques Descriptives",
  "summary": "Analyse de la distribution des variables",
  "recommendation": "Conseils pour le nettoyage et la préparation",
  "details": {{
    "qualite_donnees": "bonne/moyenne/faible",
    "alertes": ["alerte1", "alerte2"]
  }}
}}""",
            "data": json.dumps(stats_summary)
        })
        
        # ===== TASK 3: CHARTS =====
        tasks.append({
            "task_id": "charts",
            "prompt": f"""Recommande des visualisations pour {analysis_type} avec cible {target}. Réponds UNIQUEMENT avec du JSON valide:

VARIABLES DISPONIBLES:
- Numériques: {n_num} variables
- Catégorielles: {n_cat} variables

Réponds AVEC CE FORMAT:
{{
  "title": "Visualisations Recommandées",
  "summary": "Types de graphiques les plus pertinents",
  "recommendation": "Ordre de priorité pour les visualisations",
  "details": {{
    "graphiques_prioritaires": ["graph1", "graph2"],
    "variables_a_visualiser": ["var1", "var2"]
  }}
}}""",
            "data": ""
        })
        
        # ===== TASK 4: TESTS =====
        tests_count = len(tests)
        significant = len([t for t in tests if t.get('p_value', 1) < 0.05])
        
        tasks.append({
            "task_id": "tests",
            "prompt": f"""Analyse ces tests statistiques ({tests_count} tests, {significant} significatifs). Réponds UNIQUEMENT avec du JSON valide:

CONTEXTE: Analyse de {target}

RÉSULTATS:
- Tests totaux: {tests_count}
- Tests significatifs (p<0.05): {significant}
- Taux de significativité: {significant/tests_count*100 if tests_count>0 else 0:.1f}%

Réponds AVEC CE FORMAT:
{{
  "title": "Tests Statistiques",
  "summary": "Évaluation de la significativité des relations",
  "recommendation": "Interprétation des résultats significatifs",
  "details": {{
    "confiance": "élevée/moyenne/faible",
    "relations_importantes": ["relation1", "relation2"]
  }}
}}""",
            "data": json.dumps({"total": tests_count, "significant": significant})
        })
        
        # ===== TASK 5: CLUSTERING - AMÉLIORÉ + FIX None =====
        # ✅ FIX: Vérifier si multi_clustering est None ou vide
        clusterings = multi_clustering.get('clusterings', {}) if multi_clustering else {}
        n_clusterings = len(clusterings)
        
        # Collecter des informations détaillées sur le clustering
        clustering_info = {
            "total_models": n_clusterings,
            "models": [],
            "best_model": None,
            "best_score": 0
        }
        
        if clusterings:  # ✅ FIX: Vérifier si clusterings n'est pas vide
            for key, clust in clusterings.items():
                if clust is None:  # ✅ FIX: Skip si cluster None
                    continue
                    
                model_info = {
                    "name": clust.get('name', key),
                    "clusters": clust.get('n_clusters', 0),
                    "silhouette": clust.get('silhouette_score', 0),
                    "validation": clust.get('validation', {})
                }
                clustering_info["models"].append(model_info)
                
                score = clust.get('silhouette_score', 0) or 0
                if score > clustering_info["best_score"]:
                    clustering_info["best_score"] = score
                    clustering_info["best_model"] = model_info
        
        tasks.append({
            "task_id": "clustering",
            "prompt": f"""Analyse ces résultats de clustering et réponds UNIQUEMENT avec du JSON valide:

DONNÉES DE CLUSTERING DÉTAILLÉES:
{json.dumps(clustering_info, indent=2)}

POINTS CLÉS:
- {n_clusterings} modèles de clustering générés
- Meilleur modèle: {clustering_info['best_model']['name'] if clustering_info['best_model'] else 'Aucun'}
- Score silhouette du meilleur modèle: {clustering_info['best_score']:.3f}
- Groupes identifiés: {', '.join([str(m['clusters']) for m in clustering_info['models']]) if clustering_info['models'] else 'Aucun'}

Réponds AVEC CE FORMAT:
{{
  "title": "Segmentation et Groupes",
  "summary": "Analyse détaillée des clusters identifiés ou raison de l'absence de clustering",
  "recommendation": "Utilisation pratique des segments ou suggestions alternatives",
  "details": {{
    "qualite_clustering": "excellente/bonne/moyenne/faible/non_applicable",
    "nombre_groupes_optimal": 3,
    "caracteristiques_cles": ["carac1", "carac2"],
    "applications": ["application1", "application2"]
  }},
  "tts_text": "Texte complet à lire pour la synthèse vocale (200-300 mots)"
}}""",
            "data": json.dumps(clustering_info)
        })
        
        # ===== TASK 6: CORRELATIONS =====
        strong_corr = len(correlations.get('strong_correlations', []))
        moderate_corr = len(correlations.get('moderate_correlations', []))
        target_corr = len(correlations.get('target_correlations', {}))
        
        tasks.append({
            "task_id": "correlation",
            "prompt": f"""Analyse les corrélations ({strong_corr} fortes, {moderate_corr} modérées, {target_corr} avec la cible). Réponds UNIQUEMENT avec du JSON valide:

ANALYSE:
- Corrélations fortes (>0.7): {strong_corr}
- Corrélations modérées (0.4-0.7): {moderate_corr}
- Corrélations avec {target}: {target_corr}

IMPLICATIONS:
- Fortes corrélations peuvent indiquer de la multicolinéarité
- Corrélations avec la cible sont importantes pour la prédiction

Réponds AVEC CE FORMAT:
{{
  "title": "Matrice de Corrélation",
  "summary": "Analyse des relations linéaires entre variables",
  "recommendation": "Actions basées sur les corrélations",
  "details": {{
    "intensite_relations": "forte/moyenne/faible",
    "multicolinearite": "présente/absente",
    "variables_liees_cible": ["var1", "var2"]
  }}
}}""",
            "data": json.dumps({"strong": strong_corr, "moderate": moderate_corr, "target": target_corr})
        })
        
        logger.info(f"📝 {len(tasks)} tâches créées avec clustering détaillé")
        return tasks

    @staticmethod
    def create_summary_eda_data(eda_results: Dict[str, Any]) -> Dict[str, Any]:
        """Crée le dictionnaire EDA pour explications"""
        metrics = eda_results.get('metrics', {})
        univariate = metrics.get('univariate', {})
        
        n_num = len([s for s in univariate.values() if s.get('type') == 'numeric'])
        n_cat = len([s for s in univariate.values() if s.get('type') == 'categorical'])
        
        return {
            "rows": eda_results.get('summary', {}).get('total_rows', 0),
            "cols": eda_results.get('summary', {}).get('total_cols', 0),
            "univariate": univariate,
            "multi_clustering": metrics.get('multi_clustering'),  # ✅ Peut être None
            "tests": metrics.get('tests', []),
            "correlations": metrics.get('correlations', {}),
            "missing_pct": (eda_results.get('summary', {}).get('missing_values', 0) / 
                          max(1, eda_results.get('summary', {}).get('total_rows', 1))) * 100,
            "clustering_explanations": eda_results.get('charts_data', {}).get('clustering_explanations', {})
        }


async def _call_anthropic(prompt: str) -> Optional[Dict]:
    """Appel Claude avec extraction JSON robuste"""
    try:
        from config.settings import settings
        client = Anthropic(api_key=settings.anthropic_api_key_1)
        
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text
        result = _extract_json_robust(response_text)
        
        if result:
            logger.info(f"✅ Claude OK: {result.get('title', 'N/A')}")
            return result
        else:
            logger.warning(f"⚠️ Claude JSON invalid")
            return None
            
    except Exception as e:
        logger.warning(f"❌ Claude failed: {e}")
        return None


async def _call_gemini(prompt: str) -> Optional[Dict]:
    """Appel Gemini via API"""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None
        
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        
        result = _extract_json_robust(response.text)
        
        if result:
            logger.info(f"✅ Gemini OK: {result.get('title', 'N/A')}")
            return result
        else:
            logger.warning(f"⚠️ Gemini JSON invalid")
            return None
            
    except Exception as e:
        logger.warning(f"❌ Gemini failed: {e}")
        return None


async def _call_gpt4(prompt: str) -> Optional[Dict]:
    """Appel GPT-4 via OpenAI"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800
        )
        
        response_text = response.choices[0].message.content
        result = _extract_json_robust(response_text)
        
        if result:
            logger.info(f"✅ GPT-4 OK: {result.get('title', 'N/A')}")
            return result
        else:
            logger.warning(f"⚠️ GPT-4 JSON invalid")
            return None
            
    except Exception as e:
        logger.warning(f"❌ GPT-4 failed: {e}")
        return None


async def _generate_fallback_explanation(task_id: str, context: Dict, eda_data: Dict) -> Dict:
    """Génère une explication par défaut si tous les LLM échouent"""
    fallbacks = {
        "overview": {
            "title": "Synthèse du Dataset",
            "summary": f"Dataset de {eda_data.get('rows', 0)} observations avec {eda_data.get('cols', 0)} variables. Analyse {context.get('analysis_type', 'exploratoire')} focalisée sur {context.get('target_variable', 'la variable cible')}.",
            "recommendation": "Examinez les statistiques descriptives pour comprendre la distribution des données.",
            "tts_text": f"Le dataset contient {eda_data.get('rows', 0)} observations et {eda_data.get('cols', 0)} variables. L'analyse est de type {context.get('analysis_type', 'exploratoire')} avec pour cible {context.get('target_variable', 'la variable principale')}."
        },
        "stats": {
            "title": "Statistiques Descriptives",
            "summary": f"Analyse univariée des variables. {len([v for v in eda_data.get('univariate', {}).values() if v.get('type') == 'numeric'])} variables numériques et {len([v for v in eda_data.get('univariate', {}).values() if v.get('type') == 'categorical'])} catégorielles analysées.",
            "recommendation": "Vérifiez les valeurs manquantes et les distributions avant toute modélisation.",
            "tts_text": "Les statistiques descriptives montrent la distribution de chaque variable. Examinez les moyennes, écarts-types et pourcentages de données manquantes."
        },
        "charts": {
            "title": "Visualisations Recommandées",
            "summary": "Plusieurs types de visualisations sont pertinents pour explorer ces données.",
            "recommendation": "Commencez par des histogrammes pour les variables numériques et des diagrammes en secteurs pour les catégorielles.",
            "tts_text": "Pour visualiser ces données, je recommande des histogrammes pour les variables continues, des diagrammes en secteurs pour les catégories, et des nuages de points pour les relations entre variables."
        },
        "tests": {
            "title": "Tests Statistiques",
            "summary": f"{len(eda_data.get('tests', []))} tests effectués pour valider les relations entre variables.",
            "recommendation": "Concentrez-vous sur les tests avec p-value < 0.05 qui indiquent des relations significatives.",
            "tts_text": f"{len(eda_data.get('tests', []))} tests statistiques ont été réalisés. Les résultats avec p-value inférieure à 0.05 sont statistiquement significatifs."
        },
        "clustering": {
            "title": "Segmentation Intelligente",
            "summary": f"Analyse de clustering pour identifier des groupes naturels dans les données.",
            "recommendation": "Les données n'ont pas formé de clusters distincts. Essayez avec d'autres variables ou consultez les visualisations.",
            "tts_text": "L'analyse de clustering n'a pas identifié de groupes très distincts. Les données sont trop dispersées ou n'ont pas de structure de clustering claire. Essayez d'explorer d'autres variables ou d'utiliser des approches alternatives.",
            "details": {
                "qualite_clustering": "faible",
                "nombre_groupes_optimal": "Non applicable",
                "caracteristiques_cles": ["Données dispersées"],
                "applications": ["Essayer d'autres variables", "Considérer d'autres méthodes"]
            }
        },
        "correlation": {
            "title": "Matrice de Corrélation",
            "summary": f"Analyse des relations linéaires entre variables. {len(eda_data.get('correlations', {}).get('strong_correlations', []))} corrélations fortes identifiées.",
            "recommendation": "Vérifiez les corrélations avec la variable cible pour identifier les prédicteurs potentiels.",
            "tts_text": "La matrice de corrélation montre les relations linéaires entre variables. Les corrélations fortes (proches de 1 ou -1) indiquent des relations importantes."
        }
    }
    
    explanation = fallbacks.get(task_id, {
        "title": task_id.title(),
        "summary": f"Analyse pour {context.get('target_variable', 'cible')}",
        "recommendation": "Consultez les données détaillées ci-dessous",
        "tts_text": f"Voici l'analyse pour l'onglet {task_id}."
    })
    
    return explanation


async def generate_tab_explanations_async(eda_data: Dict[str, Any],
                                          context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Génère les explications avec MULTI-LLM:
    1. Claude (Anthropic)
    2. Gemini (Google)
    3. GPT-4 (OpenAI)
    4. Fallback par défaut
    
    ✅ FIX: Gère le cas où multi_clustering est None
    """
    
    tasks = TabExplanationsGenerator.create_tab_explanation_tasks(eda_data, context)
    
    logger.info(f"🤖 Génération {len(tasks)} explications avec Multi-LLM")
    
    tab_explanations = {}
    
    for task in tasks:
        logger.info(f"📝 Génération: {task['task_id']}")
        
        # Essayer les 3 LLM en parallèle
        results = await asyncio.gather(
            _call_anthropic(task["prompt"]),
            _call_gemini(task["prompt"]),
            _call_gpt4(task["prompt"]),
            return_exceptions=True
        )
        
        # Prendre le premier résultat valide
        explanation = None
        for i, (result, llm_name) in enumerate(zip(results, ["Claude", "Gemini", "GPT-4"])):
            if isinstance(result, dict) and result.get("title"):
                logger.info(f"✅ {llm_name} utilisé pour {task['task_id']}")
                explanation = result
                break
        
        # Fallback si tous échouent
        if not explanation:
            logger.warning(f"⚠️ Tous les LLM ont échoué pour {task['task_id']}, fallback")
            explanation = await _generate_fallback_explanation(task['task_id'], context, eda_data)
        
        tab_explanations[task["task_id"]] = explanation
    
    logger.info(f"✅ {len(tab_explanations)} explications générées")
    return tab_explanations