import pandas as pd
import numpy as np
import json
import re
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Tuple, Optional
from models.analysis import Insight
import logging

logger = logging.getLogger(__name__)

class EDAService:
    """
    Service EDA amélioré avec:
    - Détection robuste de la cible
    - Graphiques enrichis et variés
    - Gestion d'erreurs LLM améliorée
    - Stats complètes et visibles
    """

    # =========================================================
    # 1. DÉTECTION INTELLIGENTE DE LA CIBLE
    # =========================================================

    def _smart_target_detection(self, df: pd.DataFrame, user_prompt: str, context: dict) -> str:
        """Détecte la variable cible de manière intelligente."""
        
        # A. Si la cible est explicite dans le contexte
        target = context.get('target_variable') or context.get('focus_variable')
        if target and target in df.columns:
            logger.info(f"✓ Cible explicite trouvée: {target}")
            return target

        # B. Recherche de mots-clés dans le prompt (fr/en/mg)
        keywords_target = {
            'objectif': ['target', 'cible', 'predict', 'prédire', 'outcome', 'résultat', 'but', 'depend'],
            'monétaire': ['prix', 'prix', 'revenu', 'salaire', 'coût', 'bola', 'vidin'],
            'temporel': ['date', 'durée', 'mois', 'année', 'temps', 'fotoana'],
            'binaire': ['oui/non', 'yes/no', 'succès', 'défaut', 'présence', 'absence']
        }

        for col in df.columns:
            col_lower = col.lower()
            # Recherche de patterns directs
            if any(k in col_lower for k in ['target', 'cible', 'objectif', 'outcome', 'y_', '_y']):
                logger.info(f"✓ Cible détectée par pattern: {col}")
                return col
            
            # Recherche dans le prompt
            if col_lower in user_prompt.lower():
                logger.info(f"✓ Cible mentionnée dans le prompt: {col}")
                return col

        # C. Heuristiques automatiques pour la meilleure cible
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) == 0:
            logger.warning("⚠ Aucune colonne numérique - sélection par défaut")
            return df.columns[0] if len(df.columns) > 0 else "Non définie"

        # Évaluation des colonnes numériques (variance, ratio d'asymétrie, plage)
        candidate_scores = {}
        for col in numeric_cols:
            if df[col].nunique() < 2:
                continue

            clean_data = df[col].dropna()
            if len(clean_data) < 10:
                continue

            # Score de variabilité (variance normalisée)
            var_score = clean_data.std() / (clean_data.mean().abs() + 1e-6)
            
            # Score d'asimétrie (pas trop symétrique, pas trop extrême)
            skew = clean_data.skew()
            skew_score = 1.0 - min(abs(skew) / 3, 1.0)
            
            # Score de complétude
            completeness = clean_data.notna().sum() / len(df[col])
            
            # Colonne avec plus de variance = meilleure cible
            total_score = (
                var_score * 0.4 +      # Variabilité (important)
                skew_score * 0.3 +     # Distribution (modérée)
                completeness * 0.3     # Pas de NaN (important)
            )
            
            candidate_scores[col] = total_score

        if candidate_scores:
            target = max(candidate_scores, key=candidate_scores.get)
            logger.info(f"✓ Cible auto-sélectionnée: {target} (score: {candidate_scores[target]:.2f})")
            return target

        return "Non définie"

    # =========================================================
    # 2. EXTRACTION COMPLÈTE DES STATISTIQUES
    # =========================================================

    def _get_complete_univariate_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Statistiques univariées complètes et bien formatées."""
        stats_dict = {}

        for col in df.columns:
            col_dtype = df[col].dtype
            
            if pd.api.types.is_numeric_dtype(col_dtype):
                clean_data = df[col].dropna()
                if len(clean_data) < 2:
                    continue

                desc = clean_data.describe()
                
                # Protection contre les NaN
                safe = lambda v: 0 if (pd.isna(v) or np.isinf(v)) else float(v)
                
                stats_dict[col] = {
                    "type": "numeric",
                    "count": int(len(clean_data)),
                    "missing": int(df[col].isna().sum()),
                    "missing_pct": round(100 * df[col].isna().sum() / len(df), 1),
                    "mean": round(safe(desc.get('mean')), 2),
                    "median": round(safe(desc.get('50%')), 2),
                    "mode": float(clean_data.mode()[0]) if len(clean_data.mode()) > 0 else None,
                    "std": round(safe(desc.get('std')), 2),
                    "var": round(safe(clean_data.var()), 2),
                    "min": safe(desc.get('min')),
                    "max": safe(desc.get('max')),
                    "q1": round(safe(desc.get('25%')), 2),
                    "q3": round(safe(desc.get('75%')), 2),
                    "iqr": round(safe(desc.get('75%')) - safe(desc.get('25%')), 2) if (safe(desc.get('75%')) != 0 and safe(desc.get('25%')) != 0) else 0,
                    "skew": round(safe(clean_data.skew()), 2),
                    "kurtosis": round(safe(clean_data.kurtosis()), 2),
                    "cv": round(safe(desc.get('std')) / (abs(safe(desc.get('mean'))) + 1e-6), 2)  # Coeff variation
                }
            else:
                try:
                    top = df[col].value_counts().head(10)
                    stats_dict[col] = {
                        "type": "categorical",
                        "unique": int(df[col].nunique()),
                        "missing": int(df[col].isna().sum()),
                        "missing_pct": round(100 * df[col].isna().sum() / len(df), 1),
                        "top_values": {str(k): int(v) for k, v in top.items()},
                        "top_category": str(top.index[0]) if len(top) > 0 else None,
                        "cardinality_ratio": round(df[col].nunique() / len(df), 2)
                    }
                except Exception as e:
                    logger.error(f"Erreur stats {col}: {e}")
                    stats_dict[col] = {"type": "categorical", "error": str(e)}

        return stats_dict

    # =========================================================
    # 3. GRAPHIQUES ENRICHIS & VARIÉS
    # =========================================================

    def _get_rich_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Histogrammes + Boxplots enrichis avec labels."""
        plots = {}
        numeric_cols = df.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            if df[col].nunique() < 2:
                continue
            
            data = df[col].dropna().values
            if len(data) < 3:
                continue

            try:
                # Histogramme adaptatif
                n_bins = min(30, int(np.sqrt(len(data))))
                hist, bin_edges = np.histogram(data, bins=n_bins)

                # Boxplot complètes
                q1 = np.percentile(data, 25)
                median = np.percentile(data, 50)
                q3 = np.percentile(data, 75)
                iqr = q3 - q1
                min_val = np.min(data)
                max_val = np.max(data)

                # Outliers
                lower_whisker = max(min_val, q1 - 1.5 * iqr)
                upper_whisker = min(max_val, q3 + 1.5 * iqr)
                outliers = len([x for x in data if x < lower_whisker or x > upper_whisker])

                plots[col] = {
                    "type": "distribution",
                    "title": col,
                    "unit": "",  # Peut être enrichi
                    "histogram": [
                        {
                            "range": f"{round(bin_edges[i], 2)}-{round(bin_edges[i+1], 2)}",
                            "count": int(hist[i]),
                            "pct": round(100 * hist[i] / hist.sum(), 1)
                        }
                        for i in range(len(hist))
                    ],
                    "boxplot": {
                        "min": float(min_val),
                        "max": float(max_val),
                        "q1": float(q1),
                        "median": float(median),
                        "q3": float(q3),
                        "iqr": float(iqr),
                        "lower": float(lower_whisker),
                        "upper": float(upper_whisker),
                        "outliers": outliers
                    },
                    "density_info": {
                        "mean": round(np.mean(data), 2),
                        "std": round(np.std(data), 2),
                        "skew": round(float(pd.Series(data).skew()), 2)
                    }
                }
            except Exception as e:
                logger.error(f"Erreur distribution {col}: {e}")
                continue

        return plots

    def _get_enriched_pie_charts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Camemberts avec meilleurs labels."""
        pies = []
        cat_cols = df.select_dtypes(include=['object', 'category']).columns

        for col in cat_cols:
            unique_count = df[col].nunique()
            if 2 <= unique_count <= 10:  # Augmenté de 6 à 10
                try:
                    counts = df[col].value_counts()
                    total = counts.sum()
                    data = [
                        {
                            "name": str(k)[:50],  # Limiter la longueur
                            "value": int(v),
                            "pct": round(100 * v / total, 1)
                        }
                        for k, v in counts.items()
                    ]
                    
                    pies.append({
                        "type": "pie",
                        "title": f"Distribution: {col}",
                        "label": col,
                        "total": int(total),
                        "data": data
                    })
                    
                    if len(pies) >= 8:
                        break
                except Exception as e:
                    logger.error(f"Erreur pie {col}: {e}")
                    continue

        return pies

    def _get_enhanced_scatter_plots(self, df: pd.DataFrame, target: str) -> List[Dict[str, Any]]:
        """Nuages de points enrichis avec labels et samples."""
        scatters = []
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if not target or target not in df.columns or len(num_cols) < 2:
            return []

        try:
            # Corrélations avec la cible
            if target in num_cols:
                corr_with_target = df[num_cols].corrwith(df[target]).abs().sort_values(ascending=False)
                top_vars = [v for v in corr_with_target.index[1:6] if v != target]  # Top 5, exclure la cible
            else:
                top_vars = df[num_cols].var().nlargest(5).index.tolist()

            for var in top_vars:
                try:
                    clean_data = df[[var, target]].dropna()
                    if len(clean_data) < 5:
                        continue

                    # Sampling pour ne pas surcharger le front
                    sample_size = min(500, len(clean_data))
                    if len(clean_data) > 500:
                        sample = clean_data.sample(n=sample_size, random_state=42)
                    else:
                        sample = clean_data

                    correlation = round(float(sample[var].corr(sample[target])), 2)

                    scatters.append({
                        "type": "scatter",
                        "title": f"{target} vs {var}",
                        "x_label": var,
                        "y_label": target,
                        "correlation": correlation,
                        "sample_size": len(sample),
                        "data": sample.to_dict(orient='records')
                    })
                except Exception as e:
                    logger.error(f"Erreur scatter {var}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Erreur scatter global: {e}")

        return scatters

    # =========================================================
    # 4. CLUSTERING 3D ROBUSTE
    # =========================================================

    def _perform_smart_clustering(self, df: pd.DataFrame, n_clusters: int = 3) -> Optional[Dict[str, Any]]:
        """Clustering 3D avec profiling robuste."""
        try:
            # Sélection intelligente des features
            numeric_df = df.select_dtypes(include=np.number)
            numeric_df = numeric_df.dropna(axis=1, thresh=len(df) * 0.5)
            numeric_df = numeric_df.fillna(numeric_df.median())

            if len(numeric_df.columns) < 3 or len(numeric_df) < 30:
                logger.warning("⚠ Pas assez de données pour clustering")
                return None

            # Standardisation
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)

            # K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_data)

            # PCA 3D
            pca = PCA(n_components=3)
            coords = pca.fit_transform(scaled_data)

            # Sampling pour le front
            limit = min(1000, len(coords))
            indices = np.random.choice(len(coords), limit, replace=False)

            points_3d = [
                {
                    "x": float(coords[i, 0]),
                    "y": float(coords[i, 1]),
                    "z": float(coords[i, 2]),
                    "cluster": int(clusters[i])
                }
                for i in indices
            ]

            # PROFILING: ADN des clusters
            df_clustered = numeric_df.copy()
            df_clustered['cluster'] = clusters
            global_means = numeric_df.mean()

            cluster_dna = {}
            for cid in range(n_clusters):
                subset = df_clustered[df_clustered['cluster'] == cid].drop(columns=['cluster'])
                
                if len(subset) < 5:
                    continue
                
                local_means = subset.mean()
                std = numeric_df.std().replace(0, 1)
                z_scores = (local_means - global_means) / std

                # Top 5 features différenciantes
                top_features = z_scores.abs().nlargest(5)

                features_desc = {}
                for feat in top_features.index:
                    score = z_scores[feat]
                    value = round(local_means[feat], 2)
                    direction = "↑ HAUT" if score > 0 else "↓ BAS"
                    features_desc[feat] = {
                        "direction": direction,
                        "z_score": round(score, 1),
                        "value": value
                    }

                cluster_dna[f"Groupe {cid+1}"] = {
                    "size": int(len(subset)),
                    "features": features_desc,
                    "variance_explained": round(100 * pca.explained_variance_ratio_.sum(), 1)
                }

            return {
                "scatter_points": points_3d,
                "dna": cluster_dna,
                "explained_variance": [round(float(v), 1) for v in pca.explained_variance_ratio_]
            }

        except Exception as e:
            logger.error(f"Erreur clustering: {e}")
            return None

    # =========================================================
    # 5. TESTS STATISTIQUES ROBUSTES
    # =========================================================

    def _statistical_tests(self, df: pd.DataFrame, target: str) -> List[Dict]:
        """Tests statistiques complets."""
        tests = []

        if not target or target not in df.columns:
            return []

        try:
            target_is_num = pd.api.types.is_numeric_dtype(df[target])

            for col in df.columns:
                if col == target:
                    continue

                try:
                    col_is_num = pd.api.types.is_numeric_dtype(df[col])
                    clean = df[[target, col]].dropna()

                    if len(clean) < 10:
                        continue

                    # Cas 1: Numérique vs Numérique
                    if target_is_num and col_is_num:
                        r, p = stats.pearsonr(clean[target], clean[col])
                        if abs(r) > 0.15:  # Baissé le seuil
                            tests.append({
                                "variable": col,
                                "test": "Pearson",
                                "coefficient": round(r, 3),
                                "p_value": round(p, 5),
                                "strength": "Très forte" if abs(r) > 0.7 else "Forte" if abs(r) > 0.5 else "Modérée" if abs(r) > 0.3 else "Faible"
                            })

                    # Cas 2: Catégorique vs Numérique
                    elif target_is_num and not col_is_num:
                        # ANOVA si > 2 groupes
                        groups = [clean[clean[col] == val][target].values for val in clean[col].unique()]
                        if len(groups) >= 2:
                            f_stat, p = stats.f_oneway(*groups)
                            if p < 0.1:  # Moins strict
                                tests.append({
                                    "variable": col,
                                    "test": "ANOVA",
                                    "f_statistic": round(f_stat, 2),
                                    "p_value": round(p, 5),
                                    "strength": "Significatif" if p < 0.05 else "Tendance"
                                })

                    # Cas 3: Catégorique vs Catégorique
                    elif not target_is_num and not col_is_num:
                        ct = pd.crosstab(df[target], df[col])
                        if ct.size > 1:
                            chi2, p, _, _ = stats.chi2_contingency(ct)
                            if p < 0.1:
                                tests.append({
                                    "variable": col,
                                    "test": "Chi-2",
                                    "chi2_stat": round(chi2, 2),
                                    "p_value": round(p, 5),
                                    "strength": "Significatif" if p < 0.05 else "Tendance"
                                })

                except Exception as e:
                    logger.error(f"Test {col}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Tests statistiques global: {e}")

        return sorted(tests, key=lambda x: abs(x.get('p_value', 1)))[:15]

    # =========================================================
    # 6. ORCHESTRATION MULTI-LLM ROBUSTE
    # =========================================================

    async def _generate_ai_insights_safe(self, eda_data: Dict, context: dict, user_prompt: str) -> List[Dict]:
        """Génération d'insights avec fallback robuste."""
        from services.multi_llm_insights import multi_llm_insights
        
        insights = []
        
        try:
            # Préparation des tâches
            tasks = []

            # 1. Analyse des clusters (si disponible)
            if eda_data.get('clustering') and eda_data['clustering'].get('dna'):
                dna_json = json.dumps(eda_data['clustering']['dna'], indent=2, ensure_ascii=False)
                tasks.append({
                    "task_id": "clusters",
                    "prompt": """Tu es un data analyst expert. Analyse ces clusters et génère EXACTEMENT un JSON array avec 3 objets (un par groupe).
CHAQUE objet DOIT avoir cette structure EXACTE :
{
  "title": "Nom du groupe (10 mots max)",
  "summary": "Caractéristiques clés (30 mots max)",
  "recommendation": "Action proposée (20 mots max)"
}
Sois concis et factuel. Ne mets RIEN d'autre.""",
                    "data": f"DNA des Clusters:\n{dna_json}"
                })

            # 2. Analyse des corrélations
            if eda_data.get('tests'):
                tests_json = json.dumps(eda_data['tests'][:8], indent=2)
                tasks.append({
                    "task_id": "correlations",
                    "prompt": f"""Analyse ces tests statistiques vs la cible '{context.get('target_variable', 'UNKNOWN')}'.
Génère UN SEUL JSON object :
{{
  "title": "Facteurs d'influence principaux",
  "summary": "Variables les plus impactantes (30 mots)",
  "recommendation": "Ordre de priorité pour l'investigation (20 mots)"
}}""",
                    "data": tests_json
                })

            # 3. Vue d'ensemble
            univariate = eda_data.get('univariate', {})
            summary_stats = {
                "total_variables": len(univariate),
                "numeric": len([s for s in univariate.values() if s.get('type') == 'numeric']),
                "categorical": len([s for s in univariate.values() if s.get('type') == 'categorical']),
                "target": context.get('target_variable', 'Unknown')
            }
            
            tasks.append({
                "task_id": "overview",
                "prompt": """Résume cette structure de données en UN JSON object :
{
  "title": "Vue d'ensemble du dataset",
  "summary": "Caractéristiques principales (30 mots)",
  "recommendation": "Étapes recommandées (20 mots)"
}""",
                "data": json.dumps(summary_stats)
            })

            # Exécution avec gestion d'erreurs
            if tasks:
                logger.info(f"🚀 Lancement {len(tasks)} tâches LLM")
                try:
                    # Appel du service avec timeout
                    ai_results = await asyncio.wait_for(
                        multi_llm_insights.run_parallel_analysis(tasks),
                        timeout=120
                    )
                    
                    if ai_results and isinstance(ai_results, list):
                        for result in ai_results:
                            if isinstance(result, dict) and 'title' in result:
                                insights.append(result)
                                logger.info(f"✓ Insight généré: {result.get('title', 'N/A')}")
                            else:
                                logger.warning(f"⚠ Format invalide: {result}")
                    
                except asyncio.TimeoutError:
                    logger.error("❌ Timeout LLM")
                except Exception as e:
                    logger.error(f"❌ Erreur LLM: {e}")

        except Exception as e:
            logger.error(f"Erreur globale insights: {e}")

        # Fallback: Générer des insights basiques
        if len(insights) < 3:
            logger.warning("⚠ Génération d'insights de secours")
            insights.extend(self._generate_fallback_insights(eda_data, context))

        return insights[:5]  # Max 5 insights

    def _generate_fallback_insights(self, eda_data: Dict, context: dict) -> List[Dict]:
        """Insights de secours (sans LLM)."""
        fallback = []

        target = context.get('target_variable', 'Unknown')
        
        # 1. Insights basée sur la structure
        n_num = len([s for s in eda_data.get('univariate', {}).values() if s.get('type') == 'numeric'])
        n_cat = len([s for s in eda_data.get('univariate', {}).values() if s.get('type') == 'categorical'])
        
        fallback.append({
            "title": "Structure du Dataset",
            "summary": f"{n_num} variables numériques, {n_cat} catégories. Ensemble bien équilibré.",
            "recommendation": "Procéder à l'analyse des corrélations et clusters."
        })

        # 2. Insights basée sur les tests
        tests = eda_data.get('tests', [])
        if tests:
            top_test = tests[0]
            fallback.append({
                "title": "Relation clé trouvée",
                "summary": f"{top_test.get('variable', 'X')} montre une relation significative avec la cible.",
                "recommendation": "Investiguer cette variable en priorité."
            })

        # 3. Insights clustering
        if eda_data.get('clustering'):
            fallback.append({
                "title": "Segmentation découverte",
                "summary": "3 groupes distincts identifiés par apprentissage non-supervisé.",
                "recommendation": "Analyser les caractéristiques de chaque groupe."
            })

        return fallback

    # =========================================================
    # 7. ORCHESTRATION PRINCIPALE
    # =========================================================

    async def run_full_eda(self, df: pd.DataFrame, context: dict, user_prompt: str) -> dict:
        """Pipeline EDA complet et robuste."""
        import asyncio
        
        logger.info(f"🎯 Démarrage EDA pour {len(df)} lignes × {len(df.columns)} colonnes")

        # A. Nettoyage basique
        df = df.fillna(df.mean(numeric_only=True))

        # B. Smart Target Detection
        target = self._smart_target_detection(df, user_prompt, context)
        logger.info(f"📍 Cible: {target}")

        # C. Thématisation (optionnel)
        themes = self._detect_themes(df)

        # D. Calculs EDA complets
        try:
            univariate = self._get_complete_univariate_stats(df)
            distributions = self._get_rich_distributions(df)
            pie_charts = self._get_enriched_pie_charts(df)
            scatter_plots = self._get_enhanced_scatter_plots(df, target)
            clustering = self._perform_smart_clustering(df, n_clusters=3)
            statistical_tests = self._statistical_tests(df, target)

            logger.info(f"✓ Stats complètes: {len(univariate)} variables")
            logger.info(f"✓ Distributions: {len(distributions)} graphes")
            logger.info(f"✓ Camemberts: {len(pie_charts)}")
            logger.info(f"✓ Nuages: {len(scatter_plots)}")
            logger.info(f"✓ Tests statistiques: {len(statistical_tests)}")

        except Exception as e:
            logger.error(f"❌ Erreur calculs EDA: {e}", exc_info=True)
            return self._generate_fallback_eda(df, target)

        # E. Corrélation
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        corr_matrix = {}
        if len(numeric_cols) > 1:
            focus_cols = list(set([target] + df[numeric_cols].var().nlargest(15).index.tolist()))
            focus_cols = [c for c in focus_cols if c in numeric_cols]
            corr_matrix = df[focus_cols].corr().round(2).fillna(0).to_dict()

        # F. IA Insights (avec fallback)
        eda_data_for_ai = {
            "univariate": univariate,
            "clustering": clustering,
            "tests": statistical_tests
        }

        ai_insights = await self._generate_ai_insights_safe(eda_data_for_ai, 
                                                             {"target_variable": target, **context}, 
                                                             user_prompt)

        # G. Packaging final
        return {
            "metrics": {
                "univariate": univariate,
                "correlation": corr_matrix,
                "clustering": clustering,
                "tests": statistical_tests,
                "themes": themes
            },
            "charts_data": {
                "distributions": distributions,
                "pies": pie_charts,
                "scatters": scatter_plots
            },
            "ai_insights": ai_insights,
            "auto_target": target,
            "summary": {
                "total_rows": len(df),
                "total_cols": len(df.columns),
                "numeric_cols": len(numeric_cols),
                "missing_values": df.isna().sum().sum()
            }
        }

    def _detect_themes(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Thématisation des colonnes."""
        themes = {}
        keywords_map = {
            "Démographie": ["age", "taona", "genre", "sexe", "situation", "matrimonial", "fokontany", "commune", "region"],
            "Économie": ["vol", "revenu", "argent", "budget", "fandaniana", "vidin", "prix", "amidy", "salaire", "bola"],
            "Activité": ["asa", "metier", "job", "fambolena", "fiompiana", "vokatra", "terrain", "surface"],
            "Tech & Biens": ["tel", "phone", "radio", "jiro", "electricite", "panneau", "moto", "voiture", "maison"],
            "Temporel": ["date", "fotoana", "duree", "taona", "annee", "mois"]
        }

        assigned = set()
        for theme, keywords in keywords_map.items():
            cols = [c for c in df.columns if not c in assigned and any(k in c.lower() for k in keywords)]
            if cols:
                themes[theme] = cols
                assigned.update(cols)

        others = [c for c in df.columns if c not in assigned]
        if others:
            themes["Autres"] = others

        return themes

    def _generate_fallback_eda(self, df: pd.DataFrame, target: str) -> dict:
        """EDA minimaliste en cas d'erreur."""
        logger.warning("⚠️ Fallback EDA")
        return {
            "metrics": {
                "univariate": {},
                "correlation": {},
                "clustering": None,
                "tests": [],
                "themes": {}
            },
            "charts_data": {
                "distributions": {},
                "pies": [],
                "scatters": []
            },
            "ai_insights": [{
                "title": "⚠️ Analyse Limitée",
                "summary": "Données insuffisantes ou erreur technique.",
                "recommendation": "Vérifiez la qualité du fichier et réessayez."
            }],
            "auto_target": target,
            "summary": {
                "total_rows": len(df),
                "total_cols": len(df.columns)
            }
        }

# Instanciation
eda_service = EDAService()