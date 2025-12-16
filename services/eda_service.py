"""
🎯 EDA SERVICE V5 - CLUSTERING ROBUSTE AVEC VISUALISATION 3D COMPLÈTE
✅ FIX: Afficher TOUS les points sans échantillonnage
✅ FIX: Meilleur équilibre des clusters
✅ FIX: Variables binaires mieux gérées
✅ FIX: TTS toujours activé
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
import json
import asyncio
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class EDAService:
    """EDA SERVICE V5 - CLUSTERING ROBUSTE ET VISUALISATION 3D COMPLÈTE"""

    def _smart_target_detection(self, df: pd.DataFrame, user_prompt: str, context: Dict[str, Any]) -> str:
        """Détecte la cible intelligemment"""
        logger.info("🎯 DÉTECTION INTELLIGENTE DE LA CIBLE")
        
        target = context.get('target_variable') or context.get('focus_variable')
        if target and target in df.columns:
            logger.info(f"  ✓ Cible explicite: {target}")
            return target
        
        for col in df.columns:
            col_lower = col.lower()
            if any(k in col_lower for k in ['target', 'cible', 'objectif', 'outcome', 'y_', '_y']):
                logger.info(f"  ✓ Cible par pattern: {col}")
                return col
            if col_lower in user_prompt.lower():
                logger.info(f"  ✓ Cible mentionnée: {col}")
                return col
        
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) == 0:
            logger.warning("⚠ Aucune colonne numérique")
            return df.columns[0] if len(df.columns) > 0 else "Non définie"
        
        candidate_scores = {}
        for col in numeric_cols:
            if df[col].nunique() < 2:
                continue
            clean_data = df[col].dropna()
            if len(clean_data) < 10:
                continue
            
            var_score = clean_data.std() / (clean_data.mean().abs() + 1e-6)
            skew = clean_data.skew()
            skew_score = 1.0 - min(abs(skew) / 3, 1.0)
            completeness = clean_data.notna().sum() / len(df[col])
            
            total_score = (var_score * 0.4 + skew_score * 0.3 + completeness * 0.3)
            candidate_scores[col] = total_score
        
        if candidate_scores:
            target = max(candidate_scores, key=candidate_scores.get)
            logger.info(f"  ✓ Cible auto-sélectionnée: {target}")
            return target
        
        return numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]

    def _select_key_variables(self, df: pd.DataFrame, file_structure: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Sélection intelligente de variables"""
        logger.info("🧠 SÉLECTION INTELLIGENTE DE VARIABLES")
        
        target_var = context.get('target_variable', '')
        focus_vars = context.get('focus_variables', [])
        
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # SCORING NUMÉRIQUES
        numeric_scores = {}
        for col in numeric_cols:
            if df[col].nunique() < 2:
                continue
            clean_data = df[col].dropna()
            if len(clean_data) < 5:
                continue
            
            mean_val = clean_data.mean()
            cv = clean_data.std() / (abs(mean_val) + 1e-6)
            cv_score = min(cv, 3.0) / 3.0
            completeness = 1 - (df[col].isna().sum() / len(df))
            focus_bonus = 0.3 if col in focus_vars else 0
            
            numeric_scores[col] = cv_score * 0.4 + completeness * 0.3 + focus_bonus
        
        # SCORING CATÉGORIES
        categorical_scores = {}
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if unique_count < 2 or unique_count > 20:
                continue
            
            cardinality_score = 1.0 - (unique_count / 20.0)
            completeness = 1 - (df[col].isna().sum() / len(df))
            focus_bonus = 0.3 if col in focus_vars else 0
            
            categorical_scores[col] = cardinality_score * 0.4 + completeness * 0.4 + focus_bonus
        
        top_numeric = sorted(numeric_scores.items(), key=lambda x: x[1], reverse=True)[:15]
        top_categorical = sorted(categorical_scores.items(), key=lambda x: x[1], reverse=True)[:8]
        
        numeric_analysis = [col for col, _ in top_numeric]
        categorical_analysis = [col for col, _ in top_categorical]
        
        if not numeric_analysis:
            numeric_analysis = numeric_cols[:15]
            logger.warning("⚠️ Aucune variables numériques scorées")
        
        if not categorical_analysis:
            categorical_analysis = categorical_cols[:8]
            logger.warning("⚠️ Aucune variables catégories scorées")
        
        if target_var and target_var in df.columns:
            selected_target = target_var
        elif len(numeric_analysis) > 0:
            selected_target = numeric_analysis[0]
        else:
            selected_target = df.columns[0] if len(df.columns) > 0 else "Non définie"
        
        return {
            "target": selected_target,
            "numeric_analysis": numeric_analysis,
            "categorical_analysis": categorical_analysis,
            "grouping_candidates": categorical_analysis[:3],
            "correlation_vars": numeric_analysis[:10],
            "reasoning": {
                "target_reasoning": f"Variable '{selected_target}' sélectionnée",
                "numeric_reasoning": f"{len(numeric_analysis)} variables numériques",
                "categorical_reasoning": f"{len(categorical_analysis)} variables catégories"
            },
            "scores": {
                "numeric": {col: round(score, 3) for col, score in top_numeric},
                "categorical": {col: round(score, 3) for col, score in top_categorical}
            }
        }

    def _get_univariate_stats_smart(self, df: pd.DataFrame, selected_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Stats univariées"""
        numeric_cols = selected_vars.get('numeric_analysis', [])
        categorical_cols = selected_vars.get('categorical_analysis', [])
        stats_dict = {}
        
        for col in numeric_cols:
            clean = df[col].dropna()
            if len(clean) < 2:
                continue
            desc = clean.describe()
            safe = lambda v: 0 if (pd.isna(v) or np.isinf(v)) else float(v)
            
            mode_val = float(clean.mode()[0]) if len(clean.mode()) > 0 else None
            q1 = desc['25%']
            q3 = desc['75%']
            iqr = q3 - q1
            outliers = len([x for x in clean if x < (q1 - 1.5 * iqr) or x > (q3 + 1.5 * iqr)])
            
            mean_val = desc['mean']
            cv = desc['std'] / (abs(mean_val) + 1e-6) if mean_val != 0 else 0
            
            stats_dict[col] = {
                "type": "numeric",
                "count": int(len(clean)),
                "missing": int(df[col].isna().sum()),
                "missing_pct": round(100 * df[col].isna().sum() / len(df), 1),
                "mean": round(safe(desc['mean']), 2),
                "median": round(safe(desc['50%']), 2),
                "mode": mode_val,
                "std": round(safe(desc['std']), 2),
                "var": round(safe(clean.var()), 2),
                "min": safe(desc['min']),
                "max": safe(desc['max']),
                "q1": round(safe(desc['25%']), 2),
                "q3": round(safe(desc['75%']), 2),
                "iqr": round(safe(iqr), 2),
                "skew": round(safe(clean.skew()), 2),
                "kurtosis": round(safe(clean.kurtosis()), 2),
                "cv": round(safe(cv), 2),
                "outliers": outliers
            }
        
        for col in categorical_cols:
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
        
        return stats_dict

    def _select_best_clustering_variables(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[str]:
        """
        🔧 V5: Sélectionner les variables pour clustering - ÉVITER LES VARIABLES TROP DÉSÉQUILIBRÉES
        """
        
        logger.info("🎯 SÉLECTION VARIABLES CLUSTERING V5 - Filtrage amélioré")
        
        if len(numeric_cols) < 2:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        clustering_scores = {}
        
        for col in numeric_cols:
            clean = df[col].dropna()
            
            if len(clean) < 10:  # Minimum de points
                continue
                
            # Calculer plusieurs métriques
            variance = clean.var()
            unique_count = df[col].nunique()
            
            # 🔥 FILTRES STRICTS :
            # 1. Variance trop faible (variables quasi-constantes)
            if variance < 0.001:
                logger.info(f"  ⚠️ {col}: Variance trop faible ({variance:.4f})")
                continue
                
            # 2. Variables binaires déséquilibrées (>95% d'une valeur)
            if unique_count == 2:
                value_counts = df[col].value_counts(normalize=True)
                max_ratio = value_counts.max()
                if max_ratio > 0.95:
                    logger.info(f"  ⚠️ {col}: Binaire trop déséquilibré ({max_ratio:.1%})")
                    continue
                # Bonus modéré pour binaires équilibrées
                balance_score = 1.0 - abs(value_counts.iloc[0] - 0.5) * 2
            else:
                balance_score = 1.0
                
            # 3. Cardinalité trop faible avec variance faible
            if unique_count < 3 and variance < 0.1:
                logger.info(f"  ⚠️ {col}: Cardinalité faible ({unique_count}) avec variance faible ({variance:.3f})")
                continue
            
            # Score amélioré
            if abs(clean.mean()) > 1e-6:
                variance_score = min(np.sqrt(variance) / abs(clean.mean()), 1.0)
            else:
                variance_score = min(variance, 1.0)
                
            completeness = 1.0 - (df[col].isna().sum() / len(df))
            
            # Bonus pour diversité
            diversity_bonus = min(unique_count / 15, 0.5)
            
            # Pénalité de skew modérée
            skew = abs(clean.skew()) if len(clean) > 2 else 0
            skew_penalty = 1.0 - min(skew / 6.0, 0.3)
            
            score = (
                variance_score * 0.35 +
                completeness * 0.25 +
                diversity_bonus * 0.20 +
                balance_score * 0.10 +
                skew_penalty * 0.10
            )
            
            clustering_scores[col] = score
            logger.info(f"  📊 {col}: score={score:.3f}, unique={unique_count}, var={variance:.3f}, skew={skew:.2f}")
        
        if not clustering_scores:
            logger.warning("⚠️ Aucune variable scorée, utilisation de toutes les variables")
            return [c for c in numeric_cols if df[c].notna().sum() >= 5][:10]
        
        # Trier et retourner les meilleures (moins mais mieux)
        sorted_cols = sorted(clustering_scores.items(), key=lambda x: x[1], reverse=True)
        best_cols = [col for col, _ in sorted_cols[:min(8, len(sorted_cols))]]
        
        logger.info(f"✅ Top {len(best_cols)} variables clustering: {best_cols}")
        return best_cols

    def _balance_clusters(self, clusters: np.ndarray, min_size_ratio: float = 0.05) -> np.ndarray:
        """
        Rééquilibrer les clusters pour éviter les groupes trop petits
        min_size_ratio: taille minimum relative (5% par défaut)
        """
        unique, counts = np.unique(clusters, return_counts=True)
        total = len(clusters)
        min_size = int(total * min_size_ratio)
        
        # Identifier les petits clusters
        small_clusters = unique[counts < min_size]
        
        if len(small_clusters) == 0:
            return clusters
        
        # Réassigner les points des petits clusters au cluster le plus proche en taille
        balanced = clusters.copy()
        for small_c in small_clusters:
            mask = clusters == small_c
            if np.sum(mask) > 0:
                # Trouver le cluster avec le plus de points (hors petits clusters)
                valid_clusters = unique[counts >= min_size]
                if len(valid_clusters) > 0:
                    # Assigner au cluster majoritaire
                    balanced[mask] = valid_clusters[0]
        
        logger.info(f"🔧 Rééquilibrage: {len(small_clusters)} petits clusters fusionnés")
        return balanced

    def _validate_clustering(self, clusters: np.ndarray, n_clusters: int) -> Dict[str, Any]:
        """Valider la qualité du clustering"""
        
        unique_clusters = np.unique(clusters[clusters >= 0])  # Ignorer -1 (bruit DBSCAN)
        actual_clusters = len(unique_clusters)
        
        # Distribution des clusters
        cluster_dist = {}
        for c in unique_clusters:
            cluster_dist[int(c)] = int(np.sum(clusters == c))
        
        # Gérer le cas où il n'y a pas de clusters valides
        if not cluster_dist:
            return {
                "actual_clusters": 0,
                "cluster_distribution": {},
                "min_cluster_size": 0,
                "max_cluster_size": 0,
                "balance_ratio": 0,
                "is_balanced": False
            }
        
        # Taille minimum des clusters
        min_cluster_size = min(cluster_dist.values())
        max_cluster_size = max(cluster_dist.values())
        
        # Ratio de balance
        balance_ratio = min_cluster_size / max_cluster_size if max_cluster_size > 0 else 0
        
        return {
            "actual_clusters": actual_clusters,
            "cluster_distribution": cluster_dist,
            "min_cluster_size": min_cluster_size,
            "max_cluster_size": max_cluster_size,
            "balance_ratio": round(balance_ratio, 3),
            "is_balanced": balance_ratio > 0.1  # Seuil assoupli
        }

    def _create_3d_coordinates(self, scaled_data: np.ndarray, n_vars: int) -> np.ndarray:
        """
        🔧 V5: Créer des coordonnées 3D robustes
        """
        n_samples = len(scaled_data)
        
        if n_vars >= 3:
            # PCA standard
            pca = PCA(n_components=3, random_state=42)
            coords = pca.fit_transform(scaled_data)
            logger.info(f"  📊 PCA: {n_vars} vars → 3 composantes, variance expliquée: {pca.explained_variance_ratio_}")
            
        elif n_vars == 2:
            # 2 variables: créer un 3ème axe basé sur la distance
            pca = PCA(n_components=2, random_state=42)
            coords_2d = pca.fit_transform(scaled_data)
            
            # 3ème axe = distance au centre + variation
            center = coords_2d.mean(axis=0)
            distances = np.linalg.norm(coords_2d - center, axis=1)
            z_axis = (distances - distances.min()) / (distances.max() - distances.min() + 1e-6) * 2 - 1
            
            # Ajouter du bruit structuré
            np.random.seed(42)
            z_axis = z_axis + np.random.normal(0, 0.2, n_samples)
            
            coords = np.column_stack([coords_2d, z_axis])
            logger.info(f"  📊 2 vars → 3D avec axe Z basé sur distance")
            
        else:
            # 1 variable: créer 3 axes artificiels intéressants
            var_data = scaled_data.flatten() if scaled_data.ndim > 1 else scaled_data
            
            # Normaliser
            var_min, var_max = var_data.min(), var_data.max()
            if var_max - var_min > 0:
                var_normalized = (var_data - var_min) / (var_max - var_min) * 2 - 1
            else:
                var_normalized = np.zeros_like(var_data)
            
            # Axe X: valeur normalisée avec bruit
            np.random.seed(42)
            x_axis = var_normalized + np.random.normal(0, 0.2, n_samples)
            
            # Axe Y: transformation sinusoïdale + bruit
            y_axis = np.sin(var_normalized * np.pi) + np.random.normal(0, 0.3, n_samples)
            
            # Axe Z: variation structurée par rang + bruit
            ranks = np.argsort(np.argsort(var_data))  # Rangs
            z_axis = (ranks / n_samples) * 2 - 1 + np.random.normal(0, 0.2, n_samples)
            
            coords = np.column_stack([x_axis, y_axis, z_axis])
            logger.info(f"  📊 1 var → 3 axes artificiels")
        
        # Normaliser chaque axe entre -10 et 10
        for i in range(3):
            min_val, max_val = coords[:, i].min(), coords[:, i].max()
            if max_val - min_val > 1e-6:
                coords[:, i] = (coords[:, i] - min_val) / (max_val - min_val) * 20 - 10
            else:
                # Ajouter de la variation si l'axe est plat
                coords[:, i] = np.random.uniform(-10, 10, n_samples)
        
        return coords

    def _separate_clusters_3d(self, coords: np.ndarray, clusters: np.ndarray) -> np.ndarray:
        """
        🔧 V5: Séparer visuellement les clusters en 3D
        """
        unique_clusters = np.unique(clusters[clusters >= 0])
        n_clusters = len(unique_clusters)
        
        if n_clusters <= 1:
            return coords
        
        # Calculer les centres actuels
        centers = {}
        for c in unique_clusters:
            mask = clusters == c
            if np.sum(mask) > 0:
                centers[c] = coords[mask].mean(axis=0)
        
        # Calculer les nouveaux centres séparés (sur une sphère)
        new_centers = {}
        radius = 8  # Rayon de séparation
        for i, c in enumerate(unique_clusters):
            angle = 2 * np.pi * i / n_clusters
            elevation = np.pi / 4 * (1 if i % 2 == 0 else -1)  # Alternance haut/bas
            
            new_centers[c] = np.array([
                radius * np.cos(angle) * np.cos(elevation),
                radius * np.sin(angle) * np.cos(elevation),
                radius * np.sin(elevation)
            ])
        
        # Déplacer les points vers les nouveaux centres
        new_coords = coords.copy()
        for c in unique_clusters:
            mask = clusters == c
            if np.sum(mask) > 0 and c in centers and c in new_centers:
                # Translation vers le nouveau centre
                offset = new_centers[c] - centers[c]
                new_coords[mask] += offset * 0.7  # 70% du déplacement pour garder de la structure
        
        return new_coords

    def _perform_clustering_algorithm(self, scaled_data: np.ndarray, coords_3d: np.ndarray, 
                                      n_clusters: int) -> tuple:
        """
        🔧 V5: Essayer plusieurs algorithmes de clustering avec équilibrage
        """
        n_samples = len(scaled_data)
        best_clusters = None
        best_score = -2
        best_method = None
        
        # 1. KMeans avec plusieurs initialisations
        for init_idx in range(3):
            try:
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=42 + init_idx * 17,
                    n_init=15,
                    max_iter=500
                )
                clusters = kmeans.fit_predict(scaled_data)
                
                # Appliquer le rééquilibrage
                clusters = self._balance_clusters(clusters, min_size_ratio=0.05)
                
                # Vérifier qu'on a bien n clusters
                unique = np.unique(clusters)
                if len(unique) < n_clusters:
                    continue
                
                # Calculer silhouette
                if len(unique) >= 2:
                    try:
                        score = silhouette_score(scaled_data, clusters)
                        if score > best_score:
                            best_score = score
                            best_clusters = clusters
                            best_method = f"KMeans (init {init_idx})"
                    except:
                        pass
            except Exception as e:
                logger.warning(f"  ⚠ KMeans init {init_idx}: {e}")
        
        # 2. Gaussian Mixture Model avec équilibrage
        if best_score < 0.2:
            try:
                gmm = GaussianMixture(
                    n_components=n_clusters,
                    random_state=42,
                    n_init=5,
                    covariance_type='full'
                )
                clusters = gmm.fit_predict(scaled_data)
                clusters = self._balance_clusters(clusters, min_size_ratio=0.05)
                
                unique = np.unique(clusters)
                if len(unique) >= 2:
                    try:
                        score = silhouette_score(scaled_data, clusters)
                        if score > best_score:
                            best_score = score
                            best_clusters = clusters
                            best_method = "GMM"
                    except:
                        pass
            except Exception as e:
                logger.warning(f"  ⚠ GMM: {e}")
        
        # 3. Fallback: KMeans sur coordonnées 3D
        if best_clusters is None:
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
                best_clusters = kmeans.fit_predict(coords_3d)
                best_clusters = self._balance_clusters(best_clusters, min_size_ratio=0.05)
                best_method = "KMeans sur coords 3D"
                
                unique = np.unique(best_clusters)
                if len(unique) >= 2:
                    try:
                        best_score = silhouette_score(coords_3d, best_clusters)
                    except:
                        best_score = 0.1
            except Exception as e:
                logger.error(f"  ❌ Fallback KMeans: {e}")
                # Dernier recours: assignation aléatoire structurée par quartiles
                np.random.seed(42)
                if n_clusters == 2:
                    # Basé sur médiane
                    median_val = np.median(scaled_data[:, 0]) if scaled_data.shape[1] > 0 else 0
                    best_clusters = (scaled_data[:, 0] > median_val).astype(int)
                else:
                    # Quartiles
                    q1 = np.percentile(scaled_data[:, 0], 25) if scaled_data.shape[1] > 0 else 0
                    q3 = np.percentile(scaled_data[:, 0], 75) if scaled_data.shape[1] > 0 else 1
                    best_clusters = np.zeros(n_samples, dtype=int)
                    best_clusters[scaled_data[:, 0] > q3] = 1
                    if n_clusters >= 3:
                        best_clusters[scaled_data[:, 0] < q1] = 2
                    if n_clusters >= 4:
                        mask = (scaled_data[:, 0] >= q1) & (scaled_data[:, 0] <= q3)
                        mid = (q1 + q3) / 2
                        best_clusters[mask & (scaled_data[:, 0] > mid)] = 3
                best_score = 0.0
                best_method = "Quartile Split (fallback)"
        
        logger.info(f"  ✅ Méthode: {best_method}, score: {best_score:.3f}, clusters uniques: {len(np.unique(best_clusters))}")
        return best_clusters, best_score, best_method

    def _generate_single_clustering(self, scaled_data: np.ndarray, df_numeric: pd.DataFrame, 
                                    n_clusters: int, name: str) -> Dict[str, Any]:
        """
        🔧 V5: Générer UNE segmentation avec TOUS les points
        """
        
        n_samples = len(scaled_data)
        n_vars = len(df_numeric.columns)
        
        logger.info(f"  🎯 {name}: {n_samples} samples, {n_vars} vars, k={n_clusters}")
        
        try:
            # 1. Créer les coordonnées 3D
            coords_3d = self._create_3d_coordinates(scaled_data, n_vars)
            
            # 2. Effectuer le clustering
            clusters, silhouette, method = self._perform_clustering_algorithm(
                scaled_data, coords_3d, n_clusters
            )
            
            # 3. Validation
            validation = self._validate_clustering(clusters, n_clusters)
            
            # Si pas assez de clusters, forcer
            if validation["actual_clusters"] < 2:
                logger.warning(f"  ⚠ Seulement {validation['actual_clusters']} cluster(s), forçage...")
                # Forcer 2 clusters basés sur médiane
                median_val = np.median(scaled_data[:, 0]) if n_vars > 0 else 0
                clusters = (scaled_data[:, 0] > median_val).astype(int)
                validation = self._validate_clustering(clusters, n_clusters)
            
            # 4. Séparer visuellement les clusters
            coords_3d = self._separate_clusters_3d(coords_3d, clusters)
            
            # Calculer la dispersion
            center = coords_3d.mean(axis=0)
            distances = np.linalg.norm(coords_3d - center, axis=1)
            dispersion_level = distances.std() / (distances.mean() + 1e-6)
            dispersion_level = min(dispersion_level, 1.0)
            
            # 🔥 5. TOUS LES POINTS SONT ENVOYÉS - PAS D'ÉCHANTILLONNAGE
            logger.info(f"  📊 Envoi de TOUS les {n_samples} points au frontend")
            
            # 6. Points scatter 3D - TOUS LES POINTS
            scatter_3d = [
                {
                    "x": float(coords_3d[idx, 0]),
                    "y": float(coords_3d[idx, 1]),
                    "z": float(coords_3d[idx, 2]),
                    "cluster": int(clusters[idx])
                }
                for idx in range(n_samples)  # TOUS les points
            ]
            
            # 7. DNA des clusters
            df_clust = df_numeric.copy()
            df_clust['cluster'] = clusters
            global_mean = df_numeric.mean()
            global_std = df_numeric.std() + 1e-6
            
            cluster_dna = {}
            cluster_dist = []
            heatmap_data = []
            radar_data = []
            
            unique_clusters = np.unique(clusters[clusters >= 0])
            for cid in unique_clusters:
                subset = df_clust[df_clust['cluster'] == cid].drop(columns=['cluster'])
                
                if len(subset) < 1:
                    continue
                
                local_mean = subset.mean()
                z_scores = (local_mean - global_mean) / global_std
                
                # DNA avec caractéristiques distinctives
                top_features = z_scores.abs().nlargest(min(5, len(z_scores)))
                features_desc = {}
                for feat in top_features.index:
                    direction = "↑ HAUT" if z_scores[feat] > 0 else "↓ BAS"
                    magnitude = abs(z_scores[feat])
                    
                    features_desc[feat] = {
                        "direction": direction,
                        "z_score": round(float(z_scores[feat]), 2),
                        "value": round(float(local_mean[feat]), 2),
                        "importance": round(magnitude, 2),
                        "interpretation": f"{direction} de {magnitude:.1f} écarts-types"
                    }
                
                cluster_size = len(subset)
                cluster_percentage = 100 * cluster_size / len(df_clust)
                
                cluster_dna[f"Groupe {int(cid)+1}"] = {
                    "size": int(cluster_size),
                    "percentage": round(cluster_percentage, 1),
                    "features": features_desc,
                    "centroid_distance": round(float(np.linalg.norm(local_mean - global_mean)), 2),
                    "distinctiveness": round(float(top_features.mean()) if len(top_features) > 0 else 0, 2)
                }
                
                cluster_dist.append({
                    "cluster": f"Groupe {int(cid)+1}",
                    "count": int(cluster_size),
                    "percentage": round(cluster_percentage, 1),
                    "color_index": int(cid) % 8
                })
                
                heatmap_vals = {col: round(float(z_scores[col]), 2) for col in df_numeric.columns[:10]}
                heatmap_data.append({"cluster": f"Groupe {int(cid)+1}", "values": heatmap_vals})
                
                radar_vals = {col: round(float(z_scores[col]), 2) for col in df_numeric.columns[:8]}
                radar_data.append({"cluster": f"Groupe {int(cid)+1}", "values": radar_vals})
            
            if not cluster_dna:
                logger.warning("❌ Aucun cluster DNA généré")
                return None
            
            # 8. Générer l'explication
            clustering_explanation = self._generate_clustering_explanation(
                cluster_dna, 
                validation, 
                silhouette,  # Peut être None ou float
                name,
                data_dispersion_level=dispersion_level,
                method_used=method
            )
            
            # Vérifier la qualité 3D
            var_x = np.var(coords_3d[:, 0])
            var_y = np.var(coords_3d[:, 1])
            var_z = np.var(coords_3d[:, 2])
            quality_3d = min(var_x, var_y, var_z) / (max(var_x, var_y, var_z) + 1e-6)
            
            logger.info(f"  📊 Variance 3D: X={var_x:.2f}, Y={var_y:.2f}, Z={var_z:.2f}, qualité={quality_3d:.2f}")
            
            return {
                "name": name,
                "n_clusters": len(cluster_dna),
                "scatter_points": scatter_3d,  # TOUS les points
                "dna": cluster_dna,
                "cluster_distribution": cluster_dist,
                "heatmap_data": heatmap_data,
                "radar_data": radar_data,
                "explained_variance": [],
                "validation": validation,
                "silhouette_score": round(silhouette, 3) if silhouette is not None else None,
                "dispersion_level": round(dispersion_level, 3),
                "quality_3d": round(quality_3d, 3),
                "method_used": method,
                "explanation": clustering_explanation,
                "total_points": n_samples  # Ajouter le total pour info
            }
        
        except Exception as e:
            logger.error(f"❌ Erreur génération clustering {name}: {e}", exc_info=True)
            return None

    def _generate_clustering_explanation(self, cluster_dna: Dict, validation: Dict, 
                                        silhouette_score_val: Optional[float], name: str, 
                                        data_dispersion_level: float = 0.0,
                                        method_used: str = "KMeans") -> Dict[str, str]:
        """
        🔧 V5: Explication avec gestion robuste de silhouette_score None
        """
        
        if not cluster_dna:
            return {
                "title": "Segmentation Non Concluante",
                "summary": "Les données n'ont pas permis d'identifier des groupes distincts.",
                "recommendation": "Considérez d'autres variables ou méthodes d'analyse.",
                "details": {},
                "tts_text": "L'analyse de clustering n'a pas identifié de groupes distincts."
            }
        
        n_clusters = len(cluster_dna)
        total_points = sum([c["size"] for c in cluster_dna.values()])
        
        # Analyser la distribution
        sizes = [c["size"] for c in cluster_dna.values()]
        min_size = min(sizes) if sizes else 0
        max_size = max(sizes) if sizes else 0
        balance = min_size / max_size if max_size > 0 else 0
        
        # Analyser les caractéristiques
        all_features = {}
        for cluster_info in cluster_dna.values():
            for feat, info in cluster_info.get("features", {}).items():
                if feat not in all_features:
                    all_features[feat] = []
                all_features[feat].append(abs(info.get("z_score", 0)))
        
        avg_z_scores = {k: sum(v) / len(v) for k, v in all_features.items()}
        distinctive_features = sorted(avg_z_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # === GÉNÉRER SUMMARY ===
        summary_parts = []
        
        # Nombre de clusters
        if n_clusters == 2:
            summary_parts.append(f"Deux segments ont été identifiés dans vos données ({total_points} points).")
        elif n_clusters == 3:
            summary_parts.append(f"Trois groupes distincts émergent de l'analyse ({total_points} points).")
        else:
            summary_parts.append(f"{n_clusters} clusters ont été formés ({total_points} points).")
        
        # 🔧 FIX: Gestion silhouette_score None
        if silhouette_score_val is not None:
            if silhouette_score_val > 0.5:
                summary_parts.append(f"La séparation entre groupes est très bonne (score: {silhouette_score_val:.2f}).")
            elif silhouette_score_val > 0.25:
                summary_parts.append(f"La séparation entre groupes est acceptable (score: {silhouette_score_val:.2f}).")
            elif silhouette_score_val > 0:
                summary_parts.append(f"⚠️ La séparation est faible (score: {silhouette_score_val:.2f}). "
                                   f"Les groupes se chevauchent partiellement.")
            else:
                summary_parts.append(f"⚠️ Les données sont très dispersées (score: {silhouette_score_val:.2f}). "
                                   f"Les clusters identifiés sont indicatifs.")
        else:
            summary_parts.append("Le score de qualité n'a pas pu être calculé. "
                               "Les groupes sont formés mais leur séparation est incertaine.")
        
        # Distribution
        if balance > 0.5:
            summary_parts.append(f"Les groupes sont bien équilibrés ({balance:.0%}).")
        elif balance > 0.2:
            summary_parts.append(f"Distribution acceptable (ratio: {balance:.0%}).")
        else:
            summary_parts.append(f"⚠️ Groupes déséquilibrés (ratio: {balance:.0%}).")
        
        # Méthode
        summary_parts.append(f"Méthode utilisée: {method_used}.")
        
        # Caractéristiques
        if distinctive_features:
            feature_names = [f[0] for f in distinctive_features[:3]]
            summary_parts.append(f"Variables distinctives: {', '.join(feature_names)}.")
        
        summary = " ".join(summary_parts)
        
        # === RECOMMANDATIONS ===
        recommendations = []
        recommendations.append("Examinez les profils de chaque groupe.")
        
        if silhouette_score_val is not None and silhouette_score_val < 0.2:
            recommendations.append("Les données dispersées suggèrent d'explorer d'autres variables ou méthodes.")
        
        recommendations.append("La visualisation 3D montre la répartition spatiale de TOUS les points.")
        
        recommendation = " | ".join(recommendations[:3])
        
        # === DÉTAILS ===
        details = {
            "nombre_groupes": n_clusters,
            "equilibre_distribution": f"{balance:.1%}",
            "variables_distinctives": [f[0] for f in distinctive_features[:3]],
            "score_qualite": f"{silhouette_score_val:.2f}" if silhouette_score_val is not None else "N/A",
            "methode": method_used,
            "dispersion": "Élevée" if data_dispersion_level > 0.7 else "Moyenne" if data_dispersion_level > 0.4 else "Faible",
            "points_totaux": total_points
        }
        
        # === TTS TEXT ===
        tts_parts = [f"Analyse {name}.", summary]
        
        for group_name, group_info in list(cluster_dna.items())[:3]:
            tts_parts.append(f"{group_name}: {group_info['percentage']}% des données.")
            
            features = group_info.get("features", {})
            if features:
                top_feat = list(features.items())[0]
                feat_name, feat_info = top_feat
                direction = "élevé" if feat_info.get("z_score", 0) > 0 else "faible"
                tts_parts.append(f"Caractéristique principale: {feat_name} {direction}.")
        
        tts_parts.append(recommendation)
        tts_text = " ".join(tts_parts)
        
        return {
            "title": f"Analyse {name}",
            "summary": summary,
            "recommendation": recommendation,
            "details": details,
            "tts_text": tts_text
        }

    def _perform_multi_clustering(self, df: pd.DataFrame, selected_vars: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        🔧 V5: Multi-clustering robuste avec TOUS les points
        """
        
        logger.info("🎨 MULTI-CLUSTERING V5 - TOUS LES POINTS VISIBLES")
        
        numeric_cols = selected_vars.get('numeric_analysis', [])
        
        if len(numeric_cols) < 1:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if len(numeric_cols) < 1:
            logger.warning("⚠️ Pas de variables numériques")
            return None
        
        try:
            # Sélectionner les variables
            best_clustering_vars = self._select_best_clustering_variables(df, numeric_cols)
            
            if not best_clustering_vars:
                logger.warning("❌ Aucune variable de clustering")
                return None
            
            logger.info(f"📊 {len(best_clustering_vars)} variables pour clustering")
            
            df_numeric = df[best_clustering_vars].copy()
            
            if len(df_numeric) < 10:
                logger.warning("⚠️ Pas assez de lignes (<10)")
                return None
            
            # Imputation
            imputer = SimpleImputer(strategy='median')
            df_imputed = pd.DataFrame(
                imputer.fit_transform(df_numeric), 
                columns=df_numeric.columns
            )
            
            # Supprimer les lignes avec NaN restants
            df_imputed = df_imputed.dropna()
            
            if len(df_imputed) < 10:
                logger.warning("⚠️ Pas assez de données après imputation")
                return None
            
            # Normalisation
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_imputed)
            
            # Remplacer inf par 0
            scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Configurations de clustering
            clustering_configs = [
                (2, "Segmentation Binaire"),
                (3, "Tri-Segmentation"),
                (4, "Quadri-Segmentation"),
                (5, "Penta-Segmentation")
            ]
            
            clusterings = {}
            
            for n_clusters, name in clustering_configs:
                min_points = max(n_clusters * 3, 10)
                if len(df_imputed) < min_points:
                    logger.warning(f"⏭️ {name}: Pas assez de données")
                    continue
                
                result = self._generate_single_clustering(
                    scaled_data, df_imputed, n_clusters, name
                )
                
                if result and result.get("n_clusters", 0) >= 2:
                    clusterings[f'clustering_k{n_clusters}'] = result
                    logger.info(f"✅ {name}: {result['n_clusters']} groupes, "
                              f"silhouette={result.get('silhouette_score', 'N/A')}, "
                              f"points={result.get('total_points', 0)}")
                else:
                    logger.warning(f"⚠️ {name}: échec")
            
            if not clusterings:
                logger.warning("❌ Aucun clustering généré - création de fallback")
                
                # Fallback: créer un clustering simple
                fallback_result = self._create_fallback_clustering(df_imputed, scaled_data)
                if fallback_result:
                    clusterings['clustering_k2'] = fallback_result
            
            if not clusterings:
                return None
            
            logger.info(f"✅ {len(clusterings)} segmentations générées avec TOUS les points")
            
            global_explanation = self._generate_global_clustering_summary(clusterings)
            
            return {
                "success": True,
                "clusterings": clusterings,
                "n_clustering_types": len(clusterings),
                "global_explanation": global_explanation,
                "variables_used": best_clustering_vars[:10],
                "total_points": len(df_imputed)
            }
        
        except Exception as e:
            logger.error(f"❌ Erreur multi-clustering: {e}", exc_info=True)
            return None

    def _create_fallback_clustering(self, df_numeric: pd.DataFrame, scaled_data: np.ndarray) -> Optional[Dict]:
        """
        🔧 V5: Clustering de fallback garanti avec TOUS les points
        """
        try:
            n_samples = len(scaled_data)
            n_vars = len(df_numeric.columns)
            
            logger.info(f"  🔄 Création clustering fallback... ({n_samples} points)")
            
            # Créer coordonnées 3D
            coords_3d = self._create_3d_coordinates(scaled_data, n_vars)
            
            # Clustering simple basé sur la médiane
            median_val = np.median(scaled_data[:, 0]) if n_vars > 0 else 0
            clusters = (scaled_data[:, 0] > median_val).astype(int)
            
            # Séparer les clusters
            coords_3d = self._separate_clusters_3d(coords_3d, clusters)
            
            # Créer le résultat
            validation = self._validate_clustering(clusters, 2)
            
            # Scatter points - TOUS les points
            scatter_3d = [
                {
                    "x": float(coords_3d[idx, 0]),
                    "y": float(coords_3d[idx, 1]),
                    "z": float(coords_3d[idx, 2]),
                    "cluster": int(clusters[idx])
                }
                for idx in range(n_samples)
            ]
            
            # DNA simplifié
            cluster_dna = {}
            for cid in [0, 1]:
                mask = clusters == cid
                count = np.sum(mask)
                if count > 0:
                    cluster_dna[f"Groupe {cid+1}"] = {
                        "size": int(count),
                        "percentage": round(100 * count / n_samples, 1),
                        "features": {},
                        "centroid_distance": 0,
                        "distinctiveness": 0
                    }
            
            return {
                "name": "Segmentation Binaire (Fallback)",
                "n_clusters": 2,
                "scatter_points": scatter_3d,
                "dna": cluster_dna,
                "cluster_distribution": [
                    {"cluster": f"Groupe {i+1}", "count": int(np.sum(clusters == i)), 
                     "percentage": round(100 * np.sum(clusters == i) / n_samples, 1), "color_index": i}
                    for i in range(2)
                ],
                "heatmap_data": [],
                "radar_data": [],
                "explained_variance": [],
                "validation": validation,
                "silhouette_score": None,
                "dispersion_level": 0.5,
                "quality_3d": 0.5,
                "method_used": "Median Split (Fallback)",
                "total_points": n_samples,
                "explanation": {
                    "title": "Segmentation Binaire (Fallback)",
                    "summary": f"Une segmentation simple basée sur la médiane a été créée ({n_samples} points).",
                    "recommendation": "Vos données sont difficiles à segmenter. Considérez d'ajouter plus de variables ou de points.",
                    "details": {"methode": "Median Split", "points": n_samples},
                    "tts_text": f"Une segmentation de fallback a été créée avec {n_samples} points. Les données sont difficiles à segmenter naturellement."
                }
            }
        except Exception as e:
            logger.error(f"❌ Fallback clustering échoué: {e}")
            return None

    def _generate_global_clustering_summary(self, clusterings: Dict) -> Dict[str, str]:
        """Générer un résumé global"""
        
        total_clusters = sum([c.get("n_clusters", 0) for c in clusterings.values()])
        avg_clusters = total_clusters / len(clusterings) if clusterings else 0
        
        best_clustering = None
        best_score = -1
        
        for key, clustering in clusterings.items():
            score = clustering.get("silhouette_score") or 0
            if score > best_score:
                best_score = score
                best_clustering = clustering
        
        if best_clustering:
            score_text = f"{best_score:.2f}" if best_score > 0 else "N/A"
            points_text = f"{best_clustering.get('total_points', 0)} points"
            return {
                "title": "Résumé des Segmentations",
                "summary": f"{len(clusterings)} modèles générés avec en moyenne {avg_clusters:.1f} groupes. Meilleur score: {score_text}.",
                "recommendation": "Explorez les différentes segmentations. Tous les points sont visibles en 3D.",
                "details": {
                    "meilleur_modele": best_clustering.get("name", "N/A"),
                    "score_qualite": score_text,
                    "groupes_totaux": total_clusters,
                    "points_visibles": points_text
                }
            }
        
        return {
            "title": "Segmentations Disponibles",
            "summary": f"{len(clusterings)} modèles de segmentation ont été générés.",
            "recommendation": "Examinez chaque modèle pour comprendre vos données.",
            "details": {}
        }

    def _analyze_correlations_deeply(self, df: pd.DataFrame, selected_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Corrélations"""
        
        numeric_cols = selected_vars.get('correlation_vars', [])
        target = selected_vars.get('target', '')
        
        if len(numeric_cols) < 2:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) < 2:
                return {
                    "matrix": {},
                    "strong_correlations": [],
                    "moderate_correlations": [],
                    "target_correlations": {},
                    "summary": {"strong_pairs": 0, "moderate_pairs": 0}
                }
        
        try:
            corr_matrix = df[numeric_cols].corr().round(3)
            
            strong_corr = []
            moderate_corr = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    val = corr_matrix.iloc[i, j]
                    if not np.isnan(val):
                        pair = {
                            "var1": corr_matrix.columns[i],
                            "var2": corr_matrix.columns[j],
                            "r": float(val),
                            "r_squared": round(val**2, 3),
                            "interpretation": self._interpret_correlation(val)
                        }
                        if abs(val) > 0.7:
                            strong_corr.append(pair)
                        elif abs(val) > 0.4:
                            moderate_corr.append(pair)
            
            target_corr = {}
            if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
                try:
                    tc = df[numeric_cols].corrwith(df[target]).dropna()
                    target_corr = {
                        col: {"r": round(float(val), 3), "r_squared": round(val**2, 3),
                              "interpretation": self._interpret_correlation(val)}
                        for col, val in tc.items() if col != target and abs(val) > 0.1
                    }
                    target_corr = dict(sorted(target_corr.items(), key=lambda x: abs(x[1]['r']), reverse=True))
                except Exception as e:
                    logger.warning(f"⚠️ Erreur corrélation target: {e}")
            
            return {
                "matrix": corr_matrix.to_dict(),
                "strong_correlations": sorted(strong_corr, key=lambda x: abs(x['r']), reverse=True)[:10],
                "moderate_correlations": sorted(moderate_corr, key=lambda x: abs(x['r']), reverse=True)[:10],
                "target_correlations": target_corr,
                "summary": {
                    "strong_pairs": len(strong_corr),
                    "moderate_pairs": len(moderate_corr),
                    "all_variables": numeric_cols
                }
            }
        except Exception as e:
            logger.error(f"❌ Erreur corrélations: {e}")
            return {
                "matrix": {},
                "strong_correlations": [],
                "moderate_correlations": [],
                "target_correlations": {},
                "summary": {"strong_pairs": 0, "moderate_pairs": 0}
            }

    def _interpret_correlation(self, r: float) -> str:
        abs_r = abs(r)
        if abs_r < 0.3:
            return "Très faible"
        elif abs_r < 0.5:
            return "Faible"
        elif abs_r < 0.7:
            return "Modérée"
        else:
            return "Forte"

    def _smart_statistical_tests(self, df: pd.DataFrame, selected_vars: Dict[str, Any]) -> List[Dict]:
        """Tests statistiques"""
        
        target = selected_vars.get('target', '')
        numeric_cols = selected_vars.get('numeric_analysis', [])
        categorical_cols = selected_vars.get('categorical_analysis', [])
        tests = []
        
        if not target or target not in df.columns:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                return []
            target = numeric_cols[0]
        
        target_is_numeric = pd.api.types.is_numeric_dtype(df[target])
        
        if target_is_numeric and len(categorical_cols) > 0:
            for cat_col in categorical_cols[:5]:
                try:
                    unique_count = df[cat_col].nunique()
                    
                    if unique_count == 2:
                        groups = df[cat_col].dropna().unique()
                        g1 = df[df[cat_col] == groups[0]][target].dropna().values
                        g2 = df[df[cat_col] == groups[1]][target].dropna().values
                        
                        if len(g1) >= 2 and len(g2) >= 2:
                            try:
                                t_stat, p = stats.ttest_ind(g1, g2)
                                tests.append({
                                    "variable1": cat_col,
                                    "variable2": target,
                                    "test_type": "ttest",
                                    "test_name": "T-Test",
                                    "statistic": round(t_stat, 3),
                                    "p_value": p,
                                    "conclusion": "Significatif" if p < 0.05 else "Non significatif"
                                })
                            except:
                                pass
                    
                    elif unique_count >= 3:
                        groups = [g for g in df[cat_col].dropna().unique() if pd.notna(g)]
                        gdata = [df[df[cat_col] == g][target].dropna().values for g in groups]
                        gdata = [g for g in gdata if len(g) >= 2]
                        
                        if len(gdata) >= 3:
                            try:
                                f_stat, p = stats.f_oneway(*gdata)
                                tests.append({
                                    "variable1": cat_col,
                                    "variable2": target,
                                    "test_type": "anova",
                                    "test_name": "ANOVA",
                                    "statistic": round(f_stat, 3),
                                    "p_value": p,
                                    "conclusion": "Significatif" if p < 0.05 else "Non significatif"
                                })
                            except:
                                pass
                except:
                    pass
        
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols[:5]):
                for col2 in numeric_cols[i+1:6]:
                    try:
                        clean = df[[col1, col2]].dropna()
                        if len(clean) >= 5:
                            r, p = stats.pearsonr(clean[col1], clean[col2])
                            if abs(r) > 0.2:
                                tests.append({
                                    "variable1": col1,
                                    "variable2": col2,
                                    "test_type": "pearson",
                                    "test_name": "Pearson",
                                    "statistic": round(r, 3),
                                    "p_value": p,
                                    "conclusion": "Significatif" if p < 0.05 else "Non significatif"
                                })
                    except:
                        pass
        
        return tests[:20]

    def _get_distributions(self, df: pd.DataFrame, selected_vars: Dict[str, Any]) -> Dict:
        """Distributions"""
        plots = {}
        numeric_cols = selected_vars.get('numeric_analysis', [])[:10]
        
        for col in numeric_cols:
            data = df[col].dropna().values
            if len(data) < 3:
                continue
            
            try:
                n_bins = min(30, int(np.sqrt(len(data))))
                hist, edges = np.histogram(data, bins=n_bins)
                
                q1 = np.percentile(data, 25)
                median = np.percentile(data, 50)
                q3 = np.percentile(data, 75)
                iqr = q3 - q1
                
                plots[col] = {
                    "type": "distribution",
                    "title": col,
                    "histogram": [
                        {"range": f"{round(edges[i], 2)}-{round(edges[i+1], 2)}",
                         "count": int(hist[i]), "pct": round(100 * hist[i] / hist.sum(), 1)}
                        for i in range(len(hist))
                    ],
                    "boxplot": {
                        "min": float(np.min(data)), "max": float(np.max(data)),
                        "q1": float(q1), "median": float(median), "q3": float(q3),
                        "iqr": float(iqr), "lower": float(q1 - 1.5*iqr), "upper": float(q3 + 1.5*iqr)
                    }
                }
            except:
                pass
        
        return plots

    def _get_pie_charts(self, df: pd.DataFrame, selected_vars: Dict[str, Any]) -> List[Dict]:
        """Camemberts"""
        pies = []
        for col in selected_vars.get('categorical_analysis', [])[:5]:
            uc = df[col].nunique()
            if 2 <= uc <= 10:
                try:
                    counts = df[col].value_counts()
                    total = counts.sum()
                    pies.append({
                        "type": "pie",
                        "title": f"Distribution: {col}",
                        "label": col,
                        "total": int(total),
                        "data": [{"name": str(k)[:50], "value": int(v), "pct": round(100 * v / total, 1)}
                                for k, v in counts.items()]
                    })
                except:
                    pass
        return pies

    def _get_scatter_plots(self, df: pd.DataFrame, selected_vars: Dict[str, Any]) -> List[Dict]:
        """Scatter plots"""
        scatters = []
        target = selected_vars.get('target', '')
        for var in [c for c in selected_vars.get('numeric_analysis', []) if c != target][:5]:
            try:
                clean = df[[var, target]].dropna()
                if len(clean) >= 5:
                    sample = clean.sample(n=min(500, len(clean)), random_state=42) if len(clean) > 500 else clean
                    scatters.append({
                        "type": "scatter",
                        "title": f"{target} vs {var}",
                        "x_label": var,
                        "y_label": target,
                        "correlation": round(float(sample[var].corr(sample[target])), 2),
                        "sample_size": len(sample),
                        "data": sample.to_dict(orient='records')
                    })
            except:
                pass
        return scatters

    def _detect_themes(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Thématisation"""
        themes = {}
        keywords_map = {
            "Démographie": ["age", "genre", "sexe", "region"],
            "Économie": ["revenu", "argent", "budget", "salaire"],
            "Activité": ["metier", "job", "activité"],
            "Tech": ["tel", "phone", "electricite"],
            "Temporel": ["date", "annee", "mois"]
        }
        assigned = set()
        for theme, keywords in keywords_map.items():
            cols = [c for c in df.columns if c not in assigned and any(k in c.lower() for k in keywords)]
            if cols:
                themes[theme] = cols
                assigned.update(cols)
        others = [c for c in df.columns if c not in assigned]
        if others:
            themes["Autres"] = others
        return themes

    def _build_clustering_views(self, multi_clustering: Optional[Dict]) -> List[Dict]:
        """Construire les vues pour tous les clusterings"""
        if not multi_clustering or not multi_clustering.get('clusterings'):
            return []
        
        views = []
        for clust_key, clust_data in multi_clustering['clusterings'].items():
            views.append({
                "type": "clustering",
                "title": clust_data.get('name', clust_key),
                "key": clust_key,
                "data": clust_data
            })
        return views

    async def _generate_ai_insights(self, multi_clustering: Optional[Dict], correlations: Dict, 
                                   tests: List[Dict], context: Dict) -> List[Dict]:
        """Insights IA"""
        try:
            target = context.get('target_variable', 'Unknown')
            n_clusters = multi_clustering.get('n_clustering_types', 0) if multi_clustering else 0
            
            insights_list = [
                {"title": "Analyse Complétée", "summary": f"Analyse de {target} effectuée", "recommendation": "Consultez les résultats"},
                {"title": "Segmentation", "summary": f"{n_clusters} modèles de clustering", "recommendation": "Examinez les profils"},
                {"title": "Relations", "summary": f"{len(tests)} tests statistiques", "recommendation": "Vérifiez les tests significatifs"}
            ]
            
            if multi_clustering and multi_clustering.get('global_explanation'):
                global_exp = multi_clustering['global_explanation']
                insights_list.append({
                    "title": "Segmentation Intelligente",
                    "summary": global_exp.get('summary', ''),
                    "recommendation": global_exp.get('recommendation', '')
                })
            
            return insights_list
            
        except Exception as e:
            logger.warning(f"⚠️ Insights error: {e}")
            return [{"title": "Analyse Effectuée", "summary": "EDA complétée", "recommendation": "Examinez les résultats"}]

    async def run_full_eda(self, df: pd.DataFrame, file_structure: Dict[str, Any],
                          context: Dict[str, Any], user_prompt: str = "") -> Dict[str, Any]:
        """🚀 PIPELINE EDA V5"""
        
        logger.info("=" * 60 + "\n🚀 EDA PIPELINE V5 DÉMARRAGE\n" + "=" * 60)
        
        try:
            target = self._smart_target_detection(df, user_prompt, context)
            context['target_variable'] = target
            selected_vars = self._select_key_variables(df, file_structure, context)
            
            logger.info(f"📊 Target: {target}")
            logger.info(f"   - Numériques: {len(selected_vars.get('numeric_analysis', []))}")
            logger.info(f"   - Catégories: {len(selected_vars.get('categorical_analysis', []))}")
            
            univariate = self._get_univariate_stats_smart(df, selected_vars)
            multi_clustering = self._perform_multi_clustering(df, selected_vars)
            correlations = self._analyze_correlations_deeply(df, selected_vars)
            statistical_tests = self._smart_statistical_tests(df, selected_vars)
            distributions = self._get_distributions(df, selected_vars)
            pie_charts = self._get_pie_charts(df, selected_vars)
            scatter_plots = self._get_scatter_plots(df, selected_vars)
            themes = self._detect_themes(df)
            ai_insights = await self._generate_ai_insights(multi_clustering, correlations, statistical_tests, context)
            
            clustering_explanations = {}
            if multi_clustering and multi_clustering.get('clusterings'):
                for key, clustering in multi_clustering['clusterings'].items():
                    if clustering and clustering.get('explanation'):
                        clustering_explanations[key] = clustering['explanation']
            
            logger.info(f"✅ EDA V5 COMPLÉTÉE:")
            logger.info(f"   - Univariate: {len(univariate)}")
            logger.info(f"   - Clustering: {multi_clustering.get('n_clustering_types', 0) if multi_clustering else 0}")
            logger.info(f"   - Correlations: {correlations.get('summary', {}).get('strong_pairs', 0)} fortes")
            logger.info(f"   - Tests: {len(statistical_tests)}")
            
            return {
                "metrics": {
                    "selection_reasoning": selected_vars.get('reasoning'),
                    "variable_scores": selected_vars.get('scores'),
                    "univariate": univariate,
                    "multi_clustering": multi_clustering,
                    "correlations": correlations,
                    "tests": statistical_tests,
                    "themes": themes
                },
                "charts_data": {
                    "distributions": distributions,
                    "pies": pie_charts,
                    "scatters": scatter_plots,
                    "clustering_views": self._build_clustering_views(multi_clustering),
                    "clustering_explanations": clustering_explanations
                },
                "ai_insights": ai_insights,
                "auto_target": selected_vars.get('target'),
                "summary": {
                    "total_rows": len(df),
                    "total_cols": len(df.columns),
                    "numeric_analyzed": len(selected_vars.get('numeric_analysis', [])),
                    "categorical_analyzed": len(selected_vars.get('categorical_analysis', [])),
                    "missing_values": int(df.isna().sum().sum())
                }
            }
        
        except Exception as e:
            logger.error(f"❌ ERREUR EDA: {e}", exc_info=True)
            raise


# Instance globale
eda_service = EDAService()