"""
🔧 FEATURE FORGE SERVICE - Ingénierie des Fonctionnalités Automatisée
Phase 5 du pipeline d'analyse de données

CORRECTIONS APPLIQUÉES:
- Évite les interactions entre une colonne et elle-même
- PCA conserve au minimum 80% de variance
- Meilleure sélection des colonnes pour interactions
- Limite raisonnable du nombre de features créées
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import asyncio
import json
import re
from scipy import stats
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, PowerTransformer
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression,
    mutual_info_classif, mutual_info_regression,
    VarianceThreshold
)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FeatureForgeService:
    """
    🔧 SERVICE DE FEATURE ENGINEERING AUTOMATISÉ
    Transforme les données brutes en features intelligentes pour le ML
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.pca_models = {}
        self.feature_importance = {}
        self.transformation_log = []
        
    # ==================== PHASE 5.1: CRÉATION DE NOUVELLES VARIABLES ====================
    
    def _detect_date_columns(self, df: pd.DataFrame) -> List[str]:
        """Détecte les colonnes de type date"""
        date_cols = []
        
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_cols.append(col)
                continue
                
            # Essayer de parser comme date
            if df[col].dtype == 'object':
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    try:
                        parsed = pd.to_datetime(sample, errors='coerce')
                        valid_ratio = parsed.notna().sum() / len(sample)
                        if valid_ratio > 0.8:
                            date_cols.append(col)
                    except:
                        pass
        
        return date_cols
    
    def _create_temporal_features(self, df: pd.DataFrame, date_cols: List[str]) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        📅 Crée des features temporelles à partir des colonnes de dates
        LIMITÉ à 5 colonnes de dates max pour éviter l'explosion de features
        """
        df_result = df.copy()
        created_features = []
        
        # Limiter le nombre de colonnes de dates traitées
        date_cols = date_cols[:5]
        
        for col in date_cols:
            try:
                # Convertir en datetime
                df_result[col] = pd.to_datetime(df_result[col], errors='coerce')
                
                if df_result[col].notna().sum() < 10:
                    continue
                
                base_name = col.replace(' ', '_').replace('/', '_').lower()[:20]
                
                # Extractions temporelles ESSENTIELLES seulement
                df_result[f"{base_name}_annee"] = df_result[col].dt.year
                df_result[f"{base_name}_mois"] = df_result[col].dt.month
                df_result[f"{base_name}_jour_semaine"] = df_result[col].dt.dayofweek
                df_result[f"{base_name}_is_weekend"] = (df_result[col].dt.dayofweek >= 5).astype(int)
                
                # Ancienneté (jours depuis aujourd'hui) - souvent utile
                today = pd.Timestamp.now()
                df_result[f"{base_name}_anciennete_jours"] = (today - df_result[col]).dt.days
                
                created_features.append({
                    "source": col,
                    "type": "temporal",
                    "features_created": [
                        f"{base_name}_annee", f"{base_name}_mois",
                        f"{base_name}_jour_semaine", f"{base_name}_is_weekend",
                        f"{base_name}_anciennete_jours"
                    ],
                    "count": 5
                })
                
                logger.info(f"📅 Features temporelles créées pour '{col}': 5 nouvelles variables")
                
            except Exception as e:
                logger.warning(f"⚠️ Erreur features temporelles {col}: {e}")
        
        return df_result, created_features
    
    def _create_interaction_features(self, df: pd.DataFrame, 
                                     numeric_cols: List[str],
                                     max_interactions: int = 10) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        🔗 Crée des features d'interaction entre variables numériques DISTINCTES
        
        ⚠️ CORRIGÉ: Évite les interactions d'une colonne avec elle-même
        """
        df_result = df.copy()
        created_features = []
        
        if len(numeric_cols) < 2:
            return df_result, created_features
        
        # Filtrer les colonnes avec des noms uniques et assez de données
        unique_cols = []
        seen_base_names = set()
        
        for col in numeric_cols:
            # Nettoyer le nom pour détecter les doublons
            clean_name = re.sub(r'[^a-z0-9]', '', col.lower())
            
            if clean_name in seen_base_names:
                continue
            if df[col].notna().sum() < 20:
                continue
            if df[col].std() < 0.01:
                continue
                
            seen_base_names.add(clean_name)
            unique_cols.append(col)
        
        if len(unique_cols) < 2:
            logger.info("🔗 Pas assez de colonnes numériques distinctes pour les interactions")
            return df_result, created_features
        
        # Sélectionner les meilleures colonnes par variance
        col_scores = []
        for col in unique_cols[:15]:
            try:
                variance = df[col].var()
                completeness = df[col].notna().mean()
                score = variance * completeness if variance > 0 else 0
                col_scores.append((col, score))
            except:
                pass
        
        col_scores.sort(key=lambda x: x[1], reverse=True)
        best_cols = [c[0] for c in col_scores[:6]]  # Top 6 max
        
        interactions_created = 0
        used_pairs = set()
        
        for i, col1 in enumerate(best_cols):
            for j, col2 in enumerate(best_cols):
                if i >= j:  # Éviter col1 == col2 et doublons (a,b) vs (b,a)
                    continue
                if interactions_created >= max_interactions:
                    break
                
                pair_key = tuple(sorted([col1, col2]))
                if pair_key in used_pairs:
                    continue
                used_pairs.add(pair_key)
                    
                try:
                    # Noms courts et uniques
                    base1 = re.sub(r'[^a-zA-Z0-9]', '', col1)[:10]
                    base2 = re.sub(r'[^a-zA-Z0-9]', '', col2)[:10]
                    
                    features_for_pair = []
                    
                    # Ratio (si col2 n'a pas trop de zéros)
                    non_zero_pct = (df_result[col2] != 0).mean()
                    if non_zero_pct > 0.5:
                        ratio_name = f"ratio_{base1}_vs_{base2}"
                        safe_col2 = df_result[col2].replace(0, np.nan)
                        df_result[ratio_name] = df_result[col1] / safe_col2
                        features_for_pair.append(ratio_name)
                    
                    # Différence normalisée
                    diff_name = f"diff_{base1}_vs_{base2}"
                    df_result[diff_name] = df_result[col1] - df_result[col2]
                    features_for_pair.append(diff_name)
                    
                    if features_for_pair:
                        created_features.append({
                            "source": [col1, col2],
                            "type": "interaction",
                            "features_created": features_for_pair,
                            "count": len(features_for_pair)
                        })
                        interactions_created += len(features_for_pair)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Erreur interaction {col1}/{col2}: {e}")
        
        logger.info(f"🔗 {interactions_created} features d'interaction créées entre {len(used_pairs)} paires distinctes")
        return df_result, created_features
    
    def _create_aggregation_features(self, df: pd.DataFrame,
                                     numeric_cols: List[str],
                                     categorical_cols: List[str]) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        📊 Crée des features d'agrégation par groupe
        LIMITÉ pour éviter l'explosion de features
        """
        df_result = df.copy()
        created_features = []
        
        if not categorical_cols or not numeric_cols:
            return df_result, created_features
        
        # Limiter à 2 catégories et 3 numériques
        cat_cols_limited = categorical_cols[:2]
        num_cols_limited = numeric_cols[:3]
        
        for cat_col in cat_cols_limited:
            unique_count = df[cat_col].nunique()
            
            if unique_count < 2 or unique_count > 30:
                continue
            
            for num_col in num_cols_limited:
                try:
                    base_cat = re.sub(r'[^a-zA-Z0-9]', '', cat_col)[:8]
                    base_num = re.sub(r'[^a-zA-Z0-9]', '', num_col)[:8]
                    
                    # Moyenne par groupe seulement
                    group_mean = df_result.groupby(cat_col)[num_col].transform('mean')
                    mean_col_name = f"{base_num}_mean_by_{base_cat}"
                    df_result[mean_col_name] = group_mean
                    
                    # Écart à la moyenne du groupe
                    ecart_col_name = f"{base_num}_ecart_{base_cat}"
                    df_result[ecart_col_name] = df_result[num_col] - group_mean
                    
                    created_features.append({
                        "source": [cat_col, num_col],
                        "type": "aggregation",
                        "features_created": [mean_col_name, ecart_col_name],
                        "count": 2
                    })
                    
                except Exception as e:
                    logger.warning(f"⚠️ Erreur agrégation {cat_col}/{num_col}: {e}")
        
        logger.info(f"📊 Features d'agrégation créées: {len(created_features)} groupes")
        return df_result, created_features
    
    # ==================== PHASE 5.2: ENCODAGE DES VARIABLES CATÉGORIELLES ====================
    
    def _smart_encode_categorical(self, df: pd.DataFrame,
                                  categorical_cols: List[str],
                                  target_col: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        🔢 Encodage intelligent des variables catégorielles
        AMÉLIORÉ: Limite le nombre de colonnes créées
        """
        df_result = df.copy()
        encoding_info = {
            "one_hot": [],
            "label": [],
            "target": [],
            "skipped": []
        }
        
        # Limiter le nombre de colonnes à encoder
        categorical_cols = categorical_cols[:15]
        
        for col in categorical_cols:
            try:
                unique_count = df[col].nunique()
                
                if unique_count > 50:
                    encoding_info["skipped"].append({
                        "column": col,
                        "reason": f"Cardinalité trop élevée ({unique_count})"
                    })
                    continue
                
                # ONE-HOT ENCODING pour faible cardinalité (< 8)
                if unique_count <= 8:
                    dummies = pd.get_dummies(df_result[col], prefix=col[:12], dummy_na=False)
                    
                    # Limiter à 10 colonnes max
                    if len(dummies.columns) > 10:
                        top_cats = df[col].value_counts().head(8).index
                        dummies = dummies[[c for c in dummies.columns if any(str(cat) in c for cat in top_cats)]][:10]
                    
                    df_result = pd.concat([df_result, dummies], axis=1)
                    df_result.drop(columns=[col], inplace=True, errors='ignore')
                    
                    encoding_info["one_hot"].append({
                        "column": col,
                        "unique_values": unique_count,
                        "columns_created": list(dummies.columns)
                    })
                    
                # LABEL ENCODING pour cardinalité moyenne
                else:
                    le = LabelEncoder()
                    mask = df_result[col].notna()
                    new_col_name = f"{col[:15]}_encoded"
                    df_result[new_col_name] = -1
                    df_result.loc[mask, new_col_name] = le.fit_transform(df_result.loc[mask, col].astype(str))
                    
                    self.encoders[col] = le
                    df_result.drop(columns=[col], inplace=True, errors='ignore')
                    
                    encoding_info["label"].append({
                        "column": col,
                        "unique_values": unique_count,
                        "mapping_size": len(le.classes_)
                    })
                    
            except Exception as e:
                logger.warning(f"⚠️ Erreur encodage {col}: {e}")
                encoding_info["skipped"].append({
                    "column": col,
                    "reason": str(e)
                })
        
        return df_result, encoding_info
    
    # ==================== PHASE 5.3: MISE À L'ÉCHELLE (SCALING) ====================
    
    def _smart_scale_features(self, df: pd.DataFrame,
                              numeric_cols: List[str],
                              method: str = "auto") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        📏 Mise à l'échelle intelligente des variables numériques
        CORRIGÉ: Ne scale que les colonnes qui en ont besoin
        """
        df_result = df.copy()
        scaling_info = {
            "standard": [],
            "minmax": [],
            "robust": [],
            "skipped": []
        }
        
        # Limiter le nombre de colonnes scalées
        cols_to_scale = []
        for col in numeric_cols[:50]:  # Max 50 colonnes
            if col not in df_result.columns:
                continue
            if col.endswith('_scaled'):
                continue
            
            data = df_result[col].dropna()
            if len(data) < 20:
                continue
                
            # Ne pas scaler si déjà bien distribué (0-1 ou standardisé)
            if data.min() >= 0 and data.max() <= 1:
                continue
            if abs(data.mean()) < 0.5 and abs(data.std() - 1) < 0.5:
                continue
                
            cols_to_scale.append(col)
        
        for col in cols_to_scale[:30]:  # Limiter à 30
            try:
                data = df_result[col].dropna()
                
                # Détecter la meilleure méthode
                if method == "auto":
                    skewness = abs(data.skew()) if len(data) > 20 else 0
                    
                    # Calculer le % d'outliers
                    q1, q3 = data.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    if iqr > 0:
                        outliers_pct = ((data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)).mean()
                    else:
                        outliers_pct = 0
                    
                    if outliers_pct > 0.1:
                        chosen_method = "robust"
                    elif skewness < 1:
                        chosen_method = "standard"
                    else:
                        chosen_method = "minmax"
                else:
                    chosen_method = method
                
                # Appliquer le scaling
                col_data = df_result[[col]].copy()
                imputer = SimpleImputer(strategy='median')
                col_imputed = imputer.fit_transform(col_data)
                
                if chosen_method == "standard":
                    scaler = StandardScaler()
                elif chosen_method == "minmax":
                    scaler = MinMaxScaler()
                else:
                    scaler = RobustScaler()
                
                scaled = scaler.fit_transform(col_imputed)
                df_result[f"{col}_scaled"] = scaled.flatten()
                self.scalers[col] = scaler
                scaling_info[chosen_method].append(col)
                
            except Exception as e:
                scaling_info["skipped"].append({"column": col, "reason": str(e)})
        
        total_scaled = len(scaling_info["standard"]) + len(scaling_info["minmax"]) + len(scaling_info["robust"])
        logger.info(f"📏 Scaling appliqué: {total_scaled} variables")
        
        return df_result, scaling_info
    
    # ==================== PHASE 5.4: TRANSFORMATIONS MATHÉMATIQUES ====================
    
    def _apply_transformations(self, df: pd.DataFrame,
                               numeric_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        🔄 Applique des transformations pour normaliser les distributions
        LIMITÉ aux colonnes qui en bénéficient vraiment
        """
        df_result = df.copy()
        transform_info = {
            "log": [],
            "sqrt": [],
            "boxcox": [],
            "skipped": []
        }
        
        transforms_applied = 0
        max_transforms = 10  # Limiter
        
        for col in numeric_cols:
            if transforms_applied >= max_transforms:
                break
            if col not in df_result.columns:
                continue
                
            try:
                data = df_result[col].dropna()
                
                if len(data) < 30:
                    continue
                
                skewness = data.skew()
                min_val = data.min()
                
                # LOG TRANSFORM pour skew très positif et valeurs positives
                if skewness > 2.0 and min_val > 0:
                    new_col = f"{col[:15]}_log"
                    df_result[new_col] = np.log1p(df_result[col])
                    transform_info["log"].append({
                        "column": col,
                        "original_skew": round(skewness, 2),
                        "new_skew": round(df_result[new_col].skew(), 2)
                    })
                    transforms_applied += 1
                        
            except Exception as e:
                transform_info["skipped"].append({"column": col, "reason": str(e)})
        
        logger.info(f"🔄 Transformations: {transforms_applied} variables transformées")
        return df_result, transform_info
    
    # ==================== PHASE 5.5: RÉDUCTION DE DIMENSIONNALITÉ ====================
    
    def _apply_pca(self, df: pd.DataFrame,
                   numeric_cols: List[str],
                   variance_threshold: float = 0.80,  # ⚠️ CORRIGÉ: 80% minimum
                   max_components: int = 20) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        📉 Réduction de dimensionnalité par ACP
        
        ⚠️ CORRIGÉ: Conserve au minimum 80% de la variance
        """
        df_result = df.copy()
        pca_info = {
            "applied": False,
            "original_features": 0,
            "components_kept": 0,
            "variance_explained": [],
            "cumulative_variance": [],
            "feature_importance": {},
            "reason": ""
        }
        
        # Filtrer les colonnes numériques valides
        valid_cols = [c for c in numeric_cols if c in df.columns and 
                     pd.api.types.is_numeric_dtype(df[c]) and
                     df[c].notna().sum() > 20]
        
        if len(valid_cols) < 10:
            pca_info["reason"] = "Pas assez de variables numériques (< 10)"
            return df_result, pca_info
        
        try:
            X = df_result[valid_cols].copy()
            
            # Imputer et normaliser
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
            
            # PCA
            n_components = min(len(valid_cols), max_components, len(df) - 1)
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)
            
            # ⚠️ CORRIGÉ: Conserver au minimum 80% de variance
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            n_keep = np.argmax(cumsum >= variance_threshold) + 1
            n_keep = max(n_keep, 5)  # Minimum 5 composantes
            n_keep = min(n_keep, n_components)  # Ne pas dépasser le max
            
            # Vérifier que la variance conservée est suffisante
            variance_kept = cumsum[n_keep-1]
            if variance_kept < 0.6:
                # Si trop peu de variance, garder plus de composantes
                n_keep = min(np.argmax(cumsum >= 0.9) + 1, n_components)
            
            # Créer les colonnes PCA
            for i in range(n_keep):
                df_result[f"PCA_{i+1}"] = X_pca[:, i]
            
            self.pca_models['main'] = pca
            
            pca_info.update({
                "applied": True,
                "original_features": len(valid_cols),
                "components_kept": n_keep,
                "variance_explained": [round(v, 4) for v in pca.explained_variance_ratio_[:n_keep]],
                "cumulative_variance": [round(v, 4) for v in cumsum[:n_keep]],
                "total_variance_captured": round(cumsum[n_keep-1], 4)
            })
            
            logger.info(f"📉 PCA: {len(valid_cols)} → {n_keep} composantes ({pca_info['total_variance_captured']*100:.1f}% variance)")
            
        except Exception as e:
            pca_info["reason"] = f"Erreur PCA: {str(e)}"
            logger.error(f"❌ Erreur PCA: {e}")
        
        return df_result, pca_info
    
    # ==================== PHASE 5.6: FEATURE SELECTION ====================
    
    def _select_best_features(self, df: pd.DataFrame,
                              target_col: Optional[str],
                              numeric_cols: List[str],
                              max_features: int = 50) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        🎯 Sélection des meilleures features
        """
        df_result = df.copy()
        selection_info = {
            "original_features": 0,
            "selected_features": [],
            "removed_low_variance": [],
            "removed_redundant": [],
            "feature_scores": {},
            "method_used": ""
        }
        
        valid_cols = [c for c in numeric_cols if c in df.columns and df[c].notna().sum() > 10]
        selection_info["original_features"] = len(valid_cols)
        
        if len(valid_cols) < 5:
            selection_info["selected_features"] = valid_cols
            return df_result, selection_info
        
        try:
            X = df_result[valid_cols].copy()
            
            imputer = SimpleImputer(strategy='median')
            X_imputed = pd.DataFrame(
                imputer.fit_transform(X),
                columns=valid_cols,
                index=X.index
            )
            
            # 1. VARIANCE THRESHOLD - supprimer les quasi-constantes
            vt = VarianceThreshold(threshold=0.01)
            vt.fit(X_imputed)
            low_variance = [valid_cols[i] for i, keep in enumerate(vt.get_support()) if not keep]
            selection_info["removed_low_variance"] = low_variance
            
            remaining_cols = [c for c in valid_cols if c not in low_variance]
            
            # 2. SUPPRESSION DES FEATURES REDONDANTES (corrélation > 0.95)
            if len(remaining_cols) > 5:
                corr_matrix = X_imputed[remaining_cols].corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                
                redundant = []
                for col in upper.columns:
                    if any(upper[col] > 0.95):
                        redundant.append(col)
                
                # Limiter les suppressions
                redundant = redundant[:len(remaining_cols)//3]  # Max 1/3 des colonnes
                selection_info["removed_redundant"] = redundant
                remaining_cols = [c for c in remaining_cols if c not in redundant]
            
            # Limiter au max
            remaining_cols = remaining_cols[:max_features]
            selection_info["selected_features"] = remaining_cols
            selection_info["method_used"] = "variance_correlation"
            
            logger.info(f"🎯 Feature Selection: {len(valid_cols)} → {len(remaining_cols)} features")
            
        except Exception as e:
            logger.error(f"❌ Erreur feature selection: {e}")
            selection_info["selected_features"] = valid_cols[:max_features]
        
        return df_result, selection_info
    
    # ==================== MÉTHODE PRINCIPALE ====================
    
    async def forge_features(self, df: pd.DataFrame,
                             context: Dict[str, Any],
                             options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        🔧 PIPELINE COMPLET DE FEATURE ENGINEERING
        
        ⚠️ CORRIGÉ: Limite raisonnable du nombre de features
        """
        
        logger.info("=" * 60)
        logger.info("🔧 FEATURE FORGE - DÉMARRAGE")
        logger.info("=" * 60)
        
        options = options or {}
        self.transformation_log = []
        
        target_col = context.get('target_variable')
        
        # Colonnes initiales
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        result = {
            "success": True,
            "original_shape": df.shape,
            "final_shape": None,
            "transformations": {},
            "recommendations": [],
            "feature_summary": {}
        }
        
        df_transformed = df.copy()
        
        try:
            # PHASE 1: Features temporelles (LIMITÉ)
            logger.info("📅 Phase 1: Features temporelles...")
            date_cols = self._detect_date_columns(df_transformed)
            if date_cols:
                df_transformed, temporal_info = self._create_temporal_features(df_transformed, date_cols[:3])
                result["transformations"]["temporal"] = temporal_info
            
            # PHASE 2: Features d'interaction (CORRIGÉ)
            logger.info("🔗 Phase 2: Features d'interaction...")
            if options.get("create_interactions", True) and len(numeric_cols) >= 2:
                df_transformed, interaction_info = self._create_interaction_features(
                    df_transformed, numeric_cols, max_interactions=options.get("max_interactions", 10)
                )
                result["transformations"]["interactions"] = interaction_info
            
            # PHASE 3: Features d'agrégation (LIMITÉ)
            logger.info("📊 Phase 3: Features d'agrégation...")
            if options.get("create_aggregations", True) and categorical_cols:
                df_transformed, agg_info = self._create_aggregation_features(
                    df_transformed, numeric_cols[:5], categorical_cols[:3]
                )
                result["transformations"]["aggregations"] = agg_info
            
            # PHASE 4: Encodage catégoriel (LIMITÉ)
            logger.info("🔢 Phase 4: Encodage catégoriel...")
            remaining_cat = [c for c in categorical_cols if c in df_transformed.columns][:10]
            if remaining_cat:
                df_transformed, encoding_info = self._smart_encode_categorical(
                    df_transformed, remaining_cat, target_col
                )
                result["transformations"]["encoding"] = encoding_info
            
            # Mettre à jour les colonnes numériques
            numeric_cols = df_transformed.select_dtypes(include=np.number).columns.tolist()
            
            # PHASE 5: Transformations mathématiques (LIMITÉ)
            logger.info("🔄 Phase 5: Transformations mathématiques...")
            if options.get("apply_transforms", True):
                df_transformed, transform_info = self._apply_transformations(
                    df_transformed, numeric_cols[:20]
                )
                result["transformations"]["mathematical"] = transform_info
            
            # PHASE 6: Scaling (LIMITÉ)
            logger.info("📏 Phase 6: Mise à l'échelle...")
            if options.get("apply_scaling", True):
                scaling_cols = [c for c in numeric_cols if not c.endswith('_scaled')][:30]
                df_transformed, scaling_info = self._smart_scale_features(
                    df_transformed, scaling_cols, method=options.get("scaling_method", "auto")
                )
                result["transformations"]["scaling"] = scaling_info
            
            # PHASE 7: PCA (CORRIGÉ - 80% variance min)
            logger.info("📉 Phase 7: Réduction de dimensionnalité...")
            numeric_for_pca = df_transformed.select_dtypes(include=np.number).columns.tolist()
            if options.get("apply_pca", False) and len(numeric_for_pca) > 20:  # Désactivé par défaut
                df_transformed, pca_info = self._apply_pca(
                    df_transformed, numeric_for_pca,
                    variance_threshold=options.get("pca_variance", 0.80)
                )
                result["transformations"]["pca"] = pca_info
            
            # PHASE 8: Feature Selection
            logger.info("🎯 Phase 8: Sélection des features...")
            if options.get("feature_selection", True):
                final_numeric = df_transformed.select_dtypes(include=np.number).columns.tolist()
                df_transformed, selection_info = self._select_best_features(
                    df_transformed, target_col, final_numeric,
                    max_features=options.get("max_features", 100)
                )
                result["transformations"]["selection"] = selection_info
            
            # Résumé final
            result["final_shape"] = df_transformed.shape
            result["df_transformed"] = df_transformed
            
            # Feature summary
            final_cols = df_transformed.columns.tolist()
            features_created = max(0, df_transformed.shape[1] - df.shape[1])
            
            result["feature_summary"] = {
                "total_features": len(final_cols),
                "numeric_features": len(df_transformed.select_dtypes(include=np.number).columns),
                "categorical_features": len(df_transformed.select_dtypes(include=['object', 'category']).columns),
                "features_created": features_created,
                "columns": final_cols[:100]  # Limiter la liste
            }
            
            # Générer recommandations
            result["recommendations"] = self._generate_recommendations(result)
            
            logger.info("=" * 60)
            logger.info(f"✅ FEATURE FORGE TERMINÉ")
            logger.info(f"   Shape: {df.shape} → {df_transformed.shape}")
            logger.info(f"   Features créées: {features_created}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Erreur Feature Forge: {e}", exc_info=True)
            result["success"] = False
            result["error"] = str(e)
            result["df_transformed"] = df
        
        return result
    
    def _generate_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """Génère des recommandations basées sur les transformations"""
        recommendations = []
        
        transforms = result.get("transformations", {})
        
        if transforms.get("temporal"):
            recommendations.append("✅ Features temporelles créées - utilisez-les pour analyser les tendances")
        
        if transforms.get("pca", {}).get("applied"):
            pca_info = transforms["pca"]
            variance = pca_info.get('total_variance_captured', 0) * 100
            recommendations.append(
                f"📉 PCA appliquée: {pca_info.get('original_features', 0)} → {pca_info.get('components_kept', 0)} composantes "
                f"({variance:.1f}% variance conservée)"
            )
        
        selection = transforms.get("selection", {})
        if selection.get("removed_low_variance"):
            recommendations.append(
                f"🗑️ {len(selection['removed_low_variance'])} variables quasi-constantes supprimées"
            )
        
        if selection.get("removed_redundant"):
            recommendations.append(
                f"🔗 {len(selection['removed_redundant'])} variables redondantes supprimées"
            )
        
        return recommendations


# Instance globale
feature_forge_service = FeatureForgeService()