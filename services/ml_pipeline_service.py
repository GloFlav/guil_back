"""
🤖 ML PIPELINE SERVICE V4 - AVEC EXPLICATIONS LLM INTÉGRÉES
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import asyncio
import json
from datetime import datetime
from scipy import stats

# Sklearn imports
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold

# Modèles de Régression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Modèles de Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Modèles de Clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# Métriques
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    mean_absolute_percentage_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MLPipelineService:
    """🤖 SERVICE DE MODÉLISATION ML V4 - AVEC EXPLICATIONS LLM"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_importances = {}
        self.predictions = {}
        self.metrics_history = []
        self.llm_explainer = None
        
        # Initialiser le LLM Explainer
        try:
            from services.llm_explainer_service import llm_explainer_service
            self.llm_explainer = llm_explainer_service
            logger.info("✅ LLM Explainer Service chargé")
        except Exception as e:
            logger.warning(f"⚠️ LLM Explainer non disponible: {e}")
        
        # Modèles avec régularisation par défaut
        self.regression_models = {
            "linear_regression": {
                "model": LinearRegression,
                "params": {},
                "tuning_params": {}
            },
            "ridge": {
                "model": Ridge,
                "params": {"alpha": 1.0},
                "tuning_params": {"alpha": [0.1, 1.0, 10.0, 100.0]}
            },
            "lasso": {
                "model": Lasso,
                "params": {"alpha": 0.1},
                "tuning_params": {"alpha": [0.001, 0.01, 0.1, 1.0]}
            },
            "decision_tree": {
                "model": DecisionTreeRegressor,
                "params": {"max_depth": 5, "random_state": 42},
                "tuning_params": {"max_depth": [3, 5, 7]}
            },
            "random_forest": {
                "model": RandomForestRegressor,
                "params": {"n_estimators": 100, "max_depth": 5, "random_state": 42, "n_jobs": -1},
                "tuning_params": {"n_estimators": [50, 100], "max_depth": [3, 5, 7]}
            },
            "gradient_boosting": {
                "model": GradientBoostingRegressor,
                "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
                "tuning_params": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]}
            }
        }
        
        self.classification_models = {
            "logistic_regression": {
                "model": LogisticRegression,
                "params": {"C": 0.1, "penalty": 'l2', "max_iter": 1000, "random_state": 42},
                "tuning_params": {"C": [0.01, 0.1, 1.0], "penalty": ['l2']}
            },
            "decision_tree": {
                "model": DecisionTreeClassifier,
                "params": {"max_depth": 5, "random_state": 42},
                "tuning_params": {"max_depth": [3, 5, 7], "min_samples_split": [2, 5, 10]}
            },
            "random_forest": {
                "model": RandomForestClassifier,
                "params": {"n_estimators": 100, "max_depth": 5, "random_state": 42, "n_jobs": -1},
                "tuning_params": {"n_estimators": [50, 100], "max_depth": [3, 5, 7]}
            },
            "svm": {
                "model": SVC,
                "params": {"C": 0.1, "kernel": "linear", "probability": True, "random_state": 42},
                "tuning_params": {"C": [0.1, 1.0], "kernel": ["linear", "rbf"]}
            }
        }
    
    # ==================== DÉTECTION D'OVERFITTING ====================
    
    def _detect_overfitting_risks(self, df: pd.DataFrame, target_col: Optional[str], 
                                   X_cols: List[str]) -> Dict[str, Any]:
        """Détecte les risques d'overfitting avant l'entraînement"""
        risks = {
            "warnings": [],
            "high_risk": False,
            "recommendations": [],
            "details": {}
        }
        
        n_samples = len(df)
        n_features = len(X_cols)
        
        # 1. Ratio Features/Samples
        ratio = n_features / n_samples if n_samples > 0 else float('inf')
        risks["details"]["ratio"] = ratio
        risks["details"]["n_samples"] = n_samples
        risks["details"]["n_features"] = n_features
        
        if ratio > 1:
            risks["high_risk"] = True
            risks["warnings"].append(
                f"Ratio Features/Samples: {ratio:.2f} (> 1). Risque d'overfitting très élevé."
            )
            risks["recommendations"].append(
                f"Réduire les features à maximum {min(50, n_samples // 5)}"
            )
        elif ratio > 0.2:
            risks["warnings"].append(
                f"Ratio Features/Samples: {ratio:.2f} (> 0.2 recommandé)"
            )
            risks["recommendations"].append(
                "Considérez la réduction de dimensionnalité (PCA, SelectKBest)"
            )
        
        # 2. Vérifier la fuite de données
        if target_col and target_col in X_cols:
            risks["high_risk"] = True
            risks["warnings"].append(
                f"⚠️ FUITE DE DONNÉES: La variable cible '{target_col}' est dans les features!"
            )
            risks["recommendations"].append(
                f"Retirer '{target_col}' des features d'entraînement"
            )
        
        # 3. Variables quasi-constantes
        constant_vars = []
        for col in X_cols:
            if col in df.columns and df[col].nunique() <= 1:
                constant_vars.append(col)
        
        if constant_vars:
            risks["warnings"].append(
                f"{len(constant_vars)} variables quasi-constantes détectées"
            )
            risks["recommendations"].append(
                "Supprimer les variables avec variance nulle"
            )
            risks["details"]["constant_vars"] = constant_vars
        
        # 4. Déséquilibre des classes
        if target_col and target_col in df.columns:
            target_counts = df[target_col].dropna().value_counts()
            if len(target_counts) > 0:
                min_class = target_counts.min()
                max_class = target_counts.max()
                imbalance_ratio = min_class / max_class if max_class > 0 else 0
                
                risks["details"]["class_imbalance"] = imbalance_ratio
                
                if imbalance_ratio < 0.1:
                    risks["warnings"].append(
                        f"Déséquilibre de classes sévère: ratio {imbalance_ratio:.2f}"
                    )
                    risks["recommendations"].append(
                        "Utiliser class_weight='balanced' ou SMOTE"
                    )
        
        return risks
    
    def _auto_reduce_features(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                               max_features: int = 50) -> Tuple[np.ndarray, List[str]]:
        """Réduction automatique des features"""
        n_features = X.shape[1]
        feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f"feature_{i}" for i in range(n_features)]
        
        if n_features <= max_features:
            return X.values if hasattr(X, 'values') else X, feature_names
        
        logger.info(f"🔧 Réduction automatique: {n_features} → {max_features} features")
        
        try:
            # 1. Supprimer les features quasi-constantes
            selector = VarianceThreshold(threshold=0.01)
            X_reduced = selector.fit_transform(X)
            selected_mask = selector.get_support()
            feature_names = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
            logger.info(f"  Après VarianceThreshold: {X_reduced.shape[1]} features")
            
            # 2. Sélection des meilleures features
            if y is not None and X_reduced.shape[1] > max_features:
                selector = SelectKBest(f_classif, k=min(max_features, X_reduced.shape[1]))
                X_reduced = selector.fit_transform(X_reduced, y)
                selected_mask = selector.get_support()
                feature_names = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
                logger.info(f"  Après SelectKBest: {X_reduced.shape[1]} features")
            
            # 3. Troncature si nécessaire
            if X_reduced.shape[1] > max_features:
                X_reduced = X_reduced[:, :max_features]
                feature_names = feature_names[:max_features]
                logger.info(f"  Après troncature: {X_reduced.shape[1]} features")
            
            return X_reduced, feature_names
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur réduction features: {e}")
            X_array = X.values if hasattr(X, 'values') else X
            return X_array[:, :max_features], feature_names[:max_features]
    
    # ==================== PRÉPARATION DES DONNÉES ====================
    
    def _prepare_data(self, df: pd.DataFrame, 
                      target_col: Optional[str],
                      feature_cols: Optional[List[str]] = None,
                      test_size: float = 0.2,
                      val_size: float = 0.2) -> Dict[str, Any]:
        """📊 Prépare les données avec détection d'overfitting"""
        result = {
            "success": True,
            "problem_type": None,
            "X_train": None, "X_val": None, "X_test": None,
            "y_train": None, "y_val": None, "y_test": None,
            "feature_names": [],
            "scaler": None,
            "label_encoder": None,
            "class_names": None,
            "warning": None,
            "validation_info": {},
            "overfitting_risks": {}
        }
        
        try:
            # Déterminer les features
            if feature_cols:
                X_cols = [c for c in feature_cols if c in df.columns and c != target_col]
            else:
                X_cols = [c for c in df.select_dtypes(include=np.number).columns 
                         if c != target_col]
            
            if len(X_cols) == 0:
                result["success"] = False
                result["error"] = "Aucune feature numérique disponible"
                return result
            
            # Détection des risques d'overfitting
            overfitting_risks = self._detect_overfitting_risks(df, target_col, X_cols)
            result["overfitting_risks"] = overfitting_risks
            
            # Créer X
            X = df[X_cols].copy()
            
            # Imputer les NaN
            imputer = SimpleImputer(strategy='median')
            X_imputed = pd.DataFrame(
                imputer.fit_transform(X),
                columns=X_cols,
                index=X.index
            )
            
            # Réduction automatique si trop de features
            n_samples = len(X_imputed)
            if len(X_cols) > 100 or len(X_cols) > n_samples:
                max_features = min(50, n_samples // 5)
                y_for_selection = df[target_col] if target_col and target_col in df.columns else None
                X_reduced, feature_names = self._auto_reduce_features(X_imputed, y_for_selection, max_features)
                result["feature_names"] = feature_names
                logger.info(f"✅ Features réduites: {len(X_cols)} → {len(feature_names)}")
            else:
                X_reduced = X_imputed.values
                result["feature_names"] = X_cols
            
            # Scaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_reduced)
            result["scaler"] = scaler
            
            # Préparer y si target existe
            if target_col and target_col in df.columns:
                y = df[target_col].copy()
                
                # Validation de la cible
                result["validation_info"] = self._validate_target(y, target_col)
                
                if not result["validation_info"]["is_valid"]:
                    result["success"] = False
                    result["error"] = result["validation_info"]["reason"]
                    return result
                
                # Aligner les indices
                valid_idx = y.notna()
                X_final = X_scaled[valid_idx.values]
                y_final = y[valid_idx].values
                
                # Détecter le type de problème
                problem_type = self._detect_problem_type(df, target_col)
                result["problem_type"] = problem_type
                
                # Encoder y si classification
                if "classification" in problem_type:
                    le = LabelEncoder()
                    y_final = le.fit_transform(y_final.astype(str))
                    result["label_encoder"] = le
                    result["class_names"] = le.classes_.tolist()
                
                # Split train/val/test
                X_trainval, X_test, y_trainval, y_test = train_test_split(
                    X_final, y_final, test_size=test_size, random_state=42,
                    stratify=y_final if "classification" in problem_type else None
                )
                
                val_ratio = val_size / (1 - test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_trainval, y_trainval, test_size=val_ratio, random_state=42,
                    stratify=y_trainval if "classification" in problem_type else None
                )
                
                result.update({
                    "X_train": X_train, "X_val": X_val, "X_test": X_test,
                    "y_train": y_train, "y_val": y_val, "y_test": y_test,
                    "train_size": len(X_train),
                    "val_size": len(X_val),
                    "test_size": len(X_test)
                })
                
                logger.info(f"✅ Données préparées: {problem_type}, "
                           f"train/val/test: {len(X_train)}/{len(X_val)}/{len(X_test)}")
                
            else:
                result["problem_type"] = "clustering"
                result["X_train"] = X_scaled
                result["train_size"] = len(X_scaled)
            
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            logger.error(f"❌ Erreur préparation données: {e}")
        
        return result
    
    def _validate_target(self, y: pd.Series, target_col: str) -> Dict[str, Any]:
        """Valide la variable cible"""
        y_clean = y.dropna()
        n_samples = len(y_clean)
        
        if n_samples < 20:
            return {
                "is_valid": False,
                "reason": f"Trop peu d'échantillons non-nuls: {n_samples} (minimum 20)",
                "recommendation": "Choisissez une autre variable cible"
            }
        
        unique_values = y_clean.nunique()
        
        if unique_values < 2:
            return {
                "is_valid": False,
                "reason": f"La variable '{target_col}' n'a qu'une seule valeur unique",
                "recommendation": "Choisissez une variable avec au moins 2 valeurs"
            }
        
        if unique_values <= 10:
            value_counts = y_clean.value_counts()
            min_count = value_counts.min()
            
            if min_count < 5:
                return {
                    "is_valid": False,
                    "reason": f"Classe minoritaire trop petite: {min_count} échantillons",
                    "recommendation": "Choisissez une variable cible plus équilibrée"
                }
        
        return {
            "is_valid": True,
            "reason": f"OK: {unique_values} classes, {n_samples} échantillons",
            "n_classes": unique_values,
            "n_samples": n_samples
        }
    
    def _detect_problem_type(self, df: pd.DataFrame, target_col: str) -> str:
        """Détecte automatiquement le type de problème ML"""
        if not target_col or target_col not in df.columns:
            return "clustering"
        
        target = df[target_col].dropna()
        
        if len(target) == 0:
            return "clustering"
        
        if target.dtype == 'object' or target.dtype.name == 'category':
            return "classification"
        
        n_unique = target.nunique()
        n_total = len(target)
        
        if n_unique <= 2:
            return "binary_classification"
        elif n_unique <= 10:
            return "multiclass_classification"
        elif n_unique / n_total < 0.05:
            return "classification"
        else:
            return "regression"
    
    # ==================== ENTRAÎNEMENT ====================
    
    def _train_classification_models(self, X_train: np.ndarray, y_train: np.ndarray,
                                     X_val: np.ndarray, y_val: np.ndarray,
                                     tune_hyperparams: bool = False,
                                     is_multiclass: bool = False) -> Dict[str, Any]:
        """Entraîne les modèles de classification"""
        results = {}
        
        for name, config in self.classification_models.items():
            try:
                logger.info(f"  🔄 Entraînement {name}...")
                
                params = config["params"].copy()
                
                # Ajouter class_weight si déséquilibre
                unique, counts = np.unique(y_train, return_counts=True)
                if len(unique) > 1:
                    imbalance_ratio = counts.min() / counts.max()
                    if imbalance_ratio < 0.3 and "class_weight" not in params:
                        if hasattr(config["model"], 'class_weight'):
                            params["class_weight"] = 'balanced'
                
                model = config["model"](**params)
                
                if tune_hyperparams and config["tuning_params"]:
                    try:
                        grid_search = GridSearchCV(
                            model, config["tuning_params"],
                            cv=3, scoring='f1_weighted' if is_multiclass else 'f1', 
                            n_jobs=-1, error_score='raise'
                        )
                        grid_search.fit(X_train, y_train)
                        model = grid_search.best_estimator_
                        best_params = grid_search.best_params_
                    except Exception as e:
                        logger.warning(f"    GridSearch échoué pour {name}: {e}")
                        model.fit(X_train, y_train)
                        best_params = params
                else:
                    model.fit(X_train, y_train)
                    best_params = params
                
                # Prédictions
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                
                # Métriques
                average = 'weighted' if is_multiclass else 'binary'
                
                train_metrics = {
                    "accuracy": float(accuracy_score(y_train, y_train_pred)),
                    "precision": float(precision_score(y_train, y_train_pred, average=average, zero_division=0)),
                    "recall": float(recall_score(y_train, y_train_pred, average=average, zero_division=0)),
                    "f1": float(f1_score(y_train, y_train_pred, average=average, zero_division=0))
                }
                
                val_metrics = {
                    "accuracy": float(accuracy_score(y_val, y_val_pred)),
                    "precision": float(precision_score(y_val, y_val_pred, average=average, zero_division=0)),
                    "recall": float(recall_score(y_val, y_val_pred, average=average, zero_division=0)),
                    "f1": float(f1_score(y_val, y_val_pred, average=average, zero_division=0))
                }
                
                # Matrice de confusion
                cm = confusion_matrix(y_val, y_val_pred)
                
                results[name] = {
                    "model": model,
                    "best_params": best_params,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "confusion_matrix": cm.tolist(),
                    "predictions": {"val": y_val_pred.tolist()}
                }
                
                logger.info(f"    ✅ {name}: Train Acc={train_metrics['accuracy']:.3f}, "
                           f"Val Acc={val_metrics['accuracy']:.3f}")
                
            except Exception as e:
                logger.warning(f"    ⚠️ {name} échoué: {e}")
                results[name] = {"error": str(e)}
        
        return results
    
    def _train_regression_models(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray,
                                 tune_hyperparams: bool = False) -> Dict[str, Any]:
        """Entraîne les modèles de régression"""
        results = {}
        
        for name, config in self.regression_models.items():
            try:
                logger.info(f"  🔄 Entraînement {name}...")
                
                model = config["model"](**config["params"])
                
                if tune_hyperparams and config["tuning_params"]:
                    try:
                        grid_search = GridSearchCV(
                            model, config["tuning_params"],
                            cv=3, scoring='neg_mean_squared_error', n_jobs=-1
                        )
                        grid_search.fit(X_train, y_train)
                        model = grid_search.best_estimator_
                        best_params = grid_search.best_params_
                    except:
                        model.fit(X_train, y_train)
                        best_params = config["params"]
                else:
                    model.fit(X_train, y_train)
                    best_params = config["params"]
                
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                
                results[name] = {
                    "model": model,
                    "best_params": best_params,
                    "train_metrics": {
                        "rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
                        "mae": float(mean_absolute_error(y_train, y_train_pred)),
                        "r2": float(r2_score(y_train, y_train_pred))
                    },
                    "val_metrics": {
                        "rmse": float(np.sqrt(mean_squared_error(y_val, y_val_pred))),
                        "mae": float(mean_absolute_error(y_val, y_val_pred)),
                        "r2": float(r2_score(y_val, y_val_pred))
                    }
                }
                
                logger.info(f"    ✅ {name}: R²={results[name]['val_metrics']['r2']:.3f}")
                
            except Exception as e:
                logger.warning(f"    ⚠️ {name} échoué: {e}")
                results[name] = {"error": str(e)}
        
        return results
    
    def _train_clustering_models(self, X: np.ndarray, max_clusters: int = 5) -> Dict[str, Any]:
        """Entraîne les modèles de clustering"""
        results = {}
        
        max_clusters = min(max_clusters, len(X) // 10, 5)
        
        # Trouver le nombre optimal de clusters
        silhouette_scores = []
        for k in range(2, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                score = silhouette_score(X, labels)
                silhouette_scores.append((k, score))
            except:
                pass
        
        optimal_k = max(silhouette_scores, key=lambda x: x[1])[0] if silhouette_scores else 2
        
        # KMeans
        try:
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            labels_km = kmeans.fit_predict(X)
            
            results["kmeans"] = {
                "model": kmeans,
                "labels": labels_km.tolist(),
                "n_clusters": optimal_k,
                "metrics": {
                    "silhouette": float(silhouette_score(X, labels_km)),
                    "davies_bouldin": float(davies_bouldin_score(X, labels_km)),
                    "calinski_harabasz": float(calinski_harabasz_score(X, labels_km)),
                    "inertia": float(kmeans.inertia_)
                },
                "cluster_centers": kmeans.cluster_centers_.tolist(),
                "cluster_sizes": [int(np.sum(labels_km == i)) for i in range(optimal_k)]
            }
        except Exception as e:
            logger.warning(f"    ⚠️ KMeans échoué: {e}")
        
        return results
    
    # ==================== CROSS-VALIDATION ====================
    
    def _cross_validate(self, model, X: np.ndarray, y: np.ndarray,
                       problem_type: str, cv: int = 5) -> Dict[str, Any]:
        """Cross-validation robuste"""
        try:
            if "classification" in problem_type:
                kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
                scoring = 'f1_weighted' if "multiclass" in problem_type else 'f1'
            else:
                kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
                scoring = 'r2'
            
            scores = cross_val_score(model, X, y, cv=kfold, scoring=scoring, n_jobs=-1)
            
            return {
                "cv_scores": scores.tolist(),
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "min_score": float(np.min(scores)),
                "max_score": float(np.max(scores)),
                "cv_folds": cv,
                "scoring_metric": scoring
            }
        except Exception as e:
            return {"error": str(e)}
    
    # ==================== FEATURE IMPORTANCE ====================
    
    def _get_feature_importance(self, model, feature_names: List[str],
                                X_val: np.ndarray, y_val: np.ndarray,
                                problem_type: str) -> Dict[str, Any]:
        """Calcule l'importance des features"""
        importance_result = {
            "method": None,
            "importances": {},
            "top_features": []
        }
        
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                importance_result["method"] = "feature_importances"
            elif hasattr(model, 'coef_'):
                coefs = model.coef_
                if len(coefs.shape) > 1:
                    coefs = np.abs(coefs).mean(axis=0)
                importances = np.abs(coefs)
                importance_result["method"] = "coefficients"
            else:
                try:
                    from sklearn.inspection import permutation_importance
                    perm_importance = permutation_importance(
                        model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1
                    )
                    importances = perm_importance.importances_mean
                    importance_result["method"] = "permutation"
                except:
                    return importance_result
            
            # Normaliser
            total = np.sum(importances)
            if total > 0:
                importances = importances / total
            
            importance_result["importances"] = {
                feature_names[i]: float(importances[i])
                for i in range(min(len(feature_names), len(importances)))
            }
            
            # Trier par importance
            sorted_importance = sorted(
                importance_result["importances"].items(),
                key=lambda x: x[1], reverse=True
            )
            importance_result["top_features"] = [
                {"feature": k, "importance": round(v, 4)}
                for k, v in sorted_importance[:10]
            ]
            
        except Exception as e:
            importance_result["error"] = str(e)
        
        return importance_result
    
    # ==================== ANALYSE DES ERREURS ====================
    
    def _analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray,
                       problem_type: str) -> Dict[str, Any]:
        """Analyse détaillée des erreurs"""
        analysis = {
            "problem_type": problem_type,
            "sample_size": len(y_true)
        }
        
        try:
            if "classification" in problem_type:
                errors = y_true != y_pred
                analysis["error_rate"] = float(np.mean(errors))
                analysis["n_errors"] = int(np.sum(errors))
                
                cm = confusion_matrix(y_true, y_pred)
                analysis["confusion_matrix"] = cm.tolist()
                
                unique_classes = np.unique(y_true)
                per_class = {}
                for c in unique_classes:
                    mask = y_true == c
                    per_class[int(c)] = {
                        "total": int(np.sum(mask)),
                        "correct": int(np.sum(y_pred[mask] == c)),
                        "accuracy": float(np.mean(y_pred[mask] == c)) if np.sum(mask) > 0 else 0
                    }
                analysis["per_class_analysis"] = per_class
                
            else:
                residuals = y_true - y_pred
                analysis["residuals"] = {
                    "mean": float(np.mean(residuals)),
                    "std": float(np.std(residuals)),
                    "min": float(np.min(residuals)),
                    "max": float(np.max(residuals))
                }
                
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis
    
    # ==================== MÉTHODE PRINCIPALE ====================
    
    async def run_ml_pipeline(self, df: pd.DataFrame,
                              context: Dict[str, Any],
                              eda_results: Optional[Dict[str, Any]] = None,
                              options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """🤖 PIPELINE ML V4 - AVEC EXPLICATIONS LLM"""
        
        logger.info("=" * 60)
        logger.info("🤖 ML PIPELINE V4 - AVEC EXPLICATIONS LLM")
        logger.info("=" * 60)
        
        options = options or {}
        target_col = context.get('target_variable')
        
        result = {
            "success": True,
            "problem_type": None,
            "data_summary": {},
            "models_trained": {},
            "best_model": None,
            "cross_validation": {},
            "feature_importance": {},
            "error_analysis": {},
            "recommendations": [],
            "warnings": [],
            "timestamp": datetime.now().isoformat(),
            "validation_info": {},
            "ml_applicable": True,
            "overfitting_detected": False,
            "llm_explanation": None,
            "test_metrics": {}
        }
        
        try:
            # PHASE 1: Préparation
            logger.info("📊 Phase 1: Préparation des données...")
            data_prep = self._prepare_data(
                df, target_col,
                feature_cols=options.get("feature_cols"),
                test_size=options.get("test_size", 0.2),
                val_size=options.get("val_size", 0.2)
            )
            
            # Ajouter les warnings d'overfitting
            if data_prep.get("overfitting_risks", {}).get("warnings"):
                result["warnings"].extend(data_prep["overfitting_risks"]["warnings"])
            
            if data_prep["overfitting_risks"].get("high_risk"):
                result["overfitting_detected"] = True
            
            if not data_prep["success"]:
                result["success"] = False
                result["ml_applicable"] = False
                result["error"] = data_prep.get("error", "Erreur préparation données")
                return result
            
            result["problem_type"] = data_prep["problem_type"]
            result["validation_info"] = data_prep.get("validation_info", {})
            result["data_summary"] = {
                "total_features": len(data_prep["feature_names"]),
                "feature_names": data_prep["feature_names"],
                "train_size": data_prep.get("train_size", 0),
                "val_size": data_prep.get("val_size", 0),
                "test_size": data_prep.get("test_size", 0),
                "class_names": data_prep.get("class_names")
            }
            
            # PHASE 2: Entraînement
            logger.info(f"🔄 Phase 2: Entraînement ({data_prep['problem_type']})...")
            
            tune_hyperparams = options.get("tune_hyperparams", True)
            
            if data_prep["problem_type"] == "clustering":
                models_result = self._train_clustering_models(
                    data_prep["X_train"],
                    max_clusters=options.get("max_clusters", 5)
                )
            elif data_prep["problem_type"] == "regression":
                models_result = self._train_regression_models(
                    data_prep["X_train"], data_prep["y_train"],
                    data_prep["X_val"], data_prep["y_val"],
                    tune_hyperparams=tune_hyperparams
                )
            else:
                is_multiclass = "multiclass" in data_prep["problem_type"]
                models_result = self._train_classification_models(
                    data_prep["X_train"], data_prep["y_train"],
                    data_prep["X_val"], data_prep["y_val"],
                    tune_hyperparams=tune_hyperparams,
                    is_multiclass=is_multiclass
                )
            
            # Formater les résultats
            result["models_trained"] = {}
            for name, model_data in models_result.items():
                if "error" not in model_data:
                    result["models_trained"][name] = {
                        k: v for k, v in model_data.items() 
                        if k != "model" and k != "predictions"
                    }
            
            # PHASE 3: Sélection du meilleur modèle
            logger.info("🏆 Phase 3: Sélection du meilleur modèle...")
            best_model_name = None
            best_score = -np.inf
            
            for name, model_data in models_result.items():
                if "error" in model_data:
                    continue
                
                if data_prep["problem_type"] == "clustering":
                    score = model_data.get("metrics", {}).get("silhouette", 0)
                elif data_prep["problem_type"] == "regression":
                    score = model_data.get("val_metrics", {}).get("r2", 0)
                else:
                    score = model_data.get("val_metrics", {}).get("f1", 0)
                
                if score > best_score:
                    best_score = score
                    best_model_name = name
            
            # Détection score parfait
            if best_score == 1.0:
                result["overfitting_detected"] = True
                result["warnings"].append(
                    "⚠️ SCORE PARFAIT DÉTECTÉ (100%) - Risque élevé d'overfitting ou fuite de données"
                )
            
            if best_model_name:
                result["best_model"] = {
                    "name": best_model_name,
                    "score": best_score,
                    "metrics": result["models_trained"].get(best_model_name, {})
                }
                self.best_model = models_result[best_model_name].get("model")
                logger.info(f"  🏆 Meilleur modèle: {best_model_name} (score: {best_score:.4f})")
            
            # PHASE 4: Cross-Validation
            if data_prep["problem_type"] != "clustering" and self.best_model:
                logger.info("🔄 Phase 4: Cross-Validation...")
                X_full = np.vstack([data_prep["X_train"], data_prep["X_val"]])
                y_full = np.concatenate([data_prep["y_train"], data_prep["y_val"]])
                
                result["cross_validation"] = self._cross_validate(
                    self.best_model, X_full, y_full,
                    data_prep["problem_type"], cv=5
                )
            
            # PHASE 5: Feature Importance
            if self.best_model and data_prep["problem_type"] != "clustering":
                logger.info("📊 Phase 5: Feature Importance...")
                result["feature_importance"] = self._get_feature_importance(
                    self.best_model,
                    data_prep["feature_names"],
                    data_prep["X_val"],
                    data_prep["y_val"],
                    data_prep["problem_type"]
                )
            
            # PHASE 6: Analyse des erreurs
            if self.best_model and data_prep["problem_type"] != "clustering":
                logger.info("🔍 Phase 6: Analyse des erreurs...")
                y_val_pred = self.best_model.predict(data_prep["X_val"])
                result["error_analysis"] = self._analyze_errors(
                    data_prep["y_val"], y_val_pred,
                    data_prep["problem_type"]
                )
            
            # PHASE 7: Évaluation finale sur Test
            if data_prep.get("X_test") is not None and self.best_model:
                logger.info("📝 Phase 7: Évaluation finale sur Test...")
                y_test_pred = self.best_model.predict(data_prep["X_test"])
                
                if data_prep["problem_type"] == "regression":
                    result["test_metrics"] = {
                        "rmse": float(np.sqrt(mean_squared_error(data_prep["y_test"], y_test_pred))),
                        "mae": float(mean_absolute_error(data_prep["y_test"], y_test_pred)),
                        "r2": float(r2_score(data_prep["y_test"], y_test_pred))
                    }
                else:
                    average = 'weighted' if "multiclass" in data_prep["problem_type"] else 'binary'
                    result["test_metrics"] = {
                        "accuracy": float(accuracy_score(data_prep["y_test"], y_test_pred)),
                        "f1": float(f1_score(data_prep["y_test"], y_test_pred, average=average, zero_division=0)),
                        "precision": float(precision_score(data_prep["y_test"], y_test_pred, average=average, zero_division=0)),
                        "recall": float(recall_score(data_prep["y_test"], y_test_pred, average=average, zero_division=0))
                    }
            
            # PHASE 8: Génération des explications LLM
            logger.info("🧠 Phase 8: Génération des explications LLM...")
            if self.llm_explainer:
                try:
                    llm_explanation = await self.llm_explainer.generate_ml_explanation(
                        result,
                        result["data_summary"],
                        result.get("overfitting_detected", False)
                    )
                    result["llm_explanation"] = llm_explanation
                    logger.info("✅ Explications LLM générées")
                except Exception as e:
                    logger.warning(f"⚠️ Erreur génération LLM: {e}")
                    result["llm_explanation"] = {
                        "success": False,
                        "error": str(e)
                    }
            
            # Générer recommandations simples
            result["recommendations"] = self._generate_simple_recommendations(result)
            
            logger.info("=" * 60)
            logger.info("✅ ML PIPELINE V4 TERMINÉ")
            logger.info(f"   Meilleur modèle: {result['best_model']['name'] if result['best_model'] else 'N/A'}")
            logger.info(f"   Score: {best_score:.4f}")
            logger.info(f"   Overfitting: {result['overfitting_detected']}")
            logger.info(f"   LLM Explanation: {result['llm_explanation'] is not None}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Erreur ML Pipeline: {e}", exc_info=True)
            result["success"] = False
            result["error"] = str(e)
            result["ml_applicable"] = False
        
        return result
    
    def _generate_simple_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """Génère des recommandations simples"""
        recommendations = []
        
        overfitting_detected = result.get("overfitting_detected", False)
        data_summary = result.get("data_summary", {})
        n_features = data_summary.get("total_features", 0)
        n_samples = data_summary.get("train_size", 0)
        ratio = n_features / n_samples if n_samples > 0 else 0
        
        if overfitting_detected:
            recommendations.append("🔴 Overfitting détecté - réduire les features et augmenter la régularisation")
        
        if ratio > 0.2:
            recommendations.append(f"📉 Ratio features/samples élevé ({ratio:.2f}) - réduire à max {n_samples // 5} features")
        
        if result.get("best_model"):
            recommendations.append(f"🏆 Modèle recommandé: {result['best_model']['name']}")
        
        return recommendations[:5]


# Instance globale
ml_pipeline_service = MLPipelineService()