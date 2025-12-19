"""
🧠 LLM EXPLAINER SERVICE V2 - CORRIGÉ
Génère des explications intelligentes pour les résultats ML
CORRECTIONS:
- Détection correcte des scores parfaits (100% = CRITIQUE)
- Score de santé réaliste
- Pas de "prêt pour production" si overfitting
- Messages cohérents
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class LLMExplainerService:
    """Service de génération d'explications ML intelligentes"""
    
    # Traductions des variables malgaches
    VARIABLE_TRANSLATIONS = {
        "Inona no jiro ampiasainar": "Source d'énergie principale",
        "Aiza no maka rano fisotro": "Source d'eau potable",
        "Velaran-tany": "Surface de terrain",
        "Karazana tany": "Type de terrain",
        "Tanim-bary": "Rizière",
        "Faritra": "Région",
        "Fokontany": "Village",
        "Kaominina": "Commune",
    }
    
    def __init__(self):
        self.initialized = True
    
    async def generate_ml_explanation(
        self,
        ml_results: Dict[str, Any],
        data_summary: Dict[str, Any],
        overfitting_detected: bool = False
    ) -> Dict[str, Any]:
        """
        Génère une explication complète des résultats ML
        
        Args:
            ml_results: Résultats du pipeline ML
            data_summary: Résumé des données (features, samples, etc.)
            overfitting_detected: Flag d'overfitting du pipeline
            
        Returns:
            Dict avec explanation, recommendations, diagnostic, tts_text
        """
        try:
            # Extraire les infos clés
            best_model = ml_results.get('best_model', {})
            model_name = best_model.get('name', 'Modèle inconnu')
            problem_type = ml_results.get('problem_type', 'classification')
            test_metrics = ml_results.get('test_metrics', {})
            models_trained = ml_results.get('models_trained', {})
            warnings = ml_results.get('warnings', [])
            
            # Extraire les métriques
            train_acc = self._get_train_accuracy(models_trained, model_name)
            val_acc = self._get_val_accuracy(models_trained, model_name)
            test_acc = test_metrics.get('accuracy', test_metrics.get('f1', 0))
            
            # Données
            n_features = data_summary.get('total_features', 0)
            n_train = data_summary.get('train_size', 1)
            n_test = data_summary.get('test_size', 0)
            ratio = n_features / max(n_train, 1)
            
            # 🔴 DÉTECTION CRITIQUE: Score parfait = problème grave
            # Score parfait si TEST >= 99% (le plus important) OU train ET val >= 99%
            is_perfect_test = test_acc >= 0.99
            is_perfect_train_val = train_acc >= 0.99 and val_acc >= 0.99
            is_perfect_score = is_perfect_test or is_perfect_train_val
            has_data_leakage = is_perfect_test and is_perfect_train_val  # Tout à 99%+
            
            # Calculer l'écart train/val
            train_val_gap = abs(train_acc - val_acc) if train_acc and val_acc else 0
            
            # Déterminer le niveau de criticité
            is_critical = is_perfect_score or has_data_leakage or ratio > 0.3
            is_warning = train_val_gap > 0.15 or ratio > 0.2 or overfitting_detected
            
            # Générer les composants
            explanation = self._generate_explanation(
                model_name=model_name,
                problem_type=problem_type,
                train_acc=train_acc,
                val_acc=val_acc,
                test_acc=test_acc,
                n_features=n_features,
                n_train=n_train,
                n_test=n_test,
                ratio=ratio,
                is_perfect_score=is_perfect_score,
                has_data_leakage=has_data_leakage,
                train_val_gap=train_val_gap,
                warnings=warnings
            )
            
            diagnostic = self._generate_diagnostic(
                train_acc=train_acc,
                val_acc=val_acc,
                test_acc=test_acc,
                ratio=ratio,
                n_train=n_train,
                is_perfect_score=is_perfect_score,
                has_data_leakage=has_data_leakage,
                train_val_gap=train_val_gap,
                overfitting_detected=overfitting_detected
            )
            
            recommendations = self._generate_recommendations(
                is_perfect_score=is_perfect_score,
                has_data_leakage=has_data_leakage,
                ratio=ratio,
                train_val_gap=train_val_gap,
                test_acc=test_acc,
                train_acc=train_acc,
                val_acc=val_acc,
                n_train=n_train,
                warnings=warnings
            )
            
            tts_text = self._generate_tts(
                model_name=model_name,
                test_acc=test_acc,
                diagnostic=diagnostic,
                is_perfect_score=is_perfect_score,
                train_acc=train_acc,
                val_acc=val_acc
            )
            
            return {
                "success": True,
                "explanation": explanation,
                "recommendations": recommendations,
                "diagnostic": diagnostic,
                "tts_text": tts_text,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur génération explication: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_train_accuracy(self, models_trained: Dict, model_name: str) -> float:
        """Récupère l'accuracy d'entraînement"""
        if model_name in models_trained:
            train_metrics = models_trained[model_name].get('train_metrics', {})
            return train_metrics.get('accuracy', train_metrics.get('f1', 0))
        return 0
    
    def _get_val_accuracy(self, models_trained: Dict, model_name: str) -> float:
        """Récupère l'accuracy de validation"""
        if model_name in models_trained:
            val_metrics = models_trained[model_name].get('val_metrics', {})
            return val_metrics.get('accuracy', val_metrics.get('f1', 0))
        return 0
    
    def _generate_explanation(
        self,
        model_name: str,
        problem_type: str,
        train_acc: float,
        val_acc: float,
        test_acc: float,
        n_features: int,
        n_train: int,
        n_test: int,
        ratio: float,
        is_perfect_score: bool,
        has_data_leakage: bool,
        train_val_gap: float,
        warnings: List[str]
    ) -> Dict[str, Any]:
        """Génère l'explication structurée"""
        
        # 🔴 LOGIQUE CORRIGÉE: Différencier les cas
        if test_acc >= 0.99:
            # Vrai data leakage - test aussi à 100%
            confidence_level = "faible"
            summary = (
                f"⚠️ **ALERTE CRITIQUE** : Le modèle {model_name} affiche un score test de "
                f"{test_acc*100:.1f}%, ce qui est **très suspect**. "
                f"Un score de 100% sur le test indique généralement une **fuite de données** (data leakage). "
                f"Le modèle ne doit **PAS** être utilisé en production sans investigation approfondie."
            )
            metrics_interpretation = (
                "🔴 ANOMALIE : Score test parfait = fuite de données probable. "
                "Une feature contient l'information de la cible."
            )
        elif train_acc >= 0.99 and val_acc >= 0.99:
            # Train/val parfaits mais test inférieur = overfitting sévère
            confidence_level = "faible"
            summary = (
                f"⚠️ **OVERFITTING SÉVÈRE** : Le modèle {model_name} atteint 100% sur train/val "
                f"mais seulement **{test_acc*100:.1f}%** sur le test. "
                f"Le modèle a **mémorisé** les données d'entraînement sans apprendre à généraliser. "
                f"Une régularisation forte est nécessaire."
            )
            metrics_interpretation = (
                f"🔴 Train/Val: 100% vs Test: {test_acc*100:.1f}% = overfitting massif. "
                "Le modèle ne généralise pas du tout."
            )
        elif train_val_gap > 0.15:
            confidence_level = "faible"
            summary = (
                f"Le modèle {model_name} montre un **écart significatif** de {train_val_gap*100:.1f}% "
                f"entre l'entraînement ({train_acc*100:.1f}%) et la validation ({val_acc*100:.1f}%). "
                f"Cela indique de l'**overfitting** : le modèle mémorise les données d'entraînement "
                f"sans généraliser correctement."
            )
            metrics_interpretation = (
                f"⚠️ Écart train/val de {train_val_gap*100:.1f}% trop élevé. "
                "Simplifier le modèle ou augmenter les données."
            )
        elif ratio > 0.3:
            confidence_level = "faible"
            summary = (
                f"Le modèle {model_name} souffre d'un **ratio features/samples critique** ({ratio:.2f}). "
                f"Avec {n_features} features pour seulement {n_train} échantillons, "
                f"le modèle risque fortement de surapprendre. Réduire le nombre de features est essentiel."
            )
            metrics_interpretation = (
                f"🔴 Ratio {ratio:.2f} > 0.3 = haute dimensionnalité. "
                "Les performances sont probablement surévaluées."
            )
        elif ratio > 0.2:
            confidence_level = "modéré"
            summary = (
                f"Le modèle {model_name} atteint {test_acc*100:.1f}% sur le test. "
                f"Le ratio features/samples ({ratio:.2f}) est élevé, ce qui peut affecter la fiabilité. "
                f"Des optimisations sont recommandées."
            )
            metrics_interpretation = (
                f"⚠️ Ratio {ratio:.2f} légèrement élevé. Performances à vérifier avec validation croisée."
            )
        elif test_acc < 0.5:
            confidence_level = "faible"
            summary = (
                f"Le modèle {model_name} n'atteint que {test_acc*100:.1f}% sur le test, "
                f"ce qui est inférieur au hasard pour une classification binaire. "
                f"Les features actuelles ne sont probablement pas prédictives de la cible."
            )
            metrics_interpretation = "🔴 Performance insuffisante. Revoir la sélection de features."
        else:
            confidence_level = "modéré" if test_acc < 0.7 else "élevé"
            summary = (
                f"Le modèle {model_name} atteint {test_acc*100:.1f}% d'accuracy sur les données de test. "
                f"{'Performance acceptable mais améliorable.' if test_acc < 0.7 else 'Bonne performance générale.'}"
            )
            metrics_interpretation = (
                f"{'Performance correcte.' if test_acc < 0.7 else 'Bon équilibre train/val/test.'}"
            )
        
        # Analyse qualité données
        if ratio > 0.3:
            ratio_status = "critique"
            data_interpretation = f"🔴 Ratio CRITIQUE ({ratio:.2f}). Trop de features pour le nombre d'échantillons."
        elif ratio > 0.2:
            ratio_status = "élevé"
            data_interpretation = f"⚠️ Ratio élevé ({ratio:.2f}). Risque d'overfitting."
        elif ratio > 0.1:
            ratio_status = "modéré"
            data_interpretation = f"Ratio acceptable ({ratio:.2f})."
        else:
            ratio_status = "bon"
            data_interpretation = f"✅ Bon ratio ({ratio:.2f}). Marge suffisante pour l'apprentissage."
        
        # Analyse overfitting
        overfitting_analysis = None
        if is_perfect_score or has_data_leakage:
            overfitting_analysis = {
                "detected": True,
                "severity": "élevée",
                "causes": [
                    "🔴 Score parfait de 100% sur tous les ensembles",
                    "Fuite de données probable (la cible est dans les features)",
                    "Ou variable parfaitement corrélée avec la cible",
                ],
                "impact": "CRITIQUE : Le modèle ne généralisera pas sur de nouvelles données"
            }
        elif train_val_gap > 0.15:
            overfitting_analysis = {
                "detected": True,
                "severity": "modérée",
                "causes": [
                    f"Écart train/val de {train_val_gap*100:.1f}%",
                    "Modèle trop complexe pour les données",
                    "Possible présence de variables quasi-constantes"
                ],
                "impact": "Le modèle mémorise plutôt qu'il n'apprend"
            }
        elif ratio > 0.2:
            overfitting_analysis = {
                "detected": True,
                "severity": "modérée",
                "causes": [
                    f"Ratio features/samples élevé ({ratio:.2f})",
                    "Haute dimensionnalité",
                ],
                "impact": "Risque de surapprentissage des patterns spécifiques"
            }
        
        return {
            "summary": summary,
            "model_selected": model_name,
            "problem_type": self._format_problem_type(problem_type),
            "confidence_level": confidence_level,
            "metrics_analysis": {
                "train_accuracy": f"{train_acc*100:.1f}%",
                "validation_accuracy": f"{val_acc*100:.1f}%",
                "test_accuracy": f"{test_acc*100:.1f}%",
                "interpretation": metrics_interpretation
            },
            "data_quality": {
                "features": n_features,
                "samples_train": n_train,
                "samples_test": n_test,
                "ratio": f"{ratio:.2f}",
                "ratio_status": ratio_status,
                "interpretation": data_interpretation
            },
            "overfitting_analysis": overfitting_analysis
        }
    
    def _generate_diagnostic(
        self,
        train_acc: float,
        val_acc: float,
        test_acc: float,
        ratio: float,
        n_train: int,
        is_perfect_score: bool,
        has_data_leakage: bool,
        train_val_gap: float,
        overfitting_detected: bool
    ) -> Dict[str, Any]:
        """Génère le diagnostic avec score de santé RÉALISTE"""
        
        # 🔴 CALCUL CORRIGÉ DU SCORE DE SANTÉ
        health_score = 100
        
        # Détection des cas critiques
        is_perfect_test = test_acc >= 0.99
        is_perfect_train_val = train_acc >= 0.99 and val_acc >= 0.99
        
        # Pénalités selon gravité
        if is_perfect_test:
            # Data leakage confirmé = très grave
            health_score -= 60
        elif is_perfect_train_val:
            # Overfitting train/val mais test OK = grave mais moins
            health_score -= 35
        
        # Pénalités ratio
        if ratio > 0.3:
            health_score -= 25
        elif ratio > 0.2:
            health_score -= 15
        elif ratio > 0.15:
            health_score -= 8
        
        # Pénalités écart train/val
        if train_val_gap > 0.2:
            health_score -= 20
        elif train_val_gap > 0.1:
            health_score -= 10
        
        # Pénalités performance test
        if test_acc < 0.5:
            health_score -= 25
        elif test_acc < 0.6:
            health_score -= 15
        elif test_acc < 0.7:
            health_score -= 8
        
        # Pénalité overfitting flag
        if overfitting_detected:
            health_score -= 10
        
        # Pénalité données insuffisantes
        if n_train < 100:
            health_score -= 15
        elif n_train < 300:
            health_score -= 8
        
        # Borner le score
        health_score = max(0, min(100, health_score))
        
        # Déterminer le statut
        if health_score >= 70:
            health_status = "bon"
            summary = "Le modèle est globalement sain avec quelques points d'attention."
        elif health_score >= 50:
            health_status = "modéré"
            summary = "Le modèle nécessite des optimisations avant utilisation."
        else:
            health_status = "critique"
            if is_perfect_score:
                summary = "⚠️ CRITIQUE : Score parfait suspect. Investiguer la fuite de données."
            else:
                summary = "⚠️ CRITIQUE : Le modèle présente des problèmes majeurs à corriger."
        
        # Générer les checks
        checks = []
        
        # Check 1: Ratio features/samples
        if ratio > 0.3:
            checks.append({
                "name": "Ratio Features/Samples",
                "status": "error",
                "value": f"{ratio:.2f}",
                "message": f"CRITIQUE : {ratio:.2f} > 0.3. Trop de features!"
            })
        elif ratio > 0.2:
            checks.append({
                "name": "Ratio Features/Samples",
                "status": "warning",
                "value": f"{ratio:.2f}",
                "message": f"Élevé : {ratio:.2f} > 0.2. Réduire les features."
            })
        else:
            checks.append({
                "name": "Ratio Features/Samples",
                "status": "ok",
                "value": f"{ratio:.2f}",
                "message": "Ratio acceptable pour l'apprentissage."
            })
        
        # Check 2: Écart Train/Validation et overfitting
        is_perfect_test = test_acc >= 0.99
        is_perfect_train_val = train_acc >= 0.99 and val_acc >= 0.99
        
        if is_perfect_test:
            checks.append({
                "name": "Performance Test",
                "status": "error",
                "value": f"{test_acc*100:.1f}%",
                "message": "🔴 Score test parfait = Fuite de données probable!"
            })
        elif is_perfect_train_val:
            checks.append({
                "name": "Overfitting Train/Val",
                "status": "error",
                "value": "100% train/val",
                "message": f"🔴 100% train/val vs {test_acc*100:.1f}% test = overfitting!"
            })
        elif train_val_gap > 0.15:
            checks.append({
                "name": "Écart Train/Validation",
                "status": "error",
                "value": f"{train_val_gap*100:.1f}%",
                "message": "Overfitting détecté. Simplifier le modèle."
            })
        elif train_val_gap > 0.08:
            checks.append({
                "name": "Écart Train/Validation",
                "status": "warning",
                "value": f"{train_val_gap*100:.1f}%",
                "message": "Léger surapprentissage possible."
            })
        else:
            checks.append({
                "name": "Écart Train/Validation",
                "status": "ok",
                "value": f"{train_val_gap*100:.1f}%",
                "message": "Bon équilibre train/validation."
            })
        
        # Check 3: Performance Test (seulement si pas déjà couvert par check 2)
        if not is_perfect_test and not is_perfect_train_val:
            if test_acc >= 0.7:
                checks.append({
                    "name": "Performance Test",
                    "status": "ok",
                    "value": f"{test_acc*100:.1f}%",
                    "message": "Bonne performance sur données non vues."
                })
            elif test_acc >= 0.5:
                checks.append({
                    "name": "Performance Test",
                    "status": "warning",
                    "value": f"{test_acc*100:.1f}%",
                    "message": "Performance modeste. Amélioration possible."
                })
            else:
                checks.append({
                    "name": "Performance Test",
                    "status": "error",
                    "value": f"{test_acc*100:.1f}%",
                    "message": "Inférieur au hasard. Features non prédictives."
                })
        
        # Check 4: Taille données
        if n_train >= 500:
            checks.append({
                "name": "Volume de Données",
                "status": "ok",
                "value": f"{n_train}",
                "message": "Volume suffisant pour l'apprentissage."
            })
        elif n_train >= 200:
            checks.append({
                "name": "Volume de Données",
                "status": "warning",
                "value": f"{n_train}",
                "message": "Volume limité. Plus de données recommandé."
            })
        else:
            checks.append({
                "name": "Volume de Données",
                "status": "error",
                "value": f"{n_train}",
                "message": "Données insuffisantes pour un ML fiable."
            })
        
        return {
            "health_score": health_score,
            "health_status": health_status,
            "summary": summary,
            "checks": checks
        }
    
    def _generate_recommendations(
        self,
        is_perfect_score: bool,
        has_data_leakage: bool,
        ratio: float,
        train_val_gap: float,
        test_acc: float,
        train_acc: float,
        val_acc: float,
        n_train: int,
        warnings: List[str]
    ) -> List[Dict[str, Any]]:
        """Génère les recommandations priorisées avec code"""
        
        recommendations = []
        rec_id = 1
        
        # Détection locale
        is_perfect_test = test_acc >= 0.99
        is_perfect_train_val = train_acc >= 0.99 and val_acc >= 0.99
        
        # 🔴 PRIORITÉ 1: Score parfait = fuite de données ou overfitting sévère
        if is_perfect_test or is_perfect_train_val:
            # Différencier le message selon le cas
            if is_perfect_test:
                title = "🔴 Investiguer la fuite de données"
                description = (
                    f"Un score test de {test_acc*100:.1f}% est presque toujours le signe d'une fuite de données. "
                    "Une feature contient probablement l'information de la cible."
                )
            else:
                title = "🔴 Corriger l'overfitting sévère"
                description = (
                    f"Le modèle atteint 100% sur train/val mais seulement {test_acc*100:.1f}% sur test. "
                    "Il mémorise les données d'entraînement sans généraliser."
                )
            
            recommendations.append({
                "id": rec_id,
                "title": title,
                "priority": "haute",
                "category": "Data Leakage" if is_perfect_test else "Overfitting",
                "description": description,
                "actions": [
                    {
                        "step": 1,
                        "action": "Identifier les features parfaitement corrélées à la cible",
                        "code": """# Chercher les corrélations parfaites
import pandas as pd

# Calculer corrélation avec la cible
correlations = df.corr()[target_column].abs().sort_values(ascending=False)
print("Features les plus corrélées:")
print(correlations.head(10))

# Features avec corrélation > 0.95 = suspects
suspects = correlations[correlations > 0.95].index.tolist()
print(f"\\n🔴 Features suspectes: {suspects}")"""
                    },
                    {
                        "step": 2,
                        "action": "Vérifier si une feature est dérivée de la cible",
                        "code": """# Examiner les features suspectes
for col in suspects:
                            if col != target_column:
        print(f"\\n--- {col} ---")
        print(f"Valeurs uniques: {df[col].nunique()}")
        print(f"Corrélation: {correlations[col]:.4f}")
        # Vérifier si c'est un identifiant ou dérivé
        print(df[[col, target_column]].head(10))"""
                    },
                    {
                        "step": 3,
                        "action": "Supprimer les features problématiques et réentraîner",
                        "code": """# Exclure les features suspectes
features_clean = [f for f in features if f not in suspects]
X_clean = df[features_clean]

# Réentraîner
model.fit(X_clean, y)
print(f"Score après nettoyage: {model.score(X_test_clean, y_test)}")"""
                    }
                ],
                "expected_impact": "Éliminer l'overfitting artificiel",
                "effort": "Moyen",
                "timeline": "Immédiat"
            })
            rec_id += 1
        
        # PRIORITÉ 2: Ratio élevé
        if ratio > 0.2:
            recommendations.append({
                "id": rec_id,
                "title": "Réduire le nombre de features",
                "priority": "haute" if ratio > 0.3 else "moyenne",
                "category": "Dimensionnalité",
                "description": (
                    f"Le ratio features/samples de {ratio:.2f} est trop élevé. "
                    f"Objectif: ratio < 0.1 pour une généralisation fiable."
                ),
                "actions": [
                    {
                        "step": 1,
                        "action": "Appliquer une sélection de features basée sur l'importance",
                        "code": """from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# Sélection basée sur importance
selector = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42),
    threshold='median'  # Garde les 50% plus importantes
)
X_selected = selector.fit_transform(X_train, y_train)
print(f"Features: {X_train.shape[1]} → {X_selected.shape[1]}")"""
                    },
                    {
                        "step": 2,
                        "action": "Utiliser la variance pour éliminer les features quasi-constantes",
                        "code": """from sklearn.feature_selection import VarianceThreshold

# Supprimer features avec variance < 0.01
selector = VarianceThreshold(threshold=0.01)
X_filtered = selector.fit_transform(X_train)
kept_features = X_train.columns[selector.get_support()].tolist()
print(f"Features conservées: {len(kept_features)}")"""
                    }
                ],
                "expected_impact": f"Réduire ratio de {ratio:.2f} à < 0.1",
                "effort": "Faible",
                "timeline": "1-2 heures"
            })
            rec_id += 1
        
        # PRIORITÉ 3: Overfitting (écart train/val)
        if train_val_gap > 0.1 and not is_perfect_score:
            recommendations.append({
                "id": rec_id,
                "title": "Augmenter la régularisation",
                "priority": "haute" if train_val_gap > 0.2 else "moyenne",
                "category": "Régularisation",
                "description": (
                    f"L'écart de {train_val_gap*100:.1f}% entre train et validation "
                    "indique un surapprentissage."
                ),
                "actions": [
                    {
                        "step": 1,
                        "action": "Pour Decision Tree / Random Forest: limiter la profondeur",
                        "code": """from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Decision Tree régularisé
dt = DecisionTreeClassifier(
    max_depth=5,           # Limiter profondeur
    min_samples_split=10,  # Min samples pour split
    min_samples_leaf=5,    # Min samples par feuille
    random_state=42
)

# Random Forest régularisé  
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)"""
                    },
                    {
                        "step": 2,
                        "action": "Pour Logistic Regression: augmenter la pénalité",
                        "code": """from sklearn.linear_model import LogisticRegression

# Forte régularisation (C petit = plus régularisé)
lr = LogisticRegression(
    C=0.01,              # Forte régularisation
    penalty='l2',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)"""
                    }
                ],
                "expected_impact": f"Réduire l'écart à < 5%",
                "effort": "Faible",
                "timeline": "30 minutes"
            })
            rec_id += 1
        
        # PRIORITÉ 4: Performance faible
        if test_acc < 0.6 and not is_perfect_score:
            recommendations.append({
                "id": rec_id,
                "title": "Améliorer la qualité des features",
                "priority": "moyenne",
                "category": "Feature Engineering",
                "description": (
                    f"La performance de {test_acc*100:.1f}% suggère que les features "
                    "actuelles ne capturent pas bien le signal."
                ),
                "actions": [
                    {
                        "step": 1,
                        "action": "Analyser la distribution de la cible",
                        "code": """# Vérifier le déséquilibre des classes
print("Distribution de la cible:")
print(y.value_counts(normalize=True))

# Si déséquilibré, utiliser SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)"""
                    },
                    {
                        "step": 2,
                        "action": "Créer des features d'interaction",
                        "code": """from sklearn.preprocessing import PolynomialFeatures

# Créer interactions de degré 2
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_interactions = poly.fit_transform(X_train)
print(f"Features avec interactions: {X_interactions.shape[1]}")"""
                    }
                ],
                "expected_impact": "Améliorer accuracy de 10-20%",
                "effort": "Moyen",
                "timeline": "2-4 heures"
            })
            rec_id += 1
        
        # PRIORITÉ 5: Données insuffisantes
        if n_train < 300:
            recommendations.append({
                "id": rec_id,
                "title": "Utiliser la validation croisée",
                "priority": "moyenne",
                "category": "Validation",
                "description": (
                    f"Avec seulement {n_train} échantillons, la validation croisée "
                    "donne une estimation plus fiable des performances."
                ),
                "actions": [
                    {
                        "step": 1,
                        "action": "Implémenter une validation croisée stratifiée",
                        "code": """from sklearn.model_selection import cross_val_score, StratifiedKFold

# Validation croisée 5-fold stratifiée
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

print(f"Scores CV: {scores}")
print(f"Moyenne: {scores.mean():.3f} (+/- {scores.std()*2:.3f})"""
                    }
                ],
                "expected_impact": "Estimation fiable des performances",
                "effort": "Faible",
                "timeline": "15 minutes"
            })
            rec_id += 1
        
        # Toujours ajouter une recommandation générale si peu de recs
        if len(recommendations) < 2:
            recommendations.append({
                "id": rec_id,
                "title": "Documenter et versionner le modèle",
                "priority": "basse",
                "category": "MLOps",
                "description": "Bonnes pratiques pour la reproductibilité.",
                "actions": [
                    {
                        "step": 1,
                        "action": "Sauvegarder le modèle avec métadonnées",
                        "code": """import joblib
from datetime import datetime

# Sauvegarder avec métadonnées
model_info = {
    'model': model,
    'features': feature_names,
    'metrics': {'accuracy': test_acc, 'f1': f1_score},
    'trained_at': datetime.now().isoformat()
}
joblib.dump(model_info, 'model_v1.joblib')"""
                    }
                ],
                "expected_impact": "Reproductibilité et traçabilité",
                "effort": "Faible",
                "timeline": "30 minutes"
            })
        
        return recommendations
    
    def _generate_tts(
        self,
        model_name: str,
        test_acc: float,
        diagnostic: Dict[str, Any],
        is_perfect_score: bool,
        train_acc: float = 0,
        val_acc: float = 0
    ) -> str:
        """Génère le texte pour la synthèse vocale"""
        
        health_status = diagnostic.get('health_status', 'modéré')
        health_score = diagnostic.get('health_score', 50)
        
        is_perfect_test = test_acc >= 0.99
        is_perfect_train_val = train_acc >= 0.99 and val_acc >= 0.99
        
        if is_perfect_test:
            return (
                f"Attention, alerte critique. Le modèle {model_name} affiche un score test parfait "
                f"de {test_acc*100:.0f} pour cent, ce qui est très suspect. Cela indique probablement une fuite "
                f"de données. Le score de santé est de {health_score} sur 100, statut critique. "
                f"Il est impératif d'investiguer les features avant toute utilisation."
            )
        elif is_perfect_train_val:
            return (
                f"Attention, overfitting détecté. Le modèle {model_name} atteint 100 pour cent "
                f"sur l'entraînement mais seulement {test_acc*100:.0f} pour cent sur le test. "
                f"Score de santé: {health_score} sur 100. Le modèle mémorise les données "
                f"sans généraliser. Une régularisation est nécessaire."
            )
        elif health_status == 'critique':
            return (
                f"Le modèle {model_name} présente des problèmes critiques. "
                f"Score de santé: {health_score} sur 100. "
                f"Des corrections sont nécessaires avant utilisation."
            )
        elif health_status == 'modéré':
            return (
                f"Le modèle {model_name} atteint {test_acc*100:.0f} pour cent d'accuracy. "
                f"Score de santé: {health_score} sur 100, statut modéré. "
                f"Des optimisations sont recommandées pour améliorer la fiabilité."
            )
        else:
            return (
                f"Le modèle {model_name} est performant avec {test_acc*100:.0f} pour cent d'accuracy. "
                f"Score de santé: {health_score} sur 100. Le modèle peut être utilisé avec confiance."
            )
    
    def _format_problem_type(self, problem_type: str) -> str:
        """Formate le type de problème"""
        formats = {
            'binary_classification': 'Classification Binaire',
            'multiclass_classification': 'Classification Multi-classe',
            'regression': 'Régression',
            'clustering': 'Clustering'
        }
        return formats.get(problem_type, problem_type)


# Instance singleton
llm_explainer_service = LLMExplainerService()