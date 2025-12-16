# backend/services/relation_key_finder_improved.py
"""
Sélection INTELLIGENTE de la relation clé selon CONTEXTE
Pas juste la meilleure p-value!
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class RelationKeyFinder:
    """
    Trouve la relation VRAIMENT clé (selon contexte + analyse type)
    Considère:
    - Target variable implication
    - P-value significatif
    - Effect size
    - Pertinence pour l'analysis type
    - Non-trivialité
    """

    def find_key_relation(
        self,
        tests: List[Dict[str, Any]],
        context: Dict[str, Any],
        user_prompt: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Sélectionne la relation CLÉE selon CONTEXTE INTELLIGENT
        
        Scoring:
        - Implique target variable: +200
        - P-value < 0.05 (significatif): +100
        - P-value < 0.15 (tendance): +50
        - Effect size > 0.3: +50
        - Effect size > 0.5: +70
        - Type d'analyse pertinent: +30
        - Non-trivial (pas juste binaire): +20
        """
        
        if not tests:
            return None

        target = context.get('target_variable', '')
        analysis_type = context.get('analysis_type', 'descriptive')
        
        # Scorer chaque test
        scored_tests = []
        
        for test in tests:
            score = self._calculate_test_score(
                test,
                target,
                analysis_type,
                user_prompt
            )
            
            scored_tests.append({
                'test': test,
                'score': score,
                'reasons': self._explain_score(test, score, target, analysis_type)
            })
        
        # Retourner le meilleur
        if scored_tests:
            best = sorted(scored_tests, key=lambda x: x['score'], reverse=True)[0]
            logger.info(f"✓ Key relation selected: {best['test']['variable1']} vs {best['test']['variable2']} (score={best['score']})")
            
            # Ajouter explication
            best['test']['key_relation_score'] = best['score']
            best['test']['key_relation_reasons'] = best['reasons']
            
            return best['test']
        
        return None

    def _calculate_test_score(
        self,
        test: Dict,
        target: str,
        analysis_type: str,
        user_prompt: str
    ) -> int:
        """Calcule le score de pertinence"""
        
        score = 0

        # 1. IMPLIQUE TARGET VARIABLE? (+200 si oui)
        if test.get('variable1') == target or test.get('variable2') == target:
            score += 200
            logger.debug(f"  +200: Implique target {target}")
        elif target:
            # Vérifier si mention dans user_prompt
            test_vars = f"{test.get('variable1')} {test.get('variable2')}"
            if target.lower() in user_prompt.lower():
                score += 100  # Moins que direct implication
                logger.debug(f"  +100: Mentionné dans prompt")

        # 2. P-VALUE (SIGNIFICATIF?)
        p_value = test.get('p_value', 1)
        if p_value < 0.01:
            score += 120  # Très significatif!
            logger.debug(f"  +120: P-value très faible ({p_value:.4f})")
        elif p_value < 0.05:
            score += 100
            logger.debug(f"  +100: P-value significatif ({p_value:.4f})")
        elif p_value < 0.15:
            score += 50
            logger.debug(f"  +50: P-value tendance ({p_value:.4f})")

        # 3. EFFECT SIZE (PRATIQUEMENT SIGNIFICATIF?)
        stat = abs(test.get('statistic', 0))
        if stat > 0.5:
            score += 70
            logger.debug(f"  +70: Effect size fort ({stat:.3f})")
        elif stat > 0.3:
            score += 50
            logger.debug(f"  +50: Effect size modéré ({stat:.3f})")

        # 4. TYPE DE TEST (QUELLE QUALITÉ?)
        test_type = test.get('test_type', '')
        
        if test_type == 'pearson':
            score += 40  # Corrélations sont informatrices
            logger.debug(f"  +40: Pearson correlation (informatif)")
        elif test_type == 'anova':
            score += 30  # ANOVA sur 3+ groupes
            logger.debug(f"  +30: ANOVA (multi-groupe)")
        elif test_type == 'ttest':
            score -= 20  # T-tests souvent triviaux (binaire)
            logger.debug(f"  -20: T-test (potentiellement trivial)")
        elif test_type == 'chi2':
            score += 25
            logger.debug(f"  +25: Chi-2 (catégories)")

        # 5. PERTINENCE POUR TYPE D'ANALYSE
        if analysis_type == 'regression':
            if test_type == 'pearson':
                score += 30  # Les corrélations sont clés pour régression
                logger.debug(f"  +30: Pertinent pour régression")
        elif analysis_type == 'classification':
            if test_type in ['chi2', 'ttest', 'anova']:
                score += 30
                logger.debug(f"  +30: Pertinent pour classification")
        elif analysis_type == 'clustering':
            if test_type == 'anova':
                score += 30
                logger.debug(f"  +30: Pertinent pour clustering")

        # 6. NON-TRIVIALITÉ
        var1, var2 = test.get('variable1', ''), test.get('variable2', '')
        if not (var1.lower() in ['id', 'index'] or var2.lower() in ['id', 'index']):
            score += 20
            logger.debug(f"  +20: Relation non-triviale")

        # 7. VARIABILITÉ (pas constant)
        if 'conclusion' in test and 'Significant' in test['conclusion']:
            score += 15
            logger.debug(f"  +15: Conclusion positive")

        return score

    def _explain_score(
        self,
        test: Dict,
        score: int,
        target: str,
        analysis_type: str
    ) -> List[str]:
        """Explique pourquoi ce test a ce score"""
        
        reasons = []
        
        # Raison 1: Implique target?
        if test.get('variable1') == target or test.get('variable2') == target:
            reasons.append(f"Implique la variable cible ({target})")
        
        # Raison 2: P-value
        p_value = test.get('p_value', 1)
        if p_value < 0.01:
            reasons.append(f"Très significatif (p={p_value:.4f})")
        elif p_value < 0.05:
            reasons.append(f"Significatif (p={p_value:.4f})")
        elif p_value < 0.15:
            reasons.append(f"Tendance intéressante (p={p_value:.4f})")
        
        # Raison 3: Effect size
        stat = abs(test.get('statistic', 0))
        if stat > 0.5:
            reasons.append(f"Effet pratique fort ({stat:.3f})")
        elif stat > 0.3:
            reasons.append(f"Effet pratique modéré ({stat:.3f})")
        
        # Raison 4: Type d'analyse
        test_type = test.get('test_type', '')
        reasons.append(f"Test {test_type.upper()} approprié pour {analysis_type}")
        
        # Raison 5: Conclusion
        if 'conclusion' in test:
            reasons.append(f"Conclusion: {test['conclusion']}")
        
        return reasons

# Instance
relation_key_finder = RelationKeyFinder()