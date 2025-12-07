# backend/services/multi_llm_orchestration.py
"""
Service d'orchestration parallèle multi-LLM
Gère la génération parallèle des sections de questionnaire avec OpenAI, Anthropic, Gemini
"""

import logging
import json
import asyncio
from typing import Dict, Any, Optional, Callable, List, Union
from openai import OpenAI
import anthropic
import google.generativeai as genai
from config.settings import settings
from models.survey import Category, Question, ExpectedAnswer, QuestionType, AnswerType, ContextExtraction

logger = logging.getLogger(__name__)

class MultiLLMOrchestrationService:
    """Service pour orchestrer la génération parallèle avec plusieurs LLM"""
    
    def __init__(self):
        """Initialise les clients LLM"""
        self._init_clients()
    
    def _init_clients(self):
        """Initialise les clients pour chaque LLM"""
        openai_keys = settings.get_openai_keys()
        anthropic_keys = settings.get_anthropic_keys()
        gemini_keys = settings.get_gemini_keys()
        
        self.openai_client = OpenAI(api_key=openai_keys[0]) if openai_keys else None
        self.openai_model = settings.openai_model
        
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_keys[0]) if anthropic_keys else None
        self.anthropic_model = settings.anthropic_model
        
        if gemini_keys:
            genai.configure(api_key=gemini_keys[0])
        self.gemini_model = settings.gemini_model
        
        logger.info(f"Clients LLM initialisés: OpenAI={bool(self.openai_client)}, "
                   f"Anthropic={bool(self.anthropic_client)}, Gemini={bool(gemini_keys)}")
    
    def _get_generation_system_prompt(self) -> str:
        """Retourne le prompt système pour la génération"""
        return """Tu es un expert en création de questionnaires d'enquête professionnels.
        
Génère les sections du questionnaire basées sur les catégories fournies.

RÈGLES OBLIGATOIRES:
- Chaque question DOIT avoir un ID unique (q1, q2, q3, etc.)
- Utilise les types de questions: single_choice, multiple_choice, text, scale, yes_no, number, date
- Chaque question doit avoir 2-5 réponses possibles
- Inclus une logique conditionnelle avec next_question_id quand pertinent
- Les réponses doivent être détaillées et professionnelles
- Adapte au contexte français/malgache

Format de réponse: JSON uniquement, sans texte supplémentaire."""
    
    def _get_generation_schema(self) -> str:
        """Retourne le schéma JSON pour la génération"""
        return """{
            "categories": [
                {
                    "category_id": "cat1",
                    "category_name": "string",
                    "description": "string",
                    "order": 1,
                    "questions": [
                        {
                            "question_id": "q1",
                            "question_type": "single_choice|multiple_choice|text|scale|yes_no|date|number",
                            "question_text": "string",
                            "is_required": true,
                            "help_text": "string ou null",
                            "predecessor_answer_id": null,
                            "expected_answers": [
                                {
                                    "answer_id": "a1",
                                    "answer_type": "option|text|number|scale|boolean|date",
                                    "answer_text": "string",
                                    "next_question_id": "q2 ou null"
                                }
                            ]
                        }
                    ]
                }
            ]
        }"""
    
    async def generate_category_section(
        self,
        provider: str,
        categories: List[str],
        category_indices: List[int],
        context: Union[Dict[str, Any], ContextExtraction],
        attempt: int = 0
    ) -> Dict[str, Any]:
        """
        Génère une section de catégories avec un LLM spécifique
        
        Args:
            provider: Fournisseur LLM ('openai', 'anthropic', 'gemini')
            categories: Liste de toutes les catégories
            category_indices: Indices des catégories à générer par ce provider
            context: Contexte d'extraction (dict ou ContextExtraction)
            attempt: Numéro de tentative
        
        Returns:
            Dict avec les catégories générées
        """
        try:
            # Convertir context dict en ContextExtraction si nécessaire
            if isinstance(context, dict):
                ctx_dict = context
            else:
                ctx_dict = context.dict()
            
            # Sélectionner les catégories pour ce provider
            assigned_categories = [categories[i] for i in category_indices if i < len(categories)]
            
            logger.info(f"[{provider.upper()}] Génération des catégories: {assigned_categories}")
            
            # Construction du prompt
            prompt = f"""
Génère les questions pour ces catégories:
Objectif: {ctx_dict.get('survey_objective', 'Non spécifié')}
Catégories à générer: {', '.join(assigned_categories)}
Nombre total de questions pour {len(assigned_categories)} catégories: {ctx_dict.get('number_of_questions', 30) // max(len(categories), 1)} questions par catégorie
Audience cible: {ctx_dict.get('target_audience', 'Général')}

Schéma attendu:
{self._get_generation_schema()}

Réponds UNIQUEMENT avec un JSON valide."""
            
            # Appel au LLM approprié
            if provider == "openai":
                result = await self._generate_openai(prompt)
            elif provider == "anthropic":
                result = await self._generate_anthropic(prompt)
            elif provider == "gemini":
                result = await self._generate_gemini(prompt)
            else:
                return {"success": False, "error": f"Provider inconnu: {provider}"}
            
            if result["success"]:
                logger.info(f"[{provider.upper()}] Génération réussie")
                return result
            else:
                logger.warning(f"[{provider.upper()}] Erreur: {result.get('error')}")
                return result
        
        except Exception as e:
            logger.error(f"[{provider.upper()}] Exception: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def _generate_openai(self, prompt: str) -> Dict[str, Any]:
        """Génère avec OpenAI"""
        try:
            if not self.openai_client:
                return {"success": False, "error": "Client OpenAI non disponible"}
            
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": self._get_generation_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2500,
                temperature=0.7,
            )
            
            content = response.choices[0].message.content.strip()
            
            # Nettoyage du JSON
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            data = json.loads(content)
            return {"success": True, "data": data}
        
        except Exception as e:
            logger.error(f"Erreur OpenAI: {e}")
            return {"success": False, "error": str(e)}
    
    async def _generate_anthropic(self, prompt: str) -> Dict[str, Any]:
        """Génère avec Anthropic Claude"""
        try:
            if not self.anthropic_client:
                return {"success": False, "error": "Client Anthropic non disponible"}
            
            message = self.anthropic_client.messages.create(
                model=self.anthropic_model,
                max_tokens=2500,
                system=self._get_generation_system_prompt(),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = message.content[0].text.strip()
            
            # Nettoyage du JSON
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            data = json.loads(content)
            return {"success": True, "data": data}
        
        except Exception as e:
            logger.error(f"Erreur Anthropic: {e}")
            return {"success": False, "error": str(e)}
    
    async def _generate_gemini(self, prompt: str) -> Dict[str, Any]:
        """Génère avec Google Gemini"""
        try:
            if not settings.get_gemini_keys():
                return {"success": False, "error": "Client Gemini non disponible"}
            
            model = genai.GenerativeModel(self.gemini_model)
            
            response = model.generate_content(
                f"{self._get_generation_system_prompt()}\n\n{prompt}",
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=2500,
                    temperature=0.7
                )
            )
            
            content = response.text.strip()
            
            # Nettoyage du JSON
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            data = json.loads(content)
            return {"success": True, "data": data}
        
        except Exception as e:
            logger.error(f"Erreur Gemini: {e}")
            return {"success": False, "error": str(e)}
    
    async def generate_survey_sections_parallel(
        self,
        context: Union[Dict[str, Any], ContextExtraction],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Génère les sections du questionnaire en parallèle
        
        Args:
            context: Contexte d'extraction (dict ou ContextExtraction)
            progress_callback: Callback pour la progression
        
        Returns:
            Dict avec toutes les catégories générées
        """
        try:
            # Convertir context dict en dict si nécessaire
            if isinstance(context, dict):
                ctx_dict = context
            else:
                ctx_dict = context.dict()
            
            categories = ctx_dict.get('categories', []) or [
                "Informations générales",
                "Situation actuelle", 
                "Problèmes et défis",
                "Besoins et priorités",
                "Suggestions d'amélioration"
            ]
            
            if progress_callback:
                await progress_callback("🚀 Démarrage de la génération parallèle", "starting")
            
            # Distribution des catégories aux providers
            # OpenAI: catégories 0-1, Anthropic: catégories 2-3, Gemini: catégories 4-5
            tasks = [
                self.generate_category_section("openai", categories, [0, 1], ctx_dict),
                self.generate_category_section("anthropic", categories, [2, 3], ctx_dict),
                self.generate_category_section("gemini", categories, [4] if len(categories) > 4 else [], ctx_dict)
            ]
            
            # Exécution parallèle
            if progress_callback:
                await progress_callback("🔄 Génération OpenAI et Anthropic en parallèle", "generation")
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Fusion des résultats
            all_categories = []
            total_questions = 0
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Exception dans la génération {i}: {result}")
                    continue
                
                if isinstance(result, dict) and result.get("success"):
                    categories_data = result.get("data", {}).get("categories", [])
                    all_categories.extend(categories_data)
                    total_questions += sum(len(cat.get("questions", [])) for cat in categories_data)
                else:
                    logger.warning(f"Erreur génération {i}: {result.get('error') if isinstance(result, dict) else str(result)}")
            
            if progress_callback:
                await progress_callback(
                    f"✅ {len(all_categories)} catégories générées ({total_questions} questions)",
                    "complete"
                )
            
            return {
                "success": True,
                "categories": all_categories,
                "total_questions": total_questions
            }
        
        except Exception as e:
            logger.error(f"Erreur orchestration: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

# Instance globale
multi_llm_orchestration = MultiLLMOrchestrationService()