"""
Service d'orchestration parallèle multi-LLM
Gère la génération parallèle des sections de questionnaire avec OpenAI, Anthropic, Gemini
"""

import logging
import json
import asyncio
import re
import math
from typing import Dict, Any, Optional, Callable, List, Union
from openai import OpenAI
import anthropic
import google.generativeai as genai
from config.settings import settings
from models.survey import ContextExtraction

logger = logging.getLogger(__name__)

class MultiLLMOrchestrationService:
    """Service pour orchestrer la génération parallèle avec plusieurs LLM"""
    
    def __init__(self):
        """Initialise les clients LLM"""
        self._init_clients()
    
    def _init_clients(self):
        """Initialise les clients pour chaque LLM"""
        self.providers_status = {}

        # 1. OpenAI
        openai_keys = settings.get_openai_keys()
        if openai_keys:
            self.openai_client = OpenAI(api_key=openai_keys[0])
            self.openai_model = settings.openai_model
            self.providers_status["openai"] = True
        else:
            self.openai_client = None
            self.providers_status["openai"] = False
        
        # 2. Anthropic
        anthropic_keys = settings.get_anthropic_keys()
        if anthropic_keys:
            self.anthropic_client = anthropic.Anthropic(api_key=anthropic_keys[0])
            self.anthropic_model = settings.anthropic_model
            self.providers_status["anthropic"] = True
        else:
            self.anthropic_client = None
            self.providers_status["anthropic"] = False
        
        # 3. Gemini
        gemini_keys = settings.get_gemini_keys()
        if gemini_keys:
            genai.configure(api_key=gemini_keys[0])
            self.gemini_model = settings.gemini_model
            self.providers_status["gemini"] = True
        else:
            self.providers_status["gemini"] = False
        
        logger.info(f"Clients LLM initialisés: {self.providers_status}")
    
    def _get_generation_system_prompt(self) -> str:
        """Retourne le prompt système pour la génération"""
        return """Tu es un expert en création de questionnaires d'enquête professionnels.
Ton rôle est de générer une structure JSON stricte.

RÈGLES CRITIQUES :
1. Tu DOIS générer UNIQUEMENT du JSON valide.
2. Pas de texte introductif, pas de conclusion, pas de markdown (```json).
3. Échappe correctement les guillemets à l'intérieur des textes (ex: \\").
4. Utilise "single_choice", "multiple_choice", "text", "scale" pour les types.
"""
    
    def _get_generation_schema(self) -> str:
        """Retourne le schéma JSON pour la génération"""
        return """{
    "categories": [
        {
            "category_id": "string_unique",
            "category_name": "string",
            "description": "string",
            "order": int,
            "questions": [
                {
                    "question_id": "string_unique",
                    "question_type": "single_choice|multiple_choice|text|scale|yes_no",
                    "question_text": "string",
                    "is_required": true,
                    "expected_answers": [
                        {
                            "answer_id": "string",
                            "answer_text": "string"
                        }
                    ]
                }
            ]
        }
    ]
}"""
    
    def _robust_json_parse(self, content: str) -> Dict[str, Any]:
        """Parse le JSON de manière robuste (nettoyage markdown + extraction { })"""
        if not content:
            raise ValueError("Contenu vide reçu du LLM")

        # Nettoyage des balises Markdown
        content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
        content = re.sub(r'^```\s*', '', content, flags=re.MULTILINE)
        content = re.sub(r'\s*```$', '', content, flags=re.MULTILINE)
        
        # Extraction du bloc JSON uniquement
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx : end_idx + 1]
        
        # Nettoyage caractères invisibles
        content = "".join(ch for ch in content if (ord(ch) >= 32 or ch in "\n\r\t"))

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Erreur parsing JSON: {str(e)}")
            raise ValueError(f"Impossible de parser le JSON: {str(e)}")
    
    async def generate_category_section(
        self,
        provider: str,
        categories: List[str],
        category_indices: List[int],
        context: Dict[str, Any],
        attempt: int = 0
    ) -> Dict[str, Any]:
        """Génère une section de catégories avec un LLM spécifique"""
        try:
            assigned_categories = [categories[i] for i in category_indices if i < len(categories)]
            if not assigned_categories:
                return {"success": True, "data": {"categories": []}}

            logger.info(f"[{provider.upper()}] Génération pour: {assigned_categories}")
            
            prompt = f"""CONTEXTE DE L'ENQUÊTE:
Objectif: {context.get('survey_objective', 'Non spécifié')}
Cible: {context.get('target_audience', 'Général')}

TÂCHE:
Génère un JSON contenant EXACTEMENT ces catégories : {json.dumps(assigned_categories, ensure_ascii=False)}.
Pour chaque catégorie, crée 4 à 6 questions pertinentes et techniques.

FORMAT ATTENDU:
{self._get_generation_schema()}
"""
            
            if provider == "openai":
                result = await self._generate_openai(prompt)
            elif provider == "anthropic":
                result = await self._generate_anthropic(prompt)
            elif provider == "gemini":
                result = await self._generate_gemini(prompt)
            else:
                return {"success": False, "error": f"Provider inconnu: {provider}"}
            
            if result["success"]:
                cats = result.get("data", {}).get("categories", [])
                if not cats:
                    raise ValueError("JSON valide mais vide")
                logger.info(f"[{provider.upper()}] ✅ Succès: {len(cats)} catégories")
                return result
            else:
                logger.warning(f"[{provider.upper()}] ❌ Erreur: {result.get('error')}")
                if attempt < 2:
                    await asyncio.sleep(2)
                    return await self.generate_category_section(
                        provider, categories, category_indices, context, attempt + 1
                    )
                return result
        
        except Exception as e:
            logger.error(f"[{provider.upper()}] Exception: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def _generate_openai(self, prompt: str) -> Dict[str, Any]:
        """Génère avec OpenAI (JSON mode)"""
        try:
            if not self.openai_client: return {"success": False, "error": "OpenAI non configuré"}
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": self._get_generation_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            return {"success": True, "data": json.loads(response.choices[0].message.content)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _generate_anthropic(self, prompt: str) -> Dict[str, Any]:
        """Génère avec Anthropic (Prefill technique)"""
        try:
            if not self.anthropic_client: return {"success": False, "error": "Anthropic non configuré"}
            message = self.anthropic_client.messages.create(
                model=self.anthropic_model,
                max_tokens=4096,
                system=self._get_generation_system_prompt(),
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "{"}
                ],
                temperature=0.2
            )
            content = "{" + message.content[0].text
            return {"success": True, "data": self._robust_json_parse(content)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _generate_gemini(self, prompt: str) -> Dict[str, Any]:
        """Génère avec Google Gemini (JSON MIME Type)"""
        try:
            if not self.providers_status.get("gemini"):
                return {"success": False, "error": "Gemini non configuré"}
            
            # Utilisation de run_in_executor car l'API Python Gemini n'est pas nativement async partout
            loop = asyncio.get_event_loop()
            
            def call_gemini():
                model = genai.GenerativeModel(self.gemini_model)
                
                # Configuration spécifique pour Gemini 1.5 pour forcer le JSON
                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=4000,
                    temperature=0.2,
                    response_mime_type="application/json"  # <--- CRUCIAL pour la stabilité
                )
                
                full_prompt = f"{self._get_generation_system_prompt()}\n\n{prompt}"
                response = model.generate_content(full_prompt, generation_config=generation_config)
                return response.text

            content = await loop.run_in_executor(None, call_gemini)
            return {"success": True, "data": self._robust_json_parse(content)}
        
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def generate_survey_sections_parallel(
        self,
        context: Union[Dict[str, Any], ContextExtraction],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Génère les sections du questionnaire en parallèle en divisant la charge"""
        try:
            ctx_dict = context if isinstance(context, dict) else context.dict()
            categories = ctx_dict.get('categories', [])
            
            if not categories:
                categories = ["Informations Générales", "Besoins", "Satisfaction", "Suggestions"]

            # 1. Identifier les providers disponibles
            active_providers = [p for p, available in self.providers_status.items() if available]
            
            if not active_providers:
                return {"success": False, "error": "Aucun LLM configuré (OpenAI, Anthropic ou Gemini)"}

            if progress_callback:
                msg = f"🚀 Génération avec {', '.join([p.title() for p in active_providers])} ({len(categories)} catégories)"
                await progress_callback(msg, "starting")
            
            # 2. Distribution dynamique des tâches
            num_providers = len(active_providers)
            total_cats = len(categories)
            chunk_size = math.ceil(total_cats / num_providers)
            
            tasks = []
            
            for i, provider in enumerate(active_providers):
                start_idx = i * chunk_size
                # Si c'est le dernier provider, il prend tout le reste pour éviter les oublis
                if i == num_providers - 1:
                    indices = list(range(start_idx, total_cats))
                else:
                    end_idx = min((i + 1) * chunk_size, total_cats)
                    indices = list(range(start_idx, end_idx))
                
                if indices:
                    tasks.append(
                        self.generate_category_section(provider, categories, indices, ctx_dict)
                    )

            # 3. Exécution parallèle
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 4. Agrégation des résultats
            all_categories = []
            errors = []
            
            for res in results:
                if isinstance(res, Exception):
                    errors.append(str(res))
                elif isinstance(res, dict):
                    if res.get("success"):
                        all_categories.extend(res.get("data", {}).get("categories", []))
                    else:
                        errors.append(res.get("error"))

            if all_categories:
                # Ré-indexer proprement
                all_categories.sort(key=lambda x: x.get('order', 0)) # Tentative de garder l'ordre logique
                for idx, cat in enumerate(all_categories, 1):
                    cat['order'] = idx
                
                total_questions = sum(len(c.get('questions', [])) for c in all_categories)
                
                if progress_callback:
                    await progress_callback(f"✅ Terminé: {total_questions} questions via {num_providers} IA", "complete")
                
                return {
                    "success": True,
                    "categories": all_categories,
                    "total_questions": total_questions,
                    "partial_errors": errors if errors else None
                }
            else:
                return {"success": False, "error": f"Échec total: {'; '.join(filter(None, errors))}"}
        
        except Exception as e:
            logger.error(f"Erreur orchestration: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

# Instance globale
multi_llm_orchestration = MultiLLMOrchestrationService()