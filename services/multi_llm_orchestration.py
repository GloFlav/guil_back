"""
Service d'orchestration parallèle multi-LLM
VERSION ULTRA-ROBUSTE : SCHEMA STRICT POUR GEMINI
"""

import logging
import json
import asyncio
import re
import math
import random
from typing import Dict, Any, Optional, Callable, List, Union
from openai import AsyncOpenAI
import anthropic
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from config.settings import settings

logger = logging.getLogger(__name__)

class MultiLLMOrchestrationService:
    def __init__(self):
        self._init_clients()
    
    def _init_clients(self):
        self.providers_status = {}
        # 1. OpenAI
        openai_keys = settings.get_openai_keys()
        if openai_keys:
            self.openai_client = AsyncOpenAI(api_key=openai_keys[0])
            self.openai_model = settings.openai_model
            self.providers_status["openai"] = True
        else:
            self.openai_client = None
            self.providers_status["openai"] = False
        
        # 2. Anthropic
        anthropic_keys = settings.get_anthropic_keys()
        if anthropic_keys:
            self.anthropic_client = anthropic.AsyncAnthropic(api_key=anthropic_keys[0])
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

    def _get_system_prompt(self) -> str:
        return """Tu es un expert méthodologue.
Génère un JSON STRICT (une LISTE d'objets) correspondant aux catégories demandées.
Ne mets AUCUN texte avant ou après le JSON.

Structure OBLIGATOIRE :
{
  "categories": [
    {
      "category_name": "Nom Exact",
      "description": "...",
      "questions": [
        {
          "question_text": "...",
          "question_type": "text | single_choice | multiple_choice | numerical | date | gps",
          "is_required": true,
          "options": ["A", "B"] (si choix)
        }
      ]
    }
  ]
}"""

    async def generate_survey_sections_parallel(
        self, context: Dict[str, Any], progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        
        categories = context.get('categories', [])
        if not categories:
            categories = ["Général", "Besoins", "Attentes"]

        available_workers = [p for p, ok in self.providers_status.items() if ok]
        if not available_workers:
            return {"success": False, "error": "Aucun LLM disponible"}

        chunks = {}
        chunk_size = math.ceil(len(categories) / len(available_workers))
        
        for i, worker in enumerate(available_workers):
            start = i * chunk_size
            end = start + chunk_size
            worker_cats = categories[start:end]
            if worker_cats:
                chunks[worker] = worker_cats

        if progress_callback:
            await progress_callback(f"🚀 Lancement sur {len(chunks)} modèles...", "starting")

        tasks = []
        for provider, cats in chunks.items():
            tasks.append(
                self._execute_safe_task(provider, cats, context, progress_callback)
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_categories = []
        for res in results:
            if isinstance(res, dict) and res.get("success"):
                data = res.get("data", [])
                if isinstance(data, list):
                    final_categories.extend(data)
                elif isinstance(data, dict):
                    final_categories.append(data)
            else:
                logger.error(f"Tâche échouée définitivement: {res}")

        final_categories.sort(key=lambda x: categories.index(x.get('category_name')) if x.get('category_name') in categories else 999)

        return {"success": True, "categories": final_categories}

    async def _execute_safe_task(self, provider, cats, context, callback):
        result = await self._process_provider_task(provider, cats, context, callback)
        
        is_empty = False
        if result.get("success"):
            data = result.get("data", [])
            total_q = sum(len(c.get("questions", [])) for c in data)
            if total_q == 0:
                is_empty = True
                logger.warning(f"⚠️ {provider} a renvoyé un succès mais 0 questions ! Forçage du backup.")

        if result.get("success") and not is_empty:
            return result
        
        logger.warning(f"⚠️ Échec/Vide de {provider} sur {cats}. Basculement sur le BACKUP...")
        backup_provider = "openai" if self.providers_status.get("openai") and provider != "openai" else None
        if not backup_provider and self.providers_status.get("anthropic") and provider != "anthropic":
            backup_provider = "anthropic"
            
        if backup_provider:
            return await self._process_provider_task(backup_provider, cats, context, callback)
        else:
            return result

    async def _process_provider_task(
        self, provider: str, categories: List[str], context: Dict[str, Any], callback: Optional[Callable]
    ) -> Dict[str, Any]:
        try:
            logger.info(f"[{provider}] Start: {categories}")
            
            prompt = f"""CONTEXTE: {context.get('survey_objective')}

MISSION: Générer les questions pour les {len(categories)} catégories suivantes : {json.dumps(categories, ensure_ascii=False)}.

CONSIGNES:
1. Génère entre 5 et 7 questions par catégorie.
2. Sépare bien les questions dans des objets catégories distincts.
3. Utilise les noms exacts des catégories fournies.

FORMAT DE SORTIE (JSON LIST) :
[
  {{ "category_name": "{categories[0]}", "questions": [...] }},
  ... (Répéter pour chaque catégorie demandée)
]
"""
            content = ""
            
            if provider == "openai":
                response = await self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[{"role": "system", "content": self._get_system_prompt()}, {"role": "user", "content": prompt}],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content

            elif provider == "anthropic":
                msg = await self.anthropic_client.messages.create(
                    model=self.anthropic_model,
                    max_tokens=8000,
                    system=self._get_system_prompt(),
                    messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": "{"}]
                )
                content = "{" + msg.content[0].text

            # --- GEMINI (AVEC SCHEMA) ---
            elif provider == "gemini":
                loop = asyncio.get_event_loop()
                model = genai.GenerativeModel(self.gemini_model)
                
                # SÉCURITÉ DÉSACTIVÉE
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }

                # SCHEMA DE RÉPONSE STRICT (C'est la clé pour Gemini 2.0)
                # On définit la structure exacte attendue pour qu'il ne puisse pas renvoyer vide
                response_schema = {
                    "type": "object",
                    "properties": {
                        "categories": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "category_name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "questions": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "question_text": {"type": "string"},
                                                "question_type": {"type": "string"},
                                                "is_required": {"type": "boolean"},
                                                "options": {
                                                    "type": "array",
                                                    "items": {"type": "string"}
                                                }
                                            },
                                            "required": ["question_text", "question_type"]
                                        }
                                    }
                                },
                                "required": ["category_name", "questions"]
                            }
                        }
                    },
                    "required": ["categories"]
                }

                generation_config = genai.types.GenerationConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                    response_schema=response_schema # <--- INJECTION DU SCHÉMA
                )
                
                res = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: model.generate_content(
                        prompt, 
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )),
                    timeout=60.0
                )
                content = res.text

            raw_data = self._robust_json_extract(content)
            if not raw_data:
                raw_data = self._try_repair_json(content)

            if not raw_data:
                return {"success": False, "error": "Empty JSON"}

            # --- NORMALISATION ---
            # Pour Gemini avec Schema, le format est déjà propre, mais on repasse la normalisation
            # pour harmoniser les clés (question_id, expected_answers) avec le front.
            
            # Gestion liste plate éventuelle
            is_flat_question_list = False
            if isinstance(raw_data, list) and len(raw_data) > 0:
                first_item = raw_data[0]
                if ("question_text" in first_item or "question" in first_item) and "questions" not in first_item:
                    is_flat_question_list = True

            final_data = []

            if is_flat_question_list:
                logger.warning(f"[{provider}] Liste plate détectée ! Redistribution intelligente...")
                total_q = len(raw_data)
                cats_count = len(categories)
                q_per_cat = math.ceil(total_q / cats_count) if cats_count > 0 else total_q
                
                for i, cat_name in enumerate(categories):
                    start = i * q_per_cat
                    end = start + q_per_cat
                    cat_questions = raw_data[start:end]
                    if cat_questions:
                        final_data.append({
                            "category_name": cat_name,
                            "description": "Questions générées automatiquement",
                            "questions": cat_questions
                        })
            else:
                final_data = raw_data

            cleaned_data = []
            
            for cat in final_data:
                if not isinstance(cat, dict): continue

                cat['source_llm'] = provider
                
                if 'category_name' not in cat:
                    for k in ['name', 'title', 'category', 'categorie']:
                        if k in cat:
                            cat['category_name'] = cat[k]
                            break
                    if 'category_name' not in cat:
                        current_idx = len(cleaned_data)
                        if current_idx < len(categories):
                            cat['category_name'] = categories[current_idx]
                        else:
                            cat['category_name'] = "Section"

                raw_questions = cat.get('questions', [])
                if not isinstance(raw_questions, list):
                    for k, v in cat.items():
                        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                            if 'question' in v[0] or 'text' in v[0] or 'type' in v[0] or 'question_text' in v[0]:
                                raw_questions = v
                                break

                clean_questions = []
                for q in raw_questions:
                    if not isinstance(q, dict): continue

                    new_q = {}
                    new_q['question_text'] = q.get('question_text') or q.get('question') or q.get('text') or q.get('label') or "Question"
                    
                    raw_type = str(q.get('question_type') or q.get('type') or 'text').lower()
                    if any(x in raw_type for x in ['multi', 'plusieurs', 'checkbox']): new_q['question_type'] = 'multiple_choice'
                    elif any(x in raw_type for x in ['single', 'choix', 'radio', 'unique']): new_q['question_type'] = 'single_choice'
                    elif any(x in raw_type for x in ['num', 'chiffre', 'int']): new_q['question_type'] = 'numerical'
                    elif 'date' in raw_type: new_q['question_type'] = 'date'
                    elif any(x in raw_type for x in ['gps', 'loc', 'geo']): new_q['question_type'] = 'gps'
                    else: new_q['question_type'] = 'text'

                    new_q['options'] = q.get('options') or q.get('choices') or []
                    
                    new_q['expected_answers'] = []
                    if new_q['options']:
                        for idx, opt in enumerate(new_q['options']):
                            new_q['expected_answers'].append({"answer_id": str(idx+1), "answer_text": str(opt)})
                    
                    new_q['question_id'] = str(random.randint(10000, 99999))
                    new_q['is_required'] = q.get('is_required', True)

                    clean_questions.append(new_q)
                
                cat['questions'] = clean_questions
                cleaned_data.append(cat)

            # --- VERIFICATION CRITIQUE ---
            total_questions = sum(len(c['questions']) for c in cleaned_data)
            if total_questions == 0:
                return {"success": False, "error": "Zero questions generated"}

            if callback:
                await callback(message=f"✅ {provider} terminé", status="partial_data", payload=cleaned_data)

            return {"success": True, "data": cleaned_data}

        except Exception as e:
            logger.error(f"[{provider}] Crash: {e}")
            return {"success": False, "error": str(e)}

    def _robust_json_extract(self, text: str) -> List[Dict]:
        try:
            if not text: return []
            text = re.sub(r'```json', '', text, flags=re.IGNORECASE)
            text = re.sub(r'```', '', text)
            text = text.strip()
            try: return self._normalize_to_list(json.loads(text))
            except: pass
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                try: return self._normalize_to_list(json.loads(match.group()))
                except: pass
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try: return self._normalize_to_list(json.loads(match.group()))
                except: pass
            return []
        except Exception: return []

    def _try_repair_json(self, text: str) -> List[Dict]:
        try:
            text = re.sub(r'```json', '', text, flags=re.IGNORECASE)
            text = re.sub(r'```', '', text).strip()
            cutoff = max(text.rfind('},'), text.rfind('}'))
            if cutoff != -1:
                repaired = text[:cutoff+1]
                if not repaired.strip().endswith(']'): repaired += ']'
                try: return self._normalize_to_list(json.loads(repaired))
                except: pass
            return []
        except Exception: return []

    def _normalize_to_list(self, data: Any) -> List[Dict]:
        if isinstance(data, list): return data
        if isinstance(data, dict):
            for k in ["categories", "questions", "data", "survey"]:
                if k in data and isinstance(data[k], list): return data[k]
            if all(isinstance(v, list) for v in data.values()):
                normalized = []
                for k, v in data.items():
                    normalized.append({"category_name": k, "questions": v})
                return normalized
            return [data]
        return []

multi_llm_orchestration = MultiLLMOrchestrationService()