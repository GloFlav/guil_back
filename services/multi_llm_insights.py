import asyncio
import json
import logging
from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError
from anthropic import Anthropic
from typing import List, Dict, Any, Tuple, Optional
from models.analysis import Insight

logger = logging.getLogger(__name__)

class MultiLLMInsights:
    """
    Service multi-LLM COMPATIBLE avec l'ancien code.
    - Pas de dépendance tenacity
    - Fallback intelligent Claude→OpenAI→Manuel
    - Parsing JSON robuste
    - Retry manuel avec backoff exponentiel
    """

    def __init__(self, settings=None):
        self.settings = settings
        self.max_retries = 2
        self.retry_delay = 2  # secondes, augmente de manière exponentielle

    # =========================================================
    # RETRY MANUEL (Sans tenacity)
    # =========================================================

    async def _call_with_retry(self, func, *args, max_retries=2, **kwargs):
        """Appelle une fonction avec retry manuel et backoff exponentiel."""
        delay = self.retry_delay
        
        for attempt in range(max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                return result
            except (APIError, APIConnectionError, RateLimitError) as e:
                if attempt < max_retries:
                    logger.warning(f"⚠️ Tentative {attempt + 1} échouée. Retry dans {delay}s...")
                    await asyncio.sleep(delay)
                    delay *= 2  # Backoff exponentiel
                else:
                    logger.error(f"❌ Toutes les tentatives échouées: {e}")
                    return None
            except asyncio.TimeoutError:
                logger.error(f"❌ Timeout après {attempt + 1} tentatives")
                return None
            except Exception as e:
                logger.error(f"❌ Erreur inattendue: {e}")
                return None

        return None

    # =========================================================
    # PARSING JSON ROBUSTE
    # =========================================================

    @staticmethod
    def _extract_and_parse_json(text: str) -> Optional[Dict]:
        """Extrait le JSON du texte LLM et le parse."""
        if not text:
            return None

        # Cas 1: JSON pur
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Cas 2: JSON dans le texte (cherche {... } ou [...])
        import re
        
        # Cherche le JSON object
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Cherche JSON array
        match = re.search(r'\[[\s\S]*\]', text)
        if match:
            try:
                parsed = json.loads(match.group())
                # Si c'est une liste, retourne le premier élément ou la liste
                if isinstance(parsed, list) and len(parsed) > 0:
                    return parsed[0] if isinstance(parsed[0], dict) else None
                return parsed
            except json.JSONDecodeError:
                pass

        logger.warning(f"⚠️ Impossible de parser JSON de: {text[:80]}")
        return None

    @staticmethod
    def _validate_insight_structure(obj: Any) -> bool:
        """Valide la structure d'un insight."""
        if not isinstance(obj, dict):
            return False
        
        required_keys = {'title', 'summary', 'recommendation'}
        return all(
            key in obj and 
            isinstance(obj[key], str) and 
            len(obj[key].strip()) > 0
            for key in required_keys
        )

    @staticmethod
    def _create_fallback_insight(error_msg: str = "") -> Dict:
        """Crée un insight de fallback sans LLM."""
        return {
            "title": "⚠️ Analyse Partielle",
            "summary": "Les données ont été analysées mais les insights IA n'ont pu être générés.",
            "recommendation": "Vérifiez les graphiques et statistiques pour explorer vos données."
        }

    # =========================================================
    # APPELS LLM
    # =========================================================

    async def _call_openai(self, prompt: str, data: str, task_id: str = "task") -> Optional[Dict]:
        """Appel OpenAI avec gestion d'erreur."""
        try:
            logger.info(f"📞 Appel OpenAI pour {task_id}")
            
            from config.settings import settings
            openai_keys = settings.get_openai_keys()
            
            if not openai_keys:
                logger.warning("⚠️ Aucune clé OpenAI disponible")
                return None

            client = AsyncOpenAI(api_key=openai_keys[0])
            
            full_prompt = f"""{prompt}

Données:
{data}"""

            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[
                        {
                            "role": "user",
                            "content": full_prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=500
                ),
                timeout=30
            )

            response_text = response.choices[0].message.content
            logger.info(f"✓ Réponse OpenAI reçue ({len(response_text)} chars)")

            # Parse JSON
            parsed = self._extract_and_parse_json(response_text)
            
            if parsed and self._validate_insight_structure(parsed):
                return parsed

            logger.warning(f"⚠️ Structure invalide: {response_text[:100]}")
            return None

        except (APIError, APIConnectionError, RateLimitError, asyncio.TimeoutError) as e:
            logger.error(f"❌ Erreur API OpenAI: {type(e).__name__}")
            return None
        except Exception as e:
            logger.error(f"❌ Erreur OpenAI: {e}")
            return None

    async def _call_claude(self, prompt: str, data: str, task_id: str = "task") -> Optional[Dict]:
        """Appel Claude avec gestion d'erreur."""
        try:
            logger.info(f"📞 Appel Claude pour {task_id}")
            
            from config.settings import settings
            anthropic_keys = settings.get_anthropic_keys()
            
            if not anthropic_keys:
                logger.warning("⚠️ Aucune clé Anthropic disponible")
                return None

            client = Anthropic(api_key=anthropic_keys[0])

            full_prompt = f"""{prompt}

Données:
{data}"""

            message = client.messages.create(
                model=settings.anthropic_model,
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                timeout=30
            )

            response_text = message.content[0].text
            logger.info(f"✓ Réponse Claude reçue ({len(response_text)} chars)")

            # Parse JSON
            parsed = self._extract_and_parse_json(response_text)
            
            if parsed and self._validate_insight_structure(parsed):
                return parsed

            logger.warning(f"⚠️ Structure invalide: {response_text[:100]}")
            return None

        except Exception as e:
            logger.error(f"❌ Erreur Claude: {e}")
            return None

    # =========================================================
    # API PRINCIPALE (Compatible avec ancien code)
    # =========================================================

    async def run_parallel_analysis(self, tasks_data: List[Dict[str, Any]]) -> List[Insight]:
        """
        Lance plusieurs tâches d'analyse LLM en parallèle.
        
        Format des tasks:
        [
            {
                "prompt": "Analyse...",
                "data": "{...json...}",
                "task_id": "optional"  # Optionnel
            }
        ]
        
        Returns:
            List[Insight] : Insights valides générés ou fallback
        """
        if not tasks_data:
            logger.warning("⚠️ Aucune tâche à exécuter")
            return []

        logger.info(f"🚀 Lancement {len(tasks_data)} tâches LLM")

        results = []
        
        for task in tasks_data:
            try:
                prompt = task.get("prompt", "")
                data = task.get("data", "")
                task_id = task.get("task_id", "unknown")

                if not prompt or not data:
                    logger.warning(f"⚠️ Tâche {task_id}: prompt ou data manquant")
                    results.append(self._create_fallback_insight())
                    continue

                # Stratégie: Essayer Claude d'abord, puis OpenAI
                logger.info(f"→ Tâche {task_id}: Tentative 1 (Claude)")
                insight = await self._call_claude(prompt, data, task_id)

                # Fallback OpenAI
                if not insight:
                    logger.info(f"→ Tâche {task_id}: Tentative 2 (OpenAI)")
                    insight = await self._call_openai(prompt, data, task_id)

                # Fallback manuel
                if not insight:
                    logger.warning(f"⚠️ Tâche {task_id}: Fallback manuel")
                    insight = self._create_fallback_insight()

                # Convertir en Insight object si nécessaire
                if isinstance(insight, dict):
                    try:
                        results.append(Insight(**insight))
                        logger.info(f"✓ Tâche {task_id}: OK")
                    except TypeError:
                        # Si Insight ne peut pas être créé avec ces args, garder le dict
                        results.append(insight)
                else:
                    results.append(insight)

            except Exception as e:
                logger.error(f"❌ Erreur tâche {task.get('task_id', '?')}: {e}")
                results.append(self._create_fallback_insight(str(e)))

        logger.info(f"✓ Analyse parallèle terminée: {len(results)} insights")
        return results

    # =========================================================
    # POUR COMPATIBILITÉ: Générateur de tâches
    # =========================================================

    @staticmethod
    def create_task(prompt: str, data: str, task_id: str = "task") -> Dict:
        """Helper pour créer une tâche."""
        return {
            "prompt": prompt,
            "data": data,
            "task_id": task_id
        }

# =========================================================
# INSTANCIATION GLOBALE
# =========================================================

multi_llm_insights = None

def init_multi_llm_insights(settings=None):
    """Initialise le service."""
    global multi_llm_insights
    multi_llm_insights = MultiLLMInsights(settings)
    logger.info("✓ Service Multi-LLM initialisé (sans tenacity)")
    return multi_llm_insights

# Instanciation par défaut si besoin
if multi_llm_insights is None:
    try:
        from config.settings import settings
        multi_llm_insights = MultiLLMInsights(settings)
    except:
        multi_llm_insights = MultiLLMInsights()