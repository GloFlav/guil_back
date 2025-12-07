import json
import logging
from typing import Dict, Any, Optional, Callable
from openai import OpenAI
from config.settings import settings
from models.survey import SurveyResponse
import traceback

# Configuration du logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

class GPTService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
    
    def _get_system_prompt(self) -> str:
        return """Tu es un expert en création de questionnaires d'enquête. 
        Tu dois générer un questionnaire structuré et professionnel basé sur la demande de l'utilisateur.
        
        RÈGLES OBLIGATOIRES :
        - Génère EXACTEMENT entre 25 et 30 questions (PAS MOINS DE 25)
        - Organise en 5-6 catégories avec 5-6 questions chacune
        - Utilise des types de questions variés (multiple_choice, single_choice, text, scale, yes_no, number)
        - Chaque question doit avoir un ID unique (q1, q2, q3, etc.)
        - Crée des réponses détaillées pour les questions à choix multiples
        - Prévois une logique conditionnelle avec next_question_id
        - Adapte au contexte français/malgache
        
        STRUCTURE OBLIGATOIRE :
        - Catégorie 1: Informations générales (5-6 questions)
        - Catégorie 2: Situation actuelle (5-6 questions) 
        - Catégorie 3: Problèmes/défis (5-6 questions)
        - Catégorie 4: Besoins/priorités (5-6 questions)
        - Catégorie 5: Suggestions/améliorations (5-6 questions)
        - Catégorie 6 (optionnelle): Perspectives d'avenir (3-5 questions)
        
        IMPORTANT : Le questionnaire DOIT contenir AU MINIMUM 25 questions.
        
        Format de réponse : JSON strictement conforme au schéma SurveyResponse fourni.
        Ne renvoie que le JSON, sans texte supplémentaire."""
    
    def _get_json_schema(self) -> str:
        return """{
            "metadata": {
                "title": "string",
                "introduction": "string", 
                "survey_total_duration": "string",
                "number_of_respondents": "integer",
                "number_of_investigators": "integer", 
                "number_of_locations": "integer",
                "location_characteristics": "string",
                "target_audience": "string",
                "survey_objective": "string"
            },
            "categories": [
                {
                    "category_id": "string",
                    "category_name": "string",
                    "description": "string",
                    "order": "integer",
                    "questions": [
                        {
                            "question_id": "string",
                            "question_type": "multiple_choice|single_choice|text|scale|yes_no|date|number",
                            "question_text": "string",
                            "is_required": "boolean",
                            "help_text": "string",
                            "predecessor_answer_id": "string|null",
                            "expected_answers": [
                                {
                                    "answer_id": "string",
                                    "answer_type": "text|number|option|scale|boolean|date",
                                    "answer_text": "string",
                                    "next_question_id": "string|null"
                                }
                            ]
                        }
                    ]
                }
            ],
            "version": "string"
        }"""

    async def generate_survey(
        self, 
        user_prompt: str, 
        log_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Génère un questionnaire basé sur le prompt utilisateur avec retry si insuffisant
        """
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                if log_callback:
                    if attempt == 0:
                        log_callback("Début de la génération du questionnaire...")
                    else:
                        log_callback(f"Tentative {attempt + 1}/3...")
                
                logger.info(f"Génération d'un questionnaire (tentative {attempt + 1}) pour le prompt: {user_prompt[:100]}...")
                
                # Construction du prompt renforcé
                full_prompt = f"""
                Génère un questionnaire d'enquête détaillé basé sur cette demande :
                
                "{user_prompt}"
                
                EXIGENCES ABSOLUES - TRÈS IMPORTANT :
                - Créer EXACTEMENT entre 25 et 30 questions (PAS MOINS DE 25)
                - Organiser en 5-6 catégories thématiques
                - Chaque catégorie DOIT contenir 5-6 questions minimum
                - Utiliser tous les types de questions : multiple_choice, single_choice, text, scale, yes_no, number
                - Créer des questions précises et détaillées
                - Ajouter une logique conditionnelle avec next_question_id
                - Fournir des réponses complètes pour les choix multiples
                
                STRUCTURE OBLIGATOIRE pour "{user_prompt}" :
                1. Informations générales (6 questions) : rôle, institution, localisation, expérience, etc.
                2. État actuel (6 questions) : évaluation des infrastructures, services, ressources actuelles
                3. Problèmes et défis (5 questions) : identification des problèmes principaux et secondaires
                4. Besoins et priorités (5 questions) : besoins prioritaires, ressources nécessaires
                5. Suggestions d'amélioration (5 questions) : propositions concrètes, innovations souhaitées
                6. Perspectives d'avenir (3 questions) : vision à long terme, objectifs futurs
                
                TOTAL ATTENDU : 30 questions réparties comme indiqué ci-dessus.
                
                Le questionnaire doit suivre exactement ce schéma JSON :
                {self._get_json_schema()}
                
                RAPPEL CRITIQUE : Le questionnaire final DOIT contenir AU MINIMUM 25 questions.
                Compte bien tes questions avant de finaliser la réponse JSON.
                """
                
                if log_callback:
                    log_callback("Envoi de la requête à OpenAI...")
                
                # Appel à l'API OpenAI avec plus de tokens
                response = self.client.chat.completions.create(
                    model=self.model,  # Utilisez la variable de configuration
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": full_prompt}
                    ],
                    max_completion_tokens=2500,  # Paramètre corrigé
                    response_format={"type": "json_object"}  # Pour forcer le format JSON
                )
                
                if log_callback:
                    log_callback("Réponse reçue, traitement en cours...")
                
                # Extraction du contenu
                content = response.choices[0].message.content.strip()
                logger.info(f"Réponse brute de GPT: {content[:200]}...")
                
                # Nettoyage du JSON
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                # Parsing JSON
                try:
                    survey_data = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"Erreur de parsing JSON (tentative {attempt + 1}): {e}")
                    if attempt == max_attempts - 1:
                        return {
                            "success": False,
                            "error": f"Erreur de format dans la réponse GPT après {max_attempts} tentatives: {str(e)}"
                        }
                    continue
                
                # Compter les questions avant validation Pydantic
                total_questions = sum(len(cat["questions"]) for cat in survey_data.get("categories", []))
                logger.info(f"Questionnaire généré avec {total_questions} questions (tentative {attempt + 1})")
                
                # Si pas assez de questions, retry (sauf dernière tentative)
                if total_questions < 25 and attempt < max_attempts - 1:
                    logger.warning(f"Seulement {total_questions} questions générées, nouvelle tentative...")
                    if log_callback:
                        log_callback(f"Seulement {total_questions} questions générées, nouvelle tentative...")
                    continue
                
                # Validation avec Pydantic
                try:
                    survey_response = SurveyResponse(**survey_data)
                    if log_callback:
                        log_callback(f"Questionnaire généré avec succès! ({total_questions} questions)")
                    
                    logger.info(f"Questionnaire généré et validé avec succès - {total_questions} questions")
                    
                    return {
                        "success": True,
                        "data": survey_response.dict()
                    }
                except Exception as e:
                    logger.error(f"Erreur de validation Pydantic (tentative {attempt + 1}): {e}")
                    
                    # Si c'est juste le nombre de questions et qu'on a encore des tentatives
                    if "au moins 20 questions" in str(e) and attempt < max_attempts - 1:
                        logger.info("Retry pour générer plus de questions...")
                        continue
                    
                    # Sinon, c'est une vraie erreur de validation
                    if attempt == max_attempts - 1:
                        logger.error(f"Données problématiques: {survey_data}")
                        return {
                            "success": False,
                            "error": f"Erreur de validation des données après {max_attempts} tentatives: {str(e)}"
                        }
                    continue
                    
            except Exception as e:
                error_msg = f"Erreur lors de la génération (tentative {attempt + 1}): {str(e)}"
                logger.error(error_msg)
                
                if attempt == max_attempts - 1:
                    logger.error(traceback.format_exc())
                    if log_callback:
                        log_callback(f"Erreur: {error_msg}")
                    
                    return {
                        "success": False,
                        "error": f"Erreur après {max_attempts} tentatives: {str(e)}"
                    }
                continue
        
        # Si on arrive ici, toutes les tentatives ont échoué
        return {
            "success": False,
            "error": f"Impossible de générer un questionnaire valide après {max_attempts} tentatives"
        }

# Instance globale du service
gpt_service = GPTService()

# Fonction wrapper pour la compatibilité
async def generate_survey(user_prompt: str, log_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Fonction wrapper pour maintenir la compatibilité avec l'ancien code"""
    return await gpt_service.generate_survey(user_prompt, log_callback)
