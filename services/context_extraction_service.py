# backend/services/context_extraction_service.py

import logging
import json
import re
from typing import Dict, Any, Optional, List
import asyncio
import pandas as pd  # ← AJOUTER CET IMPORT
import numpy as np   # ← AJOUTER CET IMPORT
from config.settings import settings
from models.survey import ContextExtraction

logger = logging.getLogger(__name__)

class ContextExtractionService:
    """Service pour extraire le contexte du prompt utilisateur"""
    
    def __init__(self):
        """Initialise le service avec la première clé OpenAI"""
        self.openai_keys = settings.get_openai_keys()
        if not self.openai_keys:
            logger.warning("Aucune clé OpenAI configurée")
        
        self.model = settings.openai_model
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialise le client OpenAI de manière lazy"""
        if self.openai_keys:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.openai_keys[0])
                logger.info("Client OpenAI initialisé")
            except ImportError as e:
                logger.error(f"Erreur import OpenAI: {e}")
                self.client = None
    
    def _get_extraction_prompt(self) -> str:
        """Retourne le prompt système pour l'extraction de contexte"""
        return """Tu es un expert en méthodologie d'enquête et en collecte de données.
        
Analyse le prompt utilisateur et extrais les informations clés pour structurer un questionnaire d'enquête.

INSTRUCTIONS:
1. Identifie l'objectif principal de l'enquête
2. Détermine le nombre de questions (entre 24 et 60, sinon 50 par défaut)
3. Identifie les zones géographiques mentionnées (régions, districts, localités)
4. Estime le nombre de lieux pour l'enquête (entre 3 et 20, sinon 5 par défaut)
5. Détermine l'audience cible
6. Propose 10 catégories de questions pertinentes
7. Estime le nombre de répondants selon l'objectif
8. Estime le nombre d'enquêteurs selon l'ampleur

Retourne UNIQUEMENT un JSON valide sans texte supplémentaire."""
    
    def _get_extraction_schema(self) -> str:
        """Retourne le schéma JSON pour l'extraction"""
        return """{
            "survey_objective": "string - L'objectif principal de l'enquête",
            "number_of_questions": "integer - Entre 24 et 60",
            "number_of_locations": "integer - Entre 3 et 20",
            "target_audience": "string - Audience cible identifiée",
            "geographic_zones": "string - Zones géographiques (ex: 'Analamanga, Antananarivo')",
            "number_of_respondents": "integer - Nombre de répondants estimé",
            "number_of_investigators": "integer - Nombre d'enquêteurs estimé",
            "categories": ["string - Noms des 10 catégories proposées"]
        }"""
    
    async def extract_context(self, user_prompt: str) -> Dict[str, Any]:
        """
        Extrait le contexte et les métadonnées du prompt utilisateur
        
        Args:
            user_prompt: Prompt fourni par l'utilisateur
        
        Returns:
            Dict contenant le contexte extrait
        """
        try:
            if not self.client:
                logger.warning("Client OpenAI non disponible")
                return {
                    "success": True,
                    "data": self._get_default_context(user_prompt)
                }
            
            logger.info(f"Extraction du contexte pour le prompt: {user_prompt[:100]}...")
            
            # Construction du prompt d'extraction
            extraction_prompt = f"""
Analyse ce prompt d'enquête et extrais les informations clés:

"{user_prompt}"

Schéma attendu:
{self._get_extraction_schema()}

Réponds UNIQUEMENT avec un JSON valide, sans texte supplémentaire."""
            
            # Appel à OpenAI dans un thread (pour ne pas bloquer)
            def call_openai():
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self._get_extraction_prompt()},
                            {"role": "user", "content": extraction_prompt}
                        ],
                        max_tokens=1000,
                        temperature=0.3,
                    )
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    logger.error(f"Erreur appel OpenAI: {e}")
                    raise
            
            # Exécuter dans un executor (non-bloquant)
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(None, call_openai)
            
            logger.debug(f"Réponse extraction: {content}")
            
            # Nettoyage du JSON si nécessaire
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Parsing JSON
            context_data = json.loads(content)
            
            # Validation des valeurs
            context_data["number_of_questions"] = max(
                settings.min_questions,
                min(context_data.get("number_of_questions", 50), settings.max_questions)
            )
            
            context_data["number_of_locations"] = max(
                3,
                min(context_data.get("number_of_locations", settings.default_num_locations), 
                    settings.max_num_locations)
            )
            
            # Validation avec Pydantic
            context = ContextExtraction(**context_data)
            
            logger.info(f"Contexte extrait: {context.survey_objective[:50]}... "
                       f"({context.number_of_questions} questions, "
                       f"{context.number_of_locations} lieux)")
            
            return {
                "success": True,
                "data": context.dict()
            }
        
        except json.JSONDecodeError as e:
            logger.error(f"Erreur parsing JSON: {e}")
            return {
                "success": False,
                "error": f"Erreur d'extraction du contexte: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Erreur extraction: {e}", exc_info=True)
            # Retourner un contexte par défaut en cas d'erreur
            return {
                "success": True,
                "data": self._get_default_context(user_prompt)
            }
    
    def _get_default_context(self, user_prompt: str) -> Dict[str, Any]:
        """Retourne un contexte par défaut si l'extraction échoue"""
        return {
            "survey_objective": user_prompt[:100],
            "number_of_questions": 50,
            "number_of_locations": 5,
            "target_audience": "Général",
            "geographic_zones": "Analamanga, Antananarivo",
            "number_of_respondents": 100,
            "number_of_investigators": 5,
            "categories": [
                "Informations générales",
                "Situation actuelle",
                "Problèmes et défis",
                "Besoins et priorités",
                "Suggestions d'amélioration",
                "Commentaires supplémentaires"
            ]
        }

    async def extract_context_with_file(self, user_prompt: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extrait le contexte AVEC analyse du fichier de données.
        """
        try:
            # 1. Extraire le contexte de base
            base_context = await self.extract_context(user_prompt)
            if not base_context["success"]:
                return base_context
            
            context_data = base_context["data"]
            
            # 2. Analyser le fichier pour enrichir le contexte
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # 3. Détecter le type d'analyse probable
            analysis_type = self._infer_analysis_type(df, context_data)
            
            # 4. Détecter la variable cible probable
            target_variable = self._infer_target_variable(df, context_data, user_prompt)
            
            # 5. Identifier les variables focus
            focus_variables = self._identify_focus_variables(df, context_data, user_prompt)
            
            # 6. Enrichir le contexte
            enriched_context = {
                **context_data,
                "analysis_type": analysis_type,
                "target_variable": target_variable,
                "focus_variables": focus_variables,
                "file_stats": {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "numeric_columns": len(numeric_cols),
                    "categorical_columns": len(categorical_cols)
                }
            }
            
            return {
                "success": True,
                "data": enriched_context
            }
            
        except Exception as e:
            logger.error(f"Erreur extraction contexte enrichi: {e}")
            return {
                "success": True,
                "data": self._get_default_context(user_prompt)
            }
    
    def _infer_analysis_type(self, df: pd.DataFrame, context: Dict[str, Any]) -> str:
        """Infère le type d'analyse basé sur les données et le contexte."""
        
        objective = context.get("survey_objective", "").lower()
        numeric_cols = df.select_dtypes(include=np.number).columns
        
        # Règles de détection
        if any(word in objective for word in ["prédire", "prédiction", "forecast", "predict"]):
            if len(numeric_cols) >= 5:
                return "regression"
            else:
                return "classification"
        
        elif any(word in objective for word in ["grouper", "segmenter", "cluster", "group"]):
            return "clustering"
        
        elif any(word in objective for word in ["associer", "lier", "relation", "corrélation"]):
            return "correlation"
        
        # Par défaut: descriptive
        return "descriptive"
    
    def _infer_target_variable(self, df: pd.DataFrame, context: Dict[str, Any], 
                              user_prompt: str) -> str:
        """Infère la variable cible."""
        
        objective = context.get("survey_objective", "").lower()
        cols = df.columns.tolist()
        
        # 1. Chercher des mots-clés dans les noms de colonnes
        target_keywords = ["target", "cible", "objectif", "resultat", "outcome", 
                          "score", "note", "rating", "évaluation"]
        
        for col in cols:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in target_keywords):
                return col
        
        # 2. Chercher dans l'objectif
        for col in cols:
            if col.lower() in objective:
                return col
        
        # 3. Prendre la première colonne numérique
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            return numeric_cols[0]
        
        # 4. Fallback
        return cols[0] if len(cols) > 0 else "unknown"
    
    def _identify_focus_variables(self, df: pd.DataFrame, context: Dict[str, Any],
                                 user_prompt: str) -> List[str]:
        """Identifie les variables importantes."""
        
        objective = context.get("survey_objective", "").lower()
        cols = df.columns.tolist()
        focus_vars = []
        
        # Chercher des variables mentionnées dans le prompt
        for col in cols:
            if col.lower() in user_prompt.lower():
                focus_vars.append(col)
        
        # Limiter à 5 variables
        return focus_vars[:5]

# Instance globale du service
context_extraction_service = ContextExtractionService()