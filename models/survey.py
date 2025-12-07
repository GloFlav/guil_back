# backend/models/survey.py
"""
Modèles Pydantic pour les questionnaires et réponses
Définit la structure des données pour l'API Survey Generator
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime

class QuestionType(str, Enum):
    """Types de questions supportés"""
    SINGLE_CHOICE = "single_choice"
    MULTIPLE_CHOICE = "multiple_choice"
    TEXT = "text"
    SCALE = "scale"
    YES_NO = "yes_no"
    DATE = "date"
    NUMBER = "number"

class AnswerType(str, Enum):
    """Types de réponses"""
    TEXT = "text"
    NUMBER = "number"
    OPTION = "option"
    SCALE = "scale"
    BOOLEAN = "boolean"
    DATE = "date"

class LLMProvider(str, Enum):
    """Fournisseurs LLM supportés"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"

    # Ajouter d'autres fournisseurs si nécessaire

class ExpectedAnswer(BaseModel):
    """Modèle pour une réponse attendue"""
    answer_id: str = Field(..., description="ID unique de la réponse")
    answer_type: AnswerType = Field(..., description="Type de réponse")
    answer_text: str = Field(..., description="Texte de la réponse")
    next_question_id: Optional[str] = Field(None, description="ID de la prochaine question (logique conditionnelle)")

class Question(BaseModel):
    """Modèle pour une question"""
    question_id: str = Field(..., description="ID unique de la question")
    question_type: QuestionType = Field(..., description="Type de question")
    question_text: str = Field(..., description="Texte de la question")
    is_required: bool = Field(default=True, description="La question est-elle obligatoire?")
    help_text: Optional[str] = Field(None, description="Texte d'aide pour la question")
    predecessor_answer_id: Optional[str] = Field(None, description="Réponse qui déclenche cette question")
    expected_answers: List[ExpectedAnswer] = Field(default_factory=list, description="Réponses possibles")

class Category(BaseModel):
    """Modèle pour une catégorie de questions"""
    category_id: str = Field(..., description="ID unique de la catégorie")
    category_name: str = Field(..., description="Nom de la catégorie")
    description: str = Field(..., description="Description de la catégorie")
    order: int = Field(..., description="Ordre d'affichage")
    questions: List[Question] = Field(default_factory=list, description="Questions de la catégorie")

class SurveyMetadata(BaseModel):
    """Métadonnées du questionnaire"""
    title: str = Field(..., description="Titre du questionnaire")
    introduction: str = Field(..., description="Introduction/objectif")
    survey_total_duration: str = Field(..., description="Durée totale estimée")
    number_of_respondents: int = Field(..., description="Nombre de répondants")
    number_of_investigators: int = Field(..., description="Nombre d'enquêteurs")
    number_of_locations: int = Field(..., description="Nombre de lieux")
    location_characteristics: str = Field(..., description="Caractéristiques des lieux")
    target_audience: str = Field(..., description="Audience cible")
    survey_objective: str = Field(..., description="Objectif de l'enquête")

class Location(BaseModel):
    """Modèle pour une localisation"""
    pcode: str = Field(..., description="Code administratif unique")
    name: str = Field(..., description="Nom du lieu")
    adm1: str = Field(..., description="Région (ADM1)")
    adm2: str = Field(..., description="District (ADM2)")
    adm3: Optional[str] = Field(None, description="Lieu (ADM3)")
    latitude: Optional[float] = Field(None, description="Latitude pour Google Maps")
    longitude: Optional[float] = Field(None, description="Longitude pour Google Maps")

class ContextExtraction(BaseModel):
    """Résultat de l'extraction de contexte"""
    survey_objective: str = Field(..., description="Objectif principal")
    number_of_questions: int = Field(..., description="Nombre de questions demandées")
    number_of_locations: int = Field(..., description="Nombre de lieux demandés")
    target_audience: str = Field(..., description="Audience cible identifiée")
    geographic_zones: str = Field(default="", description="Zones géographiques identifiées")
    number_of_respondents: int = Field(default=100, description="Nombre de répondants estimé")
    number_of_investigators: int = Field(default=5, description="Nombre d'enquêteurs estimé")
    categories: List[str] = Field(default_factory=list, description="Catégories identifiées")

class SurveyResponse(BaseModel):
    """Modèle complet pour un questionnaire généré"""
    metadata: SurveyMetadata
    categories: List[Category] = Field(default_factory=list)
    locations: List[Location] = Field(default_factory=list)
    version: str = Field(default="3.0.0")
    generated_at: datetime = Field(default_factory=datetime.now)
    generated_by: Optional[str] = None
    validation: Optional[Dict[str, Any]] = None

class SurveyGenerationRequest(BaseModel):
    """Requête de génération de questionnaire"""
    user_prompt: str = Field(..., description="Prompt utilisateur")
    language: str = Field(default="fr", description="Langue (fr/en/mg)")
    enable_parallel_generation: bool = Field(default=True, description="Générer en parallèle?")
    include_locations: bool = Field(default=True, description="Inclure les lieux?")

class ProgressMessage(BaseModel):
    """Message de progression WebSocket"""
    type: str = Field(..., description="Type du message (progress/error/result)")
    message: str = Field(..., description="Message de progression")
    status: str = Field(..., description="Statut actuel")
    percentage: int = Field(default=0, description="Pourcentage de progression")
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class GenerationStats(BaseModel):
    """Statistiques de génération"""
    total_questions: int = Field(..., description="Nombre total de questions")
    num_categories: int = Field(..., description="Nombre de catégories")
    num_locations: int = Field(..., description="Nombre de lieux")
    generation_time: float = Field(..., description="Temps de génération en secondes")
    provider_used: Optional[str] = None
    backup_used: bool = Field(default=False)