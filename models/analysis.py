from pydantic import BaseModel, Field
from typing import List, Any, Dict, Optional

class FilePreviewResponse(BaseModel):
    file_id: str # id du fichier nettoyé    
    raw_file_id: str # id du fichier brut original
    filename: str
    total_rows: int
    total_columns: int
    columns_list: List[str] 
    empty_columns: List[str]
    removed_empty_columns: List[str] = []
    preview_data: List[Dict[str, Any]]
    partially_empty_columns: List[Dict[str, Any]] = []
    file_size_kb: float

class Insight(BaseModel):
    """Un insight clé généré par le LLM"""
    title: str
    summary: str
    recommendation: str
    # Optionnel: référence à une colonne ou un graphique

class TabExplanation(BaseModel):
    """Explication pour un onglet spécifique de l'analyse EDA"""
    title: str = Field(..., description="Titre de l'onglet/analyse")
    summary: str = Field(..., description="Résumé des résultats")
    recommendation: str = Field(..., description="Recommandations d'action")

class Visualization(BaseModel):
    """Référence à un graphique généré (pour affichage React)"""
    id: str # ID unique du graphique
    title: str
    type: str # 'histogram', 'boxplot', 'correlation_matrix', 'scatter'
    data_url: str # URL pour l'image ou les données du graphique

class FullAnalysisResult(BaseModel):
    """Le résultat final complet de l'analyse"""
    file_id: str
    summary_stats: Dict[str, Any] = Field(..., description="Statistiques descriptives (moyenne, médiane, etc.) de base")
    insights: List[Insight]
    tab_explanations: Dict[str, TabExplanation] = Field(..., description="Explications par onglet: overview, stats, charts, tests, clustering, correlation")
    visualizations: List[Visualization]
    model_predictions: Optional[Dict[str, Any]] = None # Si un modèle ML a été entraîné
    analysis_type: str = Field(..., description="Le type d'analyse détecté (ex: regression, classification, descriptif)")