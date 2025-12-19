from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks, WebSocketDisconnect, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
import logging
import traceback
import os
from datetime import datetime
from typing import Dict, Any, Optional
import asyncio
import json 
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import math


# ==================== CUSTOM JSON ENCODER POUR NaN/Inf ====================

class NaNSafeJSONEncoder(json.JSONEncoder):
    """
    🔧 Encodeur JSON qui gère les NaN, Inf, -Inf et types numpy
    """
    def default(self, obj):
        # Types numpy
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return self._clean_array(obj.tolist())
        
        # Pandas types
        if isinstance(obj, pd.DataFrame):
            return obj.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict('records')
        if isinstance(obj, pd.Series):
            return obj.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if pd.isna(obj):
            return None
            
        # Datetime
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # Bytes
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')
        
        return super().default(obj)
    
    def _clean_array(self, arr):
        """Nettoie récursivement une liste"""
        result = []
        for item in arr:
            if isinstance(item, (list, tuple)):
                result.append(self._clean_array(item))
            elif isinstance(item, float) and (math.isnan(item) or math.isinf(item)):
                result.append(None)
            elif isinstance(item, (np.floating, np.float64, np.float32)):
                if np.isnan(item) or np.isinf(item):
                    result.append(None)
                else:
                    result.append(float(item))
            elif isinstance(item, (np.integer, np.int64, np.int32)):
                result.append(int(item))
            else:
                result.append(item)
        return result
    
    def encode(self, obj):
        """Override encode pour nettoyer avant encodage"""
        return super().encode(self._deep_clean(obj))
    
    def _deep_clean(self, obj):
        """Nettoyage profond récursif"""
        if obj is None:
            return None
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: self._deep_clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._deep_clean(item) for item in obj]
        if isinstance(obj, np.ndarray):
            return self._deep_clean(obj.tolist())
        return obj


class SafeJSONResponse(JSONResponse):
    """
    🛡️ JSONResponse qui utilise l'encodeur NaN-safe
    """
    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            cls=NaNSafeJSONEncoder,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")


from config.settings import settings
from utils.websocket_manager import connection_manager, ProgressStreamer, WebSocketLogHandler
# import des services de genération de questionnaire
from models.survey import SurveyGenerationRequest
from services.multi_llm_orchestration import multi_llm_orchestration
from services.export_service import export_service
from services.context_extraction_service import context_extraction_service
from services.administrative_data_service import adm_service

# import des modèles et services pour l'analyse de données
from models.analysis import FilePreviewResponse, FullAnalysisResult, Insight, TabExplanation
from services.upload_service import upload_service
from services.cleaning_service import cleaning_service
from services.feature_service import feature_service
from services.context_analyst import context_analyst
from services.eda_service import eda_service
from services.multi_llm_insights import multi_llm_insights
from services.file_structure_analysis_service import file_structure_analysis_service
from services.analysis_pipeline_service import analysis_pipeline
from services.tab_explanations_generator import generate_tab_explanations_async
from services.smart_export_service import smart_export_service
# ==================== 🚀 NOUVEAUX IMPORTS - SMART ANALYTICS ====================
from services.feature_forge_service import feature_forge_service
from services.ml_pipeline_service import ml_pipeline_service
from services.insight_storyteller_service import insight_storyteller_service
from services.smart_analytics_orchestrator import (
    smart_analytics_orchestrator, 
    analyze_file_complete
)

# ==================== MODÈLES PYDANTIC (DÉFINIR EN HAUT!) ====================

class AnalyzeRequest(BaseModel):
    """Requête pour l'analyse complète ou TTS"""
    file_id: str
    user_prompt: str = ""  # Optionnel pour TTS


class CleanRequest(BaseModel):
    """Requête pour le nettoyage"""
    file_id: str
    format: str = "xlsx"
    remove_sparse: bool = False


class SmartAnalysisRequest(BaseModel):
    """Requête pour l'analyse intelligente complète"""
    file_id: str
    user_prompt: str = ""
    options: Optional[Dict[str, Any]] = None


class FeatureEngineeringRequest(BaseModel):
    """Requête pour le Feature Engineering seul"""
    file_id: str
    options: Optional[Dict[str, Any]] = None


class MLPipelineRequest(BaseModel):
    """Requête pour le ML Pipeline seul"""
    file_id: str
    target_variable: Optional[str] = None
    tune_hyperparams: bool = False
    options: Optional[Dict[str, Any]] = None


class StorytellerRequest(BaseModel):
    """Requête pour générer le rapport/storytelling"""
    file_id: str
    include_llm_enrichment: bool = True


# ==================== UTILITAIRES JSON ====================

def clean_nan_values(obj: Any) -> Any:
    """
    🧹 Nettoie récursivement les valeurs NaN/Inf pour la sérialisation JSON.
    
    Remplace:
    - NaN -> None
    - Inf -> None
    - -Inf -> None
    - numpy types -> Python types
    """
    if obj is None:
        return None
    
    # Gestion des types numpy
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return clean_nan_values(obj.tolist())
    
    # Gestion des floats Python
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    
    # Gestion des dictionnaires
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    
    # Gestion des listes/tuples
    if isinstance(obj, (list, tuple)):
        return [clean_nan_values(item) for item in obj]
    
    # Gestion des DataFrames pandas (les convertir en dict)
    if isinstance(obj, pd.DataFrame):
        return clean_nan_values(obj.replace({np.nan: None}).to_dict('records'))
    
    # Gestion des Series pandas
    if isinstance(obj, pd.Series):
        return clean_nan_values(obj.replace({np.nan: None}).to_dict())
    
    # Gestion des Timestamp pandas
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    
    # Gestion des datetime
    if isinstance(obj, datetime):
        return obj.isoformat()
    
    # Retourner tel quel pour les autres types (str, int, bool, etc.)
    return obj


# ==================== Configuration du Logging ====================

log_dir = os.path.dirname(settings.log_file)
if log_dir:
    os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# WebSocket Log Handler
try:
    ws_handler = WebSocketLogHandler(connection_manager)
    ws_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(ws_handler)
    logger.info("📡 WebSocket Log Handler activé")
except Exception as e:
    logger.error(f"Impossible d'activer le log WebSocket: {e}")

# ==================== App Setup ====================

app = FastAPI(
    title="Smart Analytics API", 
    version="4.0.0",
    description="API d'analyse de données automatisée avec Multi-LLM"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists(settings.excel_output_dir):
    app.mount("/exports", StaticFiles(directory=settings.excel_output_dir), name="exports")

# ==================== Routes Générales ====================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "version": "4.0.0",
        "services": {
            "eda": "active",
            "feature_forge": "active",
            "ml_pipeline": "active",
            "storyteller": "active",
            "smart_orchestrator": "active"
        }
    }

@app.get("/api/v1/locations")
async def get_locations():
    return {"success": True, "regions": list(adm_service.adm1_regions.keys())}

@app.get("/api/v1/exports/{filename}")
async def download_export(filename: str):
    path = os.path.join(settings.excel_output_dir, filename)
    if os.path.exists(path):
        return FileResponse(path, filename=filename)
    raise HTTPException(404, "Fichier non trouvé")

# ==================== Routes Analyse de Données ====================

@app.post("/api/v1/analyze/upload-preview", response_model=FilePreviewResponse)
async def upload_file_preview(file: UploadFile):
    """
    Analyse rapide du fichier dès l'upload pour afficher les métadonnées au front.
    Retourne nombre de lignes/cols, colonnes vides et preview des données.
    """
    logger.info(f"📤 Upload preview demandé pour: {file.filename}")
    try:
        return await upload_service.process_upload_preview(file)
    except Exception as e:
        logger.error(f"Erreur upload preview: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/analyze/file-structure-tts")
async def file_structure_tts(request: AnalyzeRequest):
    """
    ✨ ENDPOINT TTS AVEC EXPLICATION THÉMATIQUE
    """
    logger.info(f"🎤 TTS Structure demandé pour: {request.file_id}")
    
    file_path = os.path.join(settings.excel_output_dir, request.file_id)
    
    if not os.path.exists(file_path):
        logger.error(f"Fichier non trouvé: {file_path}")
        raise HTTPException(404, f"Fichier '{request.file_id}' introuvable.")
    
    try:
        try:
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.tsv'):
                df = pd.read_csv(file_path, sep='\t')
            else:
                raise ValueError(f"Format non supporté: {file_path}")
        except Exception as e:
            logger.error(f"Erreur lecture fichier: {e}")
            raise HTTPException(400, f"Impossible de lire le fichier: {str(e)}")
        
        file_stats = {
            "file_id": request.file_id,
            "filename": os.path.basename(file_path),
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "empty_columns": [],
            "partially_empty_columns": []
        }
        
        logger.info(f"📊 Fichier chargé: {file_stats['total_rows']} lignes, {file_stats['total_columns']} colonnes")
        
        result = await file_structure_analysis_service.analyze_file_structure(
            file_path=file_path,
            df=df,
            file_stats=file_stats
        )
        
        if not result.get("success", False):
            logger.error(f"Erreur analyse: {result.get('error')}")
            raise HTTPException(500, f"Erreur analyse: {result.get('error')}")
        
        logger.info(f"✅ TTS généré avec succès ({len(result['tts_text'])} chars)")
        
        return {
            "success": True,
            "file_id": request.file_id,
            "tts_text": result["tts_text"],
            "ai_summary": result["ai_summary"],
            "structure": result["structure"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur TTS: {e}", exc_info=True)
        raise HTTPException(500, f"Erreur interne: {str(e)}")

# ==================== Routes Export & Nettoyage ====================

@app.post("/api/v1/export/clean-download")
async def clean_and_download(request: CleanRequest):
    """Reprend le fichier uploadé via son ID, supprime les colonnes vides et renvoie le fichier."""
    logger.info(f"🧹 Demande de nettoyage pour le fichier {request.file_id}")
    
    file_path = os.path.join(settings.excel_output_dir, request.file_id)
    
    if not os.path.exists(file_path):
        raise HTTPException(404, "Fichier original introuvable ou expiré.")
    
    try:
        result = cleaning_service.auto_clean_file(
            file_path, 
            request.format, 
            request.remove_sparse
        )
        
        filename = os.path.basename(result['path'])
        
        return {
            "success": True,
            "download_url": f"/exports/{filename}",
            "removed_total": result['removed_total'],
            "logs": result['details']
        }
    except Exception as e:
        logger.error(f"Erreur nettoyage: {e}")
        raise HTTPException(500, str(e))

@app.post("/api/v1/export/excel")
async def export_excel(survey_data: Dict[str, Any]):
    logger.info("📊 Demande export Excel")
    try:
        return export_service.export_to_excel(survey_data)
    except Exception as e:
        logger.error(f"Erreur Excel: {e}")
        raise HTTPException(500, str(e))

@app.post("/api/v1/export/csv")
async def export_csv(survey_data: Dict[str, Any]):
    logger.info("📄 Demande export CSV")
    try:
        return export_service.export_to_csv(survey_data)
    except Exception as e:
        logger.error(f"Erreur CSV: {e}")
        raise HTTPException(500, str(e))

@app.post("/api/v1/export/kobo")
async def export_kobo(survey_data: Dict[str, Any]):
    logger.info("🌍 Demande export KoboToolbox")
    try:
        return export_service.export_to_kobo(survey_data)
    except Exception as e:
        logger.error(f"Erreur Kobo: {e}")
        raise HTTPException(500, str(e))

@app.post("/api/v1/create-google-form")
async def create_google_form(survey_data: Dict[str, Any]):
    logger.info("📝 Demande création Google Form API")
    try:
        result = export_service.create_google_form_online(survey_data)
        if not result["success"]:
            raise Exception(result.get("error"))
        return result
    except Exception as e:
        logger.error(f"Erreur Google API: {e}")
        raise HTTPException(500, str(e))

# ==================== Routes Analyse Complète (Legacy) ====================

@app.post("/api/v1/analyze/full", response_model=FullAnalysisResult)
async def full_analysis(request: AnalyzeRequest):
    """Analyse complète du fichier avec EDA + 6 explications par onglet."""
    logger.info(f"🚀 Analyse complète : {request.file_id}")
    
    clean_path = os.path.join(settings.excel_output_dir, request.file_id)
    if not os.path.exists(clean_path): 
        raise HTTPException(404, "Fichier introuvable.")
    
    try:
        if clean_path.endswith('.xlsx') or clean_path.endswith('.xls'):
            df = pd.read_excel(clean_path)
        else:
            df = pd.read_csv(clean_path)
        
        # 1. Contexte IA
        cols = df.columns.tolist()
        sample = df.head(5).copy()
        for c in sample.select_dtypes(include=['datetime64','datetimetz']): 
            sample[c] = sample[c].dt.strftime('%Y-%m-%d')
        data_sample = sample.replace({np.nan: None}).to_dict('records')
        
        context = await context_analyst.infer_analysis_goal(
            request.user_prompt, cols, data_sample
        )
        
        # 2. Feature Engineering
        df_processed = feature_service.process_features(
            df.copy(), context.get("target_variable", "")
        )
        
        # 3. EDA avec contexte complet
        eda_results = await eda_service.run_full_eda(
            df=df_processed,
            file_structure={},
            context=context,
            user_prompt=request.user_prompt
        )
        
        # 4. Explications par onglet
        logger.info("📝 Génération des 6 explications par onglet...")
        
        from services.tab_explanations_generator import TabExplanationsGenerator
        
        eda_data = TabExplanationsGenerator.create_summary_eda_data(eda_results)
        
        tab_explanations_raw = await generate_tab_explanations_async(
            eda_data=eda_data,
            context=context
        )
        
        logger.info(f"✅ {len(tab_explanations_raw)} explications générées")
        
        tab_explanations = {}
        for tab_key, explanation_data in tab_explanations_raw.items():
            if explanation_data and isinstance(explanation_data, dict):
                try:
                    tab_explanations[tab_key] = TabExplanation(
                        title=explanation_data.get("title", f"Onglet {tab_key}"),
                        summary=explanation_data.get("summary", ""),
                        recommendation=explanation_data.get("recommendation", "")
                    )
                except Exception as e:
                    logger.warning(f"Erreur création TabExplanation pour {tab_key}: {e}")
                    tab_explanations[tab_key] = TabExplanation(
                        title=f"Onglet {tab_key}",
                        summary="Explication non disponible",
                        recommendation="Consultez les données pour plus d'informations"
                    )
        
        # 5. Insights
        insights = []
        ai_insights = eda_results.get("ai_insights", [])
        if ai_insights and isinstance(ai_insights, list):
            for insight_data in ai_insights:
                if isinstance(insight_data, dict):
                    insights.append(Insight(
                        title=insight_data.get("title", "Insight"),
                        summary=insight_data.get("summary", ""),
                        recommendation=insight_data.get("recommendation", "")
                    ))
        
        # 6. Réponse
        full_response = FullAnalysisResult(
            file_id=request.file_id,
            analysis_type=context.get("analysis_type", "descriptive"),
            summary_stats={
                "target": context.get("target_variable"),
                "focus_variables": context.get("focus_variables", []),
                "rows_original": len(df),
                "cols_original": len(df.columns),
                "rows_final": len(df_processed),
                "cols_features": len(df_processed.columns),
                "eda_metrics": eda_results.get("metrics", {}), 
                "charts": eda_results.get("charts_data", {})
            },
            insights=insights,
            tab_explanations=tab_explanations,
            visualizations=[]
        )
        
        return full_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur analyse complète: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ==================== 🚀 NOUVEAUX ENDPOINTS - SMART ANALYTICS ====================

@app.post("/api/v1/smart-analyze/complete")
async def smart_analysis_complete(request: SmartAnalysisRequest):
    """
    🚀 ANALYSE INTELLIGENTE COMPLÈTE - 8 PHASES
    
    Pipeline complet:
    1. Chargement données
    2. Analyse structure
    3. Inférence contexte
    4. EDA exploratoire
    5. Feature Engineering
    6. Modélisation ML
    7-8. Storytelling & Rapports
    
    Retourne l'analyse complète avec insights, modèles ML et recommandations.
    """
    logger.info(f"🚀 Smart Analysis demandée pour: {request.file_id}")
    
    file_path = os.path.join(settings.excel_output_dir, request.file_id)
    
    if not os.path.exists(file_path):
        raise HTTPException(404, f"Fichier '{request.file_id}' introuvable.")
    
    try:
        result = await analyze_file_complete(
            file_id=request.file_id,
            file_path=file_path,
            user_prompt=request.user_prompt
        )
        
        if not result.get("success"):
            raise HTTPException(500, result.get("error", "Erreur inconnue"))
        
        # 🧹 Nettoyer les NaN et utiliser SafeJSONResponse
        cleaned_result = clean_nan_values(result)
        return SafeJSONResponse(content=cleaned_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur Smart Analysis: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/api/v1/smart-analyze/feature-engineering")
async def feature_engineering_endpoint(request: FeatureEngineeringRequest):
    """
    🔧 FEATURE ENGINEERING SEUL
    
    Transforme les données brutes en features intelligentes:
    - Features temporelles
    - Features d'interaction
    - Encodage catégoriel
    - Scaling
    - PCA
    - Feature Selection
    """
    logger.info(f"🔧 Feature Engineering demandé pour: {request.file_id}")
    
    file_path = os.path.join(settings.excel_output_dir, request.file_id)
    
    if not os.path.exists(file_path):
        raise HTTPException(404, f"Fichier '{request.file_id}' introuvable.")
    
    try:
        # Charger les données
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        # Inférer le contexte
        sample = df.head(5).replace({np.nan: None}).to_dict('records')
        context = await context_analyst.infer_analysis_goal(
            "", df.columns.tolist(), sample
        )
        
        # Feature Engineering
        result = await feature_forge_service.forge_features(
            df=df,
            context=context,
            options=request.options or {}
        )
        
        # Exclure le DataFrame pour la sérialisation
        response = {
            k: v for k, v in result.items() 
            if k != "df_transformed"
        }
        response["file_id"] = request.file_id
        
        # 🧹 Nettoyer et utiliser SafeJSONResponse
        return SafeJSONResponse(content=clean_nan_values(response))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur Feature Engineering: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/api/v1/smart-analyze/ml-pipeline")
async def ml_pipeline_endpoint(request: MLPipelineRequest):
    """
    🤖 ML PIPELINE SEUL
    
    Modélisation Machine Learning automatisée:
    - Détection du type de problème
    - Entraînement multi-algorithmes
    - Cross-validation
    - Feature importance
    - Analyse des erreurs
    """
    logger.info(f"🤖 ML Pipeline demandé pour: {request.file_id}")
    
    file_path = os.path.join(settings.excel_output_dir, request.file_id)
    
    if not os.path.exists(file_path):
        raise HTTPException(404, f"Fichier '{request.file_id}' introuvable.")
    
    try:
        # Charger les données
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        # Construire le contexte
        context = {
            "target_variable": request.target_variable,
            "analysis_type": "auto"
        }
        
        # Si pas de target, essayer de l'inférer
        if not request.target_variable:
            sample = df.head(5).replace({np.nan: None}).to_dict('records')
            inferred = await context_analyst.infer_analysis_goal(
                "", df.columns.tolist(), sample
            )
            context = inferred
        
        # ML Pipeline
        options = request.options or {}
        options["tune_hyperparams"] = request.tune_hyperparams
        
        result = await ml_pipeline_service.run_ml_pipeline(
            df=df,
            context=context,
            options=options
        )
        
        result["file_id"] = request.file_id
        
        # 🧹 Nettoyer et utiliser SafeJSONResponse
        return SafeJSONResponse(content=clean_nan_values(result))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur ML Pipeline: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/api/v1/smart-analyze/generate-report")
async def generate_report_endpoint(request: StorytellerRequest):
    """
    📖 GÉNÉRATION DE RAPPORT & STORYTELLING
    
    Génère le rapport complet avec:
    - Interprétation des résultats
    - Insights "So what?"
    - Data storytelling narratif
    - Recommandations actionnables
    - Export multi-format (MD, HTML)
    """
    logger.info(f"📖 Génération rapport demandée pour: {request.file_id}")
    
    # Vérifier si on a des résultats en cache
    cached_results = await smart_analytics_orchestrator.get_analysis_results(request.file_id)
    
    if not cached_results:
        raise HTTPException(
            404, 
            f"Aucune analyse trouvée pour '{request.file_id}'. "
            "Lancez d'abord /api/v1/smart-analyze/complete"
        )
    
    try:
        # Générer le rapport depuis les résultats cachés
        result = await insight_storyteller_service.tell_the_story(
            eda_results=cached_results.get("data", {}).get("eda", {}),
            ml_results=cached_results.get("data", {}).get("ml_pipeline", {}),
            feature_engineering=cached_results.get("data", {}).get("feature_engineering", {}),
            context=cached_results.get("data", {}).get("context", {}),
            options={"use_llm_enrichment": request.include_llm_enrichment}
        )
        
        result["file_id"] = request.file_id
        
        # 🧹 Nettoyer et utiliser SafeJSONResponse
        return SafeJSONResponse(content=clean_nan_values(result))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur génération rapport: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.get("/api/v1/smart-analyze/status/{file_id}")
async def get_smart_analysis_status(file_id: str):
    """
    📊 Statut d'une analyse Smart Analytics
    """
    status = await smart_analytics_orchestrator.get_analysis_status(file_id)
    return SafeJSONResponse(content=clean_nan_values(status))


@app.get("/api/v1/smart-analyze/results/{file_id}")
async def get_smart_analysis_results(file_id: str):
    """
    📊 Récupère les résultats complets d'une Smart Analysis
    """
    results = await smart_analytics_orchestrator.get_analysis_results(file_id)
    
    if not results:
        raise HTTPException(404, f"Aucun résultat trouvé pour '{file_id}'")
    
    # 🧹 Nettoyer les NaN/Inf ET utiliser SafeJSONResponse
    cleaned_results = clean_nan_values(results)
    
    return SafeJSONResponse(content=cleaned_results)


@app.delete("/api/v1/smart-analyze/clear/{file_id}")
async def clear_smart_analysis(file_id: str):
    """
    🗑️ Supprime les résultats en cache pour un fichier
    """
    smart_analytics_orchestrator.clear_cache(file_id)
    return {"success": True, "message": f"Cache vidé pour {file_id}"}


# ==================== WebSocket Smart Analytics ====================

# ... (le reste du fichier reste identique) ...

# ==================== WebSocket Smart Analytics ====================

@app.websocket("/ws/smart-analyze/{file_id}")
async def websocket_smart_analyze(websocket: WebSocket, file_id: str):
    """
    🔄 WebSocket pour suivre en temps réel l'analyse Smart Analytics
    
    ✅ CORRIGÉ: Gestion des déconnexions sans crash
    """
    await websocket.accept()
    
    file_path = os.path.join(settings.excel_output_dir, file_id)
    
    if not os.path.exists(file_path):
        await websocket.send_json({
            "type": "error",
            "message": f"Fichier '{file_id}' introuvable"
        })
        await websocket.close()
        return
    
    try:
        # Callback de progression
        async def progress_callback(message: str, percentage: int):
            try:
                await websocket.send_json({
                    "type": "progress",
                    "message": message,
                    "percentage": percentage,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.warning(f"⚠️ Erreur envoi progression: {e}")
        
        # Attendre un message de démarrage optionnel avec le prompt
        try:
            init_data = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
            user_prompt = init_data.get("user_prompt", "")
        except asyncio.TimeoutError:
            user_prompt = ""
        
        logger.info(f"🔄 WebSocket Smart Analysis: {file_id}")
        
        # Lancer l'analyse complète
        results = await smart_analytics_orchestrator.run_complete_analysis(
            file_id=file_id,
            file_path=file_path,
            user_prompt=user_prompt,
            progress_callback=progress_callback
        )
        
        # 🧹 Nettoyer les NaN avant d'envoyer via WebSocket
        cleaned_summary = json.loads(
            json.dumps(results.get("summary", {}), cls=NaNSafeJSONEncoder)
        )
        
        # Envoyer les résultats finaux
        await websocket.send_json({
            "type": "completed",
            "success": results.get("success", False),
            "summary": cleaned_summary,
            "phases_completed": results.get("phases_completed", []),
            "phases_skipped": results.get("phases_skipped", []),
            "execution_time": results.get("execution_time"),
            "timestamp": datetime.now().isoformat()
        })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket déconnecté: {file_id}")
        if connection_manager is not None:
            try:
                await connection_manager.disconnect(websocket)
            except Exception as e:
                logger.warning(f"Erreur lors de la déconnexion WebSocket: {e}")
        else:
            logger.warning("connection_manager est None")
    
    except Exception as e:
        logger.error(f"Erreur WebSocket Smart Analysis: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
        except:
            pass
    
    finally:
        try:
            await websocket.close()
        except:
            pass


# ==================== Routes Existantes (Legacy) ====================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await connection_manager.connect(websocket)
    progress_streamer = ProgressStreamer(connection_manager, websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "generate":
                await handle_generate_message(data, progress_streamer)
    
    except WebSocketDisconnect:
        logger.info("WebSocket déconnecté")
        if connection_manager is not None:
            try:
                await connection_manager.disconnect(websocket)
            except Exception as e:
                logger.warning(f"Erreur déconnexion WebSocket: {e}")
        else:
            logger.warning("connection_manager est None")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        if connection_manager is not None:
            try:
                await connection_manager.disconnect(websocket)
            except Exception as e:
                logger.warning(f"Erreur déconnexion WebSocket: {e}")


async def handle_generate_message(data: Dict[str, Any], progress_streamer: ProgressStreamer):
    """Gère la génération avec logs temps réel"""
    user_prompt = data.get("prompt", "")
    language = data.get("language", "fr")
    
    logger.info(f"Message WebSocket reçu: generate")
    await progress_streamer.send_progress("", "starting", 5)
    
    try:
        # 1. Extraction
        context_result = await context_extraction_service.extract_context(user_prompt)
        if not context_result["success"]: return
        context = context_result["data"]
        
        # --- INITIALISATION FRONT ---
        initial_structure = {
            "metadata": {
                "title": f"Enquête: {context.get('survey_objective')[:50]}...",
                "survey_objective": context.get("survey_objective", ""),
                "survey_total_duration": "Estimée...",
                "number_of_respondents": 100,
                "number_of_investigators": 5,
                "number_of_locations": context.get("number_of_locations", 5),
                "location_characteristics": "Zones contextuelles",
                "target_audience": context.get("target_audience", "Général")
            },
            "categories": [],
            "locations": []
        }
        await progress_streamer.websocket.send_json({
            "type": "init_structure",
            "data": initial_structure
        })
        
        logger.info(f"Contexte: {context.get('number_of_questions')} questions, {context.get('number_of_locations')} lieux")
        await progress_streamer.send_progress("", "context_ok", 15)
        
        # 2. Lieux
        logger.info(f"Recherche lieux pour: {context.get('geographic_zones')}")
        locations_result = adm_service.search_locations_by_context(
            context.get("geographic_zones", ""),
            context.get("number_of_locations", 5)
        )
        
        await progress_streamer.websocket.send_json({
            "type": "update_locations",
            "data": locations_result
        })
        await progress_streamer.send_progress("", "locations_ok", 25)
        
        # 3. Génération IA Parallèle
        logger.info("Progression IA: 🚀 Lancement génération parallèle")
        
        async def partial_callback(message, status, payload=None):
            if status == "partial_data" and payload:
                logger.info(f"📤 STREAMING: Envoi de {len(payload)} catégories au front")
                await progress_streamer.websocket.send_json({
                    "type": "append_categories",
                    "data": payload
                })
            else:
                await progress_streamer.send_progress(message, status, 50)
        
        generation_result = await multi_llm_orchestration.generate_survey_sections_parallel(
            context, progress_callback=partial_callback
        )
        
        if not generation_result["success"]: return
        
        # 4. Finalisation
        categories = generation_result.get("categories", [])
        
        survey_response = {
            "metadata": initial_structure["metadata"],
            "categories": categories,
            "locations": locations_result,
            "version": "4.0.0",
            "generated_at": datetime.now().isoformat(),
            "language": language
        }
        
        await progress_streamer.send_result(survey_response)
        
        logger.info(f"Génération terminée avec succès ({len(categories)} cats)")
        await progress_streamer.send_progress("Terminé", "success", 100)

    except Exception as e:
        logger.error(f"Erreur critique: {e}", exc_info=True)
        await progress_streamer.send_error(f"Erreur interne: {str(e)}", "INTERNAL_ERROR")


@app.get("/api/v1/files/{file_id}/preview")
async def get_file_preview(file_id: str):
    """Renvoie les 50 premières lignes du fichier pour prévisualisation."""
    logger.info(f"👀 Prévisualisation demandée pour : {file_id}")
    
    file_path = os.path.join(settings.excel_output_dir, file_id)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Fichier introuvable sur le serveur.")

    try:
        if file_id.lower().endswith('.csv'):
            try:
                df = pd.read_csv(file_path, engine='python')
            except:
                df = pd.read_csv(file_path, sep=';', encoding='latin1', engine='python')
        else:
            df = pd.read_excel(file_path)

        df_preview = df.head(500)
        df_preview = df_preview.replace({np.nan: None})
        
        for col in df_preview.select_dtypes(include=['datetime64', 'datetimetz']).columns:
            df_preview[col] = df_preview[col].dt.strftime('%Y-%m-%d %H:%M:%S')

        return {
            "file_id": file_id,
            "total_rows": len(df),
            "preview": df_preview.to_dict(orient='records')
        }

    except Exception as e:
        logger.error(f"❌ Erreur lecture preview: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de lecture du fichier : {str(e)}")


# ==================== Endpoints Legacy Pipeline ====================

@app.post("/api/v1/analyze/start-full-pipeline")
async def start_full_pipeline(request: AnalyzeRequest):
    """Lance le pipeline complet d'analyse (version legacy)"""
    logger.info(f"🚀 Demande pipeline complet: {request.file_id}")
    
    file_path = os.path.join(settings.excel_output_dir, request.file_id)
    
    if not os.path.exists(file_path):
        raise HTTPException(404, f"Fichier '{request.file_id}' introuvable.")
    
    try:
        task = asyncio.create_task(
            analysis_pipeline.run_complete_analysis_pipeline(
                file_id=request.file_id,
                file_path=file_path,
                user_prompt=request.user_prompt
            )
        )
        
        return {
            "success": True,
            "file_id": request.file_id,
            "message": "Pipeline lancé en arrière-plan",
            "status_url": f"/api/v1/analyze/status/{request.file_id}",
            "results_url": f"/api/v1/analyze/results/{request.file_id}"
        }
    
    except Exception as e:
        logger.error(f"Erreur start_full_pipeline: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/v1/analyze/status/{file_id}")
async def get_analysis_status(file_id: str):
    """Retourne l'état actuel de l'analyse d'un fichier"""
    try:
        status = await analysis_pipeline.get_analysis_status(file_id)
        return clean_nan_values(status)
    except Exception as e:
        logger.error(f"Erreur get_analysis_status: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/v1/analyze/results/{file_id}")
async def get_analysis_results(file_id: str):
    """Récupère les résultats complets d'une analyse"""
    try:
        results = await analysis_pipeline.get_analysis_results(file_id)
        
        if not results:
            raise HTTPException(404, f"Analyse non complétée ou fichier inexistant: {file_id}")
        
        return {"success": True, "data": clean_nan_values(results)}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur get_analysis_results: {e}")
        raise HTTPException(500, str(e))


@app.delete("/api/v1/analyze/clear/{file_id}")
async def clear_analysis(file_id: str):
    """Supprime l'analyse en cache pour un fichier"""
    try:
        success = analysis_pipeline.clear_analysis(file_id)
        return {"success": success, "message": f"Analyse supprimée pour {file_id}"}
    except Exception as e:
        logger.error(f"Erreur clear_analysis: {e}")
        raise HTTPException(500, str(e))


@app.websocket("/ws/analyze/{file_id}")
async def websocket_analyze(websocket: WebSocket, file_id: str):
    """WebSocket pour suivre en temps réel l'analyse d'un fichier (legacy)"""
    await websocket.accept()
    
    file_path = os.path.join(settings.excel_output_dir, file_id)
    
    if not os.path.exists(file_path):
        await websocket.send_json({
            "type": "error",
            "message": f"Fichier '{file_id}' introuvable"
        })
        await websocket.close()
        return
    
    try:
        async def progress_callback(message: str, percentage: int):
            await websocket.send_json({
                "type": "progress",
                "message": message,
                "percentage": percentage,
                "timestamp": pd.Timestamp.now().isoformat()
            })
        
        logger.info(f"WebSocket analyse: {file_id}")
        
        results = await analysis_pipeline.run_complete_analysis_pipeline(
            file_id=file_id,
            file_path=file_path,
            user_prompt="",
            progress_callback=progress_callback
        )
        
        await websocket.send_json({
            "type": "completed",
            "data": clean_nan_values(results),
            "timestamp": pd.Timestamp.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Erreur WebSocket analyze: {e}", exc_info=True)
        await websocket.send_json({
            "type": "error",
            "message": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        })
    
    finally:
        try:
            await websocket.close()
        except:
            pass


@app.post("/api/v1/analyze/quick-eda")
async def quick_eda(request: AnalyzeRequest):
    """Analyse rapide: File Structure + EDA (sans Feature Engineering)"""
    logger.info(f"⚡ Quick EDA: {request.file_id}")
    
    file_path = os.path.join(settings.excel_output_dir, request.file_id)
    
    if not os.path.exists(file_path):
        raise HTTPException(404, "Fichier introuvable.")
    
    try:
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        file_stats = {
            "file_id": request.file_id,
            "filename": os.path.basename(file_path),
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "empty_columns": [],
            "partially_empty_columns": []
        }
        
        file_structure = await file_structure_analysis_service.analyze_file_structure(
            file_path, df, file_stats
        )
        
        eda_results = await eda_service.run_full_eda(df, {}, request.user_prompt)
        
        return clean_nan_values({
            "success": True,
            "file_id": request.file_id,
            "file_structure": file_structure,
            "eda": eda_results,
            "analysis_type": "quick_eda"
        })
    
    except Exception as e:
        logger.error(f"Erreur quick_eda: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/v1/smart-analyze/export-pdf")
async def export_pdf_report(request: AnalyzeRequest):
    """
    📄 GÉNÈRE LE RAPPORT PDF PROFESSIONNEL AVEC INTERPRÉTATION DÉCISIONNELLE
    """
    logger.info(f"📄 Export PDF demandé pour: {request.file_id}")
    
    # 1. Récupération des données d'analyse complètes en cache
    cached_results = await smart_analytics_orchestrator.get_analysis_results(request.file_id)
    
    if not cached_results:
        raise HTTPException(
            404, 
            "Données d'analyse introuvables. Veuillez d'abord exécuter l'analyse intelligente."
        )
    
    try:
        # 2. Appel au service d'export professionnel
        export_result = await smart_export_service.generate_professional_report(
            cached_results, 
            request.user_prompt
        )
        
        if not export_result["success"]:
            raise HTTPException(500, export_result["error"])
            
        return export_result
        
    except Exception as e:
        logger.error(f"❌ Erreur Export PDF: {e}")
        raise HTTPException(500, str(e))


# ==================== Lancement de l'App ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=settings.debug)