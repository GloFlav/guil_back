from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks, WebSocketDisconnect, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
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

app = FastAPI(title="Survey Generator API v3", version="3.0.0")

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
    return {"status": "healthy", "version": "3.0.0"}

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
        # On délègue au service dédié qui utilise FileParser
        return await upload_service.process_upload_preview(file)
    except Exception as e:
        logger.error(f"Erreur upload preview: {e}")
        # On renvoie une 400 pour que le front sache que le fichier est illisible
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/analyze/file-structure-tts")
async def file_structure_tts(request: AnalyzeRequest):
    """
    ✨ ENDPOINT TTS AVEC EXPLICATION THÉMATIQUE
    
    Analyse la structure du fichier et génère une explication complète:
    1. Filtrage appliqué
    2. Analyse globale
    3. Variables clés
    4. Analyse technique détaillée
    
    Retourne le texte prêt pour TTS avec explication naturelle.
    """
    logger.info(f"🎤 TTS Structure demandé pour: {request.file_id}")
    
    # Construire le chemin du fichier
    file_path = os.path.join(settings.excel_output_dir, request.file_id)
    
    if not os.path.exists(file_path):
        logger.error(f"Fichier non trouvé: {file_path}")
        raise HTTPException(404, f"Fichier '{request.file_id}' introuvable.")
    
    try:
        # Lire le fichier
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
        
        # Préparer les stats
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
        
        # ✅ APPEL CORRECT AU SERVICE:
        result = await file_structure_analysis_service.analyze_file_structure(
            file_path=file_path,      # ✅ Correct!
            df=df,                    # ✅ Correct!
            file_stats=file_stats     # ✅ Correct!
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
    """
    Reprend le fichier uploadé via son ID, supprime les colonnes vides et renvoie le fichier.
    """
    logger.info(f"🧹 Demande de nettoyage pour le fichier {request.file_id} (Sparse: {request.remove_sparse})")
    
    file_path = os.path.join(settings.excel_output_dir, request.file_id)
    
    if not os.path.exists(file_path):
        raise HTTPException(404, "Fichier original introuvable ou expiré.")
    
    try:
        # Appel au service de nettoyage
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

# ==================== Routes Analyse Complète ====================
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
            file_structure={},  # file_structure vide car on a déjà le fichier
            context=context,
            user_prompt=request.user_prompt
        )
        
        # ===== 🎯 EXPLICATIONS PAR ONGLET - CORRECTION CRITIQUE =====
        logger.info("📝 Génération des 6 explications par onglet...")
        
        from services.tab_explanations_generator import TabExplanationsGenerator, generate_tab_explanations_async
        
        # Créer le dictionnaire de données EDA pour les explications
        eda_data = TabExplanationsGenerator.create_summary_eda_data(eda_results)
        
        # ✅ APPEL CORRECT: Pas de multi_llm_service!
        # La fonction appelle directement Anthropic en interne
        tab_explanations_raw = await generate_tab_explanations_async(
            eda_data=eda_data,
            context=context
        )
        
        logger.info(f"✅ {len(tab_explanations_raw)} explications générées")
        
        # ===== Convertir les explications brutes en objets TabExplanation =====
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
                    # Créer un TabExplanation par défaut
                    tab_explanations[tab_key] = TabExplanation(
                        title=f"Onglet {tab_key}",
                        summary="Explication non disponible",
                        recommendation="Consultez les données pour plus d'informations"
                    )
        
        # 4. Préparer les insights
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
        
        # 5. Réponse enrichie
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
# ==================== WebSocket ====================

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
        await connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await connection_manager.disconnect(websocket)

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
        
        # --- ENVOI LIEUX ---
        await progress_streamer.websocket.send_json({
            "type": "update_locations",
            "data": locations_result
        })
        await progress_streamer.send_progress("", "locations_ok", 25)
        
        # 3. Génération IA Parallèle
        logger.info("Progression IA: 🚀 Lancement génération parallèle")
        
        # --- CALLBACK DE STREAMING ---
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
            "version": "3.0.0",
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
    """
    Renvoie les 50 premières lignes du fichier (Raw ou Clean) pour prévisualisation.
    """
    logger.info(f"👀 Prévisualisation demandée pour : {file_id}")
    
    file_path = os.path.join(settings.excel_output_dir, file_id)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Fichier introuvable sur le serveur.")

    try:
        # 1. Lecture du fichier (CSV ou Excel)
        if file_id.lower().endswith('.csv'):
            try:
                df = pd.read_csv(file_path, engine='python') # Engine python plus robuste
            except:
                df = pd.read_csv(file_path, sep=';', encoding='latin1', engine='python')
        else:
            df = pd.read_excel(file_path)

        # 2. Limitation aux 50 premières lignes
        df_preview = df.head(500)

        # 3. Nettoyage pour le JSON (Très important !)
        # JSON ne supporte pas 'NaN' (Not a Number) de Pandas, il faut mettre 'null' (None en Python)
        df_preview = df_preview.replace({np.nan: None})
        
        # Gestion des dates pour éviter les erreurs de sérialisation
        for col in df_preview.select_dtypes(include=['datetime64', 'datetimetz']).columns:
            df_preview[col] = df_preview[col].dt.strftime('%Y-%m-%d %H:%M:%S')

        return {
            "file_id": file_id,
            "total_rows": len(df), # On renvoie la taille totale pour info
            "preview": df_preview.to_dict(orient='records')
        }

    except Exception as e:
        logger.error(f"❌ Erreur lecture preview: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de lecture du fichier : {str(e)}")




# ==================== Endpoints Analyse Complète ====================

@app.post("/api/v1/analyze/start-full-pipeline")
async def start_full_pipeline(request: AnalyzeRequest):
    """
    Lance le pipeline complet d'analyse sur un fichier
    
    Phases:
    1. File Structure Analysis (5%)
    2. Context Inference (20%)
    3. Feature Engineering (35%)
    4. EDA (55%)
    5. Full Analysis Synthesis (100%)
    
    Retourne immédiatement avec un job_id pour tracking
    """
    logger.info(f"🚀 Demande pipeline complet: {request.file_id}")
    
    file_path = os.path.join(settings.excel_output_dir, request.file_id)
    
    if not os.path.exists(file_path):
        raise HTTPException(404, f"Fichier '{request.file_id}' introuvable.")
    
    try:
        # Lancer le pipeline en background
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
    """
    Retourne l'état actuel de l'analyse d'un fichier
    
    Responses:
    {
      "file_id": "clean_uuid...",
      "file_structure": "completed|pending",
      "eda": "completed|pending",
      "full_analysis": "completed|pending",
      "progress": 0-100
    }
    """
    try:
        status = await analysis_pipeline.get_analysis_status(file_id)
        return status
    except Exception as e:
        logger.error(f"Erreur get_analysis_status: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/v1/analyze/results/{file_id}")
async def get_analysis_results(file_id: str):
    """
    Récupère les résultats complets d'une analyse
    
    Retourne:
    {
      "file_id": "...",
      "filename": "...",
      "analysis_type": "regression|classification|descriptive|clustering",
      "target_variable": "...",
      "summary": {
        "total_rows_original": 1000,
        "total_cols_original": 50,
        "total_rows_final": 950,
        "total_cols_final": 35,
        "numeric_cols": 20,
        "categorical_cols": 15,
        "missing_values": 150
      },
      "structure": {...},       # File structure analysis
      "context": {...},         # Context inferred
      "eda": {
        "univariate": {...},    # Stats par colonne
        "correlation": {...},   # Matrice corrélation
        "clustering": {...},    # Résultats clustering 3D
        "statistical_tests": [...],  # T-tests, Chi-2, ANOVA, etc.
        "themes": {...},        # Thématisation colonnes
        "distributions": {...},  # Histogrammes, boxplots
        "pie_charts": [...],    # Camemberts
        "scatter_plots": [...]  # Nuages de points
      },
      "insights": [...],        # Insights IA
      "tts_text": "...",       # Texte pour lecture vocale
      "timestamp": "..."
    }
    """
    try:
        results = await analysis_pipeline.get_analysis_results(file_id)
        
        if not results:
            raise HTTPException(404, f"Analyse non complétée ou fichier inexistant: {file_id}")
        
        return {
            "success": True,
            "data": results
        }
    
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
        return {
            "success": success,
            "message": f"Analyse supprimée pour {file_id}"
        }
    except Exception as e:
        logger.error(f"Erreur clear_analysis: {e}")
        raise HTTPException(500, str(e))


# ==================== WebSocket pour Streaming Analyse ====================

@app.websocket("/ws/analyze/{file_id}")
async def websocket_analyze(websocket: WebSocket, file_id: str):
    """
    WebSocket pour suivre en temps réel l'analyse d'un fichier
    
    Envoie des messages de progression au fil de l'analyse
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
            await websocket.send_json({
                "type": "progress",
                "message": message,
                "percentage": percentage,
                "timestamp": pd.Timestamp.now().isoformat()
            })
        
        # Lancer le pipeline
        logger.info(f"WebSocket analyse: {file_id}")
        
        results = await analysis_pipeline.run_complete_analysis_pipeline(
            file_id=file_id,
            file_path=file_path,
            user_prompt="",
            progress_callback=progress_callback
        )
        
        # Envoyer les résultats finaux
        await websocket.send_json({
            "type": "completed",
            "data": results,
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


# ==================== Endpoint Analyse Rapide (Sans Feature Engineering) ====================

@app.post("/api/v1/analyze/quick-eda")
async def quick_eda(request: AnalyzeRequest):
    """
    Analyse rapide: File Structure + EDA (sans Feature Engineering)
    Plus rapide mais moins complet que run_complete_analysis_pipeline
    """
    logger.info(f"⚡ Quick EDA: {request.file_id}")
    
    file_path = os.path.join(settings.excel_output_dir, request.file_id)
    
    if not os.path.exists(file_path):
        raise HTTPException(404, "Fichier introuvable.")
    
    try:
        # Charger
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        # File Structure
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
        
        # EDA direct (sans feature engineering)
        eda_results = await eda_service.run_full_eda(df, {}, request.user_prompt)
        
        return {
            "success": True,
            "file_id": request.file_id,
            "file_structure": file_structure,
            "eda": eda_results,
            "analysis_type": "quick_eda"
        }
    
    except Exception as e:
        logger.error(f"Erreur quick_eda: {e}")
        raise HTTPException(500, str(e))


# ==================== Lancement de l'App ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=settings.debug)