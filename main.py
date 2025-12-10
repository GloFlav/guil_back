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
from models.survey import SurveyGenerationRequest
from models.analysis import FilePreviewResponse, FullAnalysisResult, Insight
from utils.websocket_manager import connection_manager, ProgressStreamer, WebSocketLogHandler
from services.context_extraction_service import context_extraction_service
from services.administrative_data_service import adm_service
from services.multi_llm_orchestration import multi_llm_orchestration
from services.export_service import export_service
from services.upload_service import upload_service
from services.cleaning_service import cleaning_service
from services.feature_service import feature_service
from services.context_analyst import context_analyst
from services.eda_service import eda_service
from services.multi_llm_insights import multi_llm_insights



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

# ==================== Routes Export & Nettoyage (NOUVEAU) ====================

# Modèle pour la requête de nettoyage
class CleanRequest(BaseModel):
    file_id: str
    format: str = "xlsx"
    remove_sparse: bool = False


class AnalyzeRequest(BaseModel):
    file_id: str
    user_prompt: str


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


# Dans backend/main.py

@app.post("/api/v1/analyze/full", response_model=FullAnalysisResult)
async def full_analysis(request: AnalyzeRequest):
    logger.info(f"🚀 Analyse complète : {request.file_id}")
    
    clean_path = os.path.join(settings.excel_output_dir, request.file_id)
    if not os.path.exists(clean_path): raise HTTPException(404, "Fichier introuvable.")
    
    df = pd.read_excel(clean_path) 
    
    # 1. Contexte IA
    cols = df.columns.tolist()
    sample = df.head(5).copy()
    for c in sample.select_dtypes(include=['datetime64','datetimetz']): 
        sample[c] = sample[c].dt.strftime('%Y-%m-%d')
    data_sample = sample.replace({np.nan: None}).to_dict('records')
    
    context = await context_analyst.infer_analysis_goal(request.user_prompt, cols, data_sample)
    
    # 2. Feature Engineering
    df_processed = feature_service.process_features(df.copy(), context.get("target_variable", ""))
    
    # 3. EDA (Maths + Graphes + IA)
    eda_results = await eda_service.run_full_eda(df_processed, context, request.user_prompt)
    
    # 4. Réponse
    return FullAnalysisResult(
        file_id=request.file_id,
        analysis_type=context.get("analysis_type", "descriptive"),
        summary_stats={
            "target": context.get("target_variable"),
            "focus_variables": context.get("focus_variables"),
            "rows_original": len(df),
            "cols_original": len(df.columns),
            "rows_final": len(df_processed),
            "cols_features": len(df_processed.columns),
            "eda_metrics": eda_results["metrics"], 
            "charts": eda_results["charts_data"]
        },
        insights=eda_results["ai_insights"],
        visualizations=[] 
    )
# ==================== Lancement de l'App ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=settings.debug)