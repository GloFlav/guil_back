from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks, WebSocketDisconnect
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

from config.settings import settings
from models.survey import SurveyGenerationRequest
from utils.websocket_manager import connection_manager, ProgressStreamer, WebSocketLogHandler
from services.context_extraction_service import context_extraction_service
from services.administrative_data_service import adm_service
from services.multi_llm_orchestration import multi_llm_orchestration
from services.export_service import export_service

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

# ==================== Routes Export ====================

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
        # On envoie le squelette vide pour afficher l'interface immédiatement
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
            # Si c'est un message de données partielles
            if status == "partial_data" and payload:
                logger.info(f"📤 STREAMING: Envoi de {len(payload)} catégories au front")
                await progress_streamer.websocket.send_json({
                    "type": "append_categories",
                    "data": payload
                })
            else:
                # Sinon c'est un log d'avancement classique
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=settings.debug)