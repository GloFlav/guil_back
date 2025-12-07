# backend/main.py
"""
Application FastAPI - Survey Generator Madagascar v3.0.0
API pour générer des questionnaires d'enquête avec IA multi-LLM
"""

from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
import logging
import traceback
import os
from datetime import datetime
from typing import Dict, Any, Optional
import asyncio
import json

from config.settings import settings
from models.survey import SurveyGenerationRequest, SurveyResponse, ProgressMessage
from utils.websocket_manager import connection_manager, ProgressStreamer
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

# ==================== Création de l'Application ====================

app = FastAPI(
    title="Survey Generator API v3 Madagascar",
    description="API pour générer des questionnaires d'enquête avec IA multi-LLM et streaming WebSocket",
    version="3.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# ==================== CORS Configuration ====================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ==================== Serveur des fichiers statiques ====================

if os.path.exists(settings.excel_output_dir):
    app.mount("/exports", StaticFiles(directory=settings.excel_output_dir), name="exports")

# ==================== Middleware ====================

@app.middleware("http")
async def log_requests(request, call_next):
    """Middleware pour logger les requêtes HTTP"""
    start_time = datetime.now()
    
    logger.info(f"[{request.method}] {request.url}")
    
    response = await call_next(request)
    
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"[{request.method}] {request.url} - {response.status_code} - {process_time:.3f}s")
    
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """Gestionnaire global d'exceptions"""
    logger.error(f"Error on {request.url}: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Une erreur interne s'est produite",
            "detail": str(exc) if settings.debug else "Contactez l'administrateur"
        }
    )

# ==================== Routes REST ====================

@app.get("/")
async def root():
    """Point d'entrée de l'API"""
    return {
        "message": "Survey Generator API v3 Madagascar",
        "version": "3.0.0",
        "status": "active",
        "environment": settings.environment,
        "endpoints": {
            "health": "/health",
            "ws": "/ws",
            "generate": "/api/v1/generate",
            "export": "/api/v1/export",
            "locations": "/api/v1/locations",
            "docs": "/docs"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Vérification de l'état de l'API"""
    api_keys_validation = settings.validate_api_keys()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "environment": settings.environment,
        "services": {
            "openai": api_keys_validation["openai"]["available"],
            "anthropic": api_keys_validation["anthropic"]["available"],
            "gemini": api_keys_validation["gemini"]["available"],
            "administrative_data": adm_service.df is not None
        },
        "api_keys": api_keys_validation
    }

@app.post("/api/v1/generate")
async def generate_survey(request: SurveyGenerationRequest):
    """
    Endpoint REST pour générer un questionnaire
    À utiliser via WebSocket pour le streaming
    """
    try:
        return {
            "success": False,
            "message": "Utilisez le WebSocket pour la génération avec streaming",
            "ws_endpoint": "/ws"
        }
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/locations")
async def get_locations():
    """Retourne les statistiques et données de localisation"""
    try:
        stats = adm_service.get_statistics()
        regions = adm_service.get_adm1_regions()
        
        return {
            "success": True,
            "statistics": stats,
            "regions": regions
        }
    except Exception as e:
        logger.error(f"Error getting locations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/locations/{region}")
async def get_locations_by_region(region: str):
    """Retourne les lieux par région"""
    try:
        districts = adm_service.get_adm2_districts(region)
        
        locations_by_district = {}
        for district in districts:
            locations_by_district[district] = adm_service.get_adm3_locations(district)
        
        return {
            "success": True,
            "region": region,
            "districts": districts,
            "locations_by_district": locations_by_district
        }
    except Exception as e:
        logger.error(f"Error getting locations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/export/{survey_id}")
async def export_survey(survey_id: str, survey_data: Dict[str, Any]):
    """Exporte un questionnaire dans différents formats"""
    try:
        format_type = survey_data.get("format", "xlsx")
        
        # Dispatcher vers le bon format
        exporters = {
            "xlsx": export_service.export_to_excel,
            "csv": export_service.export_to_csv,
            "json": export_service.export_to_json,
            "kobo": export_service.export_to_kobo,
            "google_forms": export_service.export_to_google_forms
        }
        
        if format_type not in exporters:
            return {"success": False, "error": f"Format non supporté: {format_type}"}
        
        result = exporters[format_type](survey_data)
        
        if result["success"]:
            logger.info(f"Export {format_type} successful: {result.get('filename')}")
        
        return result
    
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/exports")
async def list_exports():
    """Liste les fichiers exportés"""
    try:
        files = export_service.list_exported_files()
        return {
            "success": True,
            "files": files,
            "count": len(files)
        }
    except Exception as e:
        logger.error(f"Error listing exports: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/exports/{filename}")
async def download_export(filename: str):
    """Télécharge un fichier exporté"""
    try:
        filepath = os.path.join(settings.excel_output_dir, filename)
        
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Fichier non trouvé")
        
        if not filename.endswith(('.xlsx', '.csv', '.pdf', '.json', '.xml')):
            raise HTTPException(status_code=400, detail="Type de fichier non autorisé")
        
        media_type_map = {
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.csv': 'text/csv',
            '.pdf': 'application/pdf',
            '.json': 'application/json',
            '.xml': 'application/xml'
        }
        
        ext = os.path.splitext(filename)[1]
        media_type = media_type_map.get(ext, 'application/octet-stream')
        
        return FileResponse(
            path=filepath,
            media_type=media_type,
            filename=filename
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== WebSocket ====================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint pour le streaming en temps réel
    
    Messages supportés:
    - {"type": "generate", "prompt": "...", "language": "fr"}
    - {"type": "ping"}
    """
    await connection_manager.connect(websocket)
    progress_streamer = ProgressStreamer(connection_manager, websocket)
    
    try:
        while True:
            # Recevoir le message
            data = await websocket.receive_json()
            message_type = data.get("type", "")
            
            logger.info(f"WebSocket message reçu: {message_type}")
            
            if message_type == "generate":
                await handle_generate_message(data, progress_streamer)
            
            elif message_type == "ping":
                await progress_streamer.send_progress("pong", "ping", percentage=0)
            
            else:
                await progress_streamer.send_error(f"Type de message inconnu: {message_type}", "UNKNOWN_TYPE")
    
    except WebSocketDisconnect:
        await connection_manager.disconnect(websocket)
        logger.info("WebSocket déconnecté normalement")
    
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        try:
            await progress_streamer.send_error(f"Erreur WebSocket: {str(e)}", "WS_ERROR")
        except:
            pass
        await connection_manager.disconnect(websocket)

async def handle_generate_message(data: Dict[str, Any], progress_streamer: ProgressStreamer):
    """Gère la génération de questionnaire via WebSocket"""
    try:
        user_prompt = data.get("prompt", "")
        language = data.get("language", "fr")
        
        if not user_prompt:
            await progress_streamer.send_error("Prompt vide", "EMPTY_PROMPT")
            return
        
        await progress_streamer.send_progress("✨ Initialisation de la génération", "starting", percentage=5)
        
        # ========== ÉTAPE 1: Extraction du contexte ==========
        await progress_streamer.send_progress("🔍 Analyse du contexte", "context_extraction", percentage=10)
        
        context_result = await context_extraction_service.extract_context(user_prompt)
        
        if not context_result["success"]:
            await progress_streamer.send_error(
                f"Erreur d'extraction: {context_result.get('error')}",
                "CONTEXT_ERROR"
            )
            return
        
        context = context_result["data"]
        
        await progress_streamer.send_progress(
            f"📊 Contexte extrait: {context['number_of_questions']} questions, "
            f"{context['number_of_locations']} lieux",
            "context_ok",
            percentage=15,
            data={"context": context}
        )
        
        # ========== ÉTAPE 2: Chargement des lieux ==========
        await progress_streamer.send_progress("📍 Chargement des lieux d'enquête", "locations_loading", percentage=20)
        
        try:
            locations_result = adm_service.search_locations_by_context(
                context.get("geographic_zones", ""),
                context.get("number_of_locations", 5)
            )
            
            await progress_streamer.send_progress(
                f"✅ {len(locations_result)} lieux chargés",
                "locations_ok",
                percentage=25,
                data={"locations_count": len(locations_result)}
            )
        except Exception as e:
            logger.warning(f"Location loading failed: {str(e)}")
            locations_result = []
            await progress_streamer.send_progress(
                "⚠️ Lieux non chargés, continuation sans lieux",
                "locations_warning",
                percentage=25
            )
        
        # ========== ÉTAPE 3: Génération parallèle multi-LLM ==========
        await progress_streamer.send_progress(
            "🚀 Démarrage de la génération parallèle multi-LLM",
            "generation_starting",
            percentage=30
        )
        
        async def progress_callback(message: str, status: str):
            """Callback pour suivre la progression"""
            percentage_map = {
                "starting": 30,
                "generation": 40,
                "complete": 85,
                "success": 95,
                "final": 100
            }
            percentage = percentage_map.get(status, 50)
            await progress_streamer.send_progress(message, status, percentage=percentage)
        
        generation_result = await multi_llm_orchestration.generate_survey_sections_parallel(
            context,
            progress_callback=progress_callback
        )
        
        if not generation_result["success"]:
            await progress_streamer.send_error(
                f"Erreur de génération: {generation_result.get('error')}",
                "GENERATION_ERROR"
            )
            return
        
        # ========== ÉTAPE 4: Assemblage du questionnaire final ==========
        await progress_streamer.send_progress("🔗 Assemblage final du questionnaire", "assembly", percentage=90)
        
        categories = generation_result.get("categories", [])
        
        survey_response = {
            "metadata": {
                "title": context.get("survey_objective", "Questionnaire d'enquête")[:100],
                "introduction": context.get("survey_objective", ""),
                "survey_total_duration": "45-60 minutes",
                "number_of_respondents": context.get("number_of_respondents", 100),
                "number_of_investigators": context.get("number_of_investigators", 5),
                "number_of_locations": len(locations_result),
                "location_characteristics": ", ".join([l.get("name", "") for l in locations_result[:3]]) if locations_result else "Non spécifiées",
                "target_audience": context.get("target_audience", "Général"),
                "survey_objective": context.get("survey_objective", "")
            },
            "categories": categories,
            "locations": locations_result,
            "version": "3.0.0",
            "generated_at": datetime.now().isoformat(),
            "language": language
        }
        
        # ========== ÉTAPE 5: Envoi du résultat ==========
        await progress_streamer.send_result(
            survey_response,
            status="complete"
        )
        
        await progress_streamer.send_progress(
            "✅ Questionnaire généré avec succès!",
            "success",
            percentage=100
        )
        
        logger.info(f"Génération complète: {len(categories)} catégories, "
                   f"{generation_result.get('total_questions')} questions, "
                   f"{len(locations_result)} lieux")
    
    except Exception as e:
        logger.error(f"Error in generate message: {str(e)}", exc_info=True)
        logger.error(traceback.format_exc())
        await progress_streamer.send_error(f"Erreur: {str(e)}", "INTERNAL_ERROR")

# ==================== Startup/Shutdown ====================

@app.on_event("startup")
async def startup_event():
    """Événement de démarrage"""
    logger.info("=" * 80)
    logger.info("🚀 Survey Generator API v3 Madagascar - Démarrage")
    logger.info("=" * 80)
    logger.info(f"Environnement: {settings.environment}")
    logger.info(f"Debug: {settings.debug}")
    logger.info(f"Host: {settings.host}:{settings.port}")
    
    api_keys = settings.validate_api_keys()
    logger.info(f"OpenAI API keys: {api_keys['openai']['count']}")
    logger.info(f"Anthropic API keys: {api_keys['anthropic']['count']}")
    logger.info(f"Gemini API keys: {api_keys['gemini']['count']}")
    logger.info(f"Données administratives chargées: {adm_service.df is not None}")
    logger.info("=" * 80)

@app.on_event("shutdown")
async def shutdown_event():
    """Événement d'arrêt"""
    logger.info("=" * 80)
    logger.info("🛑 Survey Generator API v3 Madagascar - Arrêt")
    logger.info("=" * 80)

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Démarrage du serveur sur {settings.host}:{settings.port}")
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )