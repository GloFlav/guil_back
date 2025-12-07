# backend/utils/websocket_manager.py
"""
Gestionnaire WebSocket pour la communication en temps réel
Gère la transmission des messages de progression et des résultats
"""

import logging
import json
from datetime import datetime
from typing import List, Dict, Any
from fastapi import WebSocket

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Gestionnaire des connexions WebSocket"""
    
    def __init__(self):
        """Initialise le gestionnaire"""
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accepte une nouvelle connexion"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connecté. Connexions actives: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket):
        """Ferme une connexion"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket déconnecté. Connexions actives: {len(self.active_connections)}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Envoie un message à tous les clients"""
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Erreur envoi message: {e}")
                disconnected.append(connection)
        
        # Nettoyer les connexions fermées
        for conn in disconnected:
            await self.disconnect(conn)
    
    async def send_to_connection(self, websocket: WebSocket, message: Dict[str, Any]):
        """Envoie un message à une connexion spécifique"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Erreur envoi message à connexion: {e}")
            await self.disconnect(websocket)

class ProgressStreamer:
    """Classe pour envoyer des messages de progression"""
    
    def __init__(self, connection_manager: ConnectionManager, websocket: WebSocket):
        """Initialise le streamer"""
        self.connection_manager = connection_manager
        self.websocket = websocket
    
    async def send_progress(
        self,
        message: str,
        status: str,
        percentage: int = 0,
        data: Dict[str, Any] = None
    ):
        """Envoie un message de progression"""
        try:
            # Construire le message sans utiliser ProgressMessage (pour éviter les problèmes de sérialisation)
            progress_msg = {
                "type": "progress",
                "message": message,
                "status": status,
                "percentage": percentage,
                "data": data or {},
                "timestamp": datetime.now().isoformat()  # Convertir en ISO string
            }
            
            await self.websocket.send_json(progress_msg)
            logger.debug(f"Progress envoyé: {status} - {percentage}%")
        
        except Exception as e:
            logger.error(f"Erreur envoi progression: {e}", exc_info=True)
    
    async def send_error(self, error_message: str, error_code: str):
        """Envoie un message d'erreur"""
        try:
            error_msg = {
                "type": "error",
                "message": error_message,
                "status": error_code,
                "error_code": error_code,
                "timestamp": datetime.now().isoformat()
            }
            
            await self.websocket.send_json(error_msg)
            logger.error(f"Erreur envoyée: {error_code} - {error_message}")
        
        except Exception as e:
            logger.error(f"Erreur envoi erreur: {e}", exc_info=True)
    
    async def send_result(self, survey_data: Dict[str, Any], status: str = "success"):
        """Envoie le résultat final"""
        try:
            result_msg = {
                "type": "result",
                "message": "Questionnaire généré avec succès",
                "status": status,
                "data": survey_data,
                "timestamp": datetime.now().isoformat()
            }
            
            await self.websocket.send_json(result_msg)
            logger.info("Résultat envoyé au client")
        
        except Exception as e:
            logger.error(f"Erreur envoi résultat: {e}", exc_info=True)

# Instance globale du gestionnaire
connection_manager = ConnectionManager()