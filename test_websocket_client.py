# backend/test_websocket_client.py
"""
Client de test pour le WebSocket de génération de questionnaires
Démontre comment communiquer avec l'API via WebSocket
"""

import asyncio
import json
import websockets
from datetime import datetime

async def test_survey_generation():
    """Teste la génération de questionnaire via WebSocket"""
    
    uri = "ws://localhost:8000/ws"
    
    print("=" * 70)
    print("Survey Generator WebSocket Client - Test")
    print("=" * 70)
    print()
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"✅ Connecté au serveur: {uri}")
            print()
            
            # Message de génération
            generation_request = {
                "type": "generate",
                "prompt": """Créer un questionnaire d'enquête sur l'accès aux services de santé maternelle 
                            dans les régions d'Analamanga avec focus sur Antananarivo.
                            Environ 50 questions organisées en 6 catégories.
                            Nous avons besoin de 5 lieux pour l'enquête.""",
                "language": "fr"
            }
            
            print("📤 Envoi de la demande de génération...")
            print(f"Prompt: {generation_request['prompt'][:100]}...")
            print()
            
            # Envoyer la demande
            await websocket.send(json.dumps(generation_request))
            
            # Recevoir les messages
            total_questions = 0
            num_categories = 0
            
            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=60)
                    data = json.loads(message)
                    
                    msg_type = data.get("type")
                    msg_status = data.get("status")
                    msg_message = data.get("message")
                    msg_percentage = data.get("percentage", 0)
                    
                    # Afficher selon le type de message
                    if msg_type == "progress":
                        print(f"[{msg_percentage:3d}%] {msg_status:20s} | {msg_message}")
                        
                    elif msg_type == "result":
                        print()
                        print("✅ RÉSULTAT REÇU!")
                        print("-" * 70)
                        
                        result_data = data.get("data", {})
                        
                        # Extraire les statistiques
                        metadata = result_data.get("metadata", {})
                        categories = result_data.get("categories", [])
                        locations = result_data.get("locations", [])
                        
                        # Compter les questions
                        total_questions = sum(len(cat.get("questions", [])) for cat in categories)
                        num_categories = len(categories)
                        
                        print(f"📋 Titre: {metadata.get('title')}")
                        print(f"📝 Objectif: {metadata.get('survey_objective')[:60]}...")
                        print(f"📊 Catégories: {num_categories}")
                        print(f"❓ Questions totales: {total_questions}")
                        print(f"📍 Lieux: {len(locations)}")
                        print(f"👥 Répondants: {metadata.get('number_of_respondents')}")
                        print()
                        
                        # Afficher les catégories
                        print("Catégories générées:")
                        print("-" * 70)
                        for i, cat in enumerate(categories[:6], 1):
                            num_q = len(cat.get("questions", []))
                            print(f"  {i}. {cat.get('category_name')} ({num_q} questions)")
                        
                        print()
                        print("Lieux d'enquête:")
                        print("-" * 70)
                        for loc in locations[:5]:
                            print(f"  • {loc.get('name')} ({loc.get('adm2')})")
                        
                        break
                        
                    elif msg_type == "error":
                        print(f"❌ ERREUR: {msg_message}")
                        print(f"   Code: {msg_status}")
                        break
                    
                    # Timeout après 30 messages
                    elif msg_type == "ping":
                        continue
                        
                except asyncio.TimeoutError:
                    print("⏱️  Timeout en attente de la réponse du serveur")
                    break
                except Exception as e:
                    print(f"❌ Erreur lors de la réception: {e}")
                    break
            
            print()
            print("=" * 70)
            print("✅ Test complété")
            print("=" * 70)
    
    except ConnectionRefusedError:
        print(f"❌ Impossible de se connecter à {uri}")
        print("Vérifiez que le serveur est en cours d'exécution:")
        print("  python main.py")
    except Exception as e:
        print(f"❌ Erreur: {e}")

async def test_ping():
    """Teste le ping du serveur"""
    
    uri = "ws://localhost:8000/ws"
    
    print("Testing WebSocket ping...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connecté")
            
            # Envoyer un ping
            ping_request = {"type": "ping"}
            await websocket.send(json.dumps(ping_request))
            
            # Attendre le pong
            message = await asyncio.wait_for(websocket.recv(), timeout=5)
            data = json.loads(message)
            
            if data.get("status") == "ping":
                print("✅ Pong reçu - Serveur opérationnel")
            
    except Exception as e:
        print(f"❌ Erreur ping: {e}")

def main():
    """Fonction principale"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "ping":
        asyncio.run(test_ping())
    else:
        asyncio.run(test_survey_generation())

if __name__ == "__main__":
    main()