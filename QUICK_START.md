# backend/QUICK_START.md

# 🚀 Démarrage Rapide - Survey Generator API v3

## Installation (5 minutes)

### 1. Créer l'environnement virtuel
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Vérifier la configuration
```bash
python verify_setup.py
```

### 4. Démarrer le serveur
```bash
python main.py
```

ou

```bash
bash start.sh
```

Le serveur sera disponible sur: **http://localhost:8000**

## 📚 Accès à la documentation

Une fois le serveur lancé:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 🧪 Tester la génération de questionnaire

### Option 1: Avec le client WebSocket (Recommandé)
```bash
# Terminal 1: Démarrer le serveur
python main.py

# Terminal 2: Lancer le test
python test_websocket_client.py
```

### Option 2: Avec curl
```bash
# Test du health check
curl http://localhost:8000/health

# Lister les régions
curl http://localhost:8000/api/v1/locations
```

### Option 3: Avec Postman/Insomnia
1. Importer le fichier `postman_collection.json` (à créer)
2. Sélectionner la requête "Generate Survey"
3. Cliquer sur "Send"

## 📝 Exemple d'utilisation WebSocket

```python
import asyncio
import websockets
import json

async def test():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        # Envoyer une demande
        await ws.send(json.dumps({
            "type": "generate",
            "prompt": "Créer un questionnaire sur la santé maternelle",
            "language": "fr"
        }))
        
        # Recevoir les résultats
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            print(data["message"])
            if data["type"] == "result":
                break

asyncio.run(test())
```

## 🔧 Configuration essentielles

Editer le fichier `.env`:

```env
# Clés API (OBLIGATOIRE)
OPENAI_API_KEY_1=sk-...
ANTHROPIC_API_KEY_1=sk-ant-...
GEMINI_API_KEY=AIza...

# Paramètres optionnels
PORT=8000
DEBUG=true
MIN_QUESTIONS=24
MAX_QUESTIONS=60
DEFAULT_NUM_LOCATIONS=5
```

## 📁 Structure des fichiers générés

Les fichiers exportés sont sauvegardés dans:
- `./exports/` - Fichiers exportés

Les logs sont disponibles dans:
- `./logs/app.log` - Fichier log principal

## 🐛 Dépannage

### Erreur: "Cannot connect to API"
```bash
# Vérifier que le serveur est lancé
python main.py
```

### Erreur: "No API key configured"
```bash
# Vérifier le fichier .env
cat .env | grep API_KEY
```

### Erreur: "CSV file not found"
```bash
# Vérifier que le fichier existe
ls -la data/mdg_adm3.csv
```

### Timeout WebSocket
```bash
# Augmenter le timeout dans .env
WEBSOCKET_TIMEOUT_SECONDS=300
```

## 📊 Structure réponse

La génération retourne une réponse JSON complète:

```json
{
  "metadata": {
    "title": "Questionnaire sur la santé maternelle",
    "number_of_questions": 30,
    "number_of_locations": 5
  },
  "categories": [
    {
      "category_id": "cat1",
      "category_name": "Informations générales",
      "questions": [...]
    }
  ],
  "locations": [
    {
      "name": "Alasora",
      "adm1": "Analamanga",
      "adm2": "Antananarivo Avaradrano"
    }
  ]
}
```

## 🎯 Flux typique

1. **Connexion WebSocket** → `/ws`
2. **Envoi du prompt** → `{"type": "generate", "prompt": "..."}`
3. **Progression en temps réel** → Messages de statut
4. **Résultat final** → `{"type": "result", "data": {...}}`
5. **Export optionnel** → POST `/api/v1/export/{survey_id}`

## 📞 Besoin d'aide?

- Consultez le fichier `README.md` pour la documentation complète
- Vérifiez les logs: `tail -f logs/app.log`
- Exécutez les tests: `python verify_setup.py`

---

**Status**: ✅ Prêt à développer

Pour commencer: `python main.py`