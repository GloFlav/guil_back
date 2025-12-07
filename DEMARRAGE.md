# 🚀 Survey Generator API v3 Madagascar - Guide de Démarrage

## 📦 Fichiers Téléchargés

Vous avez téléchargé l'archive `backend.tar.gz` contenant la **solution backend complète**.

## ⚡ Installation Rapide (5 minutes)

### 1️⃣ Extraire l'archive
```bash
tar -xzf backend.tar.gz
cd backend
```

### 2️⃣ Créer l'environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3️⃣ Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4️⃣ Vérifier la configuration
```bash
python verify_setup.py
```

Vous devriez voir: ✅ Tous les vérifications sont passées!

### 5️⃣ Démarrer le serveur
```bash
python main.py
```

**Accès API**: http://localhost:8000

## 📚 Documentation Principale

Trois fichiers de documentation importants dans le dossier `backend/`:

### 📄 **QUICK_START.md** ⚡
- Démarrage en 5 minutes
- Exemples de test immédiat
- Dépannage rapide

### 📄 **README.md** 📖
- Documentation complète
- Description de tous les endpoints
- Protocole WebSocket
- Formats d'export

### 📄 **ARCHITECTURE.md** 🏗️
- Architecture détaillée
- Flux de génération complet
- Structures de données
- Extensibilité future

## 🧪 Test Immédiat (WebSocket)

Ouvrir deux terminaux:

### Terminal 1: Démarrer le serveur
```bash
cd backend
source venv/bin/activate
python main.py
```

### Terminal 2: Lancer le test
```bash
cd backend
source venv/bin/activate
python test_websocket_client.py
```

Vous verrez:
- ✅ La progression en temps réel (0-100%)
- ✅ Les étapes de génération
- ✅ Le questionnaire généré avec statistiques
- ✅ Les lieux d'enquête sélectionnés

## 🔑 Configuration Clés API

Le fichier `.env` contient déjà les clés API:

```env
✅ OpenAI (2 clés - Key 1 et Key 2)
✅ Anthropic (2 clés - Key 1 et Key 2)
✅ Gemini (1 clé)
```

**Rien à ajouter pour commencer à tester!** ✨

## 📋 Structure du Projet

```
backend/
├── main.py                      🚀 Application principale
├── config/settings.py           ⚙️ Configuration
├── models/survey.py             📋 Modèles Pydantic
├── services/
│   ├── context_extraction_service.py    🔍 Analyse contexte
│   ├── administrative_data_service.py   🗺️ Données Madagascar
│   ├── multi_llm_orchestration.py       🤖 Génération parallèle
│   └── export_service.py                💾 Exports (XLSX/CSV/JSON)
├── utils/websocket_manager.py   🔌 WebSocket streaming
├── data/mdg_adm3.csv            🗺️ Données régions/districts
├── exports/                     📁 Fichiers exportés
└── logs/                        📁 Logs d'application
```

## 🌐 Endpoints Disponibles

### REST API
```
GET  http://localhost:8000/                Info API
GET  http://localhost:8000/health          État service
GET  http://localhost:8000/docs            Documentation Swagger
GET  http://localhost:8000/api/v1/locations    Régions
GET  http://localhost:8000/api/v1/locations/{region}    Lieux
```

### WebSocket
```
WS   ws://localhost:8000/ws    Génération questionnaire + streaming
```

## 💻 Utilisation WebSocket

### Format demande
```json
{
  "type": "generate",
  "prompt": "Créer un questionnaire sur la santé maternelle en Analamanga",
  "language": "fr"
}
```

### Messages progression
```json
{
  "type": "progress",
  "message": "🔍 Analyse du contexte",
  "status": "context_extraction",
  "percentage": 15
}
```

### Résultat final
```json
{
  "type": "result",
  "message": "Questionnaire généré avec succès",
  "data": {
    "metadata": {...},
    "categories": [...],
    "locations": [...]
  }
}
```

## 📤 Exports Supportés

Le système supporte **5 formats d'export**:

- ✅ **XLSX** - Excel avec feuilles séparées
- ✅ **CSV** - Format tabulaire
- ✅ **JSON** - Format complet
- ✅ **Kobo Tools** - Format XLS Form XML
- ✅ **Google Forms** - Format importable

Les fichiers sont sauvegardés dans `./exports/`

## 🎯 Flux Type d'Utilisation

1. **Connexion WebSocket** au serveur
2. **Envoi du prompt** (description enquête)
3. **Réception progression** en temps réel
4. **Réception du questionnaire** généré
5. **Export** dans le format désiré

## 🔍 Vérification Installation

Avant de démarrer, vérifier que tout est prêt:

```bash
python verify_setup.py
```

Devrait afficher:
- ✅ Python Version: OK
- ✅ Fichiers essentiels (tous verts)
- ✅ Répertoires (tous verts)
- ✅ Variables d'environnement
- ✅ Packages Python (tous verts)

## 📋 Checklist Démarrage

- [ ] Archive extraite
- [ ] Virtual environment créé
- [ ] Dépendances installées (`pip install -r requirements.txt`)
- [ ] Configuration vérifiée (`python verify_setup.py`)
- [ ] Server lancé (`python main.py`)
- [ ] Tests WebSocket passés (`python test_websocket_client.py`)
- [ ] API accessible sur http://localhost:8000

## 🆘 Dépannage Courant

### ❌ "Cannot connect to server"
```bash
# Vérifier que le serveur est lancé
python main.py
```

### ❌ "No module named fastapi"
```bash
# Installer les dépendances
pip install -r requirements.txt
```

### ❌ "CSV file not found"
```bash
# Vérifier que le fichier existe
ls -la data/mdg_adm3.csv
```

### ❌ "API key not configured"
```bash
# Vérifier le fichier .env
cat .env | grep API_KEY
```

## 📊 Exemple de Prompt pour Test

Essayer ce prompt pour générer un questionnaire:

```
"Créer un questionnaire complet sur l'accès aux services de santé maternelle 
dans les régions d'Analamanga avec focus sur Antananarivo. 
Nous avons besoin d'environ 50 questions organisées en 6 catégories: 
informations générales, accès aux services, qualité des soins, 
problèmes identifiés, besoins des bénéficiaires, et recommandations. 
L'enquête ciblera 5 lieux différents."
```

## 🚀 Prochaines Étapes

### Frontend React (à créer)
L'application a besoin d'un frontend qui:
1. Se connecte au WebSocket `/ws`
2. Affiche la progression
3. Affiche le questionnaire
4. Permet les exports
5. Visualise les lieux sur Google Maps

### Extension Données
Pour intégrer l'analyse future:
1. Créer `data_analysis_service.py`
2. Ajouter `report_generation_service.py`
3. Implémenter `visualization_service.py`

## 📞 Support

### Fichiers d'aide
- `QUICK_START.md` - Démarrage rapide
- `README.md` - Documentation complète
- `ARCHITECTURE.md` - Architecture technique
- `verify_setup.py` - Vérification automatique

### Logs
Les logs détaillés sont disponibles:
```bash
tail -f logs/app.log
```

## ✨ Caractéristiques Principales

✅ **Orchestration Multi-LLM** - OpenAI + Anthropic + Gemini en parallèle  
✅ **WebSocket Streaming** - Progression temps réel  
✅ **Export Multi-formats** - XLSX, CSV, JSON, Kobo, Google Forms  
✅ **Données Madagascar** - Régions, districts, localités  
✅ **Production-Ready** - Code professionnel et sécurisé  
✅ **Extensible** - Architecture préparée pour futures fonctionnalités  
✅ **Documenté** - Documentation complète + code commenté  
✅ **Testé** - Scripts de vérification et test inclus  

## 🎊 C'est Prêt!

L'application backend est **100% fonctionnelle** et prête à l'emploi.

### Démarrer maintenant:
```bash
cd backend
source venv/bin/activate
python main.py
```

**API accessible sur: http://localhost:8000**

---

## 📝 Notes Importantes

- Les clés API sont déjà configurées dans `.env`
- Les données administratives Madagascar sont incluses
- Le système gère automatiquement les erreurs et les retries
- Tous les fichiers générés sont sauvegardés dans `./exports/`
- Les logs détaillés sont dans `./logs/app.log`

---

**Version**: 3.0.0  
**Status**: ✅ Production-Ready  
**Créé pour**: HelloSoins Madagascar Platform  
**Par**: Yoel

Bon développement! 🚀