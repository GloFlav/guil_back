# backend/ARCHITECTURE.md

# 🏗️ Architecture Survey Generator API v3 Madagascar

## Vue d'ensemble

L'application est une **API FastAPI** pour la génération intelligente de questionnaires d'enquête avec orchestration parallèle multi-LLM.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Client Frontend (React)                      │
│              WebSocket /ws + REST /api/v1/                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │   FastAPI   │
                    │  main.py    │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐      ┌──────▼──────┐    ┌─────▼────┐
   │ REST    │      │  WebSocket  │    │ Exception│
   │ Routes  │      │  Streaming  │    │ Handler  │
   └────┬────┘      └──────┬──────┘    └──────────┘
        │                  │
        └──────────────────┼──────────────────┐
                           │                  │
                    ┌──────▼──────────────────▼────┐
                    │    Services Layer            │
                    └──────────┬───────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
       ┌────▼─────────┐  ┌────▼────────────┐ ┌──▼──────────┐
       │ Context      │  │ Multi-LLM       │ │ Export      │
       │ Extraction   │  │ Orchestration   │ │ Service     │
       │ (OpenAI)     │  │ (OpenAI +       │ │ (XLSX/CSV/  │
       └──────────────┘  │ Anthropic +     │ │ JSON/Kobo/  │
                         │ Gemini)         │ │ GoogleForms)│
                         └──────┬──────────┘ └─────┬───────┘
                                │                  │
                       ┌────────▼────────┐   ┌─────▼────────┐
                       │ Data Layer      │   │ Export Files │
                       │ (Administrative)│   │ (./exports/) │
                       └─────────────────┘   └──────────────┘
```

## 📦 Modules Principaux

### 1. **config/settings.py**
- Configuration centralisée via variables d'environnement (.env)
- Gestion des clés API multi-LLM
- Paramètres de l'application (min/max questions, timeouts, etc.)
- Validation des clés API

```python
settings = Settings()  # Instance globale
settings.get_openai_keys()      # Liste des clés OpenAI
settings.validate_api_keys()    # Vérifier disponibilité
```

### 2. **models/survey.py**
Modèles Pydantic pour validation et documentation:
- `QuestionType` - Types de questions supportés
- `Question` - Modèle d'une question
- `Category` - Catégorie de questions
- `SurveyResponse` - Questionnaire complet
- `Location` - Localisation administrative
- `ContextExtraction` - Contexte extrait du prompt

### 3. **services/context_extraction_service.py**
**Responsable**: Premier LLM (OpenAI)

Extraits du prompt utilisateur:
- Objectif de l'enquête
- Nombre de questions (24-60)
- Zones géographiques
- Audience cible
- Catégories proposées
- Nombre de lieux

```
Flux: Prompt utilisateur → OpenAI → Contexte structuré
```

### 4. **services/administrative_data_service.py**
Gestion des données administratives Madagascar (ADM1/ADM2/ADM3)

**Charge depuis**: `./data/mdg_adm3.csv`

Fonctionnalités:
```
ADM1 (Régions) → ADM2 (Districts) → ADM3 (Localités)
Analamanga → Antananarivo Avaradrano → Alasora
```

Méthodes:
- `get_adm1_regions()` - Liste des régions
- `get_adm2_districts(region)` - Districts d'une région
- `get_adm3_locations(district)` - Localités d'un district
- `search_locations_by_context()` - Sélectionner les lieux

### 5. **services/multi_llm_orchestration.py**
**Orchestration parallèle de 3 LLM + 1 backup**

Distribution des tâches:

```
Contexte
   │
   ├─► OpenAI (Gemini-like) → Catégories 0-1 (2 catégories)
   │       [gpt-4-turbo]
   │
   ├─► Anthropic (Claude) → Catégories 2-3 (2 catégories)
   │       [claude-sonnet-4-5]
   │
   ├─► Google Gemini → Catégories 4-5 (2 catégories)
   │       [gemini-1.5-pro]
   │
   └─► BACKUP (OpenAI)
           En cas d'erreur des autres

Résultat: Toutes les catégories fusionnées
```

Exécution asynchrone parallèle avec `asyncio.gather()`

### 6. **services/export_service.py**
Exporte le questionnaire en plusieurs formats:

- **XLSX**: Feuilles Excel (Métadonnées, Questions, Lieux)
- **CSV**: Format tabulaire pour traitement
- **JSON**: Format complet avec métadonnées
- **Kobo Tools**: Format XLS Form XML
- **Google Forms**: Format JSON importable

Fichiers générés dans `./exports/` avec timestamp

### 7. **utils/websocket_manager.py**
Gestion des connexions WebSocket en temps réel

Classes:
- `ConnectionManager` - Gère les connexions actives
- `ProgressStreamer` - Envoie les messages de progression

Messages:
```json
// Progression
{"type": "progress", "status": "...", "percentage": 50, "message": "..."}

// Résultat
{"type": "result", "data": {...}}

// Erreur
{"type": "error", "error_code": "...", "message": "..."}
```

### 8. **main.py**
Application FastAPI principale

Endpoints:

```
GET  /                              Info API
GET  /health                        Santé du service
GET  /api/v1/locations              Liste des régions
GET  /api/v1/locations/{region}     Lieux par région
POST /api/v1/export/{survey_id}     Exporter questionnaire
GET  /api/v1/exports                Fichiers exportés
GET  /api/v1/exports/{filename}     Télécharger fichier
WS   /ws                            WebSocket streaming
```

## 🔄 Flux de Génération Complet

### Phase 1: Initialisation (0-10%)
```
Client WebSocket
    ↓
Validation du prompt
    ↓
Connexion établie
```

### Phase 2: Extraction du Contexte (10-20%)
```
Prompt utilisateur
    ↓
OpenAI (context_extraction_service)
    ↓
{
    survey_objective: "...",
    number_of_questions: 30,
    geographic_zones: "Analamanga",
    categories: ["Général", "Situation", ...]
}
```

### Phase 3: Chargement des Lieux (20-30%)
```
geographic_zones + number_of_locations
    ↓
administrative_data_service.search_locations_by_context()
    ↓
[{name: "Alasora", adm1: "Analamanga", ...}, ...]
```

### Phase 4: Génération Parallèle (30-85%)
```
Contexte + Catégories
    ↓
┌─────────────────────────────────────┐
│   async def generate_parallel():    │
│   tasks = [                         │
│     openai(cat 0-1),                │
│     anthropic(cat 2-3),             │
│     gemini(cat 4-5)                 │
│   ]                                 │
│   results = await gather(*tasks)    │
│   return merge(results)             │
└─────────────────────────────────────┘
    ↓
[Category1, Category2, ...]
```

### Phase 5: Assemblage Final (85-95%)
```
Catégories + Métadonnées + Lieux
    ↓
SurveyResponse(
    metadata: {...},
    categories: [...],
    locations: [...],
    version: "3.0.0"
)
```

### Phase 6: Résultat (95-100%)
```
Survey JSON
    ↓
WebSocket send_result()
    ↓
Client reçoit le questionnaire complet
```

## 📊 Modèles de Données

### Question avec logique conditionnelle
```json
{
  "question_id": "q1",
  "question_type": "single_choice",
  "question_text": "Êtes-vous enceinte?",
  "expected_answers": [
    {
      "answer_id": "a1",
      "answer_text": "Oui",
      "next_question_id": "q2"  // Logique conditionnelle
    },
    {
      "answer_id": "a2",
      "answer_text": "Non",
      "next_question_id": "q5"  // Sauter les questions
    }
  ]
}
```

### Structure complète d'un questionnaire
```json
{
  "metadata": {
    "title": "Santé maternelle",
    "introduction": "...",
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
      "pcode": "MG11102010",
      "name": "Alasora",
      "adm1": "Analamanga",
      "adm2": "Antananarivo Avaradrano"
    }
  ]
}
```

## 🔌 Extensibilité Future

Architecture conçue pour ajouter facilement:

### 1. Data Analysis Service
```python
# backend/services/data_analysis_service.py
class DataAnalysisService:
    async def analyze_responses(self, survey_responses):
        """Analyser les réponses collectées"""
        pass
    
    async def generate_statistics(self, responses):
        """Générer les statistiques"""
        pass
```

### 2. Report Generation
```python
# backend/services/report_generation_service.py
class ReportGenerationService:
    async def generate_pdf_report(self, analysis_results):
        """Générer un rapport PDF"""
        pass
```

### 3. Visualization Service
```python
# backend/services/visualization_service.py
class VisualizationService:
    async def create_charts(self, data):
        """Créer des graphiques"""
        pass
```

## 🔒 Sécurité

- **CORS**: Configuration flexible avec variables d'environnement
- **Validation**: Tous les inputs validés avec Pydantic
- **Timeouts**: Protection contre les requêtes longues
- **Logging**: Tous les événements enregistrés
- **Error Handling**: Gestion globale des erreurs

## 📈 Performance

- **Parallélisation**: 3 LLM simultanément (asyncio)
- **Caching**: Les clés API sont réutilisées
- **Streaming**: WebSocket pour progression temps réel
- **Async/Await**: Code asynchrone natif

## 🧪 Tests

Structure pour tests:
```
backend/
├── tests/
│   ├── test_context_extraction.py
│   ├── test_multi_llm.py
│   ├── test_export.py
│   └── test_websocket.py
```

## 📋 Configuration Environnement

Fichier `.env` avec:
- Clés API (x5)
- Paramètres LLM (modèles, tokens)
- Paramètres génération (min/max questions)
- Paramètres serveur (host, port, debug)
- Chemins fichiers (exports, logs, data)

## 🚀 Déploiement

### Mode Développement
```bash
python main.py
```

### Mode Production
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Avec Docker (optionnel pour le futur)
```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
```

---

**Architecture créée pour**: Production-Ready + Extensibilité Future