# Survey Generator API v3 Madagascar

## 📋 Description

API backend pour générer des questionnaires d'enquête intelligents avec orchestration parallèle multi-LLM (OpenAI, Anthropic, Google Gemini).

## 🎯 Fonctionnalités

- **Génération intelligente de questionnaires** via prompt utilisateur
- **Extraction automatique du contexte** (nombre de questions, zones géographiques, audience)
- **Orchestration parallèle multi-LLM** pour optimiser les performances
- **Sélection automatique des lieux** basée sur données administratives Madagascar (ADM1, ADM2, ADM3)
- **WebSocket streaming** pour suivi en temps réel
- **Export multi-formats**: XLSX, CSV, JSON, Kobo Tools, Google Forms
- **Architecture extensible** pour analyse de données future

## 🏗️ Architecture

```
backend/
├── config/
│   ├── settings.py              # Configuration centralisée (clés API, paramètres)
│
├── models/
│   └── survey.py                # Modèles Pydantic pour questionnaires
│
├── services/
│   ├── context_extraction_service.py      # Extraction de contexte (OpenAI)
│   ├── administrative_data_service.py     # Gestion données ADM1/ADM2/ADM3
│   ├── multi_llm_orchestration.py         # Génération parallèle multi-LLM
│   └── export_service.py                  # Export multi-formats
│
├── utils/
│   └── websocket_manager.py     # Gestion WebSocket et progression
│
├── data/
│   └── mdg_adm3.csv             # Données administratives Madagascar
│
├── exports/                     # Dossier des fichiers exportés
├── logs/                        # Fichiers de log
│
├── main.py                      # Application FastAPI principale
├── requirements.txt             # Dépendances Python
├── .env                         # Configuration d'environnement
└── README.md                    # Cette documentation
```

## 🚀 Installation

### Prérequis
- Python 3.9+
- pip ou poetry

### Étapes

1. **Cloner le projet**
```bash
cd backend
```

2. **Créer un environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Configurer les variables d'environnement**
```bash
# Éditer le fichier .env avec vos clés API
nano .env
```

5. **Vérifier les données administratives**
```bash
# Vérifier que data/mdg_adm3.csv existe
ls -la data/
```

## 📝 Configuration (.env)

```env
# API Keys
OPENAI_API_KEY_1=sk-proj-...
OPENAI_API_KEY_2=sk-proj-...
ANTHROPIC_API_KEY_1=sk-ant-...
ANTHROPIC_API_KEY_2=sk-ant-...
GEMINI_API_KEY=AIza...
GOOGLE_MAPS_API_KEY=AIza...

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=true
ENVIRONMENT=development

# Generation
MIN_QUESTIONS=24
MAX_QUESTIONS=60
DEFAULT_NUM_LOCATIONS=5
```

## 🏃 Démarrage

```bash
# Mode développement avec auto-reload
python main.py

# Mode production avec Uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000
```

L'API sera disponible sur `http://localhost:8000`

## 📡 API Endpoints

### REST Endpoints

```
GET  /                          # Info API
GET  /health                    # Santé du service
GET  /api/v1/locations          # Liste des régions
GET  /api/v1/locations/{region} # Lieux par région
POST /api/v1/export/{survey_id} # Export questionnaire
GET  /api/v1/exports            # Liste fichiers exportés
GET  /api/v1/exports/{filename} # Télécharger fichier
```

### WebSocket

```
WS  /ws                         # Streaming génération questionnaire
```

**Protocole WebSocket:**

```json
// Demande génération
{
  "type": "generate",
  "prompt": "Générer un questionnaire sur la santé maternelle en Analamanga",
  "language": "fr"
}

// Messages de progression
{
  "type": "progress",
  "message": "🔍 Analyse du contexte",
  "status": "context_extraction",
  "percentage": 10,
  "data": {...}
}

// Résultat final
{
  "type": "result",
  "message": "Questionnaire généré avec succès",
  "status": "complete",
  "data": {...}
}

// Erreurs
{
  "type": "error",
  "message": "Description de l'erreur",
  "status": "ERROR_CODE"
}
```

## 🔄 Flux de Génération

1. **Initialisation** (5%)
2. **Extraction du contexte** avec OpenAI (10-15%)
   - Nombre de questions
   - Zones géographiques
   - Catégories
   - Audience cible
3. **Chargement des lieux** (20-25%)
   - Recherche dans la BD administrative
   - Sélection par régions/districts
4. **Génération parallèle** (30-85%)
   - OpenAI: Catégories 0-1
   - Anthropic: Catégories 2-3
   - Gemini: Catégories 4-5 + backup
5. **Assemblage final** (85-95%)
6. **Envoi du résultat** (100%)

## 📤 Formats d'Export Supportés

- **XLSX**: Feuilles Excel (Métadonnées, Questions, Lieux)
- **CSV**: Format tabulaire pour traitement
- **JSON**: Format complet avec toutes les métadonnées
- **Kobo Tools**: Format compatible Kobo XLS Form
- **Google Forms**: Format JSON importable dans Google Forms

## 🗂️ Structure des Données

### Question
```json
{
  "question_id": "q1",
  "question_type": "single_choice",
  "question_text": "Avez-vous accès à l'eau potable?",
  "is_required": true,
  "help_text": "Sélectionnez oui ou non",
  "expected_answers": [
    {
      "answer_id": "a1",
      "answer_type": "option",
      "answer_text": "Oui",
      "next_question_id": "q2"
    }
  ]
}
```

### Location
```json
{
  "pcode": "MG11102010",
  "name": "Alasora",
  "adm1": "Analamanga",
  "adm2": "Antananarivo Avaradrano",
  "adm3": "Alasora"
}
```

## 🧪 Tests

```bash
# Lancer les tests
pytest

# Tests avec verbosité
pytest -v

# Couverture de code
pytest --cov=.
```

## 📊 Monitoring

Les logs sont disponibles dans `./logs/app.log`

```bash
# Suivre les logs en temps réel
tail -f logs/app.log
```

## 🔌 Extensibilité

L'architecture est conçue pour permettre l'intégration future d'analyses:

- **Data Analysis Service**: Analyser les réponses collectées
- **Report Generation**: Générer des rapports automatiques
- **Visualization**: Créer des dashboards
- **ML Pipeline**: Intégrer du machine learning

Structure pour nouveau service:
```python
# backend/services/data_analysis_service.py
class DataAnalysisService:
    """Service d'analyse de données"""
    
    async def analyze_responses(self, survey_responses: List[Dict]):
        """Analyse les réponses collectées"""
        pass
```

## 🐛 Troubleshooting

### Erreur: "Aucune clé OpenAI configurée"
```bash
# Vérifier le fichier .env
cat .env | grep OPENAI_API_KEY_1
```

### Erreur: "Fichier mdg_adm3.csv non trouvé"
```bash
# Vérifier l'emplacement du fichier
ls -la data/mdg_adm3.csv
```

### Erreur WebSocket timeout
```bash
# Augmenter le timeout dans .env
WEBSOCKET_TIMEOUT_SECONDS=600
```

## 📚 Ressources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Anthropic Documentation](https://docs.anthropic.com/)
- [Google Gemini API](https://ai.google.dev/)

## 📄 Licence

MIT License

## 👥 Auteur

Survey Generator API v3 Madagascar - Yoel

## 📞 Support

Pour toute question ou problème, consultez les logs d'erreur dans `./logs/app.log`