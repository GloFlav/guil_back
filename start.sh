#!/bin/bash
# backend/start.sh
# Script de démarrage du serveur Survey Generator API v3 Madagascar

set -e

echo "============================================"
echo "Survey Generator API v3 Madagascar"
echo "============================================"

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 n'est pas installé"
    exit 1
fi

# Vérifier si l'environnement virtuel existe
if [ ! -d "venv" ]; then
    echo "📦 Création de l'environnement virtuel..."
    python3 -m venv venv
fi

# Activer l'environnement virtuel
echo "🔌 Activation de l'environnement virtuel..."
source venv/bin/activate

# Installer les dépendances si nécessaire
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "📚 Installation des dépendances..."
    pip install -r requirements.txt
fi

# Vérifier les fichiers essentiels
if [ ! -f ".env" ]; then
    echo "⚠️  Fichier .env non trouvé. Création d'une copie par défaut..."
    cp .env.example .env || echo "❌ Impossible de créer .env"
fi

if [ ! -f "data/mdg_adm3.csv" ]; then
    echo "❌ Fichier data/mdg_adm3.csv non trouvé"
    exit 1
fi

# Créer les dossiers nécessaires
mkdir -p logs exports

# Démarrer le serveur
echo "🚀 Démarrage du serveur..."
echo "API disponible sur: http://localhost:8000"
echo "Documentation: http://localhost:8000/docs"
echo ""

uvicorn main:app --host 0.0.0.0 --port 8000 --reload