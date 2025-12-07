#!/bin/bash

# Script d'installation pour le projet Survey Generator API sur macOS

echo "🚀 Installation du projet Survey Generator API"
echo "=============================================="

# Vérifier si Python 3.8+ est installé
echo "📋 Vérification de Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 n'est pas installé. Veuillez l'installer d'abord:"
    echo "   brew install python"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $PYTHON_VERSION détecté. Version $REQUIRED_VERSION ou supérieure requise."
    exit 1
fi

echo "✅ Python $PYTHON_VERSION détecté"

# Vérifier si pip est installé
echo "📋 Vérification de pip..."
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 n'est pas installé. Installation en cours..."
    python3 -m ensurepip --upgrade
fi

echo "✅ pip3 disponible"

# Créer un environnement virtuel
echo "🔧 Création de l'environnement virtuel..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Environnement virtuel créé"
else
    echo "✅ Environnement virtuel existant trouvé"
fi

# Activer l'environnement virtuel
echo "🔧 Activation de l'environnement virtuel..."
source venv/bin/activate

# Mettre à jour pip
echo "🔧 Mise à jour de pip..."
pip install --upgrade pip

# Installer les dépendances
echo "📦 Installation des dépendances..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Dépendances installées"
else
    echo "❌ Fichier requirements.txt non trouvé"
    exit 1
fi

# Créer les dossiers nécessaires
echo "📁 Création des dossiers..."
mkdir -p config
mkdir -p services
mkdir -p models
mkdir -p exports
mkdir -p logs

echo "✅ Structure de dossiers créée"

# Vérifier si le fichier .env existe
echo "⚙️ Configuration de l'environnement..."
if [ ! -f ".env" ]; then
    echo "❌ Fichier .env non trouvé. Veuillez le créer avec vos clés API:"
    echo "   cp .env.example .env"
    echo "   puis éditez .env avec vos valeurs"
else
    echo "✅ Fichier .env trouvé"
fi

# Créer un script de démarrage
echo "🚀 Création du script de démarrage..."
cat > start.sh << 'EOF'
#!/bin/bash
echo "🚀 Démarrage de Survey Generator API..."

# Vérifier que l'environnement virtuel existe
if [ ! -d "venv" ]; then
    echo "❌ Environnement virtuel non trouvé. Lancez d'abord ./install.sh"
    exit 1
fi

# Activer l'environnement virtuel
source venv/bin/activate

# Vérifier les dépendances critiques
echo "🔍 Vérification des dépendances..."
python -c "import fastapi, uvicorn, openai" 2>/dev/null || {
    echo "❌ Dépendances manquantes. Exécutez ./fix_install.sh"
    exit 1
}

# Vérifier le fichier .env
if [ ! -f ".env" ]; then
    echo "⚠️ Fichier .env non trouvé. Utilisation des valeurs par défaut."
    echo "   IMPORTANT: Configurez votre clé OpenAI dans le fichier .env"
fi

echo "✅ Prêt à démarrer!"
python main.py
EOF

chmod +x start.sh
echo "✅ Script de démarrage créé (start.sh)"

# Créer un script de développement
echo "🔧 Création du script de développement..."
cat > dev.sh << 'EOF'
#!/bin/bash
echo "🔧 Démarrage en mode développement..."

# Vérifier que l'environnement virtuel existe
if [ ! -d "venv" ]; then
    echo "❌ Environnement virtuel non trouvé. Lancez d'abord ./install.sh"
    exit 1
fi

# Activer l'environnement virtuel
source venv/bin/activate

# Vérifier que uvicorn est disponible
if ! command -v uvicorn &> /dev/null; then
    echo "❌ uvicorn non trouvé. Installation en cours..."
    pip install "uvicorn>=0.23.0,<0.25.0"
fi

# Vérifier si le fichier .env existe
if [ ! -f ".env" ]; then
    echo "⚠️ Fichier .env non trouvé. Utilisation des valeurs par défaut."
    echo "   Créez un fichier .env pour configurer l'API."
fi

echo "🚀 Démarrage du serveur de développement..."
uvicorn main:app --reload --host 0.0.0.0 --port 8000
EOF

chmod +x dev.sh
echo "✅ Script de développement créé (dev.sh)"

echo ""
echo "🎉 Installation terminée avec succès!"
echo "======================================"
echo ""
echo "📝 Prochaines étapes:"
echo "1. Configurez votre fichier .env avec votre clé OpenAI:"
echo "   OPENAI_API_KEY=your_api_key_here"
echo ""
echo "2. Démarrez l'application:"
echo "   ./start.sh (mode production)"
echo "   ./dev.sh (mode développement)"
echo ""
echo "3. Accédez à l'API:"
echo "   http://localhost:8000 (API)"
echo "   http://localhost:8000/docs (Documentation)"
echo ""
echo "💡 Pour activer l'environnement virtuel manuellement:"
echo "   source venv/bin/activate"
echo ""
echo "🆘 Besoin d'aide? Consultez le README.md"