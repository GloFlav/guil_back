#!/bin/bash

echo "🔍 Diagnostic du système Survey Generator API"
echo "============================================"

# Vérifier Python
echo "📋 Système Python:"
echo "  Python version: $(python3 --version 2>&1 || echo 'Non installé')"
echo "  Python path: $(which python3 || echo 'Non trouvé')"
echo "  Pip version: $(pip3 --version 2>&1 || echo 'Non installé')"

# Vérifier l'environnement virtuel
echo ""
echo "📋 Environnement virtuel:"
if [ -d "venv" ]; then
    echo "  ✅ Dossier venv existe"
    echo "  Python venv: $(venv/bin/python --version 2>&1 || echo 'Erreur')"
    echo "  Pip venv: $(venv/bin/pip --version 2>&1 || echo 'Erreur')"
else
    echo "  ❌ Dossier venv n'existe pas"
fi

# Vérifier les fichiers de configuration
echo ""
echo "📋 Configuration:"
if [ -f ".env" ]; then
    echo "  ✅ Fichier .env existe"
    echo "  Contenu (sans clés sensibles):"
    grep -v "API_KEY\|SECRET\|PASSWORD" .env | sed 's/^/    /'
else
    echo "  ❌ Fichier .env manquant"
fi

if [ -f "requirements.txt" ]; then
    echo "  ✅ requirements.txt existe"
else
    echo "  ❌ requirements.txt manquant"
fi

# Vérifier les dossiers
echo ""
echo "📋 Structure des dossiers:"
for dir in "config" "models" "services" "exports"; do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir/"
    else
        echo "  ❌ $dir/ manquant"
    fi
done

# Tester l'activation de l'environnement virtuel
echo ""
echo "📋 Test de l'environnement virtuel:"
if [ -d "venv" ]; then
    source venv/bin/activate 2>/dev/null
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        echo "  ✅ Activation réussie: $VIRTUAL_ENV"
        
        # Vérifier les packages installés
        echo ""
        echo "📦 Packages installés:"
        PACKAGES=("fastapi" "uvicorn" "pydantic" "openai" "pandas")
        for package in "${PACKAGES[@]}"; do
            if python -c "import $package" &> /dev/null; then
                VERSION=$(python -c "import $package; print(getattr($package, '__version__', 'Version inconnue'))" 2>/dev/null)
                echo "  ✅ $package ($VERSION)"
            else
                echo "  ❌ $package"
            fi
        done
        
        # Vérifier les packages Excel
        echo ""
        echo "📊 Support Excel:"
        for package in "openpyxl" "xlsxwriter"; do
            if python -c "import $package" &> /dev/null; then
                VERSION=$(python -c "import $package; print(getattr($package, '__version__', 'Version inconnue'))" 2>/dev/null)
                echo "  ✅ $package ($VERSION)"
            else
                echo "  ❌ $package"
            fi
        done
        
    else
        echo "  ❌ Échec de l'activation"
    fi
else
    echo "  ❌ Pas d'environnement virtuel à tester"
fi

# Vérifier les outils système (macOS)
echo ""
echo "📋 Outils système (macOS):"
if command -v brew &> /dev/null; then
    echo "  ✅ Homebrew installé"
else
    echo "  ❌ Homebrew non installé"
fi

if command -v xcode-select &> /dev/null && xcode-select -p &> /dev/null; then
    echo "  ✅ Xcode Command Line Tools installés"
else
    echo "  ❌ Xcode Command Line Tools manquants"
fi

if command -v gcc &> /dev/null; then
    echo "  ✅ Compilateur GCC disponible"
else
    echo "  ❌ Compilateur GCC manquant"
fi

# Recommandations
echo ""
echo "💡 Recommandations:"
if [ ! -d "venv" ]; then
    echo "  1. Créez l'environnement virtuel: python3 -m venv venv"
fi

if [ ! -f ".env" ]; then
    echo "  2. Créez le fichier .env avec votre clé OpenAI"
fi

if ! command -v brew &> /dev/null; then
    echo "  3. Installez Homebrew pour faciliter les dépendances système"
fi

if ! (command -v xcode-select &> /dev/null && xcode-select -p &> /dev/null); then
    echo "  4. Installez Xcode Command Line Tools: xcode-select --install"
fi

echo ""
echo "🔧 Scripts disponibles:"
for script in "install.sh" "fix_install.sh" "start.sh" "dev.sh" "start_minimal.sh"; do
    if [ -f "$script" ]; then
        echo "  ✅ ./$script"
    else
        echo "  ❌ $script manquant"
    fi
done

echo ""
echo "📞 Support:"
echo "  - Si installation échoue: ./fix_install.sh"
echo "  - Si problème avec Excel: ./start_minimal.sh"
echo "  - Pour recréer l'environnement: rm -rf venv && ./install.sh"