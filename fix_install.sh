#!/bin/bash

echo "🔧 Script de réparation de l'installation"
echo "========================================="

# Activer l'environnement virtuel
echo "🔧 Activation de l'environnement virtuel..."
source venv/bin/activate

# Vérifier si nous sommes dans le bon environnement
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Environnement virtuel activé: $VIRTUAL_ENV"
else
    echo "❌ Impossible d'activer l'environnement virtuel"
    exit 1
fi

# Mettre à jour les outils de base
echo "🔧 Mise à jour des outils de base..."
pip install --upgrade pip setuptools wheel

# Installer les dépendances de compilation si nécessaire (macOS)
echo "🔧 Installation des outils de compilation..."
if command -v brew &> /dev/null; then
    echo "Homebrew détecté, installation des dépendances système..."
    # brew install cmake  # Décommenté si nécessaire
else
    echo "⚠️ Homebrew non trouvé. Si vous avez des erreurs de compilation, installez-le:"
    echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
fi

# Installer les dépendances par étapes pour identifier les problèmes
echo "📦 Installation des dépendances par étapes..."

echo "  📦 FastAPI et Uvicorn..."
pip install "fastapi>=0.100.0,<0.105.0" "uvicorn>=0.23.0,<0.25.0"

echo "  📦 Pydantic et configuration..."
pip install "pydantic>=2.4.0,<3.0.0" "pydantic-settings>=2.0.0,<3.0.0" "python-dotenv>=1.0.0,<2.0.0"

echo "  📦 OpenAI..."
pip install "openai>=1.3.0,<2.0.0"

echo "  📦 Utilitaires..."
pip install "python-multipart>=0.0.6" "aiofiles>=23.0.0,<24.0.0"

echo "  📦 Excel export (peut prendre du temps)..."
# Installer pandas d'abord sans dépendances optionnelles
pip install "pandas>=1.5.0,<2.2.0" --no-deps
pip install "numpy>=1.21.0,<2.0.0" "pytz>=2022.1" "python-dateutil>=2.8.1"

# Puis openpyxl
pip install "openpyxl>=3.0.0,<3.2.0"

# Alternative plus légère si openpyxl pose problème
if ! python -c "import openpyxl" &> /dev/null; then
    echo "⚠️ openpyxl a échoué, tentative avec xlsxwriter..."
    pip install "xlsxwriter>=3.0.0,<4.0.0"
fi

# Vérifier les installations
echo "🧪 Vérification des installations..."

PACKAGES=("fastapi" "uvicorn" "pydantic" "openai" "pandas")
FAILED_PACKAGES=()

for package in "${PACKAGES[@]}"; do
    if python -c "import $package" &> /dev/null; then
        echo "  ✅ $package"
    else
        echo "  ❌ $package"
        FAILED_PACKAGES+=($package)
    fi
done

# Excel packages (au moins un doit fonctionner)
EXCEL_WORKING=false
for package in "openpyxl" "xlsxwriter"; do
    if python -c "import $package" &> /dev/null; then
        echo "  ✅ $package (export Excel)"
        EXCEL_WORKING=true
        break
    fi
done

if [ "$EXCEL_WORKING" = false ]; then
    echo "  ❌ Aucun package Excel ne fonctionne"
    FAILED_PACKAGES+=("excel_export")
fi

if [ ${#FAILED_PACKAGES[@]} -eq 0 ]; then
    echo ""
    echo "🎉 Toutes les dépendances sont installées correctement!"
    echo ""
    echo "🚀 Vous pouvez maintenant démarrer l'application:"
    echo "   ./dev.sh"
    echo ""
else
    echo ""
    echo "❌ Packages qui ont échoué: ${FAILED_PACKAGES[*]}"
    echo ""
    echo "💡 Solutions:"
    echo "1. Essayez le mode minimal: ./start_minimal.sh"
    echo "2. Ou installez manuellement:"
    for package in "${FAILED_PACKAGES[@]}"; do
        echo "   pip install $package"
    done
fi

# Créer un script de démarrage minimal (sans Excel si nécessaire)
echo "📝 Création d'un script de démarrage minimal..."
cat > start_minimal.sh << 'EOF'
#!/bin/bash
echo "🚀 Démarrage en mode minimal (sans export Excel)..."
source venv/bin/activate

# Créer une version temporaire de main.py sans Excel
if [ ! -f "main_minimal.py" ]; then
    cp main.py main_minimal.py
    # Commenter les imports Excel dans le fichier temporaire si nécessaire
fi

python main.py
EOF

chmod +x start_minimal.sh

echo "✅ Script minimal créé: start_minimal.sh"
echo ""
echo "🆘 En cas de problème persistant:"
echo "1. Vérifiez que Xcode Command Line Tools est installé:"
echo "   xcode-select --install"
echo "2. Essayez avec Python depuis Homebrew:"
echo "   brew install python@3.11"
echo "3. Recréez l'environnement virtuel:"
echo "   rm -rf venv && python3 -m venv venv"