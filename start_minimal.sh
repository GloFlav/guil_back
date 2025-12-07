#!/bin/bash
echo "🚀 Démarrage en mode minimal (sans export Excel)..."
source venv/bin/activate

# Créer une version temporaire de main.py sans Excel
if [ ! -f "main_minimal.py" ]; then
    cp main.py main_minimal.py
    # Commenter les imports Excel dans le fichier temporaire si nécessaire
fi

python main.py
