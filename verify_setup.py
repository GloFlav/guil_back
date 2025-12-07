# backend/verify_setup.py
"""
Script de vérification de la configuration et des dépendances
Vérifie que tout est prêt pour démarrer le serveur
"""

import os
import sys
from pathlib import Path

def check_python_version():
    """Vérifie la version de Python"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print("✅ Python version: OK")
        return True
    else:
        print("❌ Python 3.9+ requis")
        return False

def check_required_files():
    """Vérifie les fichiers essentiels"""
    required_files = [
        ".env",
        "data/mdg_adm3.csv",
        "config/settings.py",
        "models/survey.py",
        "services/context_extraction_service.py",
        "services/administrative_data_service.py",
        "services/multi_llm_orchestration.py",
        "services/export_service.py",
        "utils/websocket_manager.py",
        "main.py",
        "requirements.txt"
    ]
    
    all_ok = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - NOT FOUND")
            all_ok = False
    
    return all_ok

def check_env_variables():
    """Vérifie les variables d'environnement"""
    print("\n🔐 Vérification des clés API:")
    
    required_vars = {
        "OPENAI_API_KEY_1": "OpenAI Key 1",
        "OPENAI_API_KEY_2": "OpenAI Key 2",
        "ANTHROPIC_API_KEY_1": "Anthropic Key 1",
        "ANTHROPIC_API_KEY_2": "Anthropic Key 2",
        "GEMINI_API_KEY": "Gemini Key"
    }
    
    if not os.path.exists(".env"):
        print("❌ Fichier .env non trouvé")
        return False
    
    with open(".env", "r") as f:
        env_content = f.read()
    
    all_ok = True
    for var, name in required_vars.items():
        if var in env_content and f"={var}" not in env_content:
            print(f"✅ {name}")
        else:
            print(f"⚠️  {name} - Vérifier la configuration")
    
    return all_ok

def check_directories():
    """Vérifie les répertoires nécessaires"""
    print("\n📁 Vérification des répertoires:")
    
    directories = [
        "config",
        "models",
        "services",
        "utils",
        "data",
        "logs",
        "exports"
    ]
    
    all_ok = True
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/ - Création...")
            os.makedirs(directory, exist_ok=True)
    
    return True

def check_packages():
    """Vérifie l'installation des packages critiques"""
    print("\n📦 Vérification des dépendances critiques:")
    
    packages = [
        ("fastapi", "FastAPI"),
        ("pydantic", "Pydantic"),
        ("openai", "OpenAI"),
        ("anthropic", "Anthropic"),
        ("google.generativeai", "Google Generative AI"),
        ("pandas", "Pandas")
    ]
    
    all_ok = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - Installer avec: pip install -r requirements.txt")
            all_ok = False
    
    return all_ok

def main():
    """Exécute toutes les vérifications"""
    print("=" * 60)
    print("Survey Generator API v3 Madagascar - Setup Verification")
    print("=" * 60)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Fichiers essentiels", check_required_files),
        ("Répertoires", check_directories),
        ("Variables d'environnement", check_env_variables),
        ("Packages Python", check_packages)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n🔍 {name}:")
        print("-" * 40)
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Erreur lors de la vérification: {e}")
            results.append((name, False))
    
    # Résumé
    print("\n" + "=" * 60)
    print("📊 Résumé:")
    print("=" * 60)
    
    all_ok = True
    for name, result in results:
        status = "✅ OK" if result else "❌ ERREUR"
        print(f"{status} - {name}")
        if not result:
            all_ok = False
    
    print("\n" + "=" * 60)
    
    if all_ok:
        print("✅ Tous les vérifications sont passées!")
        print("\n🚀 Vous pouvez démarrer le serveur:")
        print("   python main.py")
        print("   ou")
        print("   bash start.sh")
        return 0
    else:
        print("❌ Certaines vérifications ont échoué.")
        print("Veuillez résoudre les problèmes et réessayer.")
        return 1

if __name__ == "__main__":
    sys.exit(main())