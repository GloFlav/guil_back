import json
from openai import AsyncOpenAI
from config.settings import settings
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# Assurez-vous que la clé API est bien configurée dans settings.py
# (Cela peut être un autre client LLM si vous n'utilisez pas OpenAI)
client = AsyncOpenAI(api_key=settings.openai_api_key_1) 

class ContextAnalyst:
    
    async def infer_analysis_goal(self, user_prompt: str, column_names: List[str], data_sample: List[Dict[str, Any]]) -> dict:
        """
        Utilise le LLM pour déterminer le type d'analyse, la cible, et les variables clés
        en utilisant le prompt utilisateur et un échantillon de données.
        """
        cleaned_sample = []
        for row in data_sample:
            cleaned_row = {}
            for key, value in row.items():
                if isinstance(value, pd.Timestamp):
                    # Convertir le Timestamp en format string JSON
                    cleaned_row[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    cleaned_row[key] = value
            cleaned_sample.append(cleaned_row)

        data_preview_str = json.dumps(cleaned_sample, indent=2, ensure_ascii=False)
        
        system_prompt = """
        Tu es un Data Scientist Senior expert en intelligence métier. Ton rôle est d'analyser la requête utilisateur, les noms de colonnes et les premières lignes du dataset pour en déduire le plan d'analyse.
        
        Ta réponse doit être un objet JSON strict pour une exécution automatique.
        
        - "analysis_type": Choisis entre "classification", "regression", "clustering", "time_series" ou "descriptive".
        - "target_variable": Le nom de la colonne du dataset qui doit être la cible de la prédiction ou l'objet principal de l'étude (Doit exister dans column_names, sinon vide).
        - "focus_variables": Une liste de 3-5 noms de colonnes pertinentes mentionnées ou implicites dans le prompt.
        - "language": La langue dominante détectée dans les colonnes/données (ex: "fr", "mg").
        """
        
        # On sérialise l'échantillon pour l'envoyer au LLM
        data_preview_str = json.dumps(data_sample, indent=2, ensure_ascii=False)

        user_content = f"""
        # Objectif utilisateur:
        {user_prompt}

        # Noms des colonnes (déjà nettoyés):
        {column_names}
        
        # Échantillon des premières lignes pour la sémantique:
        {data_preview_str}
        """

        try:
            # Assumons que settings.openai_model est correctement défini (ex: "gpt-4o")
            response = await client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.2, 
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            # En cas d'échec de l'API (clé invalide, timeout, etc.), on retourne un fallback
            print(f"Erreur inférence LLM: {e}")
            return {"analysis_type": "descriptive", "target_variable": "", "focus_variables": [], "language": "fr"}
# Instanciation du service pour être importé par main.py
context_analyst = ContextAnalyst()