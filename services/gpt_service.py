import json 
from datetime import datetime
import openai
import os
from dotenv import load_dotenv
import traceback

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def timestamped_log(callback, message):
    if callback:
        callback(message)

def clean_json_text(text):
    """Supprime les balises markdown éventuelles autour du JSON"""
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()

def generate_survey(user_prompt: str, log_callback=None):
    content = ""
    try:
        timestamped_log(log_callback, "Initialisation du client OpenAI.")

        # Structure JSON séparée pour éviter les problèmes de f-string
        json_structure = '''{
  "introduction": "...", // tout ce que tu veux dire 
  "title": "...", // titre de l'enquete selon le prompt
  "survey_type": "...",   // [Opinion | Satisfaction | Marché | Académique | Audit]
  "survey": [
    {
      "category": "...",
      "category_description": "...",
      "questions": [
        {
          "question_id": "...",
          "question_type": "...",  // [oui/non | échelle Likert | choix multiple | réponse libre | matricielle | démographique | numérique]
          "question_text": "...",
          "expected_answers": [
            {
              "answer_id": "...",
              "answer_type": "...",  // [booléen | numérique | texte | catégoriel]
              "next_question_id": "..."
            }
          ],
          "predecessor_answer_id": "..."
        }
      ]
    }
  ],
  "survey_total_duration": "...",
  "number_of_respondents": ...,
  "number_of_investigators": ...,
  "number_of_locations": ...,
  "location_characteristics": "...",
  "nombre_de_question": "..."
}'''

        prompt = f"""
🎯 TU ES EXPERT EN ENQUÊTES QUANTITATIVES. GÉNÈRE UN JSON STRICT, EN FRANÇAIS, SELON LES SPÉCIFICATIONS SUIVANTES.

✅ OBJECTIF : Générer une enquête complète **au format JSON** avec **≥ 40 questions** réparties en **≥ 5 catégories distinctes**.

⚠️ TA RÉPONSE SERA REFUSÉE SI MOINS DE 40 QUESTIONS SONT GÉNÉRÉES.

📐 STRUCTURE DU JSON STRICT :
{json_structure}

📊 CONTRAINTES FORTES :
- Nombre minimum : 40 questions
- Catégories : au moins 5
- > 80 % de questions quantitatives (Likert, choix, matricielle, numérique)
- < 20 % de réponses libres
- Tous les identifiants doivent être uniques (Qx_CATy, Ax_Qx)
- Champs `next_question_id` et `predecessor_answer_id` renseignés
- Toutes unités claires (km, Ariary, %, etc.)

🌍 CONTEXTE : Madagascar
- Culture, régions, monnaie : Ariary

⛔ AUCUN TEXTE HORS JSON. AUCUN COMMENTAIRE. UNIQUEMENT UN BLOC JSON STRICTEMENT VALIDE.

Demande : "{user_prompt}"
"""

        timestamped_log(log_callback, "Envoi de la requête à GPT...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=4000
        )

        timestamped_log(log_callback, "Réponse reçue de GPT.")
        content = response.choices[0].message.content
        print("Réponse brute GPT:\n", content)

        timestamped_log(log_callback, "Nettoyage du texte JSON...")
        cleaned = clean_json_text(content)

        timestamped_log(log_callback, "Parsing JSON...")
        data = json.loads(cleaned)

        timestamped_log(log_callback, "Parsing JSON réussi.")

        return {
            "success": True,
            "data": data
        }

    except json.JSONDecodeError as jde:
        timestamped_log(log_callback, "Erreur de parsing JSON.")
        return {
            "success": False,
            "raw_output": content,
            "error": f"JSON invalide : {str(jde)}"
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }