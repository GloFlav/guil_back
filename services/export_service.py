# backend/services/export_service.py

import logging
import json
import os
import pickle
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
import pandas as pd
from config.settings import settings

# --- IMPORTS GOOGLE ---
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

class ExportService:
    def __init__(self):
        # Création du dossier d'export s'il n'existe pas
        self.output_dir = settings.excel_output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # --- CONFIGURATION GOOGLE ---
        self.CLIENT_SECRETS_FILE = 'client_secret.json'
        self.SCOPES = [
            "https://www.googleapis.com/auth/forms.body", 
            "https://www.googleapis.com/auth/drive"
        ]
        self.TOKEN_FILE = 'token.pickle'

    def _generate_filename(self, format_type: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"survey_{timestamp}.{format_type}"

    # =========================================================================
    # 1. PARTIE GOOGLE FORMS API
    # =========================================================================

    def get_google_service(self):
        """Authentification OAuth2 pour Google"""
        creds = None
        if os.path.exists(self.TOKEN_FILE):
            with open(self.TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception:
                    if os.path.exists(self.TOKEN_FILE): os.remove(self.TOKEN_FILE)
                    return self.get_google_service()
            else:
                if not os.path.exists(self.CLIENT_SECRETS_FILE):
                    raise FileNotFoundError("client_secret.json manquant à la racine")
                
                # Port 8080 obligatoire pour correspondre à la config console Google
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.CLIENT_SECRETS_FILE, self.SCOPES)
                logger.info("Ouverture navigateur pour Auth Google sur port 8080...")
                creds = flow.run_local_server(port=8080)
            
            with open(self.TOKEN_FILE, 'wb') as token:
                pickle.dump(creds, token)

        return build('forms', 'v1', credentials=creds)

    def create_google_form_online(self, survey_data: Dict[str, Any]) -> Dict[str, Any]:
        """Crée le formulaire en ligne sur Google"""
        try:
            logger.info("Connexion à Google API...")
            service = self.get_google_service()
            
            metadata = survey_data.get("metadata", {})
            title = metadata.get("title", "Enquête IA")
            
            # 1. Création form vide
            form_body = {"info": {"title": title, "documentTitle": title}}
            form = service.forms().create(body=form_body).execute()
            form_id = form['formId']
            
            # 2. Ajout questions
            google_requests = self._map_to_google_requests(survey_data)
            if google_requests:
                service.forms().batchUpdate(formId=form_id, body={"requests": google_requests}).execute()

            # 3. Liens
            edit_uri = f"https://docs.google.com/forms/d/{form_id}/edit"
            responder_uri = form.get('responderUri', edit_uri)

            return {
                "success": True, 
                "format": "google_forms_api",
                "formId": form_id, 
                "responderUri": responder_uri, 
                "editUri": edit_uri
            }

        except Exception as e:
            logger.error(f"Erreur Google API: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _map_to_google_requests(self, survey_data: Dict[str, Any]) -> List[Dict]:
        requests = []
        index = 0
        
        desc = survey_data.get("metadata", {}).get("introduction", "")
        if desc:
            requests.append({"updateFormInfo": {"info": {"description": desc}, "updateMask": "description"}})

        for cat in survey_data.get("categories", []):
            requests.append({
                "createItem": {
                    "item": {
                        "title": cat.get("category_name", "Section"),
                        "description": cat.get("description", ""),
                        "pageBreakItem": {}
                    },
                    "location": {"index": index}
                }
            })
            index += 1

            for q in cat.get("questions", []):
                new_item = {
                    "title": q.get("question_text", ""),
                    "description": q.get("help_text", ""),
                    "questionItem": {"question": {"required": q.get("is_required", True)}}
                }
                
                q_type = q.get("question_type", "text")
                options = [a.get("answer_text") for a in q.get("expected_answers", [])]

                if q_type in ["single_choice", "yes_no"]:
                    new_item['questionItem']['question']['choiceQuestion'] = {
                        "type": "RADIO", "options": [{"value": o} for o in options]
                    }
                elif q_type == "multiple_choice":
                    new_item['questionItem']['question']['choiceQuestion'] = {
                        "type": "CHECKBOX", "options": [{"value": o} for o in options]
                    }
                elif q_type == "date":
                    new_item['questionItem']['question']['dateQuestion'] = {"includeTime": False, "includeYear": True}
                else:
                    new_item['questionItem']['question']['textQuestion'] = {"paragraph": True}

                requests.append({"createItem": {"item": new_item, "location": {"index": index}}})
                index += 1
        return requests

    # =========================================================================
    # 2. PARTIE EXCEL (Classique)
    # =========================================================================

    def export_to_excel(self, survey_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            filename = self._generate_filename("xlsx")
            filepath = Path(self.output_dir) / filename
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Feuille 1: Métadonnées
                meta = survey_data.get("metadata", {})
                pd.DataFrame([meta]).to_excel(writer, sheet_name="Infos", index=False)
                
                # Feuille 2: Questions
                q_data = []
                for cat in survey_data.get("categories", []):
                    for q in cat.get("questions", []):
                        q_data.append({
                            "Catégorie": cat.get("category_name"),
                            "Question": q.get("question_text"),
                            "Type": q.get("question_type"),
                            "Options": ", ".join([a.get("answer_text") for a in q.get("expected_answers", [])])
                        })
                pd.DataFrame(q_data).to_excel(writer, sheet_name="Questionnaire", index=False)

                # Feuille 3: Lieux
                locs = survey_data.get("locations", [])
                if locs:
                    pd.DataFrame(locs).to_excel(writer, sheet_name="Lieux", index=False)
            
            return {"success": True, "format": "xlsx", "filename": filename}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # 3. PARTIE CSV
    # =========================================================================

    def export_to_csv(self, survey_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            filename = self._generate_filename("csv")
            filepath = Path(self.output_dir) / filename
            
            data_flat = []
            for cat in survey_data.get("categories", []):
                for q in cat.get("questions", []):
                    data_flat.append({
                        "category": cat.get("category_name"),
                        "question": q.get("question_text"),
                        "type": q.get("question_type"),
                        "required": q.get("is_required")
                    })
            
            pd.DataFrame(data_flat).to_csv(filepath, index=False, encoding='utf-8-sig')
            return {"success": True, "format": "csv", "filename": filename}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # 4. PARTIE KOBOTOOLBOX (XLSForm Standard)
    # =========================================================================

    def export_to_kobo(self, survey_data: Dict[str, Any]) -> Dict[str, Any]:
        """Génère un fichier Excel formaté XLSForm pour KoboToolbox"""
        try:
            filename = self._generate_filename("xlsx") # Kobo utilise .xlsx
            # On ajoute un préfixe pour le distinguer
            filename = f"kobo_{filename}" 
            filepath = Path(self.output_dir) / filename

            # --- Feuille SURVEY ---
            survey_rows = []
            # --- Feuille CHOICES ---
            choices_rows = []
            # --- Feuille SETTINGS ---
            settings_rows = [{"form_title": survey_data.get("metadata", {}).get("title", "Kobo Form"), "form_id": "kobo_id_v1", "default_language": "French (fr)"}]

            for cat in survey_data.get("categories", []):
                # Groupe (Section)
                grp_name = f"grp_{cat.get('category_id', '0')}"[:30].replace(" ", "_").lower()
                survey_rows.append({
                    "type": "begin_group",
                    "name": grp_name,
                    "label": cat.get("category_name")
                })

                for q in cat.get("questions", []):
                    q_name = f"q_{q.get('question_id')}"
                    q_type = q.get("question_type", "text")
                    kobo_type = "text"
                    
                    if q_type == "single_choice" or q_type == "yes_no":
                        list_name = f"list_{q_name}"
                        kobo_type = f"select_one {list_name}"
                        # Ajout des choix
                        for ans in q.get("expected_answers", []):
                            choices_rows.append({
                                "list_name": list_name,
                                "name": ans.get("answer_id", "0"),
                                "label": ans.get("answer_text", "")
                            })
                    elif q_type == "multiple_choice":
                        list_name = f"list_{q_name}"
                        kobo_type = f"select_multiple {list_name}"
                        for ans in q.get("expected_answers", []):
                            choices_rows.append({
                                "list_name": list_name,
                                "name": ans.get("answer_id", "0"),
                                "label": ans.get("answer_text", "")
                            })
                    elif q_type == "number":
                        kobo_type = "integer"
                    elif q_type == "date":
                        kobo_type = "date"
                    elif "gps" in q_type:
                        kobo_type = "geopoint"

                    survey_rows.append({
                        "type": kobo_type,
                        "name": q_name,
                        "label": q.get("question_text"),
                        "required": "true" if q.get("is_required") else "false"
                    })

                survey_rows.append({"type": "end_group"})

            # Écriture du fichier XLSForm
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                pd.DataFrame(survey_rows).to_excel(writer, sheet_name="survey", index=False)
                pd.DataFrame(choices_rows).to_excel(writer, sheet_name="choices", index=False)
                pd.DataFrame(settings_rows).to_excel(writer, sheet_name="settings", index=False)

            return {"success": True, "format": "kobo_xlsx", "filename": filename}

        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # 5. PARTIE JSON
    # =========================================================================

    def export_to_json(self, survey_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            filename = self._generate_filename("json")
            filepath = Path(self.output_dir) / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(survey_data, f, ensure_ascii=False, indent=2)
            return {"success": True, "format": "json", "filename": filename}
        except Exception as e:
            return {"success": False, "error": str(e)}

# Instance globale
export_service = ExportService()