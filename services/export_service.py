# backend/services/export_service.py
"""
Service d'export des questionnaires
Supporte XLSX, CSV, JSON, Kobo Tools et Google Forms
"""

import logging
import json
import csv
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
from config.settings import settings

logger = logging.getLogger(__name__)

class ExportService:
    """Service pour exporter les questionnaires"""
    
    def __init__(self):
        """Initialise le service"""
        self.output_dir = settings.excel_output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _generate_filename(self, format_type: str) -> str:
        """Génère un nom de fichier avec timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension_map = {
            "xlsx": "xlsx",
            "csv": "csv",
            "json": "json",
            "pdf": "pdf",
            "kobo": "xml",
            "google_forms": "json"
        }
        ext = extension_map.get(format_type, "txt")
        return f"survey_{timestamp}.{ext}"
    
    def export_to_json(self, survey_data: Dict[str, Any]) -> Dict[str, Any]:
        """Exporte en JSON"""
        try:
            filename = self._generate_filename("json")
            filepath = Path(self.output_dir) / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(survey_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Export JSON: {filename}")
            return {
                "success": True,
                "format": "json",
                "filename": filename,
                "filepath": str(filepath)
            }
        except Exception as e:
            logger.error(f"Erreur export JSON: {e}")
            return {"success": False, "error": str(e)}
    
    def export_to_excel(self, survey_data: Dict[str, Any]) -> Dict[str, Any]:
        """Exporte en XLSX"""
        try:
            filename = self._generate_filename("xlsx")
            filepath = Path(self.output_dir) / filename
            
            # Créer un workbook
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Feuille 1: Métadonnées
                metadata = survey_data.get("metadata", {})
                metadata_df = pd.DataFrame([metadata])
                metadata_df.to_excel(writer, sheet_name="Métadonnées", index=False)
                
                # Feuille 2: Questions
                questions_data = []
                for category in survey_data.get("categories", []):
                    for question in category.get("questions", []):
                        questions_data.append({
                            "Catégorie": category.get("category_name", ""),
                            "ID Question": question.get("question_id", ""),
                            "Type": question.get("question_type", ""),
                            "Question": question.get("question_text", ""),
                            "Obligatoire": question.get("is_required", True),
                            "Aide": question.get("help_text", "")
                        })
                
                if questions_data:
                    questions_df = pd.DataFrame(questions_data)
                    questions_df.to_excel(writer, sheet_name="Questions", index=False)
                
                # Feuille 3: Lieux
                locations_data = []
                for location in survey_data.get("locations", []):
                    locations_data.append({
                        "Code": location.get("pcode", ""),
                        "Lieu": location.get("name", ""),
                        "Région (ADM1)": location.get("adm1", ""),
                        "District (ADM2)": location.get("adm2", ""),
                        "Localité (ADM3)": location.get("adm3", "")
                    })
                
                if locations_data:
                    locations_df = pd.DataFrame(locations_data)
                    locations_df.to_excel(writer, sheet_name="Lieux", index=False)
            
            logger.info(f"Export XLSX: {filename}")
            return {
                "success": True,
                "format": "xlsx",
                "filename": filename,
                "filepath": str(filepath)
            }
        
        except Exception as e:
            logger.error(f"Erreur export XLSX: {e}")
            return {"success": False, "error": str(e)}
    
    def export_to_csv(self, survey_data: Dict[str, Any]) -> Dict[str, Any]:
        """Exporte en CSV"""
        try:
            filename = self._generate_filename("csv")
            filepath = Path(self.output_dir) / filename
            
            # Extraire les questions
            questions_data = []
            for category in survey_data.get("categories", []):
                for question in category.get("questions", []):
                    for answer in question.get("expected_answers", []):
                        questions_data.append({
                            "Catégorie": category.get("category_name", ""),
                            "ID_Question": question.get("question_id", ""),
                            "Type_Question": question.get("question_type", ""),
                            "Question": question.get("question_text", ""),
                            "ID_Réponse": answer.get("answer_id", ""),
                            "Réponse": answer.get("answer_text", ""),
                            "Obligatoire": question.get("is_required", True)
                        })
            
            if questions_data:
                df = pd.DataFrame(questions_data)
                df.to_csv(filepath, encoding='utf-8', index=False)
            
            logger.info(f"Export CSV: {filename}")
            return {
                "success": True,
                "format": "csv",
                "filename": filename,
                "filepath": str(filepath)
            }
        
        except Exception as e:
            logger.error(f"Erreur export CSV: {e}")
            return {"success": False, "error": str(e)}
    
    def export_to_kobo(self, survey_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exporte au format Kobo Tools (XLS Form XML)
        Format simplifié pour compatibilité
        """
        try:
            filename = self._generate_filename("kobo")
            filepath = Path(self.output_dir) / filename
            
            # Créer un format XLS Form simplifié
            survey = []
            choices = []
            
            choice_list_map = {}
            
            # Construire la structure
            for category in survey_data.get("categories", []):
                # Groupe de questions
                survey.append({
                    "type": "group",
                    "name": category.get("category_id", ""),
                    "label": category.get("category_name", ""),
                    "appearance": "field-list"
                })
                
                for question in category.get("questions", []):
                    q_type = question.get("question_type", "text")
                    
                    # Mapper les types de questions
                    kobo_type = self._map_question_type_to_kobo(q_type)
                    
                    survey_item = {
                        "type": kobo_type,
                        "name": question.get("question_id", ""),
                        "label": question.get("question_text", ""),
                        "required": "yes" if question.get("is_required") else "no"
                    }
                    
                    # Ajouter les choix si nécessaire
                    if q_type in ["single_choice", "multiple_choice"]:
                        list_name = f"list_{question.get('question_id', '')}"
                        survey_item["list_name"] = list_name
                        
                        for answer in question.get("expected_answers", []):
                            choices.append({
                                "list_name": list_name,
                                "name": answer.get("answer_id", ""),
                                "label": answer.get("answer_text", "")
                            })
                    
                    survey.append(survey_item)
            
            # Créer le contenu XML Kobo
            kobo_content = self._generate_kobo_xml(survey_data, survey, choices)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(kobo_content)
            
            logger.info(f"Export Kobo: {filename}")
            return {
                "success": True,
                "format": "kobo",
                "filename": filename,
                "filepath": str(filepath)
            }
        
        except Exception as e:
            logger.error(f"Erreur export Kobo: {e}")
            return {"success": False, "error": str(e)}
    
    def _map_question_type_to_kobo(self, question_type: str) -> str:
        """Mappe les types de questions aux types Kobo"""
        mapping = {
            "single_choice": "select_one",
            "multiple_choice": "select_multiple",
            "text": "text",
            "number": "integer",
            "scale": "integer",
            "yes_no": "select_one",
            "date": "date"
        }
        return mapping.get(question_type, "text")
    
    def _generate_kobo_xml(self, survey_data: Dict[str, Any], survey: List[Dict], choices: List[Dict]) -> str:
        """Génère le XML pour Kobo Tools"""
        title = survey_data.get("metadata", {}).get("title", "Survey")
        
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<h:html xmlns:h="http://www.w3.org/1999/xhtml" xmlns:xf="http://www.w3.org/2002/xforms">
  <h:head>
    <h:title>{title}</h:title>
    <model>
      <instance>
        <data id="survey_form">
"""
        
        # Ajouter les questions
        for item in survey:
            if item["type"] != "group":
                xml += f'          <{item["name"]}/>\n'
        
        xml += """        </data>
      </instance>
      <bind nodeset="/" type="binary"/>
"""
        
        for item in survey:
            if item["type"] != "group":
                required = 'required="true()"' if item.get("required") == "yes" else ""
                xml += f'      <bind nodeset="/{item["name"]}" type="string" {required}/>\n'
        
        xml += """    </model>
  </h:head>
  <h:body>
"""
        
        # Ajouter les contrôles
        for item in survey:
            if item["type"] == "group":
                xml += f'    <group>\n      <label>{item.get("label", "")}</label>\n'
            else:
                xml += f'    <input ref="/{item["name"]}">\n      <label>{item.get("label", "")}</label>\n    </input>\n'
        
        xml += """  </h:body>
</h:html>"""
        
        return xml
    
    def export_to_google_forms(self, survey_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exporte au format Google Forms (JSON avec structure compatible)
        Retourne un JSON que Google Forms peut importer
        """
        try:
            filename = self._generate_filename("google_forms")
            filepath = Path(self.output_dir) / filename
            
            metadata = survey_data.get("metadata", {})
            
            # Format Google Forms
            google_forms_data = {
                "title": metadata.get("title", "Questionnaire"),
                "description": metadata.get("introduction", ""),
                "items": []
            }
            
            item_id = 1
            
            for category in survey_data.get("categories", []):
                # Section pour la catégorie
                google_forms_data["items"].append({
                    "type": "SECTION_HEADER",
                    "id": f"section_{category.get('category_id', '')}",
                    "title": category.get("category_name", ""),
                    "description": category.get("description", "")
                })
                
                for question in category.get("questions", []):
                    q_type = question.get("question_type", "text")
                    gf_type = self._map_question_type_to_google_forms(q_type)
                    
                    item = {
                        "id": str(item_id),
                        "title": question.get("question_text", ""),
                        "description": question.get("help_text", ""),
                        "type": gf_type,
                        "required": question.get("is_required", True)
                    }
                    
                    # Ajouter les options si nécessaire
                    if q_type in ["single_choice", "multiple_choice"]:
                        item["options"] = [
                            {
                                "value": answer.get("answer_text", ""),
                                "id": answer.get("answer_id", "")
                            }
                            for answer in question.get("expected_answers", [])
                        ]
                    
                    google_forms_data["items"].append(item)
                    item_id += 1
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(google_forms_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Export Google Forms: {filename}")
            return {
                "success": True,
                "format": "google_forms",
                "filename": filename,
                "filepath": str(filepath)
            }
        
        except Exception as e:
            logger.error(f"Erreur export Google Forms: {e}")
            return {"success": False, "error": str(e)}
    
    def _map_question_type_to_google_forms(self, question_type: str) -> str:
        """Mappe les types de questions aux types Google Forms"""
        mapping = {
            "single_choice": "MULTIPLE_CHOICE",
            "multiple_choice": "CHECKBOX",
            "text": "SHORT_ANSWER",
            "number": "SHORT_ANSWER",
            "scale": "LINEAR_SCALE",
            "yes_no": "MULTIPLE_CHOICE",
            "date": "DATE"
        }
        return mapping.get(question_type, "SHORT_ANSWER")
    
    def list_exported_files(self) -> List[Dict[str, Any]]:
        """Liste tous les fichiers exportés"""
        try:
            files = []
            output_path = Path(self.output_dir)
            
            if output_path.exists():
                for file in output_path.glob("survey_*"):
                    files.append({
                        "filename": file.name,
                        "path": str(file),
                        "size": file.stat().st_size,
                        "created": datetime.fromtimestamp(file.stat().st_ctime).isoformat()
                    })
            
            return sorted(files, key=lambda x: x["created"], reverse=True)
        
        except Exception as e:
            logger.error(f"Erreur listage fichiers: {e}")
            return []

# Instance globale
export_service = ExportService()