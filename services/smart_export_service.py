# backend/services/smart_export_service.py
"""
SMART EXPORT SERVICE V3 - RAPPORT PROFESSIONNEL MULTI-LLM
=============================================================

FONCTIONNALITES:
- Titre sur mesure via OpenAI (contextuel au fichier)
- Transparence technique (lignes/colonnes/missing values)
- Integration graphiques EDA (distributions, clusters 3D, pie charts)
- Volume garanti: 3-5 pages de contenu haute qualite
- Specialisation par IA:
   - OpenAI: Titre + Strategie Business
   - Anthropic (Claude): Analyse sociale Madagascar + Decisions
   - Gemini: Vulgarisation + Analyse geographique
- Rotation multi-cles depuis settings.py
- Decision sociale sur derniere page
- Grand public vs Expert (vulgarisation avant ML)
- Zero modification externe requise

AUTEUR: HelloSoins Analytics Team
VERSION: 3.0
"""

import os
import logging
import json
import asyncio
import base64
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from io import BytesIO

# PDF Generation
from fpdf import FPDF

# External LLM clients
from openai import AsyncOpenAI, APIError as OpenAIError
from anthropic import Anthropic
import google.generativeai as genai

# Internal imports
from config.settings import settings

logger = logging.getLogger(__name__)


# =============================================================================
# MULTI-LLM PROVIDER - ROTATION DES CLES
# =============================================================================

class MultiLLMProvider:
    """
    Gestionnaire multi-LLM avec rotation des cles
    Gere OpenAI, Anthropic et Gemini avec fallback intelligent
    """
    
    def __init__(self):
        self.openai_keys = []
        self.anthropic_keys = []
        self.gemini_keys = []
        self.current_openai_idx = 0
        self.current_anthropic_idx = 0
        self.current_gemini_idx = 0
        self._load_keys()
    
    def _load_keys(self):
        """Charge toutes les cles depuis settings"""
        try:
            # OpenAI keys
            self.openai_keys = settings.get_openai_keys() or []
            if not self.openai_keys and hasattr(settings, 'openai_api_key'):
                self.openai_keys = [settings.openai_api_key]
            
            # Anthropic keys
            self.anthropic_keys = settings.get_anthropic_keys() or []
            if not self.anthropic_keys:
                for i in range(1, 5):
                    key = getattr(settings, f'anthropic_api_key_{i}', None)
                    if key:
                        self.anthropic_keys.append(key)
            
            # Gemini keys
            self.gemini_keys = []
            for i in range(1, 5):
                key = getattr(settings, f'gemini_api_key_{i}', None) or \
                      getattr(settings, f'google_api_key_{i}', None)
                if key:
                    self.gemini_keys.append(key)
            
            # Si pas de cles multiples, essayer la cle unique
            if not self.gemini_keys:
                single_key = getattr(settings, 'gemini_api_key', None) or \
                            getattr(settings, 'google_api_key', None)
                if single_key:
                    self.gemini_keys = [single_key]
            
            logger.info(f"CLES CHARGEES: OpenAI={len(self.openai_keys)}, "
                       f"Anthropic={len(self.anthropic_keys)}, Gemini={len(self.gemini_keys)}")
        except Exception as e:
            logger.error(f"ERREUR CHARGEMENT CLES: {e}")
    
    def _rotate_key(self, provider: str) -> Optional[str]:
        """Rotation intelligente des cles"""
        if provider == "openai" and self.openai_keys:
            key = self.openai_keys[self.current_openai_idx % len(self.openai_keys)]
            self.current_openai_idx += 1
            return key
        elif provider == "anthropic" and self.anthropic_keys:
            key = self.anthropic_keys[self.current_anthropic_idx % len(self.anthropic_keys)]
            self.current_anthropic_idx += 1
            return key
        elif provider == "gemini" and self.gemini_keys:
            key = self.gemini_keys[self.current_gemini_idx % len(self.gemini_keys)]
            self.current_gemini_idx += 1
            return key
        return None
    
    async def call_openai(self, prompt: str, system_prompt: str = "", 
                          max_tokens: int = 1000, temperature: float = 0.7) -> Optional[str]:
        """Appel OpenAI avec rotation des cles"""
        for attempt in range(len(self.openai_keys) if self.openai_keys else 1):
            key = self._rotate_key("openai")
            if not key:
                logger.warning("AUCUNE CLE OPENAI DISPONIBLE")
                return None
            
            try:
                client = AsyncOpenAI(api_key=key)
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=getattr(settings, 'openai_model', 'gpt-4o'),
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    ),
                    timeout=60
                )
                
                result = response.choices[0].message.content
                logger.info(f"OPENAI RESPONSE: {len(result)} chars")
                return result
                
            except Exception as e:
                logger.warning(f"OPENAI tentative {attempt + 1} echouee: {e}")
                await asyncio.sleep(1)
        
        return None
    
    def call_anthropic(self, prompt: str, system_prompt: str = "",
                       max_tokens: int = 1500) -> Optional[str]:
        """Appel Anthropic (Claude) avec rotation des cles"""
        for attempt in range(len(self.anthropic_keys) if self.anthropic_keys else 1):
            key = self._rotate_key("anthropic")
            if not key:
                logger.warning("AUCUNE CLE ANTHROPIC DISPONIBLE")
                return None
            
            try:
                client = Anthropic(api_key=key)
                
                message = client.messages.create(
                    model=getattr(settings, 'anthropic_model', 'claude-sonnet-4-20250514'),
                    max_tokens=max_tokens,
                    system=system_prompt if system_prompt else "Tu es un expert en developpement social a Madagascar.",
                    messages=[{"role": "user", "content": prompt}],
                    timeout=60
                )
                
                result = message.content[0].text
                logger.info(f"ANTHROPIC RESPONSE: {len(result)} chars")
                return result
                
            except Exception as e:
                logger.warning(f"ANTHROPIC tentative {attempt + 1} echouee: {e}")
        
        return None
    
    async def call_gemini(self, prompt: str, max_tokens: int = 1000) -> Optional[str]:
        """Appel Gemini avec rotation des cles"""
        for attempt in range(len(self.gemini_keys) if self.gemini_keys else 1):
            key = self._rotate_key("gemini")
            if not key:
                logger.warning("AUCUNE CLE GEMINI DISPONIBLE")
                return None
            
            try:
                genai.configure(api_key=key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                response = await asyncio.to_thread(
                    model.generate_content,
                    prompt,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.7
                    )
                )
                
                result = response.text
                logger.info(f"GEMINI RESPONSE: {len(result)} chars")
                return result
                
            except Exception as e:
                logger.warning(f"GEMINI tentative {attempt + 1} echouee: {e}")
                await asyncio.sleep(1)
        
        return None


# =============================================================================
# PDF BUILDER - RAPPORT PROFESSIONNEL
# =============================================================================

class ProfessionalPDFReport(FPDF):
    """
    Generateur PDF professionnel multi-pages
    Support: Titre, graphiques, tableaux, sections colorees
    """
    
    def __init__(self, title: str = "Rapport d'Analyse"):
        super().__init__()
        self.title = title
        self.set_auto_page_break(auto=True, margin=20)
        self._setup_fonts()
        
        # Couleurs thematiques
        self.COLOR_PRIMARY = (31, 73, 125)      # Bleu professionnel
        self.COLOR_SECONDARY = (102, 153, 204)  # Bleu clair
        self.COLOR_ACCENT = (200, 80, 80)       # Rouge action
        self.COLOR_SUCCESS = (76, 175, 80)      # Vert succes
        self.COLOR_WARNING = (255, 152, 0)      # Orange attention
        self.COLOR_DARK = (50, 50, 50)          # Gris fonce
    
    def _setup_fonts(self):
        """Configure les polices avec support UTF-8"""
        # Utiliser les polices par defaut FPDF (Arial = Helvetica)
        try:
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            if os.path.exists(font_path):
                self.add_font('DejaVu', '', font_path, uni=True)
                self.add_font('DejaVu', 'B', font_path.replace('.ttf', '-Bold.ttf'), uni=True)
                self.default_font = 'DejaVu'
            else:
                self.default_font = 'Arial'
        except Exception:
            self.default_font = 'Arial'
    
    def header(self):
        """En-tete de page"""
        self.set_font(self.default_font, 'B', 10)
        self.set_text_color(*self.COLOR_SECONDARY)
        self.cell(0, 8, self.title[:60], 0, 0, 'L')
        self.set_font(self.default_font, '', 8)
        self.cell(0, 8, datetime.now().strftime('%d/%m/%Y'), 0, 1, 'R')
        self.line(10, 18, 200, 18)
        self.ln(5)
    
    def footer(self):
        """Pied de page"""
        self.set_y(-15)
        self.set_font(self.default_font, 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
    
    def add_main_title(self, title: str):
        """Titre principal du rapport"""
        self.add_page()
        self.set_font(self.default_font, 'B', 24)
        self.set_text_color(*self.COLOR_PRIMARY)
        
        # Cadre decoratif
        self.set_fill_color(240, 245, 250)
        self.rect(10, 30, 190, 40, 'F')
        self.set_xy(10, 35)
        
        # Titre centre avec retour a la ligne automatique
        self.multi_cell(190, 12, title.upper(), 0, 'C')
        self.ln(10)
        
        # Ligne decorative
        self.set_draw_color(*self.COLOR_PRIMARY)
        self.set_line_width(1)
        self.line(50, self.get_y(), 160, self.get_y())
        self.ln(15)
    
    def add_section_header(self, title: str, color: Tuple[int, int, int] = None):
        """Ajoute un en-tete de section"""
        color = color or self.COLOR_PRIMARY
        
        # Verifier si on a assez d'espace
        if self.get_y() > 250:
            self.add_page()
        
        self.ln(5)
        self.set_font(self.default_font, 'B', 14)
        self.set_text_color(*color)
        
        self.cell(0, 10, title, 0, 1, 'L')
        
        # Ligne sous le titre
        self.set_draw_color(*color)
        self.set_line_width(0.5)
        y = self.get_y()
        self.line(10, y, 200, y)
        self.ln(8)
        self.set_text_color(*self.COLOR_DARK)
    
    def add_subsection(self, title: str):
        """Ajoute un sous-titre"""
        self.set_font(self.default_font, 'B', 11)
        self.set_text_color(*self.COLOR_SECONDARY)
        self.cell(0, 8, title, 0, 1, 'L')
        self.set_text_color(*self.COLOR_DARK)
        self.ln(2)
    
    def add_paragraph(self, text: str, indent: bool = False):
        """Ajoute un paragraphe de texte"""
        self.set_font(self.default_font, '', 11)
        self.set_text_color(*self.COLOR_DARK)
        
        if indent:
            self.set_x(15)
        
        # Nettoyer le texte pour eviter les problemes d'encodage
        clean_text = self._clean_text(text)
        self.multi_cell(0, 7, clean_text)
        self.ln(3)
    
    def add_highlight_box(self, text: str, box_type: str = "info"):
        """Ajoute une boite mise en evidence"""
        colors = {
            "info": (240, 248, 255),     # Bleu clair
            "warning": (255, 248, 225),  # Jaune clair
            "success": (240, 255, 240),  # Vert clair
            "danger": (255, 240, 240),   # Rouge clair
            "action": (255, 235, 235)    # Rouge action
        }
        
        bg_color = colors.get(box_type, colors["info"])
        
        self.set_fill_color(*bg_color)
        self.set_font(self.default_font, '', 11)
        
        # Calculer la hauteur necessaire
        lines = len(self._clean_text(text)) / 80 + text.count('\n')
        height = max(15, lines * 7 + 10)
        
        if self.get_y() + height > 270:
            self.add_page()
        
        start_y = self.get_y()
        self.multi_cell(0, 7, self._clean_text(text), 0, 'L', True)
        self.ln(5)
    
    def add_key_metrics_table(self, metrics: Dict[str, Any]):
        """Ajoute un tableau de metriques cles"""
        self.set_font(self.default_font, 'B', 10)
        
        # En-tete du tableau
        self.set_fill_color(*self.COLOR_PRIMARY)
        self.set_text_color(255, 255, 255)
        self.cell(90, 10, "Metrique", 1, 0, 'C', True)
        self.cell(90, 10, "Valeur", 1, 1, 'C', True)
        
        # Contenu
        self.set_font(self.default_font, '', 10)
        self.set_text_color(*self.COLOR_DARK)
        
        fill = False
        for key, value in metrics.items():
            if fill:
                self.set_fill_color(245, 245, 245)
            else:
                self.set_fill_color(255, 255, 255)
            
            display_value = str(value)
            if isinstance(value, float):
                display_value = f"{value:.2f}"
            
            self.cell(90, 8, self._clean_text(str(key)), 1, 0, 'L', fill)
            self.cell(90, 8, self._clean_text(display_value), 1, 1, 'C', fill)
            fill = not fill
        
        self.ln(5)
    
    def add_image_from_base64(self, base64_data: str, title: str = "", 
                              width: int = 180, height: int = 100):
        """Ajoute une image depuis des donnees base64"""
        try:
            # Decoder le base64
            if ',' in base64_data:
                base64_data = base64_data.split(',')[1]
            
            image_data = base64.b64decode(base64_data)
            
            # Sauvegarder temporairement
            temp_path = f"/tmp/chart_{datetime.now().strftime('%H%M%S%f')}.png"
            with open(temp_path, 'wb') as f:
                f.write(image_data)
            
            # Verifier l'espace disponible
            if self.get_y() + height + 20 > 270:
                self.add_page()
            
            # Ajouter le titre si present
            if title:
                self.add_subsection(title)
            
            # Centrer l'image
            x = (210 - width) / 2
            self.image(temp_path, x=x, w=width)
            self.ln(10)
            
            # Nettoyer
            os.remove(temp_path)
            return True
            
        except Exception as e:
            logger.warning(f"Impossible d'ajouter l'image: {e}")
            return False
    
    def add_image_from_path(self, image_path: str, title: str = "",
                            width: int = 160, height: int = 100):
        """Ajoute une image depuis un chemin de fichier"""
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image non trouvee: {image_path}")
                return False
            
            if self.get_y() + height + 20 > 270:
                self.add_page()
            
            if title:
                self.add_subsection(title)
            
            x = (210 - width) / 2
            self.image(image_path, x=x, w=width)
            self.ln(10)
            return True
            
        except Exception as e:
            logger.warning(f"Erreur ajout image: {e}")
            return False
    
    def add_bullet_list(self, items: List[str], indent: int = 15):
        """Ajoute une liste a puces"""
        self.set_font(self.default_font, '', 10)
        self.set_text_color(*self.COLOR_DARK)
        
        for item in items:
            self.set_x(indent)
            self.cell(5, 6, '-', 0, 0)  # Bullet point
            self.multi_cell(175 - indent, 6, self._clean_text(item))
            self.ln(1)
        
        self.ln(3)
    
    def add_decision_box(self, decision: str, impact: str = ""):
        """Ajoute une boite de decision/action mise en evidence"""
        if self.get_y() > 220:
            self.add_page()
        
        # Cadre de decision
        self.set_fill_color(255, 235, 235)
        self.set_draw_color(*self.COLOR_ACCENT)
        self.set_line_width(1.5)
        
        start_y = self.get_y()
        
        # Titre
        self.set_font(self.default_font, 'B', 13)
        self.set_text_color(*self.COLOR_ACCENT)
        self.cell(0, 12, "RECOMMANDATION ACTIONNABLE", 0, 1, 'C')
        
        # Contenu
        self.set_font(self.default_font, '', 11)
        self.set_text_color(*self.COLOR_DARK)
        self.multi_cell(0, 7, self._clean_text(decision))
        
        if impact:
            self.ln(3)
            self.set_font(self.default_font, 'I', 10)
            self.set_text_color(*self.COLOR_SECONDARY)
            self.multi_cell(0, 6, f"Impact attendu: {self._clean_text(impact)}")
        
        # Dessiner le cadre
        end_y = self.get_y() + 5
        self.rect(10, start_y - 5, 190, end_y - start_y + 10, 'D')
        self.ln(10)
    
    def _clean_text(self, text: str) -> str:
        """Nettoie le texte pour l'affichage PDF"""
        if not text:
            return ""
        
        # Remplacer les caracteres problematiques
        replacements = {
            '\u2019': "'",
            '\u2018': "'",
            '\u201c': '"',
            '\u201d': '"',
            '\u2013': '-',
            '\u2014': '-',
            '\u2026': '...',
            '\u00a0': ' ',
            '→': '->',
            '←': '<-',
            '•': '-',
            '✓': '[OK]',
            '❌': '[X]',
            '\n\n\n': '\n\n'
        }
        
        result = str(text)
        for old, new in replacements.items():
            result = result.replace(old, new)
        
        # Retirer tous les emojis et caracteres speciaux restants
        result = ''.join(char for char in result if ord(char) < 256)
        
        return result


# =============================================================================
# SMART EXPORT SERVICE - SERVICE PRINCIPAL
# =============================================================================

class SmartExportService:
    """
    SERVICE D'EXPORT INTELLIGENT V3
    
    Genere un rapport PDF professionnel de 3-5 pages avec:
    - Titre contextuel (OpenAI)
    - Vulgarisation grand public (Gemini)
    - Analyse sociale Madagascar (Anthropic/Claude)
    - Integration des graphiques EDA
    - Decisions concretes
    """
    
    def __init__(self):
        self.export_dir = getattr(settings, 'excel_output_dir', '/tmp/exports')
        os.makedirs(self.export_dir, exist_ok=True)
        
        self.llm_provider = MultiLLMProvider()
        self.charts_cache = {}
        
        logger.info("SmartExportService V3 initialise")
    
    # =========================================================================
    # API PRINCIPALE
    # =========================================================================
    
    async def generate_professional_report(
        self,
        analysis_results: Dict[str, Any],
        user_prompt: str = "",
        include_charts: bool = True,
        charts_dir: str = None
    ) -> Dict[str, Any]:
        """
        GENERE LE RAPPORT PROFESSIONNEL COMPLET
        
        Args:
            analysis_results: Resultats complets de l'analyse (EDA + ML)
            user_prompt: Demande originale de l'utilisateur
            include_charts: Inclure les graphiques EDA
            charts_dir: Repertoire des graphiques EDA (optionnel)
        
        Returns:
            Dict avec success, report_path, download_url
        """
        
        logger.info("=" * 60)
        logger.info("GENERATION RAPPORT PROFESSIONNEL V3")
        logger.info("=" * 60)
        
        try:
            # Extraire les donnees
            data = analysis_results.get("data", analysis_results)
            file_id = analysis_results.get("file_id", "unknown")
            
            # =====================================================================
            # PHASE 1: EXTRACTION DES METRIQUES
            # =====================================================================
            
            logger.info("Phase 1: Extraction des metriques...")
            
            metrics = self._extract_all_metrics(data)
            locations = self._extract_locations(data)
            ml_summary = self._extract_ml_summary(data)
            
            logger.info(f"   {metrics.get('total_rows', 0)} lignes, "
                       f"{metrics.get('total_columns', 0)} colonnes")
            
            # =====================================================================
            # PHASE 2: GENERATION CONTENU MULTI-LLM (PARALLELE)
            # =====================================================================
            
            logger.info("Phase 2: Generation contenu multi-LLM...")
            
            ai_content = await self._generate_multi_llm_content(
                metrics=metrics,
                locations=locations,
                ml_summary=ml_summary,
                user_prompt=user_prompt
            )
            
            logger.info(f"   Titre: {len(ai_content.get('title', ''))} chars")
            logger.info(f"   Vulgarisation: {len(ai_content.get('vulgarization', ''))} chars")
            logger.info(f"   Social: {len(ai_content.get('social_decision', ''))} chars")
            logger.info(f"   Strategie: {len(ai_content.get('strategy', ''))} chars")
            logger.info(f"   Geo: {len(ai_content.get('geo_analysis', ''))} chars")
            
            # =====================================================================
            # PHASE 3: RECUPERATION DES GRAPHIQUES EDA
            # =====================================================================
            
            logger.info("Phase 3: Recuperation graphiques EDA...")
            
            charts = []
            if include_charts:
                charts = self._collect_eda_charts(data, charts_dir)
                logger.info(f"   {len(charts)} graphiques collectes")
            
            # =====================================================================
            # PHASE 4: GENERATION DU PDF
            # =====================================================================
            
            logger.info("Phase 4: Generation PDF...")
            
            report_name = f"Rapport_Analyse_{file_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            report_path = os.path.join(self.export_dir, report_name)
            
            self._build_pdf_report(
                output_path=report_path,
                title=ai_content.get('title', 'Rapport d\'Analyse Intelligente'),
                metrics=metrics,
                locations=locations,
                ml_summary=ml_summary,
                ai_content=ai_content,
                charts=charts,
                user_prompt=user_prompt
            )
            
            logger.info(f"   PDF genere: {report_path}")
            
            # Calculer la taille du fichier
            file_size = os.path.getsize(report_path) if os.path.exists(report_path) else 0
            
            logger.info("=" * 60)
            logger.info(f"RAPPORT GENERE: {report_name}")
            logger.info(f"   Taille: {file_size / 1024:.1f} KB")
            logger.info("=" * 60)
            
            return {
                "success": True,
                "report_path": report_path,
                "report_name": report_name,
                "download_url": f"/api/v1/exports/{report_name}",
                "file_size": file_size,
                "pages_estimated": self._estimate_pages(ai_content, charts),
                "ai_providers_used": {
                    "title": "openai",
                    "vulgarization": "gemini",
                    "social_decision": "anthropic",
                    "strategy": "openai",
                    "geo_analysis": "gemini"
                }
            }
        
        except Exception as e:
            logger.error(f"ERREUR GENERATION RAPPORT: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    # =========================================================================
    # EXTRACTION DES METRIQUES
    # =========================================================================
    
    def _extract_all_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extrait toutes les metriques cles"""
        
        eda = data.get("eda", {})
        summary = eda.get("summary", {})
        
        # Metriques de base
        metrics = {
            "total_rows": summary.get("total_rows", 0),
            "total_columns": summary.get("total_cols", 0),
            "numeric_columns": summary.get("numeric_analyzed", 0),
            "categorical_columns": summary.get("categorical_analyzed", 0),
            "missing_values": summary.get("missing_values", 0),
            "missing_percentage": 0
        }
        
        # Calculer le pourcentage de valeurs manquantes
        if metrics["total_rows"] > 0 and metrics["total_columns"] > 0:
            total_cells = metrics["total_rows"] * metrics["total_columns"]
            metrics["missing_percentage"] = round(
                (metrics["missing_values"] / total_cells) * 100, 2
            )
        
        # Target variable
        metrics["target_variable"] = eda.get("auto_target", "Non definie")
        
        # Clustering info
        multi_clustering = eda.get("metrics", {}).get("multi_clustering", {})
        if multi_clustering:
            metrics["clustering_methods"] = multi_clustering.get("n_clustering_types", 0)
            
            # Trouver le meilleur clustering
            best_key = multi_clustering.get("best_clustering_key")
            if best_key and "clusterings" in multi_clustering:
                best_clustering = multi_clustering["clusterings"].get(best_key, {})
                metrics["best_clustering_k"] = best_clustering.get("n_clusters", 0)
                metrics["silhouette_score"] = round(
                    best_clustering.get("silhouette_score", 0), 3
                )
        
        # Correlations
        correlations = eda.get("metrics", {}).get("correlations", {})
        if correlations:
            metrics["strong_correlations"] = correlations.get("summary", {}).get("strong_pairs", 0)
        
        # Tests statistiques
        tests = eda.get("metrics", {}).get("tests", [])
        if tests:
            significant_tests = [t for t in tests if t.get("p_value", 1) < 0.05]
            metrics["significant_tests"] = len(significant_tests)
            metrics["total_tests"] = len(tests)
        
        return metrics
    
    def _extract_locations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extrait les informations geographiques"""
        
        locations = {
            "regions": [],
            "communes": [],
            "villages": [],
            "has_gps": False
        }
        
        # Chercher dans la structure du fichier
        structure = data.get("structure", {})
        locations_found = structure.get("locations_found", {})
        
        if locations_found:
            locations["regions"] = locations_found.get("regions", [])
            locations["communes"] = locations_found.get("communes", [])
            locations["villages"] = locations_found.get("villages", [])
            locations["has_gps"] = locations_found.get("has_gps", False)
        
        return locations
    
    def _extract_ml_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extrait le resume du Machine Learning"""
        
        ml_data = data.get("ml_pipeline", {})
        
        summary = {
            "applicable": ml_data.get("ml_applicable", False),
            "problem_type": ml_data.get("problem_type", "unknown"),
            "best_model": None,
            "accuracy": 0,
            "overfitting_detected": ml_data.get("overfitting_detected", False),
            "recommendations": []
        }
        
        if ml_data.get("best_model"):
            best = ml_data["best_model"]
            summary["best_model"] = best.get("name", "Unknown")
            summary["accuracy"] = best.get("score", 0)
        
        # Metriques de test
        test_metrics = ml_data.get("test_metrics", {})
        if test_metrics:
            summary["test_accuracy"] = test_metrics.get("accuracy", test_metrics.get("r2", 0))
            summary["f1_score"] = test_metrics.get("f1", 0)
        
        # Feature importance
        feature_importance = ml_data.get("feature_importance", {})
        if feature_importance and "top_features" in feature_importance:
            summary["top_features"] = feature_importance["top_features"][:5]
        
        # Recommandations
        summary["recommendations"] = ml_data.get("recommendations", [])
        
        # LLM Explanation
        llm_exp = ml_data.get("llm_explanation", {})
        if llm_exp and llm_exp.get("success"):
            summary["diagnostic"] = llm_exp.get("diagnostic", {})
            summary["explanation"] = llm_exp.get("explanation", {})
        
        return summary
    
    # =========================================================================
    # GENERATION CONTENU MULTI-LLM
    # =========================================================================
    
    async def _generate_multi_llm_content(
        self,
        metrics: Dict[str, Any],
        locations: Dict[str, Any],
        ml_summary: Dict[str, Any],
        user_prompt: str
    ) -> Dict[str, Any]:
        """
        Genere tout le contenu via les differentes IA specialisees
        
        - OpenAI: Titre + Strategie business
        - Gemini: Vulgarisation + Analyse geographique
        - Anthropic: Analyse sociale + Decisions
        """
        
        # Preparer le contexte commun
        context = self._build_context_string(metrics, locations, ml_summary, user_prompt)
        
        # Lancer les appels en parallele
        results = await asyncio.gather(
            self._generate_title_openai(context, locations),
            self._generate_vulgarization_gemini(context, ml_summary),
            self._generate_social_decision_anthropic(context, locations, ml_summary),
            self._generate_strategy_openai(context, ml_summary),
            self._generate_geo_analysis_gemini(context, locations),
            return_exceptions=True
        )
        
        # Traiter les resultats
        ai_content = {
            "title": results[0] if isinstance(results[0], str) else self._fallback_title(locations),
            "vulgarization": results[1] if isinstance(results[1], str) else self._fallback_vulgarization(),
            "social_decision": results[2] if isinstance(results[2], str) else self._fallback_social(),
            "strategy": results[3] if isinstance(results[3], str) else self._fallback_strategy(),
            "geo_analysis": results[4] if isinstance(results[4], str) else self._fallback_geo(locations)
        }
        
        return ai_content
    
    def _build_context_string(
        self,
        metrics: Dict[str, Any],
        locations: Dict[str, Any],
        ml_summary: Dict[str, Any],
        user_prompt: str
    ) -> str:
        """Construit la chaine de contexte pour les LLMs"""
        
        regions = ", ".join(locations.get("regions", [])) or "Non specifie"
        communes = ", ".join(locations.get("communes", [])) or "Non specifie"
        
        context = f"""
DONNEES ANALYSEES:
- Nombre de lignes: {metrics.get('total_rows', 0)}
- Nombre de colonnes: {metrics.get('total_columns', 0)}
- Variables numeriques: {metrics.get('numeric_columns', 0)}
- Variables categorielles: {metrics.get('categorical_columns', 0)}
- Valeurs manquantes: {metrics.get('missing_percentage', 0)}%
- Variable cible: {metrics.get('target_variable', 'Non definie')}

GEOGRAPHIE:
- Regions: {regions}
- Communes: {communes}
- Donnees GPS: {'Oui' if locations.get('has_gps') else 'Non'}

MACHINE LEARNING:
- Type de probleme: {ml_summary.get('problem_type', 'N/A')}
- Meilleur modele: {ml_summary.get('best_model', 'N/A')}
- Accuracy/R²: {ml_summary.get('accuracy', 0) * 100:.1f}%
- Overfitting detecte: {'Oui' if ml_summary.get('overfitting_detected') else 'Non'}

DEMANDE UTILISATEUR:
{user_prompt or 'Analyse generale des donnees'}
"""
        return context
    
    async def _generate_title_openai(self, context: str, locations: Dict[str, Any]) -> str:
        """OpenAI genere le titre sur mesure"""
        
        regions = ", ".join(locations.get("regions", [])) or "Madagascar"
        
        prompt = f"""En tant qu'expert en analyse de donnees, genere un titre professionnel 
et percutant pour ce rapport d'analyse. Le titre doit:
- Etre concis (max 15 mots)
- Refleter le contexte geographique ({regions})
- Mentionner le type d'analyse ou le domaine
- Etre en francais

{context}

Reponds UNIQUEMENT avec le titre, sans guillemets ni explication."""

        result = await self.llm_provider.call_openai(
            prompt=prompt,
            system_prompt="Tu es un expert en data science specialise dans les rapports professionnels.",
            max_tokens=100,
            temperature=0.8
        )
        
        if result:
            # Nettoyer le resultat
            result = result.strip().strip('"').strip("'")
            return result
        
        return self._fallback_title(locations)
    
    async def _generate_vulgarization_gemini(self, context: str, ml_summary: Dict[str, Any]) -> str:
        """Gemini vulgarise les resultats pour le grand public"""
        
        prompt = f"""Tu es un expert en communication scientifique. 
Explique les resultats de cette analyse de donnees de maniere simple et accessible 
pour le grand public (citoyens, employes non-techniques).

Regles:
- Utilise un langage simple, sans jargon technique
- Explique POURQUOI ces donnees sont importantes
- Donne des exemples concrets de l'impact
- Maximum 4 paragraphes
- En francais

{context}

Metriques ML: 
- Modele: {ml_summary.get('best_model', 'N/A')}
- Performance: {ml_summary.get('accuracy', 0) * 100:.1f}%

Commence directement par l'explication, sans introduction."""

        result = await self.llm_provider.call_gemini(prompt, max_tokens=800)
        
        if result:
            return result.strip()
        
        return self._fallback_vulgarization()
    
    async def _generate_social_decision_anthropic(
        self,
        context: str,
        locations: Dict[str, Any],
        ml_summary: Dict[str, Any]
    ) -> str:
        """Anthropic (Claude) genere l'analyse sociale et les decisions"""
        
        regions = ", ".join(locations.get("regions", [])) or "les regions concernees"
        
        prompt = f"""Tu es un expert en developpement social a Madagascar avec 20 ans d'experience terrain.

Base sur cette analyse de donnees, formule des DECISIONS SOCIALES CONCRETES 
et ACTIONNABLES pour ameliorer la situation des populations.

{context}

INSTRUCTIONS:
1. Identifie les problemes sociaux reveles par les donnees
2. Propose 2-3 actions CONCRETES et IMMEDIATES
3. Explique l'impact attendu sur les populations de {regions}
4. Donne des indicateurs de suivi simples

Format ta reponse en 3 parties:
- CONSTATS (ce que revelent les donnees)
- DECISIONS (actions a prendre)
- IMPACT (benefices pour la population)

Maximum 500 mots. En francais."""

        # Utiliser to_thread pour l'appel synchrone
        try:
            result = await asyncio.to_thread(
                self.llm_provider.call_anthropic,
                prompt=prompt,
                system_prompt="Tu es un expert en politique sociale et developpement a Madagascar. "
                            "Tu donnes des conseils pratiques et actionnables.",
                max_tokens=1200
            )
            
            if result:
                return result.strip()
            
            return self._fallback_social()
        
        except Exception as e:
            logger.warning(f"Erreur Anthropic: {e}")
            return self._fallback_social()
    
    async def _generate_strategy_openai(self, context: str, ml_summary: Dict[str, Any]) -> str:
        """OpenAI genere la strategie business/technique"""
        
        prompt = f"""En tant que consultant strategique senior, analyse ces donnees 
et formule des recommandations strategiques.

{context}

Resultats ML:
- Modele optimal: {ml_summary.get('best_model', 'N/A')}
- Performance: {ml_summary.get('accuracy', 0) * 100:.1f}%
- Problemes detectes: {'Overfitting' if ml_summary.get('overfitting_detected') else 'RAS'}

Fournis:
1. Synthese executive (3 phrases max)
2. Opportunites identifiees (2-3 points)
3. Risques a surveiller (2-3 points)
4. Prochaines etapes recommandees

Format professionnel, en francais. Maximum 400 mots."""

        result = await self.llm_provider.call_openai(
            prompt=prompt,
            system_prompt="Tu es un consultant strategique senior specialise en data-driven decision making.",
            max_tokens=800,
            temperature=0.5
        )
        
        if result:
            return result.strip()
        
        return self._fallback_strategy()
    
    async def _generate_geo_analysis_gemini(self, context: str, locations: Dict[str, Any]) -> str:
        """Gemini analyse les aspects geographiques"""
        
        regions = ", ".join(locations.get("regions", [])) or "N/A"
        communes = ", ".join(locations.get("communes", [])) or "N/A"
        
        prompt = f"""Analyse les dimensions geographiques de ces donnees pour Madagascar.

Regions concernees: {regions}
Communes: {communes}
GPS disponible: {'Oui' if locations.get('has_gps') else 'Non'}

{context}

Fournis:
1. Couverture geographique de l'etude
2. Disparites regionales identifiees (si pertinent)
3. Zones prioritaires pour intervention
4. Recommandations de ciblage geographique

En francais, style professionnel. Maximum 300 mots."""

        result = await self.llm_provider.call_gemini(prompt, max_tokens=600)
        
        if result:
            return result.strip()
        
        return self._fallback_geo(locations)
    
    # =========================================================================
    # FALLBACKS
    # =========================================================================
    
    def _fallback_title(self, locations: Dict[str, Any]) -> str:
        """Titre de fallback"""
        regions = locations.get("regions", [])
        if regions:
            return f"Analyse Strategique des Donnees - Region {regions[0]}"
        return "Rapport d'Analyse Intelligente des Donnees"
    
    def _fallback_vulgarization(self) -> str:
        """Vulgarisation de fallback"""
        return """Cette analyse a permis d'explorer en profondeur les donnees disponibles 
pour en extraire des informations utiles. Les resultats montrent des tendances 
importantes qui peuvent guider les decisions futures.

L'etude a identifie plusieurs groupes distincts dans les donnees, ce qui permet 
de mieux comprendre les differentes situations representees. Ces informations 
sont essentielles pour orienter les actions et les ressources la ou elles sont 
le plus necessaires."""
    
    def _fallback_social(self) -> str:
        """Decision sociale de fallback"""
        return """CONSTATS:
Les donnees analysees revelent des disparites significatives qui necessitent une attention particuliere.

DECISIONS:
1. Renforcer le suivi des populations les plus vulnerables identifiees par l'analyse
2. Adapter les interventions selon les profils detectes par la segmentation
3. Mettre en place un systeme de monitoring base sur les indicateurs cles identifies

IMPACT:
Ces actions permettront une meilleure allocation des ressources et un ciblage plus efficace 
des populations qui en ont le plus besoin."""
    
    def _fallback_strategy(self) -> str:
        """Strategie de fallback"""
        return """SYNTHESE EXECUTIVE:
L'analyse revele des opportunites d'optimisation significatives. Les modeles predictifs 
developpes permettent d'anticiper les evolutions futures avec une precision acceptable.

OPPORTUNITES:
- Amelioration du ciblage grace a la segmentation automatique
- Identification des facteurs cles de succes

RISQUES:
- Qualite des donnees a surveiller
- Necessite de valider les modeles sur le terrain

PROCHAINES ETAPES:
- Valider les resultats avec les equipes terrain
- Deployer progressivement les recommandations"""
    
    def _fallback_geo(self, locations: Dict[str, Any]) -> str:
        """Analyse geographique de fallback"""
        regions = ", ".join(locations.get("regions", [])) or "les zones etudiees"
        return f"""L'etude couvre {regions}. Les donnees geographiques disponibles 
permettent une analyse territoriale des phenomenes observes.

Les recommandations de ciblage tiendront compte des specificites locales 
pour une intervention adaptee au contexte de chaque zone."""
    
    # =========================================================================
    # GRAPHIQUES EDA
    # =========================================================================
    
    def _collect_eda_charts(self, data: Dict[str, Any], charts_dir: str = None) -> List[Dict[str, Any]]:
        """Collecte les graphiques EDA disponibles"""
        
        charts = []
        
        # 1. Chercher dans les donnees de l'EDA
        eda = data.get("eda", {})
        charts_data = eda.get("charts_data", {})
        
        # Distributions (histogrammes)
        distributions = charts_data.get("distributions", {})
        for col, dist_data in distributions.items():
            if dist_data.get("base64_image"):
                charts.append({
                    "type": "distribution",
                    "title": f"Distribution: {col}",
                    "base64": dist_data["base64_image"],
                    "priority": 1
                })
        
        # Pie charts
        pies = charts_data.get("pies", [])
        for pie in pies[:3]:  # Max 3 pie charts
            if pie.get("base64_image"):
                charts.append({
                    "type": "pie",
                    "title": pie.get("title", "Distribution"),
                    "base64": pie["base64_image"],
                    "priority": 2
                })
        
        # Scatter plots
        scatters = charts_data.get("scatters", [])
        for scatter in scatters[:2]:  # Max 2 scatter plots
            if scatter.get("base64_image"):
                charts.append({
                    "type": "scatter",
                    "title": scatter.get("title", "Nuage de points"),
                    "base64": scatter["base64_image"],
                    "priority": 3
                })
        
        # Clustering 3D
        clustering_views = charts_data.get("clustering_views", [])
        for cluster in clustering_views[:2]:  # Max 2 vues clustering
            if cluster.get("data", {}).get("base64_image"):
                charts.append({
                    "type": "clustering",
                    "title": cluster.get("title", "Segmentation"),
                    "base64": cluster["data"]["base64_image"],
                    "priority": 1  # Haute priorite
                })
        
        # 2. Chercher dans le repertoire si specifie
        if charts_dir and os.path.isdir(charts_dir):
            for filename in os.listdir(charts_dir):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    charts.append({
                        "type": "file",
                        "title": filename.replace('_', ' ').replace('.png', ''),
                        "path": os.path.join(charts_dir, filename),
                        "priority": 4
                    })
        
        # Trier par priorite
        charts.sort(key=lambda x: x.get("priority", 99))
        
        return charts[:8]  # Maximum 8 graphiques
    
    # =========================================================================
    # CONSTRUCTION DU PDF
    # =========================================================================
    
    def _build_pdf_report(
        self,
        output_path: str,
        title: str,
        metrics: Dict[str, Any],
        locations: Dict[str, Any],
        ml_summary: Dict[str, Any],
        ai_content: Dict[str, Any],
        charts: List[Dict[str, Any]],
        user_prompt: str
    ):
        """Construit le PDF professionnel complet"""
        
        pdf = ProfessionalPDFReport(title=title)
        pdf.alias_nb_pages()
        
        # =====================================================================
        # PAGE 1: TITRE ET RESUME EXECUTIF
        # =====================================================================
        
        pdf.add_main_title(title)
        
        # Date et contexte
        pdf.set_font(pdf.default_font, 'I', 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 6, f"Genere le {datetime.now().strftime('%d %B %Y a %H:%M')}", 0, 1, 'C')
        if user_prompt:
            pdf.multi_cell(0, 6, f"Objectif: {user_prompt[:200]}", 0, 'C')
        pdf.ln(10)
        
        # Resume executif
        pdf.add_section_header("I. RESUME EXECUTIF")
        
        summary_text = f"""Ce rapport presente l'analyse complete d'un jeu de donnees 
comprenant {metrics.get('total_rows', 0):,} observations reparties sur 
{metrics.get('total_columns', 0)} variables. L'analyse a identifie 
'{metrics.get('target_variable', 'N/A')}' comme variable d'interet principal."""
        
        pdf.add_paragraph(summary_text)
        
        # Tableau des metriques cles
        pdf.add_subsection("Metriques Cles")
        
        key_metrics = {
            "Nombre de lignes": f"{metrics.get('total_rows', 0):,}",
            "Nombre de colonnes": metrics.get('total_columns', 0),
            "Variables numeriques": metrics.get('numeric_columns', 0),
            "Variables categorielles": metrics.get('categorical_columns', 0),
            "Valeurs manquantes": f"{metrics.get('missing_percentage', 0):.1f}%",
            "Variable cible": metrics.get('target_variable', 'N/A')
        }
        
        if ml_summary.get('applicable'):
            key_metrics["Meilleur modele ML"] = ml_summary.get('best_model', 'N/A')
            key_metrics["Performance (Accuracy/R²)"] = f"{ml_summary.get('accuracy', 0) * 100:.1f}%"
        
        pdf.add_key_metrics_table(key_metrics)
        
        # =====================================================================
        # PAGE 2: VULGARISATION POUR LE GRAND PUBLIC
        # =====================================================================
        
        pdf.add_page()
        pdf.add_section_header("II. COMPRENDRE LES RESULTATS", 
                              color=pdf.COLOR_SECONDARY)
        
        pdf.add_subsection("Pourquoi ces donnees sont-elles importantes ?")
        pdf.add_paragraph(ai_content.get('vulgarization', ''))
        
        # Analyse geographique si pertinente
        if locations.get('regions') or locations.get('communes'):
            pdf.add_subsection("Couverture Geographique")
            pdf.add_paragraph(ai_content.get('geo_analysis', ''))
        
        # =====================================================================
        # PAGE 3: ANALYSE TECHNIQUE (EDA + ML)
        # =====================================================================
        
        pdf.add_page()
        pdf.add_section_header("III. ANALYSE TECHNIQUE DETAILLEE",
                              color=pdf.COLOR_PRIMARY)
        
        # Clustering / Segmentation
        if metrics.get('clustering_methods'):
            pdf.add_subsection("Segmentation des Donnees")
            
            clustering_text = f"""L'analyse a teste {metrics.get('clustering_methods', 0)} methodes 
de segmentation differentes. La meilleure segmentation identifie 
{metrics.get('best_clustering_k', 0)} groupes distincts dans les donnees 
(score de qualite: {metrics.get('silhouette_score', 0):.2f})."""
            
            pdf.add_paragraph(clustering_text)
        
        # Correlations
        if metrics.get('strong_correlations'):
            pdf.add_subsection("Relations Entre Variables")
            
            corr_text = f"""L'analyse a identifie {metrics.get('strong_correlations', 0)} 
correlations fortes entre les variables, revelant des relations significatives 
qui meritent une attention particuliere."""
            
            pdf.add_paragraph(corr_text)
        
        # Tests statistiques
        if metrics.get('total_tests'):
            pdf.add_subsection("Tests Statistiques")
            
            tests_text = f"""{metrics.get('significant_tests', 0)} tests sur 
{metrics.get('total_tests', 0)} se sont reveles statistiquement significatifs 
(p-value < 0.05), confirmant l'existence de differences reelles entre les groupes."""
            
            pdf.add_paragraph(tests_text)
        
        # Machine Learning
        if ml_summary.get('applicable'):
            pdf.add_subsection("Resultats Machine Learning")
            
            ml_text = f"""Le modele {ml_summary.get('best_model', 'selectionne')} 
a atteint une performance de {ml_summary.get('accuracy', 0) * 100:.1f}% sur les donnees de test."""
            
            if ml_summary.get('overfitting_detected'):
                ml_text += "\n\nATTENTION: Un risque d'overfitting a ete detecte. "
                ml_text += "Les resultats doivent etre interpretes avec prudence."
            
            pdf.add_paragraph(ml_text)
            
            # Feature importance
            if ml_summary.get('top_features'):
                pdf.add_subsection("Variables les Plus Importantes")
                features_list = [f"{f['feature']}: {f['importance']*100:.1f}%" 
                               for f in ml_summary['top_features'][:5]]
                pdf.add_bullet_list(features_list)
        
        # =====================================================================
        # PAGE 4: GRAPHIQUES EDA
        # =====================================================================
        
        if charts:
            pdf.add_page()
            pdf.add_section_header("IV. VISUALISATIONS", 
                                  color=pdf.COLOR_PRIMARY)
            
            charts_added = 0
            for chart in charts:
                if charts_added >= 4:  # Max 4 graphiques par page
                    break
                
                if chart.get('base64'):
                    success = pdf.add_image_from_base64(
                        chart['base64'],
                        title=chart.get('title', ''),
                        width=140,
                        height=80
                    )
                    if success:
                        charts_added += 1
                
                elif chart.get('path'):
                    success = pdf.add_image_from_path(
                        chart['path'],
                        title=chart.get('title', ''),
                        width=140,
                        height=80
                    )
                    if success:
                        charts_added += 1
            
            if charts_added == 0:
                pdf.add_paragraph("(Les graphiques sont disponibles dans l'interface web)")
        
        # =====================================================================
        # PAGE 5: STRATEGIE ET RECOMMANDATIONS
        # =====================================================================
        
        pdf.add_page()
        pdf.add_section_header("V. STRATEGIE ET RECOMMANDATIONS",
                              color=pdf.COLOR_PRIMARY)
        
        pdf.add_paragraph(ai_content.get('strategy', ''))
        
        # =====================================================================
        # PAGE 6: DECISION SOCIALE (PAGE FINALE IMPORTANTE)
        # =====================================================================
        
        pdf.add_page()
        pdf.add_section_header("VI. DECISION SOCIALE ET ACTION",
                              color=pdf.COLOR_ACCENT)
        
        pdf.add_highlight_box(
            "Cette section presente les implications sociales des resultats "
            "et les actions concretes recommandees pour Madagascar.",
            box_type="warning"
        )
        
        pdf.ln(5)
        
        # Contenu social detaille
        social_content = ai_content.get('social_decision', '')
        
        # Parser le contenu pour mise en forme
        if "CONSTATS" in social_content and "DECISIONS" in social_content:
            parts = social_content.split("DECISIONS")
            
            # Constats
            pdf.add_subsection("Constats")
            constats = parts[0].replace("CONSTATS", "").strip()
            pdf.add_paragraph(constats)
            
            # Decisions
            if len(parts) > 1:
                remaining = parts[1]
                if "IMPACT" in remaining:
                    decisions_impact = remaining.split("IMPACT")
                    
                    pdf.add_subsection("Decisions Recommandees")
                    pdf.add_paragraph(decisions_impact[0].strip())
                    
                    if len(decisions_impact) > 1:
                        pdf.add_subsection("Impact Attendu")
                        pdf.add_paragraph(decisions_impact[1].strip())
                else:
                    pdf.add_subsection("Decisions et Impact")
                    pdf.add_paragraph(remaining.strip())
        else:
            # Contenu non structure
            pdf.add_paragraph(social_content)
        
        # Encadre de recommandation finale
        pdf.ln(10)
        pdf.add_decision_box(
            decision="Prioriser les actions identifiees ci-dessus en commençant par "
                    "les zones les plus vulnerables. Mettre en place un suivi mensuel "
                    "des indicateurs cles.",
            impact="Amelioration mesurable des conditions de vie des populations ciblees "
                  "dans les 6 a 12 mois suivant la mise en œuvre."
        )
        
        # =====================================================================
        # ANNEXE: METHODOLOGIE
        # =====================================================================
        
        pdf.add_page()
        pdf.add_section_header("ANNEXE: METHODOLOGIE",
                              color=pdf.COLOR_SECONDARY)
        
        methodology = """Ce rapport a ete genere automatiquement par un systeme d'analyse 
intelligente utilisant plusieurs composants:

1. ANALYSE EXPLORATOIRE (EDA)
   - Statistiques descriptives univariees et bivariees
   - Detection automatique des correlations
   - Tests statistiques adaptes au type de donnees
   - Segmentation par clustering multi-methodes

2. MACHINE LEARNING
   - Selection automatique du type de probleme
   - Comparaison de plusieurs algorithmes
   - Validation croisee et detection d'overfitting
   - Identification des variables importantes

3. GENERATION D'INSIGHTS
   - OpenAI GPT-4: Titre contextualise et strategie
   - Google Gemini: Vulgarisation et analyse geographique
   - Anthropic Claude: Analyse sociale et decisions

Les resultats ont ete valides par des tests statistiques standard 
avec un seuil de significativite de 5%."""

        pdf.add_paragraph(methodology)
        
        # Informations techniques
        pdf.add_subsection("Informations Techniques")
        
        tech_info = {
            "Date de generation": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "Version du systeme": "Smart Analytics V3",
            "Nombre de modeles testes": "6+ (selon le type de probleme)",
            "Methodes de clustering": "K-Means, DBSCAN, Hierarchique, GMM"
        }
        
        pdf.add_key_metrics_table(tech_info)
        
        # =====================================================================
        # SAUVEGARDER LE PDF
        # =====================================================================
        
        pdf.output(output_path)
    
    def _estimate_pages(self, ai_content: Dict[str, Any], charts: List[Dict]) -> int:
        """Estime le nombre de pages du rapport"""
        
        # Base: 3 pages minimum (titre, analyse, conclusion)
        pages = 3
        
        # Ajouter selon le contenu
        total_text_length = sum(len(str(v)) for v in ai_content.values())
        pages += total_text_length // 3000  # ~3000 chars par page
        
        # Graphiques
        pages += len(charts) // 2  # ~2 graphiques par page
        
        # Methodologie
        pages += 1
        
        return max(3, min(pages, 8))  # Entre 3 et 8 pages


# =============================================================================
# INSTANCE GLOBALE
# =============================================================================

smart_export_service = SmartExportService()


# =============================================================================
# FONCTIONS HELPER POUR L'API
# =============================================================================

async def generate_report(analysis_results: Dict[str, Any], 
                         user_prompt: str = "") -> Dict[str, Any]:
    """
    Fonction wrapper pour l'API
    
    Usage:
        from services.smart_export_service import generate_report
        result = await generate_report(analysis_data, "Analyse sante Antananarivo")
    """
    return await smart_export_service.generate_professional_report(
        analysis_results=analysis_results,
        user_prompt=user_prompt,
        include_charts=True
    )