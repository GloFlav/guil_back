# backend/services/file_structure_analysis_service.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
import re
from anthropic import Anthropic

logger = logging.getLogger(__name__)

class FileStructureAnalysisService:
    
    def _clean_markdown_for_tts(self, text: str) -> str:
        """Nettoie le markdown pour la lecture vocale"""
        text = re.sub(r'#{1,6}\s+', '', text)
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        text = re.sub(r'`(.+?)`', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _extract_locations_from_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extrait les localisations RÉELLES des données"""
        locations = {
            "regions": set(),
            "communes": set(),
            "villages": set(),
            "has_gps": False
        }
        
        col_lower = [col.lower() for col in df.columns]
        
        # Chercher colonnes géographiques
        region_cols = [df.columns[i] for i, col in enumerate(col_lower) 
                      if any(x in col for x in ['région', 'region', 'district'])]
        commune_cols = [df.columns[i] for i, col in enumerate(col_lower) 
                       if any(x in col for x in ['commune', 'localité', 'locality'])]
        village_cols = [df.columns[i] for i, col in enumerate(col_lower) 
                       if any(x in col for x in ['village', 'fokontany', 'fokont'])]
        gps_cols = [df.columns[i] for i, col in enumerate(col_lower) 
                   if any(x in col for x in ['latitude', 'longitude', 'gps', 'coord'])]
        
        # Extraire valeurs uniques (limité pour éviter la surcharge)
        for col in region_cols:
            regions = df[col].dropna().unique()
            locations["regions"].update([str(r).strip() for r in regions if r])
        
        for col in commune_cols:
            communes = df[col].dropna().unique()
            locations["communes"].update([str(c).strip() for c in communes if c])
        
        for col in village_cols:
            villages = df[col].dropna().unique()
            locations["villages"].update([str(v).strip() for v in villages if v])
        
        if gps_cols:
            locations["has_gps"] = True
        
        return {
            "regions": sorted(list(locations["regions"]))[:3],
            "communes": sorted(list(locations["communes"]))[:5],
            "villages": sorted(list(locations["villages"]))[:5],
            "has_gps": locations["has_gps"]
        }
    
    def _format_locations_natural(self, locations_data: Dict[str, Any]) -> str:
        """Formate les localisations pour le TTS NATUREL"""
        regions = locations_data.get("regions", [])
        communes = locations_data.get("communes", [])
        villages = locations_data.get("villages", [])
        
        location_text = ""
        
        if regions:
            if len(regions) == 1:
                location_text += f"dans la région de {regions[0]}"
            else:
                region_str = " et ".join([f"{r}" for r in regions])
                location_text += f"dans les régions de {region_str}"
        
        if communes:
            if location_text: location_text += ", "
            if len(communes) == 1:
                location_text += f"notamment la commune de {communes[0]}"
            else:
                commune_str = " et ".join(communes)
                location_text += f"notamment les communes de {commune_str}"
        
        return location_text if location_text else "sur plusieurs localités"
    
    async def analyze_file_structure(self, 
                                     file_path: str, 
                                     df: pd.DataFrame,
                                     file_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        📊 Analyse fichier nettoyé
        """
        try:
            logger.info(f"📊 Analyse fichier final: {len(df)} lignes, {len(df.columns)} colonnes")
            
            # Extraire localisations
            locations_data = self._extract_locations_from_data(df)
            
            structure_data = {
                "final_rows": len(df),
                "final_columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "column_samples": {},
                "locations_found": locations_data
            }
            
            # Samples pour l'IA
            for col in df.columns[:40]: # Limite aux 40 premières colonnes pour le prompt
                samples = df[col].dropna().unique()[:3]
                structure_data["column_samples"][col] = [str(s)[:50] for s in samples]
            
            # Prompt pour Claude
            prompt = self._build_prompt(structure_data, locations_data)
            
            # Appeler Claude
            ai_summary_raw = await self._generate_ai_summary(prompt)
            ai_summary_clean = self._clean_markdown_for_tts(ai_summary_raw)
            
            # Générer TTS (Nouvelle version : focus sur ce qu'on a)
            tts_text = self._build_tts_direct(structure_data, locations_data, ai_summary_clean)
            
            return {
                "success": True,
                "structure": structure_data,
                "ai_summary": ai_summary_clean,
                "tts_text": tts_text
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def _build_prompt(self, structure_data: Dict[str, Any], 
                      locations_data: Dict[str, Any]) -> str:
        """Prompt pour analyser le fichier"""
        
        cols_desc = ""
        for col in list(structure_data['column_samples'].keys()):
            samples = structure_data['column_samples'][col]
            samples_str = ', '.join(samples[:2]) if samples else 'N/A'
            cols_desc += f"- {col}: {samples_str}\n"
        
        return f"""Analyse ce fichier de données.

DONNÉES DISPONIBLES:
- {structure_data['final_rows']} lignes
- {structure_data['final_columns']} colonnes exploitables

LOCALISATIONS:
- Régions: {", ".join(locations_data['regions']) or 'N/A'}
- Communes: {", ".join(locations_data['communes']) or 'N/A'}

COLONNES (Echantillon):
{cols_desc}

INSTRUCTIONS:
Fais une analyse courte et percutante en 3 points (style conversationnel pour lecture vocale):
1. CONTEXTE : De quel type de données s'agit-il ? (Enquête, Recensement, Ventes...)
2. GÉOGRAPHIE : Où cela se passe-t-il ? (Intègre les lieux naturellement dans la phrase)
3. POTENTIEL : Quelle analyse intéressante peut-on faire ?

TON: Professionnel mais fluide. Pas de listes à puces.
"""
    
    async def _generate_ai_summary(self, prompt: str) -> str:
        """Appelle Claude"""
        try:
            from config.settings import settings
            client = Anthropic(api_key=settings.anthropic_api_key_1)
            message = client.messages.create(
                model=settings.anthropic_model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"❌ Erreur IA: {e}")
            return "Analyse du contenu en cours."
    
    def _build_tts_direct(self, structure_data: Dict[str, Any],
                         locations_data: Dict[str, Any],
                         ai_summary: str) -> str:
        """
        🎤 TTS FINAL
        Ne mentionne PLUS les suppressions. Focus sur le contenu disponible.
        """
        rows = structure_data["final_rows"]
        cols_final = structure_data["final_columns"]
        
        # Introduction positive
        intro = f"Analyse terminée. Le fichier contient {cols_final} colonnes et {rows} lignes exploitables."
        
        # Combinaison avec l'analyse IA
        script = f"{intro} {ai_summary} Les données sont prêtes. Quelle analyse souhaitez-vous lancer ?"
        
        return script

file_structure_analysis_service = FileStructureAnalysisService()