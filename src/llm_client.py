import os
import time
import re
import logging
from typing import Optional
from dotenv import load_dotenv

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from huggingface_hub import InferenceClient

# Načtení environment proměnných
load_dotenv()

logger = logging.getLogger(__name__)

class LLMClassifier:
    """
    Wrapper pro volání LLM (Gemini, HuggingFace) pro binární klasifikaci.
    """
    
    def __init__(self, provider: str = "gemini", model_name: str = "gemini-1.5-flash"):
        self.provider = provider
        self.model_name = model_name
        
        # Setup Google
        if provider == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("⚠️ Chybí GOOGLE_API_KEY v .env souboru!")
            genai.configure(api_key=api_key)
            
        # Setup Hugging Face
        elif provider == "huggingface":
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                logger.warning("⚠️ Chybí HF_TOKEN v .env! Některé modely nemusí fungovat.")
            self.hf_client = InferenceClient(token=hf_token)

    def _clean_response(self, text: str) -> Optional[int]:
        """
        Vyparsuje 0 nebo 1 z odpovědi LLM.
        Zvládá: "0", "1", "**0**", "Answer: 1", "[0]" atd.
        """
        if not text:
            return None
        
        clean_text = text.strip()
        
        # 1. Přímá shoda
        if clean_text in ["0", "1"]:
            return int(clean_text)
            
        # 2. Hledání čísla pomocí Regexu (hledá samostatné 0 nebo 1)
        # Hledá 0/1 obklopené mezerami, uvozovkami, závorkami nebo hvězdičkami
        match = re.search(r'(?:^|[\s\(\[\"\*])([01])(?:$|[\s\)\]\"\*])', clean_text)
        if match:
            return int(match.group(1))
            
        return None

    def predict(self, text: str, retries: int = 3, sleep_time: int = 2) -> Optional[int]:
        """
        Pošle prompt do modelu a vrátí 0 nebo 1.
        """
        prompt = (
            "You are a linguistic expert detecting bias in news headlines. "
            "Analyze the following sentence and decide if it contains subjective bias (opinion, emotion) "
            "or if it is neutral.\n\n"
            "Rules:\n"
            "- Return ONLY the digit '0' if the sentence is Neutral.\n"
            "- Return ONLY the digit '1' if the sentence is Biased.\n"
            "- Do not explain. Do not write 'Answer:'. Just the digit.\n\n"
            f"Sentence: \"{text}\"\n\n"
            "Label:"
        )

        for attempt in range(retries):
            try:
                raw_response = ""
                
                # --- GOOGLE GEMINI ---
                if self.provider == "gemini":
                    model = genai.GenerativeModel(self.model_name)
                    
                    # Konfigurace pro deterministický výstup (teplota 0)
                    generation_config = genai.types.GenerationConfig(
                        temperature=0.0
                    )
                    
                    # Vypnutí Safety filtrů (pro výzkum biasu nutné)
                    safety_settings = {
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    }
                    
                    response = model.generate_content(
                        prompt,
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )
                    raw_response = response.text

                # --- HUGGING FACE ---
                elif self.provider == "huggingface":
                    # HF Inference API
                    raw_response = self.hf_client.text_generation(
                        prompt, 
                        model=self.model_name, 
                        max_new_tokens=5, 
                        temperature=0.1, # Nízká teplota
                        return_full_text=False
                    )

                # --- ZPRACOVÁNÍ ---
                result = self._clean_response(raw_response)
                
                if result is not None:
                    return result
                
                # Pokud vrátil něco divného, zkusíme to ještě jednou (možná halucinoval)
                logger.warning(f"⚠️ Unparseable response ({self.model_name}): '{raw_response}'")

            except Exception as e:
                # Ošetření přetížení (Rate Limit 429)
                error_msg = str(e)
                if "429" in error_msg or "Model is overloaded" in error_msg:
                    wait = sleep_time * (2 ** attempt) # Exponenciální čekání
                    print(f"⏳ Rate limit ({self.model_name}). Sleeping {wait}s...")
                    time.sleep(wait)
                    continue
                else:
                    logger.error(f"❌ Error ({self.provider}): {e}")
                    return None # Jiná chyba (např. auth), nemá smysl opakovat
        
        return None