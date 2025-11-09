import pandas as pd
import re
from pathlib import Path

INPUT_FILE = Path('concat_file/concatened_data_back_translation.csv')
OUTPUT_FILE = Path('concat_file/dados_limpos_back_translation.csv')

def clean_text_for_depression_detection(text):
    if not isinstance(text, str):
        return ""

    # 1. Converter para minúsculas
    text = text.lower()

    # --- ETAPA 1: REMOÇÃO CIRÚRGICA DE RUÍDO TÉCNICO ---
    # Remove variações de tags de sincronização que apareceram nos seus dados
    text = re.sub(r'\[syncing\]|<sync>|sync\s*[.,]', ' ', text)
    text = re.sub(r'\bscrubbed_entry\b', ' ', text)
    text = re.sub(r'\bxxx\b', ' ', text)

    # --- ETAPA 2: NORMALIZAÇÃO DE REPETIÇÕES ---
    # 2a. Colapso de caracteres repetidos excessivamente (ex: "chffffffff" -> "chff")
    # Mantém no máximo 2 letras iguais seguidas (bom para 'cool', 'bee', mas remove exageros)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # 2b. Colapso de palavras repetidas (ex: "no, no, no, no" -> "no")
    # O '{2,}' significa que se a palavra aparecer 3+ vezes seguidas, vira uma só.
    text = re.sub(r'\b(\w+)(?:[,\s]+\1\b){2,}', r'\1', text)

    # 2c. Remove sequências de símbolos repetidos (ex: # # # #)
    text = re.sub(r'(?:[#\.\-\=]\s*){3,}', ' ', text)

    # --- LIMPEZA FINAL ---
    # Mantém letras, números, pontuação básica E os símbolos < > para não quebrar as tags
    text = re.sub(r"[^a-z0-9\s,.?!'<>]", ' ', text)

    # Remove espaços duplos gerados pela limpeza
    text = re.sub(r'\s+', ' ', text).strip()

    return text

try:
    print("Lendo arquivo original...")
    df = pd.read_csv(INPUT_FILE)

    print("Limpando as respostas...")
    df['participant_responses'] = df['participant_responses'].apply(clean_text_for_depression_detection)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Salvando apenas dados limpos em: {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)
    print("Sucesso! O arquivo final contém apenas as respostas higienizadas.")

except Exception as e:
    print(f"Erro crítico: {e}")