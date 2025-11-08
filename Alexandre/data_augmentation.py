from multiprocessing import freeze_support
from pathlib import Path
import pandas as pd
import nlpaug.augmenter.word as naw
import torch
from tqdm import tqdm

def filter_hallucinations(original_text, augmented_text):
    fillers = ['um', 'uh', 'ah', 'er', 'mm-hm', 'mhm']
    orig_clean = str(original_text).strip().lower()
    aug_clean = str(augmented_text).strip()

    if len(orig_clean) < 3 or orig_clean in fillers:
        return original_text
    if "no, no, no" in aug_clean.lower() and "no, no, no" not in orig_clean:
        return original_text
    return augmented_text

def main():
    cleaned_data = Path('dados_tratados_geral')
    augmented_csv_path = Path('dados_aumentados_back_translation')
    augmented_csv_path.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")

    processed_files = {f.name for f in augmented_csv_path.glob('*.csv')}
    print(f"Arquivos já processados encontrados: {len(processed_files)}")

    all_csv_files = list(cleaned_data.glob('*.csv'))
    
    files_to_process = [f for f in all_csv_files if f.name not in processed_files]
    
    print(f"Arquivos restantes para processar: {len(files_to_process)}")

    if len(files_to_process) == 0:
        print("\nTodos os arquivos já foram processados! Nada a fazer.")
        return

    BATCH_SIZE = 32
    print("\nCarregando modelo de tradução...")
    back_trans_aug = naw.BackTranslationAug(
        from_model_name='Helsinki-NLP/opus-mt-en-de',
        to_model_name='Helsinki-NLP/opus-mt-de-en',
        max_length=150,
        device=device,
        batch_size=BATCH_SIZE
    )

    for csv in tqdm(files_to_process, desc="Processando arquivos restantes"):
        try:
            df = pd.read_csv(csv)
            column = 'value'
            original_texts = df[column].fillna("").astype(str).tolist()
            
            augmented_candidates = back_trans_aug.augment(original_texts)

            final_texts = []
            for orig, aug in zip(original_texts, augmented_candidates):
                final_texts.append(filter_hallucinations(orig, aug))

            df_augmented = df.copy()
            df_augmented[column] = final_texts
            
            new_csv_path = augmented_csv_path / csv.name
            df_augmented.to_csv(new_csv_path, index=False)
            
        except Exception as e:
            print(f"\nErro ao processar {csv.name}: {e}")
            continue

if __name__ == '__main__':
    freeze_support()
    main()