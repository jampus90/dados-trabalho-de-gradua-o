import pandas as pd
from pathlib import Path

original_csv_paths = Path('Gabriel/DadosTG/Original')
cleaned_csv_path = Path('Gabriel/DadosTG/Tratados')

cleaned_csv_path.mkdir(parents=True, exist_ok=True)

processed_files = {f.name for f in cleaned_csv_path.glob('*.csv')}
print(f"Arquivos já tratados encontrados: {len(processed_files)}")

all_csv_files = list(original_csv_paths.glob('*.csv'))

files_to_process = [f for f in all_csv_files if f.name not in processed_files]
print(f"Arquivos restantes para tratar: {len(files_to_process)}")

if not files_to_process:
    print("Todos os arquivos já foram tratados! Nada a fazer.")
else:
    for csv in files_to_process:
        try:
            df = pd.read_csv(csv, sep='\t')
            
            df_cleaned = df[df['speaker'] != 'Ellie']
            cols_to_drop = ['start_time', 'stop_time', 'speaker']
            existing_cols_to_drop = [col for col in cols_to_drop if col in df_cleaned.columns]
            df_cleaned = df_cleaned.drop(existing_cols_to_drop, axis=1)

            new_csv_path = cleaned_csv_path / csv.name

            df_cleaned.to_csv(new_csv_path, index=False)
            print(f"Processado: {csv.name}")

        except Exception as e:
            print(f"Erro ao processar {csv.name}: {e}")

    print("Done!")