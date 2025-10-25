import pandas as pd
from pathlib import Path

original_csv_paths = Path('Alexandre/dados_originais')
cleaned_csv_path = Path('Alexandre/dados_tratados')

#Lista o nome dos arquivos csv
csv_files_list = list(original_csv_paths.glob('*.csv'))


for csv in csv_files_list:
    # Le dados
    df = pd.read_csv(csv, sep='\t')
    
    # Tira as linhas com a Ellie
    df_cleaned = df[df['speaker'] != 'Ellie']

    # Tira as colunas que não queremos usar
    df_cleaned = df_cleaned.drop(['start_time', 'stop_time', 'speaker'], axis=1)

    new_csv_path = cleaned_csv_path / csv.name

    df_cleaned.to_csv(new_csv_path, index=False)

print("Done!")