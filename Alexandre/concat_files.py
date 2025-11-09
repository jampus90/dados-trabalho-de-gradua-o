import pandas as pd
from pathlib import Path

DESTINY_PATH = Path('concat_file')
FILE_PATH = Path('dados_aumentados_back_translation')
FILE_TYPE = '*.csv'

DESTINY_PATH.mkdir(parents=True, exist_ok=True)

all_csv_file = list(FILE_PATH.glob(FILE_TYPE))
all_data = []

# 1. Carrega e limpa referências
file_1 = pd.read_csv('dev_split_Depression_AVEC2017.csv')
file_2 = pd.read_csv('train_split_Depression_AVEC2017.csv')
file_3 = pd.read_csv('full_test_split.csv') 

ref_df = pd.concat([file_1, file_2, file_3])

ref_df['Participant_ID'] = pd.to_numeric(ref_df['Participant_ID'], errors='coerce')
ref_df = ref_df.dropna(subset=['Participant_ID'])
ref_df['Participant_ID'] = ref_df['Participant_ID'].astype(int)

ref_df = ref_df.drop_duplicates(subset=['Participant_ID'])
ref_df = ref_df.set_index('Participant_ID')
# -------------------------------

print(f"Referência carregada com {len(ref_df)} participantes únicos.")

for csv in all_csv_file:
    try:
        df = pd.read_csv(csv)
        concatenated_string = df['value'].astype(str).str.cat(sep=', ')

        parts = csv.name.split("_")
        # Força o ID do arquivo a ser inteiro também
        file_id_int = int(parts[0])
        
        # Busca segura: se não achar, retorna None em vez de travar com erro
        if file_id_int in ref_df.index:
            phq8_score = ref_df.loc[file_id_int, 'PHQ8_Score']
        else:
            # print(f"Aviso: ID {file_id_int} não está nas planilhas de referência.")
            phq8_score = None # Ou use np.nan se preferir

        all_data.append({
            'participant_id': file_id_int,
            'participant_responses': concatenated_string,
            'phq8_score': phq8_score
        })
        
    except Exception as e:
        print(f"Erro ao processar {csv.name}: {e}")

new_df = pd.DataFrame(all_data)
# Verifica quantos ficaram sem score antes de salvar
print(f"Total processado: {len(new_df)}")
print(f"Scores encontrados: {new_df['phq8_score'].notnull().sum()}")
print(f"Scores faltando: {new_df['phq8_score'].isnull().sum()}")

new_df.to_csv(DESTINY_PATH / 'concatened_data_back_translation.csv', index=False)