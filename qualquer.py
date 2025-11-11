import pandas as pd

TEST_FILE = 'dados_teste.csv'
TRAIN_FILE = 'dados_treino.csv'

df_treino = pd.read_csv(TRAIN_FILE)
df_teste = pd.read_csv(TEST_FILE)

ids_treino = df_treino['participant_id']
ids_teste = df_teste['participant_id']

set_treino = set(ids_treino)
set_teste = set(ids_teste)

commom_ids = set_teste.intersection(set_treino)

print(f'Common ids: {commom_ids}')