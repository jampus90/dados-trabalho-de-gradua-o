import pandas as pd
from sklearn.model_selection import train_test_split

# ================= CONFIGURAÇÃO =================
FILE_ORIGINAL = 'concat_file/dados_limpos_original_data.csv'
FILE_AUGMENTED = 'concat_file/dados_limpos_back_translation.csv'
OUTPUT_TRAIN = 'dados_treino.csv'
OUTPUT_TEST = 'dados_teste.csv'

# Colunas chave
COL_ID = 'participant_id'
COL_TEXT = 'participant_responses'
COL_LABEL = 'phq8_score'
# ================================================

print("--- INICIANDO PROCESSAMENTO ---")

# 1. Carregar os dados
try:
    df_orig = pd.read_csv(FILE_ORIGINAL)
    df_aug = pd.read_csv(FILE_AUGMENTED)
    print(f"Original: {len(df_orig)} linhas | Back-translation: {len(df_aug)} linhas")
except FileNotFoundError as e:
    print(f"\nERRO: Arquivo não encontrado: {e.filename}")
    exit()

# 2. Agrupar por participante e criar categorias de score para estratificação
unique_participants = df_orig[COL_ID].nunique()
print(f"Total de participantes únicos: {unique_participants}")

# Garante um score por participante
if unique_participants < len(df_orig):
    print("AVISO: Participantes com múltiplas linhas detectados. Agrupando...")
    participants = df_orig.groupby(COL_ID)[COL_LABEL].first().reset_index()
else:
    participants = df_orig[[COL_ID, COL_LABEL]].copy()

# Função para criar categorias do PHQ-8 (bins) para resolver o problema de estratificação
def categorize_phq8(score):
    if score <= 4: return '0-4 (Minima)'
    elif score <= 9: return '5-9 (Leve)'
    elif score <= 14: return '10-14 (Moderada)'
    elif score <= 19: return '15-19 (Mod. Severa)'
    else: return '20-24 (Severa)'

participants['category'] = participants[COL_LABEL].apply(categorize_phq8)

print("\nDistribuição por categorias PHQ-8 para estratificação:")
print(participants['category'].value_counts().sort_index())

# Verifica se alguma categoria tem menos de 2 membros
vcont = participants['category'].value_counts()
use_stratify = True
if any(vcont < 2):
    print("\nAVISO CRÍTICO: Categorias com menos de 2 participantes encontradas.")
    print("Tentando estratificação binária simplificada (>=10 vs <10)...")
    participants['category_bin'] = participants[COL_LABEL].apply(lambda x: 1 if x >= 10 else 0)
    vcont_bin = participants['category_bin'].value_counts()
    print(vcont_bin)
    
    if any(vcont_bin < 2):
         print("ERRO: Impossível estratificar mesmo com binário. Usando divisão aleatória simples.")
         use_stratify = False
    else:
         participants['stratify_col'] = participants['category_bin']
else:
    participants['stratify_col'] = participants['category']

# 3. Dividir os PARTICIPANTES
print(f"\nRealizando divisão (Estratificação: {'ATIVADA' if use_stratify else 'DESATIVADA'})...")
train_ids, test_ids = train_test_split(
    participants[COL_ID],
    test_size=0.2,
    random_state=42,
    stratify=participants['stratify_col'] if use_stratify else None
)

print(f"Participantes no treino: {len(train_ids)}")
print(f"Participantes no teste: {len(test_ids)}")

# 4. Criar DataFrames finais
df_test_final = df_orig[df_orig[COL_ID].isin(test_ids)].copy()

# Treino: dados originais dos participantes de treino + suas back-translations
df_train_orig = df_orig[df_orig[COL_ID].isin(train_ids)]
# FILTRO CRUCIAL: Só usa back-translation de participantes que estão no TREINO
df_train_aug = df_aug[df_aug[COL_ID].isin(train_ids)] 

df_train_final = pd.concat([df_train_orig, df_train_aug]).sample(frac=1, random_state=42).reset_index(drop=True)

# 5. Validação e Salvamento
print("\n--- RESUMO FINAL ---")
print(f"ARQUIVO DE TREINO: {len(df_train_final)} linhas")
print(f"  - Originais: {len(df_train_orig)}")
print(f"  - Aumentados: {len(df_train_aug)}")
print(f"ARQUIVO DE TESTE: {len(df_test_final)} linhas (100% Puras)")

# Verificação de vazamento (deve ser 0)
leak_check = set(df_train_final[COL_ID]).intersection(set(df_test_final[COL_ID]))

if len(leak_check) == 0:
    df_train_final.to_csv(OUTPUT_TRAIN, index=False)
    df_test_final.to_csv(OUTPUT_TEST, index=False)
    print(f"\nSucesso! Arquivos salvos: '{OUTPUT_TRAIN}' e '{OUTPUT_TEST}'")
else:
    print(f"\nERRO CRÍTICO: Vazamento de {len(leak_check)} participantes detectado! Arquivos NÃO salvos.")