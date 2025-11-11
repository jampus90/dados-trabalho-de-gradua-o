import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# ================= CONFIGURAÇÃO =================
TRAIN_FILE = 'dados_treino.csv'
TEST_FILE = 'dados_teste.csv'
COL_TEXT = 'participant_responses'
COL_SCORE = 'phq8_score'
PHQ8_CUTOFF = 5
# ================================================

print("--- 1. CARREGANDO DADOS ---")
df_train = pd.read_csv(TRAIN_FILE)
df_test = pd.read_csv(TEST_FILE)

print(f"Treino: {len(df_train)} linhas")
print(f"Teste:  {len(df_test)} linhas")

# Criando a coluna alvo binária (0 = Sem Depressão, 1 = Com Depressão)
y_train = (df_train[COL_SCORE] >= PHQ8_CUTOFF).astype(int)
y_test = (df_test[COL_SCORE] >= PHQ8_CUTOFF).astype(int)

print("\nDistribuição das classes no TREINO:")
print(y_train.value_counts(normalize=True))

# ================================================
print("\n--- 2. VETORIZAÇÃO (TF-IDF) ---")
# Transformando texto em números.
# max_features=5000: limita às 5000 palavras mais importantes para evitar excesso de dados
tfidf = TfidfVectorizer(max_features=5000, lowercase=True, stop_words=None, ngram_range=(1, 2))

# AJUSTAMOS (fit) apenas nos dados de treino para aprender o vocabulário
X_train = tfidf.fit_transform(df_train[COL_TEXT].fillna(''))

# Apenas TRANSFORMAMOS os dados de teste usando o vocabulário aprendido no treino
X_test = tfidf.transform(df_test[COL_TEXT].fillna(''))

print(f"Vocabulário aprendido: {len(tfidf.get_feature_names_out())} palavras")

# ================================================
# print("\n--- 3. TREINAMENTO DO MODELO BASELINE ---")
# # class_weight='balanced' é CRUCIAL aqui pois dados de depressão são desbalanceados.
# # Ele diz para o modelo prestar mais atenção na classe minoritária (quem tem depressão).
# model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
# model.fit(X_train, y_train)
# print("Modelo treinado com Regressão Logística.")

# print("\n--- 3. TREINAMENTO DO MODELO (SVM) ---")
# # LinearSVC é mais rápido e geralmente melhor para texto que o SVC padrão
# model = LinearSVC(class_weight='balanced', random_state=42, max_iter=2000)
# model.fit(X_train, y_train)
# print("Modelo treinado com SVM (LinearSVC).")

# print("\n--- 3. OTIMIZANDO O SVM ---")
# # Vamos testar 3 valores de C
# for c_value in [0.1, 1.0, 10.0]:
#     print(f"\nTreinando com C={c_value}...")
#     model = LinearSVC(C=c_value, class_weight='balanced', random_state=42, max_iter=5000) # Aumentei max_iter para garantir convergência
#     model.fit(X_train, y_train)
    
#     y_pred_opt = model.predict(X_test)
#     f1_opt = f1_score(y_test, y_pred_opt)
#     print(f"--> F1-Score com C={c_value}: {f1_opt:.4f}")

# print("\n--- 3. TREINAMENTO DO MODELO (RANDOM FOREST) ---")
# # n_estimators=200: usa 200 árvores (mais robusto que o padrão de 100)
# # n_jobs=-1: usa todos os núcleos do seu processador para ser mais rápido
# model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
# model.fit(X_train, y_train)
# print("Modelo treinado com Random Forest.")

print("\n--- 3. TREINAMENTO DO MODELO (REDE NEURAL MLP) ---")
# Configuração básica para começar:
# hidden_layer_sizes=(100,): Uma camada oculta com 100 neurônios
# activation='relu': Função de ativação padrão moderna
# solver='adam': Otimizador padrão que funciona bem na maioria dos casos
# early_stopping=True: Para o treino se o modelo parar de melhorar (evita overfitting)
model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                      alpha=0.0001, batch_size='auto', learning_rate='constant',
                      learning_rate_init=0.001, max_iter=1000, early_stopping=True,
                      random_state=42, verbose=False) # verbose=True mostra o progresso do treino

model.fit(X_train, y_train)
print("Modelo treinado com MLPClassifier.")

# ================================================
print("\n--- 4. AVALIAÇÃO NO CONJUNTO DE TESTE ---")
# O momento da verdade: prevendo dados que o modelo NUNCA viu
y_pred = model.predict(X_test)

print("\n>>> RELATÓRIO DE CLASSIFICAÇÃO <<<")
# O F1-Score da classe '1' (com depressão) é geralmente a métrica mais importante
print(classification_report(y_test, y_pred, target_names=['Sem Depressão (0)', 'Com Depressão (1)']))

# Matriz de Confusão Simples
cm = confusion_matrix(y_test, y_pred)
print("\n>>> MATRIZ DE CONFUSÃO <<<")
print(f"Verdadeiros Negativos (Acertou 'Sem Depressão'): {cm[0][0]}")
print(f"Falsos Positivos (Errou: Disse que tinha, mas não tinha): {cm[0][1]}")
print(f"Falsos Negativos (Errou: Disse que NÃO tinha, mas TINHA - ERRO CRÍTICO): {cm[1][0]}")
print(f"Verdadeiros Positivos (Acertou 'Com Depressão'): {cm[1][1]}")

f1 = f1_score(y_test, y_pred)
print(f"\nBASELINE F1-SCORE (Classe Positiva): {f1:.4f}")
print("Guarde este número. Seus próximos modelos precisam superá-lo.")