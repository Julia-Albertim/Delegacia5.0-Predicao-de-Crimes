#importando bibliotecas
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# importando modelos
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Carregando os dados do CSV correto
df = pd.read_csv('ocorrencias.csv')  # CSV gerado, separador padrão ','

# Exibindo primeiras linhas e informações do DataFrame
print(df.head())
print(df.info())

# Exemplo: contar crimes por cidade
sns.countplot(data=df, x="cidade")
plt.title("Ocorrências por cidade")
plt.show()

# ---------------------------
# 2️⃣ Preprocessing manual de variáveis categóricas (listas)
# ---------------------------
# Criando listas únicas para codificação
cidades = df['cidade'].unique().tolist()
bairros = df['bairro'].unique().tolist()
tipos = df['tipo_de_crime'].unique().tolist()
status = df['status_investigacao'].unique().tolist()
dias_semana = df['dia_semana'].unique().tolist()

# Função para codificar
df['cidade_num'] = df['cidade'].apply(lambda x: cidades.index(x))
df['bairro_num'] = df['bairro'].apply(lambda x: bairros.index(x))
df['tipo_num'] = df['tipo_de_crime'].apply(lambda x: tipos.index(x))
df['status_num'] = df['status_investigacao'].apply(lambda x: status.index(x))
df['dia_semana_num'] = df['dia_semana'].apply(lambda x: dias_semana.index(x))

# ---------------------------
# 3️⃣ Agregar dados por dia e bairro
# ---------------------------
df_grouped = df.groupby(['dia','bairro_num']).size().reset_index(name='num_ocorrencias')

X = df_grouped[['bairro_num']]  # Pode adicionar 'dia_semana_num', 'cidade_num' etc
y = df_grouped['num_ocorrencias']

# ---------------------------
# 4️⃣ Separar treino/teste
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 5️⃣ Linear Regression
# ---------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("=== Linear Regression ===")
print("R²:", r2_score(y_test, y_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))

# ---------------------------
# 6️⃣ Random Forest Regressor
# ---------------------------
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n=== Random Forest ===")
print("R²:", r2_score(y_test, y_pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))

# Importância das variáveis
feat_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
plt.figure(figsize=(6,4))
sns.barplot(x=feat_importances.values, y=feat_importances.index)
plt.title("Importância das variáveis (Random Forest)")
plt.show()

# ---------------------------
# 7️⃣ Decision Tree Regressor
# ---------------------------
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("\n=== Decision Tree ===")
print("R²:", r2_score(y_test, y_pred_dt))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_dt)))

# ---------------------------
# 8️⃣ Comparando predições vs valores reais
# ---------------------------
plt.figure(figsize=(12,6))
plt.scatter(range(len(y_test)), y_test, label='Real', alpha=0.6)
plt.scatter(range(len(y_test)), y_pred_lr, label='Linear Regression', alpha=0.6)
plt.scatter(range(len(y_test)), y_pred_rf, label='Random Forest', alpha=0.6)
plt.scatter(range(len(y_test)), y_pred_dt, label='Decision Tree', alpha=0.6)
plt.legend()
plt.title("Comparação: Valores Reais vs Preditos")
plt.xlabel("Amostras")
plt.ylabel("Número de ocorrências")
plt.show()