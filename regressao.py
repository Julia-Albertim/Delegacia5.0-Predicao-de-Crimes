# ---------------------------
# 0️⃣ Bibliotecas
# ---------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm  # Poisson
from xgboost import XGBRegressor

# ---------------------------
# 1️⃣ Carregar CSV
# ---------------------------
df = pd.read_csv('ocorrencias.csv')

# Mostrar primeiras linhas e info
print(df.head())
print(df.info())

# ---------------------------
# 2️⃣ Valores nulos e tipos
# ---------------------------
print("Valores faltantes por coluna:")
print(df.isnull().sum())

# Ajustando tipos de dados
df['dia'] = pd.to_datetime(df['dia'], format='%Y-%m-%d')
df['hora'] = pd.to_datetime(df['hora'], format='%H:%M') - pd.Timedelta(hours=3)  # horário BR

# Traduzindo dia da semana
dias_pt = {
    'Monday':'Segunda-feira', 'Tuesday':'Terça-feira', 'Wednesday':'Quarta-feira',
    'Thursday':'Quinta-feira', 'Friday':'Sexta-feira', 'Saturday':'Sábado', 'Sunday':'Domingo'
}
df['dia_semana'] = df['dia_semana'].map(dias_pt)

# ---------------------------
# 3️⃣ Preparar dados para regressão
# ---------------------------
# Vamos prever número de ocorrências por bairro/dia
df_grouped = df.groupby(['dia','bairro','cidade','tipo_de_crime','status_investigacao']).size().reset_index(name='num_ocorrencias')

# Codificação manual de variáveis categóricas
bairros = df_grouped['bairro'].unique().tolist()
cidades = df_grouped['cidade'].unique().tolist()
tipos = df_grouped['tipo_de_crime'].unique().tolist()
status = df_grouped['status_investigacao'].unique().tolist()

df_grouped['bairro_num'] = df_grouped['bairro'].apply(lambda x: bairros.index(x))
df_grouped['cidade_num'] = df_grouped['cidade'].apply(lambda x: cidades.index(x))
df_grouped['tipo_num'] = df_grouped['tipo_de_crime'].apply(lambda x: tipos.index(x))
df_grouped['status_num'] = df_grouped['status_investigacao'].apply(lambda x: status.index(x))

# Features e target
X = df_grouped[['bairro_num','cidade_num','tipo_num','status_num']]
y = df_grouped['num_ocorrencias']

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 4️⃣ Random Forest Regressor
# ---------------------------
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print("=== Random Forest ===")
print(f"R²: {r2_rf:.3f}, RMSE: {rmse_rf:.3f}, MAE: {mae_rf:.3f}")

# ---------------------------
# 5️⃣ Poisson Regression
# ---------------------------
X_train_poisson = sm.add_constant(X_train)
X_test_poisson = sm.add_constant(X_test)

poisson_model = sm.GLM(y_train, X_train_poisson, family=sm.families.Poisson())
poisson_results = poisson_model.fit()
y_pred_poisson = poisson_results.predict(X_test_poisson)

r2_p = r2_score(y_test, y_pred_poisson)
rmse_p = np.sqrt(mean_squared_error(y_test, y_pred_poisson))
mae_p = mean_absolute_error(y_test, y_pred_poisson)

print("\n=== Poisson Regression ===")
print(f"R²: {r2_p:.3f}, RMSE: {rmse_p:.3f}, MAE: {mae_p:.3f}")

# ---------------------------
# 6️⃣ XGBoost Regressor
# ---------------------------
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=200, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

r2_xgb = r2_score(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

print("\n=== XGBoost Regression ===")
print(f"R²: {r2_xgb:.3f}, RMSE: {rmse_xgb:.3f}, MAE: {mae_xgb:.3f}")

# ---------------------------
# 7️⃣ Comparando modelos
# ---------------------------
resultados = pd.DataFrame({
    'Modelo': ['Random Forest','Poisson Regression','XGBoost'],
    'R2':[r2_rf, r2_p, r2_xgb],
    'RMSE':[rmse_rf, rmse_p, rmse_xgb],
    'MAE':[mae_rf, mae_p, mae_xgb]
})

print("\n=== Comparação dos modelos ===")
print(resultados)

# Melhor modelo baseado em R²
melhor_modelo = resultados.loc[resultados['R2'].idxmax(),'Modelo']
print(f"\nMelhor modelo: {melhor_modelo}")
