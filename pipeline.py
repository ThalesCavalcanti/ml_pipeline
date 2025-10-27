# pipeline.py
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Carregar dataset
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construir pipeline
ml_pipeline = Pipeline([
    ('scaler', StandardScaler()),          # Etapa 1: normalização dos dados
    ('regressor', LinearRegression())      # Etapa 2: modelo de regressão
])

# Treinar o modelo
ml_pipeline.fit(X_train, y_train)

# Fazer previsões
y_pred = ml_pipeline.predict(X_test)

# Avaliar desempenho
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(" Resultados do Modelo:")
print(f" - MSE: {mse:.4f}")
print(f" - R² : {r2:.4f}")

# Visualização
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Valores Reais")
plt.ylabel("Previsões")
plt.title("Regressão Linear - Pipeline de ML")
plt.tight_layout()
plt.show()
