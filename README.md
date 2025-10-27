# Mini Pipeline de Machine Learning

Este projeto demonstra a criação de um **pipeline de aprendizado de máquina automatizado** em Python, utilizando o dataset *California Housing*.

## Etapas do Pipeline
1. **Carregamento de dados:** usando `fetch_california_housing` do scikit-learn.  
2. **Pré-processamento:** normalização com `StandardScaler`.  
3. **Treinamento:** modelo de regressão linear (`LinearRegression`).  
4. **Avaliação:** cálculo de MSE e R².  
5. **Visualização:** gráfico comparando valores reais e previstos.

## Como executar
```bash
pip install -r requirements.txt
python pipeline.py
