from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import json

def evaluate_model(model, X_test, y_test, metrics_path="metrics.json"):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")

    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Real")
    plt.ylabel("Previsto")
    plt.title("Avaliação do Modelo")
    plt.show()

    metrics = {"MSE": mse, "R2": r2}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Métricas salvas em {metrics_path}")
    print(f"MSE: {mse:.4f} | R²: {r2:.4f}")