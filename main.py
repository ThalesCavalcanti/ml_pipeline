from data_loader import load_data
from preprocessing import preprocess_data
from model_train import train_model
from evaluate import evaluate_model

def run_pipeline():
    print("Iniciando pipeline de Machine Learning...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    print("Pipeline concluído com sucesso!")

if __name__ == "__main__":
    run_pipeline()
