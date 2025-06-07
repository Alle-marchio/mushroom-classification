from src.preprocessing import load_and_preprocess_data
from src.models import get_models_and_parameters
from src.evaluation import evaluate_model
from src.utils import plot_target_distribution, plot_model_accuracies
from sklearn.model_selection import GridSearchCV
import os

def main():
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/mushrooms.csv")
    plot_target_distribution(y_train, save_path=f"{output_dir}/target_distribution.png")

    models_dict = get_models_and_parameters()

    best_model = None
    best_score = 0
    best_name = ""
    results = {}

    for name, (model, params) in models_dict.items():
        print(f"\n Grid search per: {name}")
        grid = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        print(f"Migliore CV accuracy: {grid.best_score_:.4f}")
        print("Migliori parametri:", grid.best_params_)

        results[name] = grid.best_score_

        if grid.best_score_ > best_score:
            best_model = grid.best_estimator_
            best_score = grid.best_score_
            best_name = name

    # Salvataggio grafico confronto modelli
    plot_model_accuracies(results, title="Accuracy Cross-Validation", save_path=f"{output_dir}/model_accuracies.png")

    print(f"\n modello migliore: {best_name} ({best_score:.4f}) — evaluating sul test set...")
    evaluate_model(best_model, X_test, y_test,output_dir=output_dir)

if __name__ == "__main__":
    main()
