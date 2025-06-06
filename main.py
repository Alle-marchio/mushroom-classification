from src.preprocessing import load_and_preprocess_data
from src.models import get_models_and_parameters
from src.evaluation import evaluate_model

from sklearn.model_selection import GridSearchCV

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/mushrooms.csv")

    models_dict = get_models_and_parameters()

    best_model = None
    best_score = 0
    best_name = ""

    for name, (model, params) in models_dict.items():
        print(f"\n Grid search per: {name}")
        grid = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        print(f"Migliore CV accuracy: {grid.best_score_:.4f}")
        print("Migliori parametri:", grid.best_params_)

        if grid.best_score_ > best_score:
            best_model = grid.best_estimator_
            best_score = grid.best_score_
            best_name = name

    print(f"\n modello migliore: {best_name} ({best_score:.4f}) — evaluating sul test set...")
    evaluate_model(best_model, X_test, y_test)

if __name__ == "__main__":
    main()
