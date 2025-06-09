from sklearn.metrics import accuracy_score
from src.utils import save_confusion_matrix

def evaluate_model(model, X_test, y_test,output_dir=None):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    if output_dir:
        save_confusion_matrix(y_test, y_pred,  save_path=f"{output_dir}/confusion_matrix.png")