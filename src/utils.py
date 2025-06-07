import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_target_distribution(y, save_path=None):

    plt.figure(figsize=(6,4))
    sns.countplot(x=y, hue=y, palette="Set2", legend=False)
    plt.title("Distribuzione della variabile target (class)")
    plt.xlabel("Classe")
    plt.ylabel("Frequenza")
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_model_accuracies(results_dict, title="Accuracy per modello", save_path=None):

    if not results_dict:  # Controllo errori importante
        return

    plt.figure(figsize=(8, 5))
    colors = sns.color_palette("viridis", len(results_dict))
    plt.bar(results_dict.keys(), results_dict.values(), color=colors)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.ylim(0.8, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def save_confusion_matrix(y_true, y_pred, labels=None, save_path=None):

    cm = confusion_matrix(y_true, y_pred)

    if labels is None:
        labels = np.unique(y_true)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
    plt.close()
