from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def get_models_and_parameters():
    models = {
        'Decision Tree': (DecisionTreeClassifier(), {
            'max_depth': [3, 5, 10, None],
            'criterion': ['gini', 'entropy']
        }),
        'Random Forest': (RandomForestClassifier(), {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None]
        }),
        'KNN': (KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 7],
            'p': [1, 2],
            'weights': ['uniform', 'distance']
        }),
        'SVM': (SVC(), {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        })
    }
    return models
