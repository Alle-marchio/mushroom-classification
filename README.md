# Mushroom Classification (UCI Dataset) 

Progetto accademico di Machine Learning per classificare funghi come commestibili o velenosi.

### [link al dataset](https://archive.ics.uci.edu/dataset/73/mushroom)

## Struttura
- `data/`: contiene il dataset mushrooms.csv
- `src/`: codice modularizzato per preprocessing, modelli, valutazione
- `main.py`: script principale
- `figures` : grafici dei vari modelli
- `requirements.txt`: dipendenze del progetto
## Tecniche usate
- Label Encoding
- Train/Test split
- Standard Scaler
- GridSearchCV con 4 modelli:
  - Decision Tree
  - Random Forest
  - KNN
  - SVM
- Confusion Matrix + classification report

## Come eseguire
```bash
  python main.py