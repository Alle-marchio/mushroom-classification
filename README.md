# Mushroom Classification (UCI Dataset)

Progetto di Machine Learning per classificare funghi come commestibili o velenosi.

## 📁 Struttura
- `src/`: codice modularizzato per preprocessing, modelli, valutazione
- `main.py`: script principale
- `data/`: contiene il dataset mushrooms.csv

## ⚙️ Tecniche usate
- Label Encoding
- Train/Test split
- Standard Scaler
- GridSearchCV con 4 modelli:
  - Decision Tree
  - Random Forest
  - KNN
  - SVM
- Confusion Matrix + classification report

## ▶️ Come eseguire
```bash
python main.py