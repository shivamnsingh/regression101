# Regression 101

Two beginner-friendly notebooks that introduce the two classic regression models with scikit-learn: **linear regression** for predicting a number and **logistic regression** for predicting a class.

## Notebooks

### 1. Linear regression: `linear_regg.ipynb`

- **Dataset:** `auto-mpg.csv` (fuel economy and specs of cars: cylinders, displacement, horsepower, weight, acceleration, model year, origin).
- **Task:** predict `mpg` from `acceleration` with simple linear regression.
- **Method:** exploratory plots with seaborn and matplotlib, an 80/20 train/test split, then evaluation with R², MSE and MAE.
- **Result:** R² of about 0.28 (MSE about 38.5, MAE about 4.99). Acceleration alone explains only a small share of mileage, which makes this a good starting point for adding more features.

### 2. Logistic regression: `logistic_regg.ipynb`

- **Dataset:** `breast-cancer.csv` (tumor measurements with a `diagnosis` label).
- **Task:** classify tumors using all numeric measurements as features.
- **Method:** label-encode the diagnosis, 80/20 train/test split, standardize features with `StandardScaler`, train `LogisticRegression`.
- **Result:** about 97.4% test accuracy (confusion matrix `[[70, 1], [2, 41]]` on 114 test samples), with precision and recall reported per class.

## Repository layout

```
auto-mpg.csv
breast-cancer.csv
linear_regg.ipynb
logistic_regg.ipynb
```

## Getting started

```bash
git clone https://github.com/shivamnsingh/regression101.git
cd regression101
pip install jupyter pandas numpy matplotlib seaborn scikit-learn
jupyter notebook
```

## Ideas to extend

- Use several features (weight, horsepower, model year) for the mpg model and compare R².
- Compare logistic regression with other classifiers on the breast cancer data.
