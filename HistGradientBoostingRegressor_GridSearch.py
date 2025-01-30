import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

# Load the training data
TrainingData = np.load(r"pathto\TrainingData.npy")

# Split the data
X = TrainingData[:, [3, 4, 5]]
y = TrainingData[:, [0, 1, 2]]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.85, random_state=42)

# HistGradientBoostingRegressor setup
HGB = HistGradientBoostingRegressor()

# MultiOutputRegressor setup
MultHGB = MultiOutputRegressor(HGB)

# Define the parameter grid
param_grid = {
    'estimator__learning_rate': [0.01, 0.1, 0.2],
    'estimator__max_iter': [100, 500, 1000],
    'estimator__max_leaf_nodes': [31, 51, 101],
    'estimator__min_samples_leaf': [10, 20, 30]
}

# Configure GridSearchCV
grid_search = GridSearchCV(MultHGB, param_grid, cv=5, verbose=3, scoring='neg_mean_squared_error')

# perform grid search
grid_search.fit(X_train, y_train)

# Make predictions with the best estimator
y_pred = grid_search.best_estimator_.predict(X_test)
