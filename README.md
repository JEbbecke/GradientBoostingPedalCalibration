# Enhanced Dynamic Calibration in an Instrumented Cycling Pedal: A Gradient Boosting Approach
## Project Overview
This project aims to improve the accuracy of force measurements in custom made instrumented cycling pedals through a gradient boosting approach. Specifically, we implement scikit learn's histogram-based Gradient Boosting Machine (GBM) model to calibrate the instrumented cycling pedal dynamically and multiaxial, enhancing the precision of force data under various cycling conditions. This repository contains the essential Python scripts used in the data selection process and the model training process, as outlined in our related publication.

## Repository Contents
```DataSelection.py:``` This script handles the selection of relevant data points from a larger dataset in order to have a equal distribution over the full calibration space and higher density of datapoints in cycling specific ranges. This ensures that the training data used is representative of typical cycling dynamics, but covers untypical force exertion.

```HistGradientBoostingRegressor_GridSearch.py:``` This script includes the setup for a HistGradientBoostingRegressor, its configuration through GridSearchCV to fine-tune hyperparameters, and subsequent model training and validation.

## Installation
To run the scripts contained in this repository, you will need Python 3.x and several dependencies:
```pip install numpy scikit-learn```

## Usage
### Data Selection Process
The ```DataSelection.py:``` script is designed to filter and select data from a comprehensive dataset based on predefined calibration spaces and voxel definitions. Ensure that the paths to your data files are correctly set in the script:
```python
pilot_data = np.load(r"path_to\PilotData.npy")
training_data = np.load(r"path_to\TrainingData.npy")
```

Run the script to process and select the data:
```
python DataSelection.py
```

### Model Training and Grid Search
The ```HistGradientBoostingRegressor_GridSearch.py``` script sets up the GBM model using scikit-learn's ```HistGradientBoostingRegressor``` wrapped in a ```MultiOutputRegressor``` for multi-dimensional targets. The grid search explores various hyperparameters to find the best model settings:
```python
grid_search = GridSearchCV(MultHGB, param_grid, cv=5, verbose=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
```

Execute the script to train the model and evaluate its performance:
```
python HistGradientBoostingRegressor_GridSearch.py
```

## License
This project is licensed under the [MIT](https://mit-license.org/) License 
