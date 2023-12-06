import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from CSV file
file_path = 'loan_data_nov2023.csv'
df = pd.read_csv(file_path)

# Features and target variable
X = df.drop('default', axis=1)
y = df['default']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for hyperparameter tuning (Note: Linear Regression doesn't have many hyperparameters to tune)
param_grid = {}

# Create a LinearRegression model
linear_reg = LinearRegression()

# Instantiate GridSearchCV (Note: In the case of Linear Regression, GridSearchCV may not be as critical as in tree-based models)
grid_search = GridSearchCV(linear_reg, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Perform grid search on the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters (Note: Linear Regression doesn't have hyperparameters like trees)
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate the model with the best hyperparameters on the test set
best_linear_reg = grid_search.best_estimator_
y_pred = best_linear_reg.predict(X_test)

# Evaluate the model's performance
print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))
