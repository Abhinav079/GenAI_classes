import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset from CSV file
file_path = 'loan_data_nov2023.csv'
df = pd.read_csv(file_path)

# Preprocessing
le = LabelEncoder()
df['grade'] = le.fit_transform(df['grade'])
df['ownership'] = le.fit_transform(df['ownership'])

# Features and target variable
X = df.drop('default', axis=1)
y = df['default']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'scale_pos_weight': [1, 5, 10]
}

# Create an XGBClassifier
xgb_classifier = XGBClassifier(random_state=42)

# Instantiate GridSearchCV
grid_search = GridSearchCV(xgb_classifier, param_grid, cv=5, scoring='recall', n_jobs=-1)

# Perform grid search on the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate the model with the best hyperparameters on the test set
best_xgb_classifier = grid_search.best_estimator_
y_pred = best_xgb_classifier.predict(X_test)

# Evaluate the model's performance
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
