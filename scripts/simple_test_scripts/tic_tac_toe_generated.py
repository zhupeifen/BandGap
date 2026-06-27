import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load the official dataset directly from the web
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data"
columns = ['top_left', 'top_mid', 'top_right', 
           'mid_left', 'mid_mid', 'mid_right', 
           'bot_left', 'bot_mid', 'bot_right', 'class']
df = pd.read_csv(url, names=columns)

# 2. Encode the features ('x'->1, 'o'->-1, 'b'->0)
mapping_features = {'x': 1, 'o': -1, 'b': 0}
for col in columns[:-1]:
    df[col] = df[col].map(mapping_features)

# 3. Encode the target ('positive'->1, 'negative'->0)
df['class'] = df['class'].map({'positive': 1, 'negative': 0})

# 4. Split into inputs (X) and outputs (y)
X = df.drop('class', axis=1)
y = df['class']

# Separate into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the XGBoost Model
# Because our output is 0 or 1, we use 'binary:logistic'
model = xgb.XGBClassifier(
    n_estimators=50,
    max_depth=5,
    objective='binary:logistic',
    random_state=42
)
model.fit(X_train, y_train)

# Test how accurate it is
predictions = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

import numpy as np

# This board state encoded: Top row (X, O, blank), Mid row (blank, X, O), Bot row (blank, blank, blank)
# If X plays in the bottom right, X will win. Let's see if the model predicts a positive outcome.
future_board = np.array([[1, 1, -1,  
                          1, -1, 0,  
                          -1, 1, 0]]) 

# Get the raw prediction (0 or 1)
prediction = model.predict(future_board)
print(f"Prediction (1=Win, 0=Loss): {prediction[0]}")

# Get the probability percentage (e.g., 95% chance of winning)
probability = model.predict_proba(future_board)
print(f"Probability of X winning: {probability[0][1] * 100:.2f}%")
