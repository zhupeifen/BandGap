from ucimlrepo import fetch_ucirepo
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

  
# fetch dataset 
tic_tac_toe_endgame = fetch_ucirepo(id=101) 
  
# data (as pandas dataframes) 
X = tic_tac_toe_endgame.data.features 
y = tic_tac_toe_endgame.data.targets

squares = ['top-left-square', 'top-middle-square', 'top-right-square', 'middle-left-square', 'middle-middle-square',
           'middle-right-square', 'bottom-left-square', 'bottom-middle-square', 'bottom-right-square']

total_outcomes = 958

'''
X['top-left-square'][0] gives the top left square of the first game

there are 9 training parameters (squares) that I can use
my goal output is for the AI to be able to give me the game's outcome based on the squares
'''

print(y)
print(y['class'][0])

train_X = np.zeros((total_outcomes, 9))
train_y = np.zeros((total_outcomes))

for i in range(total_outcomes):
    for j, square in enumerate(squares):
        square_value = X[square][i]
        
        if square_value == 'x':
            train_X[i, j] = 1
        if square_value == 'o':
            train_X[i, j] = -1
        else:
            train_X[i, j] = 0
        
    out_value = y['class'][i]
    
    if out_value == 'positive':
        train_y[i] = 1
    if out_value =='negative':
        train_y[i] = 0

X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define hyperparameters as a standard Python dictionary
params = {
    'max_depth': 8,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

'''
# Train using the native xgb.train function (instead of model.fit)
native_model = xgb.train(
    params, 
    dtrain, 
    num_boost_round=50
)

# Predict (Note: native predict outputs raw probabilities automatically)
probabilities = native_model.predict(dtest)
'''

native_model = xgb.XGBClassifier()
native_model.load_model('tic_tac_toe.json')
probabilities = native_model.predict(X_test)


correct = 0
total = len(probabilities)
print(X_test)
print(y_test)

for i in range(len(probabilities)):
    if round(probabilities[i]) == y_test[i]:
        correct += 1

print('testing_accuracy: ' + str(correct / total))

future_board = np.array([[1, 1, 0,  
                          -1, -1, -1,  
                          0, 1, 0]]) 

# Should predict 0, since it's a loss
prediction = native_model.predict_proba(future_board)
print(f"Prediction (1=Win, 0=Loss): {prediction[0]}")

json_data = native_model.get_booster().get_dump(dump_format='dot')[0]
print(json_data)
with open('tree_data.dot', 'w') as file:
    file.write(json_data)

#native_model.save_model('tic_tac_toe.json')

'''
# Train the model
# Because our output is 0 or 1, we use 'binary:logistic'

# This can work instead of the above code

model = xgb.XGBClassifier(
    n_estimators=50,
    max_depth=5,
    objective='binary:logistic',
    random_state=42
)
model.fit(X_train, y_train)
'''
