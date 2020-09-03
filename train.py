import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
import pickle
import os


X = pd.read_csv("data/train_features.csv")
y = pd.read_csv("data/train_y.csv")

# Train a model
reg = LinearRegression().fit(X, y)
# Print out training r2
print(reg.score(X,y))

# Write the model to a file
if not os.path.isdir("models/"):
    os.mkdir("models")

filename = 'models/model.pkl'
pickle.dump(reg, open(filename, 'wb'))

