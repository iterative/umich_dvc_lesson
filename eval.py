import pickle
from sklearn.datasets import make_regression
import json
import pandas as pd

model = pickle.load(open("models/model.pkl", "rb"))

# Load the test data
X_test = pd.read_csv("data/test_features.csv")
y_test = pd.read_csv("data/test_y.csv").values

# Test on the model
test_r2 = model.score(X_test, y_test.ravel())

with open('test_metrics.json', 'w') as f:
    json.dump({'r2': test_r2}, f)

