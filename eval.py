import pickle
from sklearn.datasets import make_regression
import json
model = pickle.load(open("model.pkl", "rb"))

# Make some new random data to evaluate on
X_test, y_test = make_regression(1000,n_features = 10)

# Test on the model
test_r2 = model.score(X_test, y_test)

with open('test_metrics.json', 'w') as f:
    json.dump({'r2': test_r2}, f)

