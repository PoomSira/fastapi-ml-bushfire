# import pickle
# import numpy as np
# from sklearn.svm import SVC

# # Load the model
# model = pickle.load(open('BB_svm_model.pkl', 'rb'))

# # Create some test data
# test_data = np.array([[9,10.7,32.1,0.0,23,4.0,15]]).reshape(1, -1)

# # Test the prediction and probability functions
# prediction = model.predict(test_data)
# probabilities = model.predict_proba(test_data)

# print("Prediction:", prediction)
# print("Probabilities:", probabilities)

import numpy as np
from sklearn.svm import SVC
import joblib

# Load the model
loaded_model = joblib.load('BB_svm_model.pkl')

X_test = [9,10.7,32.1,0.0,23,4.0,20]

# Reshape it to a 2D array
X_test = np.array(X_test).reshape(1, -1)

# Use the loaded model to make predictions
y_pred = loaded_model.predict(X_test)

# Get probability estimates
probabilities = loaded_model.predict_proba(X_test)
print("Probabilities:", probabilities)
