# import pickle
# model = pickle.load(open('BB_svm_model.pkl', 'rb'))
# print(type(model))

# if hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
#     print("Model supports both predict and predict_proba.")
# else:
#     print("Model does not support predict_proba.")

from sklearn.svm import SVC
model = SVC(probability=True)  # Set probability=True
model.fit(X_train, y_train)

# Save the model with probability support
with open('BB_svm_model.pkl', 'wb') as f:
    pickle.dump(model, f)

