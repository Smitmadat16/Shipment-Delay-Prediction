# Saving the Model, after training the model.
# Using this in Streamlit app
import pickle
with open('rf_model.pkl', 'wb') as f:  # Saving trained Model
    pickle.dump(ranfrst, f)
with open('model_columns.pkl', 'wb') as f: # saving cols name
    pickle.dump(list(X.columns), f)
print("Model saved as rf_model.pkl")
print("Feature columns saved as model_columns.pkl")
print(f"Total features:{len(X.columns)}")
