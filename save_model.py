"""
save_model.py
Run this script ONCE in your Jupyter notebook (or as a cell) 
AFTER you have trained your models.
It saves the Random Forest model and feature columns to disk.
"""

import pickle
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

with open('model_columns.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("✅ Model saved as rf_model.pkl")
print("✅ Feature columns saved as model_columns.pkl")
print(f"Total features: {len(X.columns)}")
