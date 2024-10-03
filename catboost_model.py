import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

# Load your dataset
df = pd.read_csv('train.csv')

# Define the categorical columns
categorical_cols = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']

# Ensure categorical columns are treated as strings and fill missing values
for col in categorical_cols:
    df[col] = df[col].fillna('Unknown').astype(str)

X = df.drop(columns=['PassengerId', 'Name', 'Transported'])
y = df['Transported'].astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X.columns)
# Initialize and train the CatBoostClassifier model
model = CatBoostClassifier(
    learning_rate=0.1, 
    depth=6, 
    l2_leaf_reg=3, 
    iterations=500, 
    early_stopping_rounds=50, 
    verbose=100
)

#Fit the model
model.fit(X_train, y_train, cat_features=categorical_cols, eval_set=(X_test, y_test))

# Make predictions
y_pred = model.predict(X_test)

# Save the model
joblib.dump(model, 'catboost_model.pkl')

# Calculate accuracy, F1 score, and AUC-ROC
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc_roc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1
Score: {f1:.4f}")
print(f"AUC-ROC Score: {auc_roc:.4f}")
