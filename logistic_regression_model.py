import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

data = pd.read_csv('train.csv')

data_TotalExpenses = data[['RoomService', 'FoodCourt','ShoppingMall','Spa', 'VRDeck']]
data_TotalExpenses['Total Expenses'] = data_TotalExpenses.sum(axis=1)
data['Total Expenses'] = data_TotalExpenses['Total Expenses']
data_updated = data[['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'Transported', 'Total Expenses']]
data_updated[['Deck','Num', 'Side']] = data_updated['Cabin'].str.split('/', expand=True)
data_updated = data_updated[['PassengerId', 'HomePlanet', 'CryoSleep', 'Deck', 'Num', 'Side', 'Destination', 'Age', 'VIP', 'Total Expenses','Transported']]
data_updated.loc[(data_updated['Total Expenses'] == 0) & (data_updated['CryoSleep'].isnull()), 'CryoSleep'] = True
data_updated.loc[(data_updated['Total Expenses'] != 0) & (data_updated['CryoSleep'].isnull()), 'CryoSleep'] = False
data_updated.loc[(data_updated['Age'] < 18) & (data_updated['VIP'].isnull()), 'VIP'] = False
data_updated.loc[(data_updated['HomePlanet'] == 'Earth') & (data_updated['VIP'].isnull()), 'VIP'] = False
data_updated[['Group','Person']] = data_updated['PassengerId'].str.split('_', expand=True)
data_updated['PassengerId'] = data['PassengerId']
data_updated = data_updated[['PassengerId', 'Group', 'HomePlanet', 'CryoSleep', 'Deck', 'Num', 'Side', 'Destination', 'Age', 'VIP', 'Total Expenses','Transported']]
data_updated.loc[:, 'HomePlanet'] = data_updated.groupby('Group')['HomePlanet'].ffill()
data_updated.loc[:, 'HomePlanet'] = data_updated.groupby('Group')['HomePlanet'].bfill()
data_updated.loc[data_updated['Deck'].isin(['A', 'B', 'C', 'T']) & data_updated['HomePlanet'].isnull(), 'HomePlanet'] = 'Europa'
data_updated.loc[(data_updated['Deck'] == 'D') & data_updated['HomePlanet'].isnull(), 'HomePlanet'] = 'Mars'
data_imputed = data_updated.copy()
features = ['HomePlanet','Age', 'Deck', 'Num', 'Side', 'Destination', 'VIP']
data_subset = data_imputed[features]
data_encoded = pd.get_dummies(data_subset, columns=['HomePlanet', 'Deck', 'Side', 'Destination', 'VIP'], dummy_na=False)
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5) 
data_imputed_array = imputer.fit_transform(data_encoded)
data_imputed = pd.DataFrame(data_imputed_array, columns=data_encoded.columns)
categorical_features = ['HomePlanet', 'Deck', 'Side', 'Destination', 'VIP']
for feature in categorical_features:
    #Get a list of columns related to the feature
    feature_cols = [col for col in data_imputed.columns if feature in col]

    #Find the column with the maximum value for each row and extract the category name
    data_imputed[feature] = data_imputed[feature_cols].idxmax(axis=1).str.replace(f'{feature}_', '')

    #Remove the one-hot encoded columns for the current feature
    data_imputed = data_imputed.drop(feature_cols, axis=1)

# Update the original DataFrame with imputed values
for feature in features:
    data_updated[feature] = data_imputed[feature]

data_updated = data_updated[['PassengerId', 'HomePlanet', 'CryoSleep', 'Deck', 'Num', 'Side', 'Destination', 'Age', 'VIP', 'Total Expenses','Transported']]
import category_encoders as ce
target_encoder = ce.TargetEncoder(cols=['HomePlanet', 'Destination', 'Deck', 'CryoSleep', 'Side'])
data_encoded = target_encoder.fit_transform(data_updated[['HomePlanet', 'Destination', 'Deck', 'CryoSleep', 'Side']], data_updated['Transported'])
data_updated['HomePlanet'] = data_encoded['HomePlanet']
data_updated['Destination'] = data_encoded['Destination']
data_updated['Deck'] = data_encoded['Deck']
data_updated['CryoSleep'] = data_encoded['CryoSleep']
data_updated['Side'] = data_encoded['Side']
data_updated['VIP'] = data_updated['VIP'].replace({'True': 1, 'False': 0})

X = data_updated[['PassengerId','HomePlanet','CryoSleep','Deck','Num','Side','Destination','Age','VIP','Total Expenses']]
y = data_updated[['Transported']]
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size = 0.2)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


joblib.dump(model, 'logistic_model.pkl')