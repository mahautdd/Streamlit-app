import pandas as pd 
#Lecture du fichier préparé dans un DataFrame
df=pd.read_csv("df_prep.csv", sep=",")

#Séparation du jeu de données
feats = df.drop('Comptage_h', axis=1)
target = df['Comptage_h']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25, random_state=42)

#Encodage de 'Orientation' (variable catégorielles)
X_train_reshaped = X_train[['Orientation']]
X_test_reshaped = X_test[['Orientation']]
from sklearn.preprocessing import OneHotEncoder
oneh = OneHotEncoder( drop="first", sparse_output=False)
X_train_encoded = oneh.fit_transform(X_train_reshaped)
X_test_encoded = oneh.transform(X_test_reshaped)
feature_names = oneh.get_feature_names_out(X_train_reshaped.columns)
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=feature_names, index=X_train.index)
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=feature_names, index=X_test.index)
X_train = pd.concat([X_train.drop('Orientation', axis=1), X_train_encoded_df], axis=1)
X_test = pd.concat([X_test.drop('Orientation', axis=1), X_test_encoded_df], axis=1)

# Standardisation des données
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled= scaler.transform(X_test)

# Entraînement des différents modèles et calcul des métriques
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(X_train_scaled, y_train)
Score1_train = model1.score(X_train_scaled, y_train)
Score1_test = model1.score(X_test_scaled, y_test)
MAE1 = mean_absolute_error(y_test, model1.predict(X_test_scaled))

from sklearn.tree import DecisionTreeRegressor
model2 = DecisionTreeRegressor()
model2.fit(X_train_scaled, y_train)
MAE2 = mean_absolute_error(y_test, model2.predict(X_test_scaled))

model2bis = DecisionTreeRegressor(min_samples_leaf = 5, random_state=42)
model2bis.fit(X_train_scaled, y_train)
MAE2bis = mean_absolute_error(y_test, model2bis.predict(X_test_scaled))

from sklearn.ensemble import RandomForestRegressor
model3 = RandomForestRegressor()
model3.fit(X_train_scaled, y_train)
MAE3 = mean_absolute_error(y_test, model3.predict(X_test_scaled))

# Sauvegarde dans joblib
from joblib import dump
dump(model1, 'model1.joblib')
dump(Score1_train, 'Score1_train.joblib')
dump(Score1_test, 'Score1_test.joblib')
dump(MAE1, 'MAE1.joblib')
dump(model2, 'model2.joblib')
dump(MAE2, 'MAE2.joblib')
dump(model2bis, 'model2bis.joblib')
dump(MAE2bis, 'MAE2bis.joblib')
dump(model3, 'model3.joblib')
dump(MAE3, 'MAE3.joblib')
dump(X_train, 'X_train.joblib')