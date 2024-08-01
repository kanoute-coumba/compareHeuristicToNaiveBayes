import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Chargement des données
#data = pd.read_csv('phishing_url.xlsx')

# Lecture du fichier Excel
data = pd.read_excel('phishing_url.xlsx')

# Affichage des premières lignes du dataframe
print(data.head())

# Sélection des caractéristiques
X = data[['IsDomainIP', 'HasObfuscation', 'IsHTTPS', 'HasExternalFormSubmit', 'HasCopyrightInfo']]
y = data['label']

# Séparation en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle Naïve Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Calcul des métriques de performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
