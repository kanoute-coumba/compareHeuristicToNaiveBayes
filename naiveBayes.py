import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Chrgement des données : lecture du fichier Excel
data = pd.read_excel('phishing_url.xlsx')

# Sélection des caractéristiques et du label
features = ['IsDomainIP', 'HasObfuscation', 'IsHTTPS', 'HasExternalFormSubmit', 'HasCopyrightInfo']
X = data[features]
y = data['label']

# Calcul des probabilités a priori
P_phishing = y.value_counts(normalize=True)[0]
P_non_phishing = y.value_counts(normalize=True)[1]

print(f'P(phishing): {P_phishing}')
print(f'P(non phishing): {P_non_phishing}')

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisation et entraînement du modèle
model = MultinomialNB()
model.fit(X_train, y_train)

# Prédiction des étiquettes
y_pred = model.predict(X_test)

# Calcul des métriques
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=0)
recall = recall_score(y_test, y_pred, pos_label=0)
f1 = f1_score(y_test, y_pred, pos_label=0)

# Calcul de la matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

# Extraction des valeurs
TP = cm[0, 0]  # Vrais Positifs (phishing correctement classés)
FP = cm[0, 1]  # Faux Positifs (sites sûrs incorrectement classés comme phishing)
FN = cm[1, 0]  # Faux Négatifs (phishing incorrectement classés comme sites sûrs)
TN = cm[1, 1]  # Vrais Négatifs (sites sûrs correctement classés)

# Affichage des résultats
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Vrais Positifs (TP): {TP}')
print(f'Faux Positifs (FP): {FP}')
print(f'Faux Négatifs (FN): {FN}')
print(f'Vrais Négatifs (TN): {TN}')
