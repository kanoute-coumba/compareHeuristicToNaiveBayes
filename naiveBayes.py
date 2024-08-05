import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Chargement des données : lecture du fichier Excel
data = pd.read_excel('phishing_url.xlsx')

# Sélection des caractéristiques et du label
features = ['IsDomainIP', 'HasObfuscation', 'IsHTTPS', 'HasExternalFormSubmit', 'HasCopyrightInfo']
X = data[features]
y = data['label']

# Calcul des probabilités a priori pour les classes 'phishing' et 'non phishing'
P_phishing = y.value_counts(normalize=True)[0]  # Probabilité a priori de la classe 'phishing'
P_non_phishing = y.value_counts(normalize=True)[1]  # Probabilité a priori de la classe 'non phishing'

# Affichage des probabilités a priori
print(f'P(phishing): {P_phishing:.2f}')
print(f'P(non phishing): {P_non_phishing:.2f}')

# Séparation des données en ensembles d'entraînement et de test (80% entraînement, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisation et entraînement du modèle Naive Bayes multinomial
model = MultinomialNB()
model.fit(X_train, y_train)  # Entraînement du modèle avec les données d'entraînement

# Prédiction des étiquettes pour les données de test
y_pred = model.predict(X_test)

# Calcul des métriques de performance du modèle
accuracy = accuracy_score(y_test, y_pred)  # Précision globale
precision = precision_score(y_test, y_pred, pos_label=0)  # Précision pour la classe 'phishing'
recall = recall_score(y_test, y_pred, pos_label=0)  # Rappel pour la classe 'phishing'
f1 = f1_score(y_test, y_pred, pos_label=0)  # Score F1 pour la classe 'phishing'

# Calcul de la matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

# Extraction des valeurs de la matrice de confusion
TP = cm[0, 0]  # Vrais Positifs (phishing correctement classés)
FP = cm[1, 0]  # Faux Positifs (phishing incorrectement classés comme sites sûrs)
FN = cm[0, 1]  # Faux Négatifs (sites sûrs incorrectement classés comme phishing)
TN = cm[1, 1]  # Vrais Négatifs (sites sûrs correctement classés)

# Assurer que la somme des TP, FP, FN et TN correspond au nombre total des exemples testés
total = len(y_test)
assert TP + FP + FN + TN == total, "La somme des TP, FP, FN et TN ne correspond pas au nombre total des URLs."

# Affichage des résultats des métriques de performance
print(f'Accuracy: {accuracy:.2f}')  # Précision globale
print(f'Precision: {precision:.2f}')  # Précision pour la classe 'phishing'
print(f'Recall: {recall:.2f}')  # Rappel pour la classe 'phishing'
print(f'F1 Score: {f1:.2f}')  # Score F1 pour la classe 'phishing'

# Affichage des valeurs de la matrice de confusion avec les pourcentages
print(f'Vrais Positifs (TP): {TP} ({TP/total*100:.2f}%)')
print(f'Faux Positifs (FP): {FP} ({FP/total*100:.2f}%)')
print(f'Faux Négatifs (FN): {FN} ({FN/total*100:.2f}%)')
print(f'Vrais Négatifs (TN): {TN} ({TN/total*100:.2f}%)')