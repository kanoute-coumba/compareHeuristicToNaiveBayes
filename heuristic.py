import pandas as pd  
import requests  
from io import BytesIO  # Importer BytesIO pour traiter les fichiers en mémoire
from urllib.parse import urlparse  # Importer urlparse pour analyser les URL

# URL du fichier Excel sur GitHub (remplacez par l'URL correcte)
url = "https://github.com/kanoute-coumba/compareHeuristicToNaiveBayes/blob/main/phishing_url.xlsx"

# Téléchargement du fichier Excel depuis GitHub
response = requests.get(url)  # Effectuer une requête GET pour télécharger le fichier
file = BytesIO(response.content)  # Lire le contenu de la réponse dans un objet BytesIO

# Chargement des données : lecture du fichier Excel depuis le contenu téléchargé
data = pd.read_excel(file)  # Lire le fichier Excel en un DataFrame pandas

# Fonction pour extraire les caractéristiques nécessaires des URLs
def extract_features(url):
    parsed_url = urlparse(url)  # Analyser l'URL pour extraire ses composants
    url_length = len(url)  # Longueur de l'URL
    num_dots = url.count('.')  # Nombre de points dans l'URL
    num_slashes = url.count('/')  # Nombre de barres obliques dans l'URL
    has_at_symbol = '@' in url  # Vérifier la présence du symbole '@'
    https_check = 1 if parsed_url.scheme == 'https' else 0  # Vérifier si l'URL utilise HTTPS
    has_special_chars = any(char in url for char in ['-', '_', ',', ';'])  # Vérifier la présence de caractères spéciaux

    return {
        'URL Length': url_length,
        'Num Dots': num_dots,
        'Num Slashes': num_slashes,
        'Has @ Symbol': has_at_symbol,
        'HTTPS': https_check,
        'Has Special Chars': has_special_chars
    }

# Extraire les caractéristiques pour chaque URL
data['features'] = data['URL'].apply(extract_features)  # Appliquer la fonction extract_features à chaque URL

# Fonction heuristique pour déterminer si une URL est phishing
def heuristic_prediction(features):
    # Règle 1 : Longueur de l'URL
    if features['URL Length'] < 54:
        url_length_feature = 'NotLong'
    elif 54 <= features['URL Length'] <= 75:
        url_length_feature = 'Suspicious'
    else:
        url_length_feature = 'VeryLong'

    # Règle 2 : Nombre de points et barres obliques
    if features['Num Dots'] >= 5 or features['Num Slashes'] >= 5:
        url_complexity_feature = 'Phishy'
    else:
        url_complexity_feature = 'Legitimate'

    # Règle 3 : Présence du symbole @
    at_symbol_feature = 'True' if features['Has @ Symbol'] else 'False'

    # Règle 4 : Vérification HTTP et SSL
    if features['HTTPS'] == 1:
        # Supposons que nous avons une information sur l'âge et l'émetteur du certificat, par exemple dans un champ 'SSL Info'
        # Pour simplification, nous ne l'utilisons pas ici
        http_ssl_feature = 'Low'
    else:
        http_ssl_feature = 'High'

    # Règle 5 : Caractères spéciaux
    special_chars_feature = 'Suspicious' if features['Has Special Chars'] else 'Legitimate'

    # Décision heuristique basée sur les règles appliquées
    if (url_length_feature == 'VeryLong' or 
        url_complexity_feature == 'Phishy' or 
        at_symbol_feature == 'True' or 
        http_ssl_feature == 'High' or 
        special_chars_feature == 'Suspicious'):
        return 0  # Classer comme phishing
    else:
        return 1  # Classer comme non phishing

# Appliquer les règles heuristiques aux données
data['prediction'] = data['features'].apply(heuristic_prediction)  # Appliquer la fonction heuristic_prediction aux caractéristiques de chaque URL

# Calcul des métriques de performance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

y = data['label']  # Vérités terrain
y_pred_heuristic = data['prediction']  # Prédictions heuristiques

accuracy = accuracy_score(y, y_pred_heuristic)  # Calcul de l'accuracy
precision = precision_score(y, y_pred_heuristic, pos_label=0)  # Calcul de la précision pour la classe 'phishing'
recall = recall_score(y, y_pred_heuristic, pos_label=0)  # Calcul du rappel pour la classe 'phishing'
f1 = f1_score(y, y_pred_heuristic, pos_label=0)  # Calcul du score F1 pour la classe 'phishing'

cm = confusion_matrix(y, y_pred_heuristic, labels=[0, 1])  # Calcul de la matrice de confusion
TP = cm[0, 0]  # Vrais Positifs
FP = cm[0, 1]  # Faux Positifs
FN = cm[1, 0]  # Faux Négatifs
TN = cm[1, 1]  # Vrais Négatifs

# Affichage des résultats
print(f'P(phishing): {y.value_counts(normalize=True)[0]}')  # Probabilité a priori pour le phishing
print(f'P(non phishing): {y.value_counts(normalize=True)[1]}')  # Probabilité a priori pour le non-phishing
print(f'Accuracy: {accuracy}')  # Affichage de l'accuracy
print(f'Precision: {precision}')  # Affichage de la précision
print(f'Recall: {recall}')  # Affichage du rappel
print(f'F1 Score: {f1}')  # Affichage du score F1
print(f'Vrais Positifs (TP): {TP} ({TP / len(y) * 100:.2f}%)')  # Affichage des vrais positifs avec pourcentage
print(f'Faux Positifs (FP): {FP} ({FP / len(y) * 100:.2f}%)')  # Affichage des faux positifs avec pourcentage
print(f'Faux Négatifs (FN): {FN} ({FN / len(y) * 100:.2f}%)')  # Affichage des faux négatifs avec pourcentage
print(f'Vrais Négatifs (TN): {TN} ({TN / len(y) * 100:.2f}%)')  # Affichage des vrais négatifs avec pourcentage
