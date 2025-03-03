from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Charger le modèle et le tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

"""Inputs to import"""

from google.colab import files

# Cela ouvrira un sélecteur de fichiers pour téléverser ton fichier
uploaded = files.upload()

# Une fois téléchargé, tu peux lire le fichier CSV (en supposant qu'il s'appelle 'Act_as_a.csv')
import pandas as pd

# Remplace 'Act_as_a.csv' par le nom du fichier téléchargé
df = pd.read_csv('Act_as_a_modif.csv',index_col=None,header=None,sep=';')
#or use df generated in prompt_generation.py script
# Afficher les premières lignes du tableau
print(df)

"""Definition des fonctions"""

def generate_response(input_text):
    """
    Génère une réponse à partir d'un texte d'entrée en utilisant Llama-3.2-1B et renvoie la réponse générée,
    les matrices des embeddings des tokens de l'input et de l'output, ainsi que les tokens correspondants.
    """
    # Encoder le texte d'entrée
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    input_length = input_ids.shape[-1]

    # Convertir les input_ids en tokens textuels
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])  # Liste des tokens pour l'entrée

    # Récupérer les hidden states pour le texte d'entrée
    with torch.no_grad():
        input_outputs = model(input_ids, output_hidden_states=True)

    # Hidden states pour les tokens d'entrée (dernière couche)
    input_embeddings = input_outputs.hidden_states[-1]  # Shape: (1, input_seq_len, hidden_size)

    # Retirer la dimension du batch pour obtenir une matrice
    input_matrix = input_embeddings.squeeze(0)  # Shape: (input_seq_len, hidden_size)

    # Générer une réponse avec les paramètres ajustés
    output_ids = model.generate(
        input_ids,
        max_length=100,  # Augmenter la longueur maximale de la séquence générée
        pad_token_id=tokenizer.eos_token_id
    )

    # Décoder et générer le texte complet
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Si vous souhaitez uniquement le texte généré après l'input
    generated_text_after_input = generated_text[len(input_text):]

    # Récupérer les hidden states pour le texte généré
    with torch.no_grad():
        output_outputs = model(output_ids, output_hidden_states=True)

    # Hidden states pour les tokens générés (dernière couche)
    output_embeddings = output_outputs.hidden_states[-1]  # Shape: (1, output_seq_len, hidden_size)

    # Retirer la dimension du batch pour obtenir des matrices
    output_matrix = output_embeddings.squeeze(0)  # Shape: (output_seq_len, hidden_size)
    output_matrix = output_matrix[len(input_matrix):]

    # Convertir les output_ids en tokens textuels
    output_tokens = tokenizer.convert_ids_to_tokens(output_ids[0])  # Liste des tokens pour la sortie
    output_tokens = output_tokens[len(input_tokens):]

    return generated_text_after_input, input_matrix, output_matrix, input_tokens, output_tokens

import numpy as np
from scipy.spatial.distance import pdist, squareform
#pour des phrases ayant au moins 10 tokens (input sentences)
def least_squares_estimates(input_matrix):
    """
    Calcul complet de la courbure de Ricci (R_icj) à partir d'une matrice d'entrée.

    Paramètres :
    ------------
    input_matrix : np.ndarray
        Matrice d'entrée où chaque ligne représente un token et chaque colonne ses dimensions/features.

    Retour :
    --------
    dict :
        - Ricci_scalar : np.ndarray
            Valeurs de la courbure de Ricci pour chaque ligne.
        - n_values : np.ndarray
            Exposants n pour chaque ligne.
    """
    if isinstance(input_matrix, torch.Tensor):
        input_matrix = input_matrix.cpu().numpy()
    # Étape 1 : Calcul des distances euclidiennes 2 à 2
    distances_condensed = pdist(input_matrix, metric='euclidean')  # Vecteur des distances
    distances_matrix = squareform(distances_condensed)  # Matrice des distances
    sorted_distances_matrix = np.sort(distances_matrix, axis=1)  # Trier les distances pour chaque ligne
    matrix_without_first_column = sorted_distances_matrix[:, 1:]  # Supprimer la première colonne (distance à soi-même)
    log_r = np.log(matrix_without_first_column)

    # Étape 2 : Subset des rayons et création du vecteur de volumes
    volume = np.arange(1, len(input_matrix), 1)  # Vecteur des volumes de 1 à 10 car input a un petit nombre de tokens
    log_v = np.log(volume)  # Logarithme des volumes
    log_r_subset = log_r  # Rayon 10 à 49 (40 valeurs)


    # Étape 3 : Estimation des paramètres (log_K_values, n_values, std_errors_all)
    log_K_values = []
    n_values = []
    #std_errors_all = []

    for i in range(log_r_subset.shape[0]):
        # Matrice X pour la régression linéaire
        X = np.vstack([np.ones(log_r_subset.shape[1]), log_r_subset[i, :]]).T
        beta = np.linalg.inv(X.T @ X) @ X.T @ log_v  # Coefficients de régression
        #residuals = log_v - X @ beta
        n, p = X.shape
        #for a corrected K :
        #sigma_squared = np.sum(residuals**2) / (n - p)  # Variance résiduelle
        #cov_beta = sigma_squared * np.linalg.inv(X.T @ X)  # Matrice de covariance
        #std_errors = np.sqrt(np.diag(cov_beta))  # Erreurs standards des paramètres
        #std_errors_all.append(std_errors)
        log_K_values.append(beta[0])
        n_values.append(beta[1])
        

    std_errors_all = np.array(std_errors_all)
    n_values = np.array(n_values)
    #K_prime = np.exp(log_K_values) * np.exp((std_errors_all[:, 0]**2) / 2)
    #log_K_prime = np.log(K_prime)

    # Étape 4 : Calcul de la courbure de Ricci (R_icj)
    R_icj_values = []
    for j in range(log_r_subset.shape[0]):
        summation = 0
        for i in range(log_r_subset.shape[1]):
            log_r_ij = log_r_subset[j, i]
            log_v_ij = log_v[i]
            r_ij_squared = np.exp(2 * log_r_ij)
            term = (6 * n_values[j] + 2) / r_ij_squared
            term *= (log_K_values[j] + n_values[j] * log_r_ij - log_v_ij)
            summation += term
        p = log_r_subset.shape[1]
        R_icj = summation / p
        R_icj_values.append(R_icj)

    R_icj_array = np.array(R_icj_values)

    # Retourner uniquement Ricci_scalar et n_values
    return R_icj_array,n_values,log_K_values, #K_prime

"""Loop to retreive local dim, K and Ricci for each input sentences"""

# Définir une liste ou un tableau pour stocker les résultats
import torch

# Fixer une seed pour PyTorch
torch.manual_seed(42)


response = []
input_token=[]
output_token=[]
input_matrix=[]
output_matrix=[]
Ricci_input=[]
Ricci_output=[]
dim_input=[]
dim_output=[]
log_K_prime_in=[]
log_K_prime_out=[]
K_prime_in=[]
K_prime_out=[]

# Boucle sur les lignes du DataFrame
for i in range(len(df)):  # Range jusqu'au nombre de lignes dans le DataFrame
    # Accéder à l'élément de la première colonne à la ligne i
    input_text_i = df.iloc[i, 0]  # Première colonne à la ligne i

    # Appeler la fonction generate_response() avec input_text_i
    response_i, input_matrix_i, output_matrix_i, input_token_i, output_token_i = generate_response(input_text_i)

    #least squares estimates for each token in input sentence and output one
    Ric_input_i, dim_input_i, log_K_prime_in_i, K_prime_in_i = least_squares_estimates(input_matrix_i)
    Ric_output_i, dim_output_i, log_K_prime_out_i, K_prime_out_i = least_squares_estimates(output_matrix_i)

    response.append(response_i)
    input_matrix.append(input_matrix_i)
    output_matrix.append(output_matrix_i)
    output_token.append(output_token_i)
    input_token.append(input_token_i)
    Ricci_input.append(Ric_input_i)
    Ricci_output.append(Ric_output_i)
    dim_input.append(dim_input_i)
    dim_output.append(dim_output_i)
    log_K_prime_in.append(log_K_prime_in_i)
    log_K_prime_out.append(log_K_prime_out_i)
    K_prime_in.append(K_prime_in_i)
    K_prime_out.append(K_prime_out_i)

response=np.array(response,dtype=object)
input_token=np.array(input_token,dtype=object)
output_token=np.array(output_token,dtype=object)
Ricci_input=np.array(Ricci_input,dtype=object)
Ricci_output=np.array(Ricci_output, dtype=object)
dim_input=np.array(dim_input,dtype=object)
dim_output=np.array(dim_output,dtype=object)
log_K_prime_in=np.array(log_K_prime_in,dtype=object)
log_K_prime_out=np.array(log_K_prime_out,dtype=object)
K_prime_in=np.array(K_prime_in,dtype=object)
K_prime_out=np.array(K_prime_out,dtype=object)

input_matrix = [tensor.cpu().numpy() for tensor in input_matrix]
output_matrix = [tensor.cpu().numpy() for tensor in output_matrix]

