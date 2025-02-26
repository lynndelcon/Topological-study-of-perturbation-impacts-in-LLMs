#first run token_space.py script to produce the output and input matrix of token embeddings
import numpy as np
from sklearn.manifold import Isomap
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
distances=[]

for i in range(len(matrix)):
  distances_i=squareform(pdist(matrix))
  distances.append(distances_i)



isomap = Isomap(n_neighbors=2, n_components=2)#k=2=log(N) en dimension 2  # Choisir un nombre de voisins adapté à vos données


geodesic=[]
for i in range(len(oput_matrix)):
  data_isomap_i = isomap.fit_transform(matrix[i])
  geodesic_i = isomap.dist_matrix_
  geodesic.append(geodesic_i)



"""Persistent image"""

from itertools import product

import time
import numpy as np
from sklearn import datasets
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt


import gudhi


import math

###intra-strata
diag=[]
for i in range(len(matrix)):
  rips_complex_i = gudhi.RipsComplex(distance_matrix=distances[i], max_edge_length=1000)
# Construire le complexe simplicial
  simplex_tree_i = rips_complex_i.create_simplex_tree(max_dimension=4)

# Calcul du diagramme de persistance
  diag_i = simplex_tree_i.persistence()

  diag.append(diag_i)

#diag=list

flat_diag = [point for sublist in diag for point in sublist]
gudhi.plot_persistence_diagram(flat_diag)  # ax=axes[0] pour le premier sous-graphe
plt.show()



#separation of the 4 prompts


#Paired test version of the NHST from Robinson and Turner (2017)
import numpy as np
from gudhi.wasserstein import wasserstein_distance
from itertools import combinations


group_names = ["and", "bracket", "colon", "period"]
# séparer les diagrammes selon les dimensions des trous 
pdgms_0= []
pdgms_1= []
pdgms_2= []

# Créer un diagramme de persistance (array) pour chaque dimension
for i in range(len(diag)):  # Inclut les dimensions de 0 à max_dimension
      pdgm_0_i = np.array([pair[1] for pair in diag[i] if pair[0] == 0])  # Filtrer par dimension
      pdgm_1_i = np.array([pair[1] for pair in diag[i] if pair[0] == 1])
      pdgm_2_i = np.array([pair[1] for pair in diag[i] if pair[0] == 2])
      pdgms_0.append(pdgm_0_i)
      pdgms_1.append(pdgm_1_i)
      pdgms_2.append(pdgm_2_i)
#diag_2=handle_empty_sublist(pdgms_2)
#séparer les diagrammes dimensionels par groupe
d_0_and = pdgms_0[0:5]
d_0_bracket = pdgms_0[6:11]
d_0_colon = pdgms_0[12:17]
d_0_period = pdgms_0[18:23]
d_1_and = pdgms_1[0:5]
d_1_bracket = pdgms_1[6:11]
d_1_colon = pdgms_1[12:17]
d_1_period = pdgms_1[18:23]
d_2_and = pdgms_2[0:5]
d_2_bracket = pdgms_2[6:11]
d_2_colon = pdgms_2[12:17]
d_2_period = pdgms_2[18:23]

groups_0 = [d_0_and, d_0_bracket, d_0_colon, d_0_period]
groups_1 = [d_1_and, d_1_bracket, d_1_colon, d_1_period]
groups_2 = [d_2_and, d_2_bracket, d_2_colon, d_2_period]

import numpy as np

def permutation_paired_test(diagrams1, diagrams2, num_permutations=1000):
    """
    Test de permutation conditionnelle pour données appariées.
    Ce test permet d'appliquer un signe négatif à chaque distance avec probabilité 1/2,
    comme dans Robinson et Turner 2017.
    """
    distances = [wasserstein_distance(d1, d2) for d1, d2 in zip(diagrams1, diagrams2)]
    stat_obs = np.mean(distances)

    # Distribution nulle via permutations
    perm_stats = []
    for _ in range(num_permutations):
        shuffled_diagrams = diagrams1.copy()  # Copie des diagrammes pour éviter d'écraser les données
        np.random.shuffle(shuffled_diagrams)  # Mélange des paires

        # Calcul des distances permutées
        permuted_distances = [wasserstein_distance(d1, d2) for d1, d2 in zip(shuffled_diagrams, diagrams2)]
        
        # Appliquer un signe aléatoire (positif ou négatif) avec probabilité 1/2
        random_signs = np.random.choice([-1, 1], size=len(permuted_distances))  # Choix aléatoire entre -1 et 1
        permuted_distances_with_sign = np.multiply(permuted_distances, random_signs)  # Multiplier les distances par le signe

        perm_stats.append(np.mean(permuted_distances_with_sign))

    # Calcul de la p-valeur empirique
    p_value = np.mean(np.array(perm_stats) >= stat_obs)
    return p_value

# Comparaisons 2 à 2 et correction de Bonferroni
p_values = []
comparisons = list(combinations(range(4), 2))  # Toutes les paires (i, j)

for i, j in comparisons:
    p = permutation_paired_test(groups_0[i], groups_0[j])
    p_values.append((group_names[i], group_names[j], p))

# Correction de Bonferroni
p_val_corr = [(g1, g2, p * 6 if p * 6 <= 1 else 1) for g1, g2, p in p_values]
# Affichage des résultats
for g1, g2, p in p_val_corr:
    print(f"Comparaison {g1} vs {g2}: p-value corrigée = {p:.4f}")

# Comparaisons 2 à 2 et correction de Bonferroni
p_values = []
comparisons = list(combinations(range(4), 2))  # Toutes les paires (i, j)

for i, j in comparisons:
    p = permutation_paired_test(groups_1[i], groups_1[j])
    p_values.append((group_names[i], group_names[j], p))

# Correction de Bonferroni
corrected_p_values = [(g1, g2, p * 6 if p * 6 <= 1 else 1) for g1, g2, p in p_values]

# Affichage des résultats
for g1, g2, p in corrected_p_values:
    print(f"Comparaison {g1} vs {g2}: p-value corrigée = {p:.4f}")
