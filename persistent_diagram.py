#first run token_space.py script to produce the output and input matrix of token embeddings
import numpy as np
from sklearn.manifold import Isomap
from scipy.spatial.distance import pdist, squareform

distances=[]
distances_intra=[]
for i in range(len(output_matrix)):
  distances_i = pdist(output_matrix[i])
  #distances_i = squareform(distances_i)
  distances_intra_i = squareform(distances_i)

# clustering based on dimension [n, n+1)
  class_assignments = []
  for value in dim_output[i]:
    # Assigner à la classe [floor(value), floor(value)+1)
      class_assignments.append(np.floor(value))  # La classe est déterminée par la partie entière de la valeur

# Convertir les résultats en tableau numpy pour plus de facilité
  class_assignments = np.array(class_assignments)


#  Remplace inter cluster token distance with infinite distance
  for i in range(len(class_assignments)):
    for j in range(len(class_assignments)):
        if class_assignments[i] != class_assignments[j]:
            distances_intra_i[i, j] = np.inf  # Remplacer la distance par l'infini si les classes sont différentes
  distances_ii = squareform(distances_i)
  distances.append(distances_ii)
  distances_intra.append(distances_intra_i)


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
for i in range(len(output_matrix)):
  rips_complex_i = gudhi.RipsComplex(distance_matrix=distances_intra[i], max_edge_length=1000)
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
diag_and = diag[0:5]
diag_bracket = diag[6:11]
diag_colon = diag[12:17]
diag_period = diag[18:23]
flat_diag_and = [point for sublist in diag_and for point in sublist]
flat_diag_bracket = [point for sublist in diag_bracket for point in sublist]
flat_diag_colon = [point for sublist in diag_colon for point in sublist]
flat_diag_period = [point for sublist in diag_period for point in sublist]
gudhi.plot_persistence_diagram(flat_diag_and)  # ax=axes[0] pour le premier sous-graphe
gudhi.plot_persistence_diagram(flat_diag_bracket)  # ax=axes[0] pour le premier sous-graphe
gudhi.plot_persistence_diagram(flat_diag_colon)  # ax=axes[0] pour le premier sous-graphe
gudhi.plot_persistence_diagram(flat_diag_period)  # ax=axes[0] pour le premier sous-graphe
plt.show()

