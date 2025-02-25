# Implementation of the silhouette coefficient, a quality measure for NLDR embeddings.
# For more details on the measure, see Rousseeuw, P. J. (1987). 
# Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal of computational and applied mathematics, 20, 53-65.

import numpy as np

from sklearn.metrics import silhouette_score

# The higher the score, the best is the visualization.
def compute(visu, labels):
	return silhouette_score(visu, labels)
