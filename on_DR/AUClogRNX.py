# Implementation of AUClogRNX, a quality measure for NLDR embeddings.
# For more details on the measure, see Lee, J. A., Peluffo-Ordonez, D. H., & Verleysen, M. (2015). 
# Multi-scale similarities in stochastic neighbour embedding: Reducing dimensionality while preserving both local and global structure. Neurocomputing, 169, 246-261.
# This implementation has been written by Adrien Bibal (University of Namur).

import math
import pickle
import numpy as np

from scipy.spatial.distance import pdist, squareform
from scipy.spatial import KDTree

def QNX_K(dataset, projection, vK, nK):
	acc = 0.0
	N   = len(projection)
	K   = len(vK[0])

	acc = sum(map(lambda vnK: np.intersect1d(vnK[0], vnK[1]).size, zip(vK, nK)))

	return acc / (K*N)

def RNX_K(dataset, projection, vK, nK):
	N   = len(projection)
	K   = len(vK[0])

	QNX = QNX_K(dataset, projection, vK, nK)

	numerator   = ((N - 1)*QNX) - K
	denominator = N - 1 - K

	return numerator / denominator

def logRNX(dataset, projection):
	N   = len(projection)

	D_dataset    = squareform(pdist(dataset))
	D_projection = squareform(pdist(projection))

	numerator   = 0.0
	denominator = 0.0

	I_dataset = np.argsort(D_dataset, 1)[:, 1:]
	I_projection = np.argsort(D_projection, 1)[:, 1:]

	for k in range(1, N-1):
		vK = I_projection[:, :k]
		nK = I_dataset[:, :k]

		numerator += (RNX_K(dataset, projection, vK, nK) / k)
		denominator += (1.0 / k)

	return numerator / denominator

def compute(data, visu):
	""" Compute AUClogRNX
	Inputs
	------
	data = high dimensional data
	visu = low dimensional data
	Output
	------
	return the log of the AUC of K neighborhoods for a growing K
	"""

	return logRNX(data, visu)