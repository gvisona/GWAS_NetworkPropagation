import numpy as np

ALPHAS = np.concatenate(([0.0],np.linspace(0.001, 0.01, 10), np.linspace(0.02, 0.9, 89)))

Ks = [20, 50, 100]
NETWORKS = ["BioPlex3", "HumanNet", "PCNet", "ProteomeHD", "STRING"]
DISEASES = ["asthma", "autism", "schizophrenia"]