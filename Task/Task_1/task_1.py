# Import of libraries

import numpy as np
import pandas as pd


# Creation of the integer array 3x4 using the numpy library
A = np.random.randint(10, size=(4,4))

# Rank of the matrix
rank_matrix = np.linalg.matrix_rank(A)

# Trace of the matrix
trace_matrix = np.trace(A)

# Determinant of the matrix
det_matrix = np.linalg.det(A)

# inverse of the matrix
inv_matrix = np.linalg.inv(A)

ATA = A.T @ A  # product A^T*A
AAT = A @ A.T  # product A*A^T

# Eigenvalues and eigenvectors of A^TA
eigenvalues_ATA, eigenvectors_ATA = np.linalg.eig(ATA)

# Eigenvalues and eigenvectors of AA^T
eigenvalues_AAT, eigenvectors_AAT = np.linalg.eig(AAT)

# Results
print("Rank of the matrix: ", rank_matrix)
print("Trace of the matrix: ", trace_matrix)
print("Determinant: ",det_matrix)
print("\nInverse: ", inv_matrix)
print("\nEigenvalues of A^TA: ",eigenvalues_ATA)
print("\nEigenvectors of A^TA :", eigenvectors_ATA)
print("Eigenvalues of AA^T: ", eigenvalues_AAT)
print("\nEigenvectors of AA^T: ", eigenvectors_AAT)