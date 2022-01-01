import matplotlib.pyplot as plt
import numpy as np
import scipy.io

mat_contents = scipy.io.loadmat('allFaces.mat')
faces = mat_contents['faces']
# m = 168, n = 192, nxm b/w pixels
m = int(mat_contents['m'])
n = int(mat_contents['n'])
nfaces = np.ndarray.flatten(mat_contents['nfaces'])
trainingFaces = faces[:, :np.sum(nfaces[:36])]
avgFace = np.mean(trainingFaces, axis=1)  # size n*m by 1
X = trainingFaces - np.tile(avgFace, (trainingFaces.shape[1], 1)).T
U, S, VT = np.linalg.svd(X, full_matrices=0)

# Project person 2 and 7 onto PC5 and PC6
P1num = 2  # Person number 2
P2num = 7  # Person number 7

P1 = faces[:, np.sum(nfaces[:(P1num - 1)]):np.sum(nfaces[:P1num])]
P2 = faces[:, np.sum(nfaces[:(P2num - 1)]):np.sum(nfaces[:P2num])]

P1 = P1 - np.tile(avgFace, (P1.shape[1], 1)).T
P2 = P2 - np.tile(avgFace, (P2.shape[1], 1)).T

PCAmodes = [5, 6]  # Project onto PCA modes 5 and 6
PCACoordsP1 = U[:, PCAmodes - np.ones_like(PCAmodes)].T @ P1
PCACoordsP2 = U[:, PCAmodes - np.ones_like(PCAmodes)].T @ P2

plt.plot(PCACoordsP1[0, :], PCACoordsP1[1, :], 'd', Color='k', label='Person 2')
plt.plot(PCACoordsP2[0, :], PCACoordsP2[1, :], '^', Color='r', label='Person 7')

plt.legend()
plt.savefig("projection.png")
plt.show()
