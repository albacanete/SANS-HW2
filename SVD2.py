# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 11:13:59 2021

@author: jorge
"""

import matplotlib.pyplot as plt
import numpy as np

import scipy.io

# set matplotlib settings
plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})

# plt.rcParams['figure.figsize'] = [10, 10] # already done before
# plt.rcParams.update({'font.size': 18}) # already done before

# Explore the faces in allFaces.mat
mat_contents = scipy.io.loadmat('allFaces.mat')
faces = mat_contents['faces']
# m = 168, n = 192, nxm b/w pixels
m = int(mat_contents['m'])
n = int(mat_contents['n'])
nfaces = np.ndarray.flatten(mat_contents['nfaces'])

allPersons = np.zeros((n * 6, m * 6))
count = 0

for j in range(6):
    for k in range(6):
        allPersons[j * n: (j + 1) * n, k * m: (k + 1) * m] = np.reshape(faces[:, np.sum(nfaces[:count])], (m, n)).T
        count += 1

img = plt.imshow(allPersons)
img.set_cmap('gray')
plt.axis('off')
plt.show()

# We use the first 36 people for training data
trainingFaces = faces[:, :np.sum(nfaces[:36])]
avgFace = np.mean(trainingFaces, axis=1)  # size n*m by 1

# Compute eigenfaces on mean-subtracted training data
X = trainingFaces - np.tile(avgFace, (trainingFaces.shape[1], 1)).T
U, S, VT = np.linalg.svd(X, full_matrices=0)

fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
img_avg = ax1.imshow(np.reshape(avgFace, (m, n)).T)
img_avg.set_cmap('gray')
plt.axis('off')

ax2 = fig1.add_subplot(122)
img_u1 = ax2.imshow(np.reshape(U[:, 0], (m, n)).T)
img_u1.set_cmap('gray')
plt.axis('off')

plt.show()

## Now show eigenface reconstruction of image that was omitted from test set
testFace = faces[:, np.sum(nfaces[:36])]  # First face of person 37
plt.imshow(np.reshape(testFace, (m, n)).T)
plt.set_cmap('gray')
plt.title('Original Image')
plt.axis('off')
plt.show()

testFaceMS = testFace - avgFace
r_list = [25, 50, 100, 200, 400, 800, 1600]  # 1600 takes a lot of CPU time

for r in r_list:
    reconFace = avgFace + U[:, :r] @ U[:, :r].T @ testFaceMS
    img = plt.imshow(np.reshape(reconFace, (m, n)).T)
    img.set_cmap('gray')
    plt.title('r = ' + str(r))
    plt.axis('off')
    plt.show()

## Project person 2 and 7 onto PC5 and PC6
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
plt.show()
