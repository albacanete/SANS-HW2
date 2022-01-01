import matplotlib.pyplot as plt
import numpy as np
import scipy.io


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

plt.savefig('allPersons.png')
plt.show()
