import matplotlib.pyplot as plt
import numpy as np
import scipy.io

mat_contents = scipy.io.loadmat('allFaces.mat')
faces = mat_contents['faces']
# m = 168, n = 192, nxm b/w pixels
m = int(mat_contents['m'])
n = int(mat_contents['n'])
nfaces = np.ndarray.flatten(mat_contents['nfaces'])
trainingFaces = faces[:,:np.sum(nfaces[:36])]
avgFace = np.mean(trainingFaces, axis=1)    # size n*m by 1
X = trainingFaces - np.tile(avgFace, (trainingFaces.shape[1], 1)).T
U, S, VT = np.linalg.svd(X, full_matrices=0)

# Now show eigenface reconstruction of image that was omitted from test set
testFace = faces[:, np.sum(nfaces[:36])]  # First face of person 37
plt.imshow(np.reshape(testFace, (m, n)).T)
plt.set_cmap('gray')
plt.title('Original Image')
plt.axis('off')

plt.savefig("37OriginalFace.png")
plt.show()

testFaceMS = testFace - avgFace
r_list = [25, 50, 100, 200, 400, 800, 1600]  # 1600 takes a lot of CPU time
# r_list = [25, 50, 100, 200, 400]  # 1600 takes a lot of CPU time

for r in r_list:
    reconFace = avgFace + U[:, :r] @ U[:, :r].T @ testFaceMS
    img = plt.imshow(np.reshape(reconFace, (m, n)).T)
    img.set_cmap('gray')
    plt.title('r = ' + str(r))
    plt.axis('off')
    plt.savefig('r' + str(r))
    plt.show()
