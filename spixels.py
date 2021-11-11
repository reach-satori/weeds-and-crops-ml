import cv2
import numpy as np
from load_images import load_images
from matplotlib import pyplot as plt
from sklearn.cluster import MiniBatchKMeans
NCLUSTERS = 3

if __name__ == "__main__":

    img = next(iter(load_images()))
    img = img[:1000, :1000]
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX) # superpixels work with 0-255
    seeds = cv2.ximgproc.createSuperpixelSLIC(img.astype(np.uint8),
                                              algorithm=cv2.ximgproc.MSLIC,
                                              region_size=60)
    seeds.iterate(8)
    seeds.enforceLabelConnectivity(5)
    out = seeds.getLabelContourMask()
    labels = seeds.getLabels()
    numpix = seeds.getNumberOfSuperpixels()
    print(f"Number of superpixels: {numpix}")
    data = np.empty((numpix, 5))
    for i in range(numpix):
        data[i] = np.array([*np.mean(img[labels == i], axis=0)])#, *np.std(img[labels == i], axis=0)])

    # remove nan numbers (why are they there?)
    data = data[~np.isnan(data).any(axis=1)]
    numpix = data.shape[0]

    km = MiniBatchKMeans(NCLUSTERS)
    preds = km.fit_predict(data)
    for i in range(numpix):
        labels[labels == i] = preds[i]

    cmap = plt.get_cmap("gist_rainbow")
    labels[labels > NCLUSTERS-1] = 0
    output = np.empty(img[...,:3].shape)
    for i in np.unique(labels):
        # colmap = np.zeros_like(img[...,:3]) + \
        b, g, r, _ = cmap(float(i)/NCLUSTERS, bytes=True)
        output[labels == i] = np.array([b,g,r])
    img[...,:3] = cv2.addWeighted(output.astype(np.float32), 0.1, img[...,:3], 0.9, 0)
    plt.imshow(img[...,:3].astype(np.uint8))
    plt.show()
