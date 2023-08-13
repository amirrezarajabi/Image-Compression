from kmeans.kmeans import Kmeans
from utils import read_image, calculate_psnr
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm.auto import tqdm

orginal_img, _ = read_image("img/slider_puffin_before_mobile.jpg")
img = orginal_img.reshape(-1, 3).astype(np.float64)

def clustring(n_clusters, tol):
    tic = time.time()
    kmeans = Kmeans(n_clusters=n_clusters, init='k-means++', n_init=1, tol=tol)
    kmeans.fit(img)
    y = kmeans.replace_with_nearest_centroid(img).reshape(800, 800, 3).astype(np.uint8)
    return y, time.time() - tic, int(img.shape[0] * np.log2(n_clusters))

results = {
}

for i in tqdm(range(1, 4)):
    k = 2 ** i
    results[k] = clustring(k, 2 ** i)

fig, axes = plt.subplots(1, 3, figsize=(11, 11))

keys = list(results.keys())


for i, ax in enumerate(axes.flat):

    image, t, bit = results[keys[i]]
    ax.imshow(image)
    psnr = calculate_psnr(image, orginal_img)
    ax.set_title(f"n_clusters:{keys[i]}")
    print (f"n_clusters:{keys[i]}, time:{t}, bits:{bit}, psnr:{psnr}")

plt.tight_layout()

plt.show()