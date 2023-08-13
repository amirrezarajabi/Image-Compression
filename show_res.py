import os
import matplotlib.pyplot as plt
from matplotlib.image import imread

folder_path = './img/'
image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
fig, axes = plt.subplots(2, 3, figsize=(8, 8))

for i, ax in enumerate(axes.flat):
    if i < len(image_files):
        image_path = os.path.join(folder_path, image_files[i])
        img = imread(image_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(image_files[i])
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()