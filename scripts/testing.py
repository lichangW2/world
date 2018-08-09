import selectivesearch as ss
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
img=cv2.imread("/Users/cj/Desktop/n01798484_16449.JPEG")
img_lb, regions = ss.selective_search(img, scale=200, sigma=0.8, min_size=50)
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
print(regions[0]["rect"],"shape: ",img.shape)
ax.imshow(img)
rect = mpatches.Rectangle(
            (regions[0]["rect"][0], regions[0]["rect"][1]), regions[0]["rect"][2], regions[0]["rect"][3], fill=False, edgecolor='red', linewidth=1)
ax.add_patch(rect)
plt.show()