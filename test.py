import cv2
import matplotlib.pyplot as plt
import numpy as np


# 图片的分辨率为200*300，这里b, g, r设为随机值，注意dtype属性
b = np.random.randint(0, 255, (200, 300), dtype=np.uint8)
g = np.random.randint(0, 255, (200, 300), dtype=np.uint8)
r = np.random.randint(0, 255, (200, 300), dtype=np.uint8)

# 合并通道，形成图片
img = cv2.merge([b, g, r])

# 显示图片
#   cv2.imshow('test', img)
#cv2.waitKey(0)
#cv2.destroyWindow('test')
#plt.show()
#input()

image_dir=r'/data/apps/efficientDet-Pytorch/datasets/test.png'
#image_dir=r'/data/apps/efficientDet-Pytorch/datasets/Gnome.jpg'
#img_bgr = cv2.cv.LoadImage(image_dir)
#img0 =np.fromfile(image_dir, dtype=np.uint8)
#print(img0)
#img = cv2.imdecode(img0, cv2.IMREAD_COLOR)
#
img = cv2.imread(image_dir, cv2.IMREAD_COLOR)
print(img)
plt.imshow(img)
plt.show()