from utilities import *
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

# dtrain, y_train, x_train = get_data_train("./Data/train_set/train/", "/*.jpg", 150, 150)

#print(type(x_train))
data_path = "./Data/seg_test"
rd_label = random.choice(os.listdir(data_path))
lst = os.path.join(data_path, rd_label)

img = cv2.imread(os.path.join(lst, random.choice(os.listdir(lst))))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (200, 150))

plt.imshow(img)
plt.show()
