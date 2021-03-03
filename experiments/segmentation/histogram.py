import cv2
from matplotlib import pyplot as plt

img = cv2.imread("/home/grvc/.encoding/data/ADEChallengeData2016/annotations/training/ADE_train_00000001.png", cv2.IMREAD_GRAYSCALE)

counter = 0
[fil,col] = img.shape
for i in range(fil):
    for j in range(col):
        if(img[i][j]==0):
            counter+= 1 

print(counter)
hist = cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(hist)
plt.xlim([0,256])
plt.show()