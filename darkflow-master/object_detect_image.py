import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt

options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolo.weights',
    'threshold': 0.3,
}

tfnet = TFNet(options)

img = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = tfnet.return_predict(img)
print(result)
print(len(result))

for res in result:
    tl = (res['topleft']['x'], res['topleft']['y'])
    br = (res['bottomright']['x'], res['bottomright']['y'])
    label = res['label']
    img = cv2.rectangle(img, tl, br, (2,255,0), 7)
    img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)

plt.imshow(img)
plt.show()
cv2.imwrite('result.jpg',img)
