import torch
from torch.autograd import Variable
import cv2
from layers.ssd import build_ssd
import numpy as np

weight_path = 'weights/ssd300_VOC.pth'
net = build_ssd('test', 300, 21)    # initialize SSD
net.load_state_dict(torch.load(weight_path))
net = net.eval().cuda()
pic = 'data/000023.jpg'
labelmap = ('aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x
class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels

transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

def predict(frame):
    height, width = frame.shape[:2]
    x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    x = x.cuda()
    y = net(x)  # forward pass
    detections = y.data
    scale = torch.Tensor([width, height, width, height])
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            cv2.rectangle(frame,
                          (int(pt[0]), int(pt[1])),
                          (int(pt[2]), int(pt[3])),
                          255, 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])),
                        1, 2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
    return frame


frame = cv2.imread(pic)
frame = predict(frame)
cv2.imshow('frame', frame)
cv2.waitKey()