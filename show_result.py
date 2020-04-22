import torch
from torch.backends import cudnn

from backbone import EfficientDetBackbone
import cv2
import matplotlib.pyplot as plt
import numpy as np
from efficientnet import EfficientNet as EffNet
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess
import time

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    print("_rebuild_tensor_v2")
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


compound_coef = 0
force_input_size = None  # set None to use default size
img_path = r'/data/apps/efficientDet-Pytorch/datasets/000000026564.jpg'
#img_path = 'datasets/000000026564.jpg'
img_path=r'datasets/computer.png'
#a=cv2.imread(r'/data/apps/efficientDet-Pytorch/datasets/000000026564.jpg')

threshold = 0.2
iou_threshold = 0.2

use_cuda = False  
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
           'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
           'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
           'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
           'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
           'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
           'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
           'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
           'toothbrush']

# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

if use_cuda:
    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
else:
    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

#model = efficientdet.from_pretrained('efficientnet-b1')
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),

                             # replace this part with your project's anchor config
                             ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                             scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

model.load_state_dict(torch.load('weights/efficientdet-d'+str(compound_coef)+'.pth'))
#model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

start_time = time.time()
last_loop  = 0
for loop_cnt in range(0,1000):
  with torch.no_grad():
      features, regression, classification, anchors = model(x)

      regressBoxes = BBoxTransform()
      clipBoxes = ClipBoxes()

      out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)

  out = invert_affine(framed_metas, out)

  for i in range(len(ori_imgs)):
      if len(out[i]['rois']) == 0:
          continue

      for j in range(len(out[i]['rois'])):

          (x1, y1, x2, y2) = out[i]['rois'][j].astype(np.int)
          cv2.rectangle(ori_imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
          obj = obj_list[out[i]['class_ids'][j]]
          score = float(out[i]['scores'][j])

          cv2.putText(ori_imgs[i], '{}, {:.3f}'.format(obj, score),
                      (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                      (255, 255, 0), 1)

          #plt.imshow(ori_imgs[i])

  #plt.show()
  if time.time() - start_time >=30:
    
    print("Frams:",loop_cnt,"used:",time.time() - start_time,"s,FPS:",(loop_cnt-last_loop)/(time.time() - start_time))
    last_loop = loop_cnt
    plt.imshow(ori_imgs[i])
    plt.show()
    #input()
    start_time =time.time()
    #break