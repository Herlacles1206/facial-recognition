import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from .model import Net

import sys

# setting path
sys.path.append('../')
from utils import xywh_to_xyxy


class Extractor(object):
    def __init__(self, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load("tracker/checkpoint/ckpt.t7", map_location=torch.device(self.device))[
            'net_dict']
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format("tracker/checkpoint/ckpt.t7"))
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(
            0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()
    
    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        h, w, _ = ori_img.shape
        for box in bbox_xywh:
            x1, y1, x2, y2 = xywh_to_xyxy(box, w, h)
            # print('------------------------------_get_features-------------------------------\n')
            # print('x1:{}, y1:{}, x2:{}, y2:{}'.format(x1, y1, x2, y2))
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self(im_crops)
        else:
            features = np.array([])
        return features


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)
