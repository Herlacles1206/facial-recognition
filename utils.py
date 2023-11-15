import numpy as np
import torch


def getFileNameOnly(name):

    i = name.rfind('.')
    if 0 < i < len(name) - 1:
        return name[0:i]
    else:
        return ''
    
# Define distance function
def findCosineDistance(vector1, vector2):
    """
    Calculate cosine distance between two vector
    """
    vec1 = vector1.flatten()
    vec2 = vector2.flatten()

    a = np.dot(vec1.T, vec2)
    b = np.dot(vec1.T, vec1)
    c = np.dot(vec2.T, vec2)
    return 1 - (a/(np.sqrt(b)*np.sqrt(c)))

def CosineSimilarity(test_vec, source_vecs):
    """
    Verify the similarity of one vector to group vectors of one class
    """
    cos_dist = 0
    for source_vec in source_vecs:
        cos_dist += findCosineDistance(test_vec, source_vec)
    return cos_dist/len(source_vecs)


def xywh_to_tlwh(bbox_xywh):
    if isinstance(bbox_xywh, np.ndarray):
        bbox_tlwh = bbox_xywh.copy()
    elif isinstance(bbox_xywh, torch.Tensor):
        bbox_tlwh = bbox_xywh.clone()
    bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
    bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
    return bbox_tlwh

def xywh_to_xyxy(bbox_xywh, t_w, t_h):
    x, y, w, h = bbox_xywh
    x1 = max(int(x - w / 2), 0)
    x2 = min(int(x + w / 2), t_w - 1)
    y1 = max(int(y - h / 2), 0)
    y2 = min(int(y + h / 2), t_h - 1)
    return x1, y1, x2, y2

def tlwh_to_xyxy(bbox_tlwh, t_w, t_h):

    x, y, w, h = bbox_tlwh
    x1 = max(int(x), 0)
    x2 = min(int(x+w), t_w - 1)
    y1 = max(int(y), 0)
    y2 = min(int(y+h), t_h - 1)
    return x1, y1, x2, y2


def xyxy_to_tlwh(bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy

    t = x1
    l = y1
    w = int(x2 - x1)
    h = int(y2 - y1)
    return t, l, w, h

