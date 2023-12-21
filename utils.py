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
    bbox_tlwh[0] = bbox_xywh[0] - bbox_xywh[2] / 2.
    bbox_tlwh[1] = bbox_xywh[1] - bbox_xywh[3] / 2.
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

def iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    # print('tl {}'.format(tl))
    # print('br {}'.format(br))
    # print('wh {}'.format(wh))

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)

    # print('area_intersection {}'.format(area_intersection))
    # print('area_bbox {}'.format(area_bbox))
    # print('area_candidates {}'.format(area_candidates))

    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou_cost(track, detections, 
             detection_indices=None):
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.ones((len(detection_indices)))

    bbox = track
    candidates = np.asarray(
            [detections[i] for i in detection_indices])    
    
    # print('track_box: {}'.format(bbox))
    # print('candidates: {}'.format(candidates))
    # print('cost matrix: {}'.format(cost_matrix))

    cost_matrix[:] = 1. - iou(bbox, candidates)

    return cost_matrix