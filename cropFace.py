
import argparse
import os
from pathlib import Path
import cv2
from ultralytics import YOLO
import time
from datetime import datetime, timezone
from align import *
from threading import Thread
from sqlalchemy import null
import torch
from torchvision import transforms
from insightface.insight_face import iresnet100
from recogTrain import *
from utils import *
from sklearn.preprocessing import LabelEncoder
from softmax_nn import SoftMax
import torch.nn.functional as F
from moviepy.editor import *

from net.tracker import SiamRPNTracker

# Setup some useful arguments
cosine_threshold = 0.8
proba_threshold = 0.85
comparing_num = 5

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

detect_model = None
recog_model = None

device_recog = torch.device("cuda" if torch.cuda.is_available() else "cpu")

min_iou_distance = 0.25
score_thres = 0.6
target_names = ['Person 7', 'Person 3']  
trackers = {}

dest_folder = ''
# variable to count frame in case of face cropping
face_save_flag = 0 

def detectAndRecognition(srcImg, _imgsz, _conf, _iou, _augment, _device, isVid = True):
    # try:
        results = detect_model.predict(source=srcImg, imgsz=_imgsz, conf=_conf, iou=_iou, augment=_augment, device=_device)
        result = results[0].cpu().numpy()
        srcimg_copy = srcImg.copy()
        img_h, img_w, _ = srcImg.shape

        global face_save_flag, dest_folder  # Use the global keyword to access the variable
        face_save_flag = (face_save_flag + 1) % 10

        # detection matching flag
        det_match_flag = [0] * len(result.boxes)
        # tracker matching flag
        tra_match_flag = [0] * len(target_names)
        detections = []

        for i, (box, keypoint) in enumerate(zip(result.boxes, result.keypoints)):
            
            conf = box.conf[0]
            cls  = box.cls[0]
            xyxy = box.xyxy[0]
            x1 = int(xyxy[0] + 0.5)
            y1 = int(xyxy[1] + 0.5)
            x2 = int(xyxy[2] + 0.5)
            y2 = int(xyxy[3] + 0.5)

            if x1 < 0 : x1 = 0
            if y1 < 0 : y1 = 0
            if x2 > img_w - 1 : x2 = img_w - 1
            if y2 > img_h - 1 : y2 = img_h - 1

            if y2 - y1 == 0 or x2 - x1 == 0:
                continue
            
            
            face = srcImg[y1:y2, x1:x2]
            face_copy = face.copy()

            # face alignment
            align_face = alignFace(face_copy, keypoint.xy[0])

            current_time = datetime.utcnow()
            milliseconds = int(current_time.timestamp() * 1000)

            cv2.imwrite(os.path.join(opt.output, '{}.jpg'.format(milliseconds)), align_face)
           
        
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='./database/source', help="image path")
    parser.add_argument('--output', default='./database/cropped', type=str, help='result path')
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov8n-face.pt', help='model.pt path(s)')
    parser.add_argument('--recog_weights', nargs='+', type=str, default="./static/feature/my_model.pth", help='model.pt path(s)')
    parser.add_argument('--img-size', nargs= '+', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', type=str, default='0', help='augmented inference')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    
    opt = parser.parse_args()

    # Initialize YOLOv8_face object detector
    detect_model = YOLO(opt.weights)


    # Check if the directory exists
    if not os.path.exists(opt.output):
        # Create the directory if it doesn't exist
        os.makedirs(opt.output)
    else:
        # Get a list of all the files in the directory
        files = os.listdir(opt.output)

        # Loop over each file and delete it
        for file in files:
            # Construct the full file path
            file_path = os.path.join(opt.output, file)

            # Check if the file is a regular file (not a directory)
            if os.path.isfile(file_path):
                # Delete the file
                os.remove(file_path)

    #cutomized code for folder
    files = os.listdir(opt.imgpath)
    print(files)
    totalCnt = len(files)

    dest_folder = opt.output
    for i, item in enumerate(files):
        # if i <= 10 and i > 18:
        #     continue
        is_img = str.lower(Path(item).suffix[1:]) in (img_formats)
        is_vid = str.lower(Path(item).suffix[1:]) in (vid_formats)
        srcName = os.path.join(opt.imgpath, item)
        dstName = os.path.join(opt.output, item)
        
        startTime = time.time()

        if is_img:
            
            srcImg = cv2.imread(srcName)
            dstImg = detectAndRecognition(srcImg, opt.img_size, opt.conf_thres, opt.iou_thres, opt.augment, device_recog, False)
            
        elif is_vid:

            # clip.close()
            # print('input: ', srcName)
            cap = cv2.VideoCapture(srcName)

            frame_count = 0
            # Read until video is completed
            while(cap.isOpened()):
                # Capture frame-by-frame
                _, frame = cap.read()
                if not _:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break

                # height, width, _ = frame.shape

                # if height == 0 or width == 0:
                #     print("None frame")
                #     continue
                # Count fps 
                frame_count += 1
                # temp code for testing
                # if frame_count < 2434:
                #     continue
                # cv2.imshow('{} frame'.format(frame_count), frame)
                # cv2.waitKey()
                dstImg = detectAndRecognition(frame, opt.img_size, opt.conf_thres, opt.iou_thres, opt.augment, device_recog)

                print('{} frame'.format(frame_count))

            cap.release()
           

        endTime = time.time()
        print("[{} {}]\t{} seconds".format(i, totalCnt, round(endTime - startTime, 2)))





    