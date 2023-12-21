
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
score_thres = 0.4
target_names = ['Person 7', 'Adrian Robinson']  
trackers = {}


# variable to count frame in case of face cropping
face_save_flag = 0 

def detectAndRecognition(srcImg, _imgsz, _conf, _iou, _augment, _device, isVid = True):
    # try:
        results = detect_model.predict(source=srcImg, imgsz=_imgsz, conf=_conf, iou=_iou, augment=_augment, device=_device)
        result = results[0].cpu().numpy()
        srcimg_copy = srcImg.copy()
        img_h, img_w, _ = srcImg.shape

        global face_save_flag  # Use the global keyword to access the variable
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


            # score, name = recognition(face)
                
            # if name == null:
            #     continue
            # else:
            #     if score < 0.25:
            #         caption= "UN_KNOWN"
            #     else:
            #         caption = f"{name.split('_')[0].upper()}:{score:.2f}"
            #     t_size = cv2.getTextSize(caption, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                        
            #     cv2.rectangle(srcimg_copy, (x1, y1), (x1 + t_size[0], y1 + t_size[1]), (0, 146, 230), -1)
            #     cv2.putText(srcimg_copy, caption, (x1, y1 + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)  

            score, name = recognitionWithCosineSimilarity(align_face)

            
            # save cropped face images
            # if not face_save_flag:
            #     # save face images to ./database/cropped
            #     file_path = r'./database/cropped'
            #     file_name = str(int(datetime.now(timezone.utc).timestamp() * 1000)) + '.jpg'
            #     file_name = os.path.join(file_path, file_name)
            #     cv2.imwrite(file_name, align_face)
            caption= name
            if name != "UN_KNOWN":
                caption = f"{name.split('_')[0].upper()}:{score:.2f}"

            if not isVid:
                if name in trackers.keys():
                    maskFace(srcimg_copy, face, (x1, y1), (x2, y2), caption)
                continue
            # Tracking if video
            bbox_tlwh = [x1, y1, x2-x1, y2-y1]
            bbox_tlwh = np.array(bbox_tlwh)
            detections.append(bbox_tlwh)
            # Initialize tracker if it is recognized
            if name in trackers.keys():
                init_box = [x1, y1, x2-x1, y2-y1]
                t_tracker = trackers[name]['model']
                t_tracker.init(srcImg, init_box)
                trackers[name]['model'] = t_tracker
                trackers[name]['initialized'] = True
                # print('model for {} is initialized'.format(name))
                det_match_flag[i] = 1
                tra_match_flag[target_names.index(name)] = 1

                maskFace(srcimg_copy, face, (x1, y1), (x2, y2), caption)

        # Updating tracker
        if not isVid:
            return srcimg_copy
        for i, name in enumerate(target_names):
            if tra_match_flag[i] == 1:
                continue
            tracker = trackers[name]['model']
            if trackers[name]['initialized'] == False:
                continue
            bbox, score = tracker.update(srcImg)
            if score < score_thres:
                trackers[name]['initialized'] = False
            bbox = np.array(bbox)
            # print('bbox: {}, score: {}'.format(bbox, score))
            tlwh_box = xywh_to_tlwh(bbox)
            detection_indices = []
            for j, det in enumerate(detections):
                if det_match_flag[j] == 0: detection_indices.append(j)

            if len(detection_indices) == 0: 
                if trackers[name]['initialized']:
                    x1, y1, x2, y2 = tlwh_to_xyxy(tlwh_box, img_w, img_h)
                    face = srcImg[y1:y2, x1:x2]
                    maskFace(srcimg_copy, face, (x1, y1), (x2, y2), name)
                continue
            # print('detection_indices: {}'.format(detection_indices))
            cost_matrix = iou_cost(tlwh_box, detections, detection_indices)
            # print('cost matrix {}'.format(cost_matrix))
            min_index = np.argmin(cost_matrix)
            if cost_matrix[min_index] > min_iou_distance:
                # x1, y1, x2, y2 = tlwh_to_xyxy(tlwh_box, img_w, img_h)
                # face = srcImg[y1:y2, x1:x2]
                # maskFace(srcimg_copy, face, (x1, y1), (x2, y2), name)
                trackers[name]['initialized'] = False
                continue
            else:
                tracker.init(srcImg, detections[detection_indices[min_index]])
                tra_match_flag[i] = 1
                det_match_flag[detection_indices[min_index]] = 1
                box = detections[detection_indices[min_index]]
                x1, y1, x2, y2 = tlwh_to_xyxy(box, img_w, img_h)
                face = srcImg[y1:y2, x1:x2]
                trackers[name]['model'] = tracker
                maskFace(srcimg_copy, face, (x1, y1), (x2, y2), name)

       
        return srcimg_copy
    # except:
    #     print('error occuired')
    #     return srcImg
        
    


def recognition(face_image):
    
    # Get feature from face
    query_emb = (get_feature(face_image, training=False))
    
    # Read features
    images_names, images_embs = read_features()   

    scores = (query_emb @ images_embs.T)[0]

    id_min = np.argmax(scores)
    score = scores[id_min]
    name = images_names[id_min]
    isThread = True
    # print("successful")
    return score, name

def recognitionWithCosineSimilarity(face_image):
    st_time = time.time()
    # Get feature from face
    query_emb = (get_feature(face_image, training=False))
    end_time = time.time()

    # print('embedding time is {} ms'.format((end_time - st_time)* 1000))
    # Read features
    images_names, images_embs = read_features()   

    le = LabelEncoder()
    labels = le.fit_transform(images_names)

    st_time = time.time()

    # Pytorch
    # print('device: {}'.format(device_recog))
    query_emb = torch.tensor(query_emb)
    # print(query_emb.to(device_recog).shape)
    preds = recog_model(query_emb.to(device_recog))
    # print('preds: {}'.format(preds))
    preds = F.softmax(preds, dim=1)
    preds = preds.cpu().detach().numpy().flatten()
    # print('softmax preds: {}'.format(preds))

    # Get the highest accuracy embedded vector
    j = np.argmax(preds)
    # print('j: {} '.format(j))
    proba = preds[j]
    # print('proba: {}'.format(proba))

    # Compare this vector to source class vectors to verify it is actual belong to this class
    match_class_idx = (labels == j)
    match_class_idx = np.where(match_class_idx)[0]
    # print('match_class_idx: {}'.format(match_class_idx))

    selected_idx = np.random.choice(match_class_idx, comparing_num)
    compare_embeddings = images_embs[selected_idx]
    # Calculate cosine similarity
    name = 'UN_KNOWN'
    cos_similarity = CosineSimilarity(query_emb, compare_embeddings)
    # print('cos_similarity: {}'.format(cos_similarity))
    if cos_similarity < cosine_threshold and proba > proba_threshold:
        name = le.classes_[j]
        text = "{}".format(name)
        # print("Recognized: {} <{:.2f}>".format(name, proba*100))
    
    end_time = time.time()

    # print('classification time is {} ms'.format((end_time - st_time)* 1000))

    return cos_similarity, name



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='Videos', help="image path")
    parser.add_argument('--output', default='result', type=str, help='result path')
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

    # Initialize recognition model
    # recog_model = load_model(opt.recog_weights)
    # Load the model
    checkpoint = torch.load(opt.recog_weights, map_location=device_recog)
    recog_model = SoftMax(input_shape=(checkpoint['input_shape'], ), num_classes=checkpoint['num_classes']).to(device_recog)
    recog_model.load_state_dict(checkpoint['state_dict'])

    # print(recog_model)
    
    # Assign trackers for each name
    model_path = 'weights/SiamRPNVOT.model'
    for name in target_names:
        trackers[name] = dict(model=SiamRPNTracker(model_path), initialized = False)

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

    for i, item in enumerate(files):
        # if i <= 10 and i > 18:
        #     continue
        is_img = str.lower(Path(item).suffix[1:]) in (img_formats)
        is_vid = str.lower(Path(item).suffix[1:]) in (vid_formats)
        srcName = os.path.join(opt.imgpath, item)
        dstName = os.path.join(opt.output, item)
        tempDstName = os.path.join(opt.output, 'temp'+item)
        startTime = time.time()

        if is_img:
            
            srcImg = cv2.imread(srcName)
            dstImg = detectAndRecognition(srcImg, opt.img_size, opt.conf_thres, opt.iou_thres, opt.augment, device_recog, False)
            
            cv2.imwrite(dstName, dstImg)
            
        elif is_vid:
            # Initialize trackers for new video
            for name in target_names:
                trackers[name]['initialized'] = False

            # Loading video dsa gfg intro video
            clip = VideoFileClip(srcName)

            # Getting audio from the clip
            audioclip = clip.audio

            # clip.close()
            # print('input: ', srcName)
            cap = cv2.VideoCapture(srcName)


            frame_count = 0
            fps = -1
            
            # Save video
            # Get the video codec and frame rate
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Get the width and height of the video frames
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # frame_width = int(cap.get(3))
            # frame_height = int(cap.get(4))
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            video = cv2.VideoWriter(dstName, fourcc, fps, (width, height))
            print('{} is created'.format(dstName))
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

                #Save video
                video.write(dstImg)
                print('{} frame'.format(frame_count))
            video.release()
            cap.release()
           
            clip = VideoFileClip(dstName)
            # Combining audio with the video clip
            clip_with_audio = clip.set_audio(audioclip)

            # Saving the combined video with audio
            clip_with_audio.write_videofile(tempDstName, codec="libx264", fps=fps)

            clip.close()
            audioclip.close()

            # Delete the original video file
            os.remove(dstName)
            
            # Rename the temporary file to the desired output filename
            os.rename(tempDstName, dstName)

        endTime = time.time()
        print("[{} {}]\t{} seconds".format(i, totalCnt, round(endTime - startTime, 2)))





    