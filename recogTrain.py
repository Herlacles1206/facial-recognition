from threading import Thread
from sqlalchemy import null
import torch
from torchvision import transforms
from insightface.insight_face import iresnet100
import cv2
import numpy as np
import os
from ultralytics import YOLO
import shutil
import argparse
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from softmax_nn import SoftMax
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import copy




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight = torch.load("insightface/resnet100_backbone.pth", map_location = device)
model_emb = iresnet100()

## Case 2: 
# from insightface.insight_face import iresnet18
# weight = torch.load("insightface/resnet18_backbone.pth", map_location = device)
# model_emb = iresnet18()

model_emb.load_state_dict(weight)
model_emb.to(device)
model_emb.eval()

YOLOv8_face_detector = YOLO('weights/yolov8n-face.pt')

face_preprocess = transforms.Compose([
                                    transforms.ToTensor(), # input PIL => (3,56,56), /255.0
                                    transforms.Resize((112, 112), antialias=True),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                    ])



def get_feature(face_image, training = True): 
    # Convert to RGB
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    # Preprocessing image BGR
    face_image = face_preprocess(face_image).to(device)
    
    # Via model to get feature
    with torch.no_grad():
        if training:
            emb_img_face = model_emb(face_image[None, :])[0].cpu().numpy()
        else:
            emb_img_face = model_emb(face_image[None, :]).cpu().numpy()
    
    # Convert to array
    images_emb = emb_img_face/np.linalg.norm(emb_img_face)
    return images_emb

def read_features(root_fearure_path = "static/feature/face_features.npz"):
    data = np.load(root_fearure_path, allow_pickle=True)
    images_name = data["arr1"]
    images_emb = data["arr2"]
    
    return images_name, images_emb

def training(full_training_dir, additional_training_dir, 
             faces_save_dir, features_save_dir, is_add_user):
    
    # Init results output
    images_name = []
    images_emb = []
    
    # Check mode full training or additidonal
    if is_add_user == True:
        source = additional_training_dir
    else:
        source = full_training_dir
    
    totalCnt = len(os.listdir(source))
    # Read train folder, get and save face 
    for index, name_person in enumerate(os.listdir(source)):
        person_image_path = os.path.join(source, name_person)
        
        # Create path save person face
        person_face_path = os.path.join(faces_save_dir, name_person)
        os.makedirs(person_face_path, exist_ok=True)
        
        for image_name in os.listdir(person_image_path):
            if image_name.endswith(("png", 'jpg', 'jpeg')):
                image_path = person_image_path + f"/{image_name}"
                # print('image path: ', image_path)
                input_image = cv2.imread(image_path)  # BGR 

                # Get faces
                # bboxs = get_face(input_image)
                # cv2.imshow('input', input_image)
                # cv2.waitKey()

                results = YOLOv8_face_detector.predict(input_image)
                result = results[0].cpu().numpy()
                # CropFace(srcimg, boxes, scores, kpts)

                # Get boxs
                for i in range(len(result.boxes)):
                    # Get number files in person path
                    number_files = len(os.listdir(person_face_path))

                    # Get location face
                    x1, y1, x2, y2 = result.boxes[i]
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    # print("x:{} y:{} w:{} h{}".format(x1, y1, x2, y2))
                    # Get face from location
                    face_image = input_image[y1:y1+y2, x1:x1+x2]
                    # cv2.imshow('face', face_image)
                    # cv2.waitKey()

                    # Path save face
                    path_save_face = person_face_path + f"/{number_files}.jpg"
                    
                    # Save to face database 
                    cv2.imwrite(path_save_face, face_image)
                    
                    # Get feature from face
                    images_emb.append(get_feature(face_image, training=True))
                    images_name.append(name_person)
            
        print('[{} {}] completed'.format(index, totalCnt))
    
    # Convert to array
    images_emb = np.array(images_emb)
    images_name = np.array(images_name)
    
    features = read_features(features_save_dir) 
    if features == null or is_add_user== False:
        pass
    else:        
        # Read features 
        old_images_name, old_images_emb = features  
       
        # Add feature and name of image to feature database
        images_name = np.hstack((old_images_name, images_name))
        images_emb = np.vstack((old_images_emb, images_emb))
        
        print("Update feature!")
    
    # Save features
    np.savez_compressed(features_save_dir, 
                        arr1 = images_name, arr2 = images_emb)
    
    # Move additional data to full train data
    if is_add_user == True:
        for sub_dir in os.listdir(additional_training_dir):
            dir_to_move = os.path.join(additional_training_dir, sub_dir)
            shutil.move(dir_to_move, full_training_dir, copy_function = shutil.copytree)
    
def training_2(full_training_dir,   features_save_dir):
    
    # Init results output
    images_name = []
    images_emb = []
    
    # Check mode full training or additidonal

    source = full_training_dir
    
    totalCnt = len(os.listdir(source))
    # Read train folder, get and save face 
    for index, name_person in enumerate(os.listdir(source)):
        person_image_path = os.path.join(source, name_person)
        
        
        for image_name in os.listdir(person_image_path):
            if image_name.endswith(("png", 'jpg', 'jpeg')):
                image_path = person_image_path + f"/{image_name}"
                # print('image path: ', image_path)
                input_image = cv2.imread(image_path)  # BGR 

                # Get faces
                # bboxs = get_face(input_image)
                # cv2.imshow('input', input_image)
                # cv2.waitKey()

                # Get boxs
                images_emb.append(get_feature(input_image, training=True))
                images_name.append(name_person)
                    
            
        print('[{} {}] completed'.format(index, totalCnt))
    
    # Convert to array
    images_emb = np.array(images_emb)
    images_name = np.array(images_name)
    
    # Save features
    np.savez_compressed(features_save_dir, 
                        arr1 = images_name, arr2 = images_emb)
    
        
def training_3(full_training_dir,   features_save_dir, model_name):
    
    # Init results output
    images_name = []
    images_emb = []
    
    # Check mode full training or additidonal

    source = full_training_dir
    
    totalCnt = len(os.listdir(source))
    # Read train folder, get and save face 
    for index, name_person in enumerate(os.listdir(source)):
        person_image_path = os.path.join(source, name_person)
        
        
        for image_name in os.listdir(person_image_path):
            if image_name.endswith(("png", 'jpg', 'jpeg')):
                image_path = person_image_path + f"/{image_name}"
                # print('image path: ', image_path)
                input_image = cv2.imread(image_path)  # BGR 

                # Get faces
                # bboxs = get_face(input_image)
                # cv2.imshow('input', input_image)
                # cv2.waitKey()

                # Get boxs
                images_emb.append(get_feature(input_image, training=True))
                images_name.append(name_person)
                    
            
        print('[{} {}] completed'.format(index, totalCnt))
    
    # Convert to array
    embeddings = np.array(images_emb)
    images_name = np.array(images_name)
    
    # print(images_name)
    # print(len(images_emb[0]))

    le = LabelEncoder()
    labels = le.fit_transform(images_name)
    # print('labels: {}'.format(labels))
    num_classes = len(np.unique(labels))
    labels = labels.reshape(-1, 1)
    one_hot_encoder = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
    labels = one_hot_encoder.fit_transform(labels).toarray()
    # print('labels: {}'.format(labels))
    # print('num_classes: {}'.format(num_classes))

    # Initialize Softmax training model arguments
    BATCH_SIZE = 32
    EPOCHS = 20
    input_shape = embeddings.shape[1]

    # Create KFold
    cv = KFold(n_splits = 5, random_state = 42, shuffle=True)
    history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}



################################################### Pytorch ########################################################
    best_acc = 0   # init to negative infinity
    # Define your loss function and optimizer
    model = SoftMax(input_shape = (input_shape, ), num_classes = num_classes).to(device)
    criterion = model.get_loss()
    optimizer = model.get_optimizer()
    # print(model)

    # Train

    for train_idx, valid_idx in cv.split(embeddings):

        X_train, X_val, y_train, y_val = embeddings[train_idx], embeddings[valid_idx], labels[train_idx], labels[valid_idx]
  
        # Convert your data to PyTorch tensors and move them to the device
        X_train = torch.tensor(X_train).to(device)
        y_train = torch.tensor(y_train).to(device)
        X_val = torch.tensor(X_val).to(device)
        y_val = torch.tensor(y_val).to(device)

        # Create DataLoaders for training and validation data
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

        # Training loop
        for epoch in range(EPOCHS):
            model.train()  # Set model to training mode
            train_loss = 0.0
            correct = 0

            for inputs, _labels in train_loader:
                inputs = inputs.to(device)
                _labels = _labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, _labels)
                correct += (torch.argmax(outputs, 1) == torch.argmax(_labels, 1)).float().mean()
                
                # print('labels {}'.format(_labels))
                # print('outputs {}'.format(outputs))
                # print('loss {}'.format(loss))

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_acc = correct / len(train_loader)
            # Validation
            model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, _labels in val_loader:
                    inputs = inputs.to(device)
                    _labels = _labels.to(device)

                    outputs = model(inputs)
                    val_loss += criterion(outputs, _labels).item()

                    # _, predicted = outputs.max(1)
                    # _, predicted = torch.max(outputs.data, 1)
                    # # print("outputs: \n {}".format(outputs))
                    correct += (torch.argmax(outputs, 1) == torch.argmax(_labels, 1)).float().mean()
                    total += _labels.size(0)
                    # correct += (predicted == labels).long().sum().item()

            epoch_loss = train_loss / len(train_loader)
            epoch_val_loss = val_loss / len(val_loader)
            val_acc = correct / len(val_loader)
            # epoch_acc = correct / total
            
            # print('epoch acc is {}'.format(epoch_acc))
            if val_acc > best_acc:
                best_acc = val_acc
                best_weights = copy.deepcopy(model.state_dict())

            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Val Loss: {epoch_val_loss:.4f} - Val Acc: {val_acc:.4f}")

            history['acc'].append(train_acc.cpu().numpy())
            history['val_acc'].append(val_acc.cpu().numpy())
            history['loss'].append(epoch_loss)
            history['val_loss'].append(epoch_val_loss)

    # Save the model state dictionary and other necessary information
    torch.save({
        'state_dict': best_weights,
        'input_shape': input_shape,
        'num_classes': num_classes
    }, model_name)




################################################# Pytorch #########################################################

    np.savez_compressed(features_save_dir, 
                        arr1 = images_name, arr2 = images_emb)
    
    # Plot
    plt.figure(1)
    # Summary history for accuracy
    plt.subplot(211)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # Summary history for loss
    plt.subplot(212)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./static/feature/accuracy_loss.png')
    plt.show()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-training-dir', type=str, default='./database/full-training-datasets/', help='dir folder full training')
    parser.add_argument('--additional-training-dir', type=str, default='./database/additional-training-datasets/', help='dir folder additional training')
    parser.add_argument('--faces-save-dir', type=str, default='./database/face-datasets/', help='dir folder save face features')
    parser.add_argument('--features-save-dir', type=str, default='./static/feature/face_features.npz', help='dir folder save face features')
    parser.add_argument('--is-add-user', type=bool, default=False, help='Mode add user or full training')
    parser.add_argument("--model", default="./static/feature/my_model.pth",
                help="path to output trained model")
    
    opt = parser.parse_args()
    return opt

def main(opt):
    # training(**vars(opt))
    training_3('./database/face-datasets/', './static/feature/face_features.npz', opt.model)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)