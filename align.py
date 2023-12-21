
import numpy as np
import math
import cv2


def findEuclideanDistance(source_representation, test_representation):
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def alignment_procedure(img, left_eye, right_eye):
    # this function aligns given face in img based on left and right eye coordinates

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    # -----------------------
    # find rotation direction

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = 1  # rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = -1  # rotate inverse direction of clock

    # -----------------------
    # find length of triangle edges

    a = findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    c = findEuclideanDistance(np.array(right_eye), np.array(left_eye))

    # -----------------------

    # apply cosine rule

    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation
        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # -----------------------
        # rotate base image

        if direction == -1:
            angle = 90 - angle

        # img = Image.fromarray(img)
        # img = np.array(img.rotate(direction * angle))

        # Get the image dimensions
        height, width = img.shape[:2]

        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)

        # Apply the rotation to the image using cv2.warpAffine
        img = cv2.warpAffine(img, rotation_matrix, (width, height))

    # -----------------------

    return img  # return img anyway

def alignment_procedure_2(img, left_eye, right_eye):
    # this function aligns given face in img based on left and right eye coordinates

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    # -----------------------
    # find rotation direction

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  #  rotate inverse direction of clock
        b = findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
        c = findEuclideanDistance(np.array(right_eye), np.array(left_eye))
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate same direction to clock
        b = findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
        c = findEuclideanDistance(np.array(right_eye), np.array(left_eye))

    # apply cosine rule

    if c != 0:  # this multiplication causes division by zero in cos_a calculation
        cos_b = b / c
        angle = np.arccos(cos_b)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # rotate base image

        if direction == 1:
            angle = -angle

        # print('angle is {}'.format(int(angle)))
        # print('direction is {}'.format(direction))

        # Get the image dimensions
        height, width = img.shape[:2]

        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)

        # Apply the rotation to the image using cv2.warpAffine
        img = cv2.warpAffine(img, rotation_matrix, (width, height))


    return img  # return img anyway

def cropFaces(image, boxes, scores, kpts, align=False):
    face_images = []
    for box, score, kp in zip(boxes, scores, kpts):
        x, y, w, h = box.astype(int)
        # Crop face region
        face_image = image[y:y+h, x:x+w]

        #print left eye and right eye position
        left_eye = (int(kp[0 * 3]), int(kp[0 * 3 + 1]))
        right_eye = (int(kp[1 * 3]), int(kp[1 * 3 + 1]))
        # print("right eye: {}".format(right_eye))
        # print("left eye: {}".format(left_eye))

        # cv2.imshow('origin', face_image)
        # cv2.waitKey()

        if align:
            # rotated_image = alignment_procedure(face_image, left_eye, right_eye)
            rotated_image = alignment_procedure_2(face_image, left_eye, right_eye)
            # cv2.imshow('rotated', rotated_image)
            # cv2.waitKey()
        face_images.append(face_image)
    
    return face_images

def alignFace(image, kp):

     #print left eye and right eye position
    right_eye = (int(kp[0][0]), int(kp[0][1]))
    left_eye = (int(kp[1][0]), int(kp[1][1]))
    # print("right eye: {}".format(right_eye))
    # print("left eye: {}".format(left_eye))

    # cv2.imshow('origin', image)
    
    rotated_image = alignment_procedure_2(image, left_eye, right_eye)
    # cv2.imshow('rotated', rotated_image)
    # cv2.waitKey()

    
    return rotated_image

def maskFace(image, face, tl, br, caption, landmarks = [], show_landmarks = False, show_caption = True, mask_face = True):
    # draw a rectangle along the bounding box of face
    # cv2.rectangle(image, tl, br, (0, 0, 255), 2)
    (t, l) = tl

    if show_landmarks:
        for i, landmark in enumerate(landmarks):
            cv2.circle(image, (int(landmark[0]), int(landmark[1])), 4, (0, 0, 255), thickness=-1)

    if mask_face:
        # bluring faces
        # apply gaussian blur to face rectangle
        roi = cv2.GaussianBlur(face, (35, 35), 60)
                
        # add blurred face on original image to get final image
        # print('x1 {} y1 {} x2 {} y2 {}'.format(t, t+roi.shape[1], l, l+roi.shape[0]))
        # cv2.imshow('face', face)
        # cv2.waitKey()

        image[l:l+roi.shape[0], t:t+roi.shape[1]] = roi

    if show_caption:
        t_size = cv2.getTextSize(caption, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        
        cv2.rectangle(image, tl, (t + t_size[0], l + t_size[1]), (0, 146, 230), -1)
        cv2.putText(image, caption, (t, l + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)




