import os
import numpy as np
import face_recognition
import cv2
from scipy.spatial import distance


def process_face_images(img_dir, verbose=True):

    image_counter = 0
    all_encodings = []
    all_filepaths = []

    for root, dirs, files in os.walk(img_dir):
        for image_file in files:
            if image_file.endswith('jpg'):
                # print(image_file)
                image_counter += 1

                img = face_recognition.api.load_image_file(os.path.join(root, image_file), mode='RGB')
                encoding = face_recognition.face_encodings(img)
                # img = cv2.imread(image_file)
                # encoding = face_recognition.face_encodings(img)

                if encoding:
                    all_filepaths.append(image_file)
                    all_encodings.append(encoding)

                    # if image_counter > 1:
                    #     dist = distance.euclidean(reference, encoding)
                    #     if dist > max_dist:
                    #         max_dist = dist
                    #         if verbose:
                    #             print('Max Distance =', str(dist))
                    # else:
                    #     reference = encoding
                else:
                    print('No face found!')

                if verbose:
                    if image_counter % 100 == 0:
                        print(str(image_counter), 'images processed')

    return all_filepaths, all_encodings
