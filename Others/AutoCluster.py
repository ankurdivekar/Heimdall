import dlib
import face_recognition
import os
from glob import glob

input_dir = 'C:/WorkData/FaceRecognition/Datasets/All Employees/'

output_dir = 'C:/WorkData/FaceRecognition/Datasets/AutoCluster/'

images_jpg = [y for x in os.walk(input_dir) for y in glob(os.path.join(x[0], '*.jpg'))]
images_png = [y for x in os.walk(input_dir) for y in glob(os.path.join(x[0], '*.png'))]
images_tif = [y for x in os.walk(input_dir) for y in glob(os.path.join(x[0], '*.tif'))]
all_images = images_jpg + images_png + images_tif

print(str(len(images_jpg)), 'JPG images')
print(str(len(images_png)), 'PNG images')
print(str(len(images_tif)), 'TIF images')
print(str(len(all_images)), 'total images')

image_counter = 0
max_images = 125
id_encodings = []
id_source_filepaths = []
id_dest_filepaths = []

dist_threshold = 0.25

for image in all_images:

    img = face_recognition.load_image_file(image)
    encoding = face_recognition.face_encodings(img)
    if encoding:
        encoding = encoding[0]

        face_distances = face_recognition.face_distance(id_encodings, encoding)

        if any(x < dist_threshold for x in face_distances):
            val, idx = min((val, idx) for (idx, val) in enumerate(face_distances))
            print(image)
            print('Match found with')
            print(id_source_filepaths[idx])
            print('Distance = ', str(val))

            # print(face_distances)

        else:
            id_encodings.append(encoding)
            id_source_filepaths.append(image)

            dest_filepath = output_dir + str(len(id_encodings)) + '.jpg'

    image_counter += 1
    if image_counter == max_images:
        break

print(str(len(id_encodings)), 'unique identities found')
