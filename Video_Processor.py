import os
import math
import cv2
import face_recognition
import dlib
from PIL import Image


def get_faces_from_vid(vid_path, img_dir, identity='Face', img_h=300, img_w=300, skip_frames=200, max_frames=300, verbose=True):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('C:/WorkData/ZZZ_Resources/ImgProcessing/shape_predictor_68_face_landmarks.dat')

    # Open the input movie file
    input_movie = cv2.VideoCapture(vid_path)
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

    # # Create an output movie file (make sure resolution/frame rate matches input video!)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # output_movie = cv2.VideoWriter(output_vid_path, fourcc, 29.97, (568, 320))

    # Start face number as no of files present from last vid in order to not over-write
    face_number = len(os.listdir(img_dir))
    frame_number = 0


    if verbose:
        print('Reading video file')

    while frame_number < max_frames+skip_frames:
        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_number += 1

        # Quit when the input video file ends
        if not ret:
            break

        if verbose:
            if frame_number % 100 == 0:
                print('Frame no:', str(frame_number))

        if frame_number > skip_frames:
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_frame = frame[:, :, ::-1]

            # Find facial landmarks
            face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

            if face_landmarks_list:

                face_number += 1

                left_eye_points = face_landmarks_list[0]['left_eye']
                right_eye_points = face_landmarks_list[0]['right_eye']

                left_eye = get_centroid(left_eye_points)
                right_eye = get_centroid(right_eye_points)

                # for (x, y) in left_eye_points:
                #     cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                # cv2.circle(frame, (left_eye[0], left_eye[1]), 1, (0, 255, 0), -1)
                #
                # for (x, y) in right_eye_points:
                #     cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                # cv2.circle(frame, (right_eye[0], right_eye[1]), 1, (255, 0, 255), -1)

                # cv2.imshow('ImageWindow', rgb_frame)
                # cv2.waitKey()

                pil_frame = Image.fromarray(rgb_frame)

                file_name = os.path.join(img_dir, identity + str(face_number) + '.jpg')
                # file_name = img_dir + identity + str(face_number) + '.jpg'
                crop_face(pil_frame, eye_left=left_eye, eye_right=right_eye, offset_pct=(0.35, 0.35),
                          dest_sz=(img_h, img_w)).save(file_name)

                # if verbose:
                #     # Write the resulting image to the output video file
                #     print("Writing frame {} / {}".format(frame_number, length))
                # output_movie.write(frame)

    # All done!
    input_movie.release()
    cv2.destroyAllWindows()


def crop_face(image, eye_left=(0, 0), eye_right=(0, 0), offset_pct=(0.2, 0.2), dest_sz=(70, 70)):
    # calculate offsets in original image
    offset_h = math.floor(float(offset_pct[0])*dest_sz[0])
    offset_v = math.floor(float(offset_pct[1])*dest_sz[1])
    # get the direction
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    # calc rotation angle in radians
    rotation = -math.atan2(float(eye_direction[1]),float(eye_direction[0]))
    # distance between them
    dist = get_distance(eye_left, eye_right)
    # calculate the reference eye-width
    reference = dest_sz[0] - 2.0*offset_h
    # scale factor
    scale = float(dist)/float(reference)
    # rotate original around the left eye
    image = scale_rotate_translate(image, center=eye_left, angle=rotation)
    # crop the rotated image
    crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
    crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
    image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
    # resize it
    image = image.resize(dest_sz, Image.ANTIALIAS)
    return image


def get_centroid(coords):
    x_coords = [p[0] for p in coords]
    y_coords = [p[1] for p in coords]
    _len = len(coords)
    centroid_x = round(sum(x_coords)/_len)
    centroid_y = round(sum(y_coords)/_len)
    return [centroid_x, centroid_y]


def get_distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx*dx+dy*dy)


def scale_rotate_translate(image, angle, center=None, new_center=None, scale=None, resample=Image.BICUBIC):
    if (scale is None) and (center is None):
        return image.rotate(angle=angle, resample=resample)
    nx, ny = x, y = center
    sx = sy =1.0
    if new_center:
        (nx, ny) = new_center
    if scale:
        (sx, sy) = (scale, scale)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine/sx
    b = sine/sx
    c = x-nx*a-ny*b
    d = -sine/sy
    e = cosine/sy
    f = y-nx*d-ny*e
    return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=resample)




