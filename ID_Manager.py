import os
import pickle
from Video_Processor import get_faces_from_vid
from Image_Processor import process_face_images
from Face_Clusterer import get_N_centroids


def regenerate_centroids(data_dir, n_centroids=10):
    # Get immediate subdirectories
    dir_list = next(os.walk(data_dir))[1]

    for id_dir in dir_list:
        process_id_folder(os.path.join(data_dir, id_dir), n_centroids=n_centroids, process_vid=False, process_imgs=False, process_centroids=True)


def get_master_identities(data_dir):

    # Get immediate subdirectories
    dir_list = next(os.walk(data_dir))[1]

    id_names = []
    id_encodings = []

    for folder in dir_list:
        pickle_file = data_dir + '/' + folder + '/' + folder + '.pickle'
        if os.path.isfile(pickle_file):
            encodings = read_from_pickle(pickle_file)

            for enc in encodings:
                id_names.append(folder)
                id_encodings.append(enc)
        else:
            print('Not Found:', pickle_file)

    id_dict = {"encodings": id_encodings, "names": id_names}
    return id_dict


def process_id_folder(id_dir, n_centroids=10, process_vid=False, process_imgs=False, process_centroids=False):

    # Get name of folder = identity
    id_str = get_id_str(id_dir)

    # Set all folder and file paths
    vid_dir = os.path.join(id_dir, 'Videos')
    img_dir = os.path.join(id_dir, 'Images')
    pickle_paths = os.path.join(id_dir, 'Img_Paths.pickle')
    pickle_encodings = os.path.join(id_dir, 'Embeddings.pickle')
    pickle_centroids = os.path.join(id_dir, id_str + '.pickle')

    if process_vid:

        # Check if IMAGES folder path exists, or create
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        # Process if video files exist
        for root, dirs, files in os.walk(vid_dir):
            for file in files:
                vid_path = os.path.join(root, file)
                print('Processing:', file)
                try:
                    get_faces_from_vid(vid_path, img_dir, skip_frames=0, max_frames=2500)
                except:
                    print('Error:', vid_path)
    else:
        print('Videos processing turned off')

    if process_imgs:

        # Process the face images to generate 3-D vectors
        paths, encodings = process_face_images(img_dir)

        # Save encodings and file paths to file
        save_to_pickle(encodings, pickle_encodings)
        save_to_pickle(paths, pickle_paths)
    else:
        print('Images processing turned off')

    if process_centroids:
        print('Generating', n_centroids, 'centroids from encodings')
        if not process_imgs:
            encodings = read_from_pickle(pickle_encodings)
        # Cluster and get N centroids
        ref_centroids = get_N_centroids(encodings, n_clusters=n_centroids, show_plot=True)

        # Save centroids to file
        save_to_pickle(ref_centroids, pickle_centroids)
    else:
        print('Centroids processing turned off')


def get_id_str(path):
    path_split = os.path.dirname(path).split('/')
    return path_split[-1]


def read_from_pickle(pickle_path):
    with open(pickle_path, 'rb') as handle:
        variable = pickle.load(handle)
        return variable


def save_to_pickle(variable, pickle_path):
    # If pickle exists, delete pickle
    try:
        os.remove(pickle_path)
    except OSError:
        pass

    with open(pickle_path, 'wb') as handle:
        pickle.dump(variable, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    id_dir = 'C:/WorkData/FaceRecognition/Datasets/Pilot_1/Dharmendra/'
    process_id_folder(id_dir, n_centroids=20, process_vid=True, process_imgs=True, process_centroids=True)

    # data_dir = 'C:/WorkData/FaceRecognition/Datasets/Pilot_1/'
    # regenerate_centroids(data_dir, n_centroids=20)
    # id_dict = get_master_identities(data_dir)
