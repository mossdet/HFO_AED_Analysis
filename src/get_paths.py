import os

path = os.path.dirname(os.path.abspath(__file__))
cutIdx = path.rfind(os.path.sep)
workspacePath = path[:cutIdx]
data_path = workspacePath + os.path.sep + 'Data' + os.path.sep
images_path = workspacePath + os.path.sep + 'Images' + os.path.sep
annotations_path = data_path+"Annotations"+os.path.sep
edfs_path = data_path+"EEG"+os.path.sep

os.makedirs(data_path, exist_ok=True)
os.makedirs(images_path, exist_ok=True)
os.makedirs(annotations_path, exist_ok=True)


def get_edf_files():
    files = os.listdir(edfs_path)
    return files
