import os
import shutil
import numpy as np

if __name__ == '__main__':
    data_dir: str = "./data/"
    dest_folder: str = "./datasets/ds2/cls1"
    os.makedirs(dest_folder, exist_ok=True)
    # select N random samples from each folder in
    # data_dir and copy them to datasets
    for folder in os.listdir(data_dir)[5:10]:
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            np.random.shuffle(files)
            for file in files[:10]:
                shutil.copy(os.path.join(folder_path, file), dest_folder)
