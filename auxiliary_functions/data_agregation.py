import os
import shutil

source_dir: str = "../data/age-regression/"
dest_dir: str = "../data/aggregated_data/"

def group_subfolders(source: str, dest: str) -> None:
    for i in range(5, 31,2):
        src_folder_1 = os.path.join(source, f'{i:03d}')
        src_folder_2 = os.path.join(source, f'{i+1:03d}')
        dest_folder = os.path.join(dest, f'{i:03d}')
        os.makedirs(dest_folder, exist_ok=True)

        for src_folder in [src_folder_1, src_folder_2]:
            print(src_folder)
            if os.path.exists(src_folder):
                for file_name in os.listdir(src_folder):
                    src_file = os.path.join(src_folder, file_name)
                    print(src_file)
                    if os.path.isfile(src_file):
                        shutil.copy(src_file, dest_folder)
            else:
                print(f"{src_folder} not founded.")


group_subfolders(source_dir, dest_dir)

