import os
import shutil


def reset_folder(directory_path: str):
    if os.path.exists(directory_path):
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)


def count_file_and_folder(folder_path: str):
    folder_count = 0
    total_file_count = 0
    for root, dirs, files in os.walk(folder_path):
        folder_count += len(dirs)
        folder_file_count = len(files)
        total_file_count += folder_file_count
    return folder_count, total_file_count
