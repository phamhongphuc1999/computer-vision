import json
import os


def get_encodings(data_directory: str):
    files = os.listdir(data_directory)
    known_encodings = []
    known_names = []
    for file_name in files:
        src_path = os.path.join(data_directory, file_name)
        file = open(src_path)
        data = json.load(file)
        file.close()
        _len = len(data)
        known_encodings += data
        count = 0
        while count < _len:
            count += 1
            known_names.append(file_name[0:-5])
    return known_encodings, known_names
