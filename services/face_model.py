from typing import List

import torch
import torch.nn as nn
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder


def get_train_data(data_path: str):
    embeddings = []
    labels = []

    for label_folder in os.listdir(data_path):
        label_path = os.path.join(data_path, label_folder)
        if os.path.isdir(label_path):
            label = label_folder
            embeddings_per_label = []
            for file_name in os.listdir(label_path):
                file_path = os.path.join(label_path, file_name)
                if file_name.endswith(".npy"):
                    embedding = np.load(file_path)
                    embeddings_per_label.append(embedding)
                    labels.append(label.split("_")[1])
            embeddings.append(embeddings_per_label)
    embeddings = [item for sublist in embeddings for item in sublist]
    return embeddings, np.array(labels)


def encode_labels(labels: List[str]):
    full_encoder = LabelEncoder()
    full_encoded_labels = torch.tensor(
        full_encoder.fit_transform(labels), dtype=torch.long
    )
    result = {}
    for label in labels:
        result[label] = True
    unique_labels = list(result.keys())
    return full_encoded_labels, unique_labels


class FaceModel(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super(FaceModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x
