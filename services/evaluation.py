import json
import os

import numpy as np
from keras.models import load_model
from keras.preprocessing import image

from models.vgg16_model import load_metadata
from utils import find_best_solution
import matplotlib.pyplot as plt


class ResultStorage:
    def __init__(self):
        self.result = {}

    def increase(self, label: str):
        if label not in self.result:
            self.result[label] = 1
        else:
            self.result[label] += 1

    def get(self, label: str):
        if label not in self.result:
            return 0
        else:
            return self.result[label]


class Evaluation:
    @staticmethod
    def calculate_positive_and_negative(validate_path: str):
        saved_model = load_model("resources/rgb/vgg16_1.h5")
        num_classes, class_indices, labels = load_metadata("resources/metadata.json")
        tp_result = ResultStorage()
        fp_result = ResultStorage()
        fn_result = ResultStorage()
        precision_result = {}
        recall_result = {}
        names = []
        for folder_name in os.listdir(validate_path):
            names.append(folder_name)
            sub_folder_path = os.path.join(validate_path, folder_name)
            for image_name in os.listdir(sub_folder_path):
                src_path = os.path.join(sub_folder_path, image_name)
                img = image.load_img(src_path, target_size=(224, 224))
                img = np.asarray(img)
                img = np.expand_dims(img, axis=0)
                output = saved_model.predict(img)
                predicted_index, _ = find_best_solution(output)
                real_index = class_indices[folder_name]
                predicted_name = labels[str(predicted_index)]
                real_name = folder_name
                if predicted_index == real_index:
                    tp_result.increase(real_name)
                else:
                    fp_result.increase(predicted_name)
                    fn_result.increase(real_name)
        for name in names:
            tp = tp_result.get(name)
            fp = fp_result.get(name)
            fn = fn_result.get(name)
            precision_result[name] = tp / (tp + fp)
            recall_result[name] = tp / (tp + fn)
        return precision_result, recall_result

    @staticmethod
    def draw_loss():
        file = open("resources/loss.json")
        data = json.load(file)
        file.close()
        train_loss = data["loss"]
        val_loss = data["val_loss"]

        plt.plot(train_loss)
        plt.plot(val_loss)

        plt.xlabel("Epoch")
        plt.legend(["Training Loss", "Validate Loss"])
        plt.show()

    @staticmethod
    def metrics():
        file = open("resources/val_test.json")
        data = json.load(file)
        file.close()
        precision_result = data["precision_result"]
        recall_result = data["recall_result"]
        counter = 0
        precision_total = 0
        recall_total = 0
        f1_total = 0
        for key, value in precision_result.items():
            counter += 1
            precision_total += value
            recall_total += recall_result[key]
            f1_total += (2 * value * recall_result[key]) / (value + recall_result[key])
        return precision_total / counter, recall_total / counter, f1_total / counter
