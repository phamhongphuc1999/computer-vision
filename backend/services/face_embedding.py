import cv2
from facenet_pytorch import InceptionResnetV1, extract_face
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os


class FaceEmbedding:
    @staticmethod
    def embedding(rgb_img: cv2.UMat, faces: list, face_embedder: InceptionResnetV1):
        embeddings = [
            face_embedder(extract_face(rgb_img, face["box"]).unsqueeze(0))
            .squeeze()
            .detach()
            .numpy()
            for face in faces
        ]
        return embeddings

    @staticmethod
    def images_embedding(
        folder_path: str,
        face_embedder: InceptionResnetV1,
        device: str,
        output_directory: str,
    ):
        embeddings = {}
        data_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
                ),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
            ]
        )
        label = os.path.basename(folder_path)
        for image_name in tqdm(
            os.listdir(folder_path), desc=f"Processing {folder_path}"
        ):
            image_path = os.path.join(folder_path, image_name)
            try:
                img = Image.open(image_path)
                img_tensor = transforms.ToTensor()(img).unsqueeze(0).float().to(device)

                embedding = face_embedder(img_tensor).squeeze().detach().cpu().numpy()
                embeddings[image_name] = embedding

                output_emb_path = os.path.join(
                    output_directory,
                    f"{label}_{os.path.splitext(image_name)[0]}_embedding.npy",
                )
                np.save(output_emb_path, embedding)

                # Apply data augmentation
                augmented_img = data_transform(img)
                img_tensor_augmented = augmented_img.unsqueeze(0).float().to(device)
                embedding_augmented = (
                    face_embedder(img_tensor_augmented).squeeze().detach().cpu().numpy()
                )
                embeddings[
                    f"{os.path.splitext(image_name)[0]}_augmented_embedding.npy"
                ] = embedding_augmented
                output_emb_path_augmented = os.path.join(
                    output_directory,
                    f"{label}_{os.path.splitext(image_name)[0]}_augmented_embedding.npy",
                )
                np.save(output_emb_path_augmented, embedding_augmented)
            except Exception as e:
                print(f"Error processing {image_name}: {str(e)}")
        return label, embeddings
