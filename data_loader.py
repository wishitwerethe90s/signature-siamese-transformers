# In data_loader.py

import os
import random
import torch
from torch.utils.data import Dataset
from utils import preprocess_signature

class TripletSignatureDataset(Dataset):
    """
    Custom Dataset for generating triplets for training the Siamese-Transformer.
    A triplet consists of an (anchor, positive, negative) sample.
    - Anchor: A genuine signature of a person.
    - Positive: A different genuine signature from the same person.
    - Negative: A signature from a different person OR a forged signature of the same person.
    """
    def __init__(self, data_dir, image_size=(224, 224), forgery_prob=0.7, augment=True):
        super().__init__()

        # TODO: Play with probailities.
        self.forgery_prob = forgery_prob  # (!!!) Prefer forgeries as negatives
        self.augment = augment
        self.data_dir = data_dir
        self.image_size = image_size
        self.users = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.user_to_images = self._create_user_to_images_map()
        self.anchor_list = self._create_anchor_list()

    def _create_user_to_images_map(self):
        """Creates a map from user ID to their genuine and forged signatures."""
        user_to_images = {}
        for user in self.users:
            user_path = os.path.join(self.data_dir, user)
            genuine_files = [os.path.join(user_path, f) for f in os.listdir(user_path) if 'forged' not in f.lower()]
            forged_files = [os.path.join(user_path, f) for f in os.listdir(user_path) if 'forged' in f.lower()]
            
            # Ensure user has at least 2 genuine signatures to form a positive pair
            if len(genuine_files) >= 2:
                user_to_images[user] = {'genuine': genuine_files, 'forged': forged_files}
        return user_to_images

    def _create_anchor_list(self):
        """Creates a flat list of all possible anchor images."""
        anchor_list = []
        for user, files in self.user_to_images.items():
            for genuine_path in files['genuine']:
                anchor_list.append((user, genuine_path))
        return anchor_list

    def __len__(self):
        """Returns the total number of anchor images."""
        return len(self.anchor_list)

    def __getitem__(self, idx):
        """
        Generates and returns a single triplet (anchor, positive, negative).
        """
        anchor_user, anchor_path = self.anchor_list[idx]

        # --- Select a Positive Sample ---
        # It must be a different genuine signature from the same user.
        positive_list = self.user_to_images[anchor_user]['genuine']
        positive_path = anchor_path
        while positive_path == anchor_path:
            positive_path = random.choice(positive_list)

        # --- Select a Negative Sample ---
        # With "forgery_prob" probability, pick a forged signature from the same user.
        # With "1-forgery_prob" probability, pick a genuine signature from a different user.
        if random.random() < self.forgery_prob and len(self.user_to_images[anchor_user]['forged']) > 0:
            # Forgery from same user (harder negative)
            negative_path = random.choice(self.user_to_images[anchor_user]['forged'])
        else:
            # Different user's signature
            other_users = [u for u in self.user_to_images.keys() if u != anchor_user]
            negative_user = random.choice(other_users)
            # Mix genuine and forged from other users
            all_negatives = (self.user_to_images[negative_user]['genuine'] + 
                           self.user_to_images[negative_user]['forged'])
            negative_path = random.choice(all_negatives)

        # --- Preprocess Images ---
        anchor_img = preprocess_signature(anchor_path, self.image_size)
        positive_img = preprocess_signature(positive_path, self.image_size)
        negative_img = preprocess_signature(negative_path, self.image_size)

        return (
            torch.from_numpy(anchor_img),
            torch.from_numpy(positive_img),
            torch.from_numpy(negative_img)
        )