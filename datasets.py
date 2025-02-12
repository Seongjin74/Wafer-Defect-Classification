# datasets.py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class LabeledDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        images: numpy 배열, shape=(N, H, W, 3), 값은 0~1 범위
        labels: numpy 배열, 각 이미지에 해당하는 정수형 라벨
        transform: PIL.Image → Tensor 등 데이터 증강 변환 함수
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        # [0,1] 범위의 numpy 배열을 [0,255] uint8로 변환 후 PIL 이미지 생성
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

class UnlabeledDataset(Dataset):
    def __init__(self, images, weak_transform=None, strong_transform=None):
        """
        images: numpy 배열, shape=(N, H, W, 3), 값은 0~1 범위
        weak_transform: 약한 증강 함수 (예: flip, crop)
        strong_transform: 강한 증강 함수 (예: RandAugment 포함)
        """
        self.images = images
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        weak_img = self.weak_transform(img) if self.weak_transform else img
        strong_img = self.strong_transform(img) if self.strong_transform else img
        return weak_img, strong_img

class TestDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        images: numpy 배열, shape=(N, H, W, 3)
        labels: numpy 배열, 각 이미지의 라벨
        transform: 테스트 시 변환 (보통 ToTensor() 정도)
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
