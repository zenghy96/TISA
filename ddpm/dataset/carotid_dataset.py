from torch.utils.data import Dataset
from torchvision import transforms as T
import cv2
from .utils import list_files


class CUBSDataset(Dataset):
    def __init__(self, data_dir, resolution):
        super(CUBSDataset, self).__init__()
        self.images_path = list_files(data_dir)
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize((resolution, resolution), antialias=False),
            T.RandomHorizontalFlip(),
            T.Normalize(mean=[.5], std=[.5])
        ])

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        image_path = self.images_path[item]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = self.transforms(image)
        return image

