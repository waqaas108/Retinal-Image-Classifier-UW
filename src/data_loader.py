import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

class RetinalDataset(Dataset):
    def __init__(self, img_dir, label_path, transform=None):
        self.img_dir = img_dir
        self.labels = torch.load(label_path)  # Load the .pt file with labels
        self.img_filenames = list(self.labels.keys())  # Image filenames are the keys in the dict
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_filename = self.img_filenames[idx]
        img_path = os.path.join(self.img_dir, img_filename)
        
        # Open image
        image = Image.open(img_path).convert('RGB')  # Ensure it's 3-channel RGB
        if self.transform:
            image = self.transform(image)
        
        # Get the corresponding label
        label = self.labels[img_filename]
        
        # Convert label to class index if it's one-hot or probability vector
        if isinstance(label, (list, np.ndarray, torch.Tensor)) and len(label) > 1:
            label = torch.tensor(label, dtype=torch.float32)
            label = torch.argmax(label).long()
        else:
            label = torch.tensor(label, dtype=torch.long)
        
        return image, label

# split_dir: Directory to save or load splits
def save_splits(train_indices, val_indices, test_indices, split_dir):
    os.makedirs(split_dir, exist_ok=True)
    np.save(os.path.join(split_dir, 'train_indices.npy'), train_indices)
    np.save(os.path.join(split_dir, 'val_indices.npy'), val_indices)
    np.save(os.path.join(split_dir, 'test_indices.npy'), test_indices)

def load_splits(split_dir):
    train_indices = np.load(os.path.join(split_dir, 'train_indices.npy'))
    val_indices = np.load(os.path.join(split_dir, 'val_indices.npy'))
    test_indices = np.load(os.path.join(split_dir, 'test_indices.npy'))
    return train_indices, val_indices, test_indices

def get_dataloaders(img_dir, label_path, split_dir=None, batch_size=32, val_split=0.2, test_split=0.1, shuffle=True, num_workers=4, save_split=True):
    # Dataset with transformations
    dataset = RetinalDataset(img_dir, label_path, transform=transforms.Compose([
        transforms.Resize((256, 256)),              # Resize images to 256x256
        transforms.RandomHorizontalFlip(),          # Random horizontal flip
        transforms.RandomRotation(30),              # Random rotation within [-30, 30] degrees
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ]))

    # Check if split files exist
    if split_dir and all([os.path.exists(os.path.join(split_dir, f'{split}.npy')) for split in ['train_indices', 'val_indices', 'test_indices']]):
        # Load saved splits
        train_indices, val_indices, test_indices = load_splits(split_dir)
    else:
        # Train, validation, test split with stratification
        total_indices = list(range(len(dataset)))
        # Extract labels for stratification
        labels = [dataset.labels[fn] for fn in dataset.img_filenames]
        train_indices, temp_indices = train_test_split(
            total_indices, 
            test_size=(val_split + test_split), 
            random_state=42, 
            stratify=labels
        )
        # Further split temp into validation and test
        temp_labels = [labels[i] for i in temp_indices]
        val_size = int(val_split / (val_split + test_split) * len(temp_indices))
        val_indices, test_indices = train_test_split(
            temp_indices, 
            test_size=(len(temp_indices) - val_size), 
            random_state=42, 
            stratify=temp_labels
        )

        # Save splits if necessary
        if save_split and split_dir:
            save_splits(train_indices, val_indices, test_indices, split_dir)

    # Create subset datasets using indices
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    img_dir = '../data/final/'
    label_path = '../data/final/labels.pt'
    split_dir = '../splits/'

    train_loader, val_loader, test_loader = get_dataloaders(img_dir, label_path, split_dir=split_dir)
    for img, label in train_loader:
        print(img.size(), label.dtype, label)
        break  # Print only the first batch for verification