import torch
from timm import create_model
from torchvision import transforms, datasets
import lightning as L
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
import matplotlib.image as mpimg
from tqdm import tqdm
import torchvision

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((100, 100)), # set size to 10 by 10
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class LitClassification(L.LightningModule):
    def __init__(self,lr=0.005):
        super().__init__()
        self.lr = lr
        
        print("Initialize LitClassification object...")
        print("self.lr: " + str(self.lr))
        self.model = create_model('resnet34', num_classes=5) # num_classes = 5
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(),lr=self.lr) # lr=0.005)  (1. lr = 0.1 2. 0.01, 3.0.005)

    def training_step(self, batch):
        images, targets = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        # implement your own
        out = self.model(x)
        labels_hat = torch.argmax(out, dim=1)
        test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        # log the outputs!
        self.log_dict({'test_acc': test_acc})


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        print("Initialize CustomImageDataset object...")        
        full_data = pd.read_csv(annotations_file)

        # delete images which are not RGB images
        drop_rows_idx = []
        for i in tqdm(range(len(full_data))):
            specific_image_file_name = full_data["image"].iloc[i]
            # print(specific_image_file_name)
            full_path = img_dir + specific_image_file_name
            img = mpimg.imread(full_path)
            if len(img.shape) != 3:
                drop_rows_idx.append(i)

        full_data.drop(drop_rows_idx, inplace=True)
        full_data.reset_index(inplace=True)
        full_data.drop(["index"], axis=1, inplace=True)
        full_data["category"] = full_data["category"] - 1 # very important to get categories between 0 and 4
        self.img_labels = full_data.head(5000) # select only 100 images
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image_tensor = read_image(img_path)
        image_pil = to_pil_image(image_tensor)
        label = self.img_labels.iloc[idx, 1]
        # print(label)
        # print(idx)
        if self.transform:
            image = self.transform(image_pil)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

class CustomImageDataset_Test(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        print("Initialize CustomImageDataset object...")        
        full_data = pd.read_csv(annotations_file)

        # delete images which are not RGB images
        drop_rows_idx = []
        for i in tqdm(range(len(full_data))):
            specific_image_file_name = full_data["image"].iloc[i]
            # print(specific_image_file_name)
            full_path = img_dir + specific_image_file_name
            img = mpimg.imread(full_path)
            if len(img.shape) != 3:
                drop_rows_idx.append(i)

        full_data.drop(drop_rows_idx, inplace=True)
        full_data.reset_index(inplace=True)
        full_data.drop(["index"], axis=1, inplace=True)
        full_data["category"] = full_data["category"] - 1 # very important to get categories between 0 and 4
        self.img_labels = full_data.tail(1000) # select only 100 images
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image_tensor = read_image(img_path)
        image_pil = to_pil_image(image_tensor)
        label = self.img_labels.iloc[idx, 1]
        # print(label)
        # print(idx)
        if self.transform:
            image = self.transform(image_pil)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

class ClassificationData(L.LightningDataModule):

    def train_dataloader(self):
        print("Run Train Data Loader")     
        train_dataset = CustomImageDataset(annotations_file="../data/archive/train/train.csv", img_dir="../data/archive/train/images/", transform=DEFAULT_TRANSFORM)
        return torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True,  num_workers=1, persistent_workers=True) #,
    
    # def test_dataloader... :)
    def test_dataloader(self):
        print("Run Test Data Loader")     
        test_dataset = CustomImageDataset_Test(annotations_file="../data/archive/train/train.csv", img_dir="../data/archive/train/images/", transform=DEFAULT_TRANSFORM)
        return torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=True,  num_workers=1, persistent_workers=True) #,



if __name__ == "__main__":

    
    list_epochs = [5,10,15,20,25,30,35,40,45,50]

    lr_list = [0.005, 0.01, 0.1]

    for lr in lr_list:
        for epochs in list_epochs:
            model = LitClassification(lr=lr)
            data = ClassificationData()
            print("+++++++++++++++++++++++++++++")
            print("EPOCH_NUMBER: " + str(epochs))
            trainer = L.Trainer(max_epochs=epochs)
            trainer.fit(model, data)
            trainer.test(model, data)
            print("EPOCH_NUMBER: " + str(epochs))
            print("with lr: " + str(lr))
            print("+++++++++++++++++++++++++++++")
    


# am besten 6x pro Wert laufen lassen und dann Mittelwert bestimmen und Abweichung