import sys

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import time
from datetime import timedelta

class App_Dataset(Dataset):
  def __init__(self, features, labels, device):
      self.features = torch.FloatTensor(features).to(device)
      self.labels = torch.LongTensor(labels).to(device)

  def __len__(self):
      return len(self.labels)

  def __getitem__(self, idx):
      return self.features[idx], self.labels[idx]

def load_cifar10(path):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)

    return train_set, test_set

class Data:
    def __init__(self, batch_size, train_set, test_set):
        #self.name = name
        
        #if self.name == "CIFAR10":
        #    self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        #    self.train_set, self.test_set = load_cifar10(path)
        #elif self.name == 'Custom':

        self.train_set = train_set
        self.test_set = test_set
        w = 0
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=w)
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, shuffle=False, num_workers=w)


def dict_to_class(class_name, data_dict):

   def __init__(self, **kwargs):
        # Set default values from the original dictionary
        for key, value in data_dict.items():
            setattr(self, key, value)

        # Allow overriding defaults with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

class App:
    def __init__(self, params):
        print("App Starting...")

        #
        # Picking Device
        #
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"Available GPUs:")
            for i in range(num_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"Device: {self.device}")

        #
        # Adding "global" parameters
        #

        # Define default parameters
        default_params = {
            'data_path': './data',
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 100,
            'model_name': 'default_model',
            "model_output": "./default.model",
            'dataset_name': None
        }

        if params is None:
            # If no params provided, use defaults
            self.params = default_params
        else:
            # Merge provided params with defaults (provided params override defaults)
            self.params = {**default_params, **params}

        for key, value in self.params.items():
            setattr(self, key, value)

        #
        # Setting up data
        #
        if self.dataset_name != 'Custom':
            self.data = Data(self.data_path, self.dataset_name, self.batch_size)


    def main(self):
        self.start_time = time.time()
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        if self.device == 'cuda':
            self.model = torch.nn.DataParallel(self.model)
            cudnn.benchmark = True

        if len(sys.argv) != 2:
            print("Usage: <program.py> <train/test>")
            return

        if sys.argv[1] == "train":
            self.train_func(self)
            torch.save(self.model.state_dict(), self.model_output)
        elif sys.argv[1] == "test":
            self.model.load_state_dict(torch.load(self.model_output, weights_only=True))
            self.test_func(self)

        self.quit()

    def set_optimizer(self):
        # OPTIMIZER
        if self.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        elif self.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        else:
            print("NO OPTIMIZER SELECTED")


    def quit(self):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        print(f'Running Time: {timedelta(seconds=elapsed_time)}')

    def pbar(self, epoch):
        return tqdm(self.data.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch", leave=False)

    def test_pbar(self):
        return tqdm(self.data.test_loader, desc=f"Testing", unit="batch", leave=False)

    def create_dataset(self, features, labels):
        return App_Dataset(features, labels, self.device)