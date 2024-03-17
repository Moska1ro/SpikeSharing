from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

#args:gpu data_path train_batch_size eval_batch_size
class Data:
    def __init__(self, batch_size, data_path):
        pin_memory = True

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

        trainset = CIFAR10(root=data_path, train=True, download=True, transform=transform_train)

        self.trainLoader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=pin_memory# win num_workers=0
        )

        testset = CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
        self.testlen = len(testset)
        self.testLoader = DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=pin_memory)# same as above
        # torch.cuda.OutOfMemoryError:CUDA out of memory.