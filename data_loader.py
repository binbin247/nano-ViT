from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import os

def get_data_loader(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    """Build training and evaluation loaders with dataset-specific transforms.

    Args:
        args: Configuration containing dataset (MNIST, FashionMNIST, CIFAR10,
            CIFAR100, or ImageNet-1K via imagenet1k/imagenet-1k/imagenet_1k),
            data_path (parent dataset directory), image_size (output
            ``(height, width)``), batch_size (training batch size), and
            num_workers.
            The dataset name is converted to lowercase in place. Datasets
            are downloaded as needed except ImageNet-1K, which must already
            be available under data_path/dataset.

    Returns:
        A (train_loader, test_loader) tuple. Training uses augmentation,
        shuffling, and drops the final incomplete batch. Evaluation uses
        twice the training batch size, preserves order, and keeps all samples;
        ImageNet-1K evaluation uses the validation split.

    Raises:
        ValueError: If the dataset name is unsupported.
    """
    args.dataset = args.dataset.lower()

    if args.dataset == "mnist":
        # MNIST: 60,000 train + 10,000 test, 28x28 grayscale, about 55 MB uncompressed.
        root = os.path.join(args.data_path, args.dataset)
        mnist_mean = (0.5,)
        mnist_std = (0.5,)

        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mnist_mean, std=mnist_std),
        ])
        train = datasets.MNIST(
            root=root,
            train = True,
            transform = train_transform,
            download = True
        )

        test_transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mnist_mean, std=mnist_std),
        ])
        test = datasets.MNIST(
            root=root,
            train = False,
            transform = test_transform,
            download = True
        )

    elif args.dataset == "fashionmnist":
        # FashionMNIST: 60,000 train + 10,000 test, 28x28 grayscale, about 55 MB uncompressed.
        root = os.path.join(args.data_path, args.dataset)
        fashionmnist_mean = (0.5,)
        fashionmnist_std = (0.5,)

        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=fashionmnist_mean, std=fashionmnist_std),
        ])
        train = datasets.FashionMNIST(
            root=root,
            train=True,
            transform=train_transform,
            download=True
        )

        test_transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=fashionmnist_mean, std=fashionmnist_std),
        ])
        test = datasets.FashionMNIST(
            root=root,
            train=False,
            transform=test_transform,
            download=True
        )

    elif args.dataset == "cifar10":
        # CIFAR-10: 50,000 train + 10,000 test, 10 classes, 32x32 RGB, about 170 MB download.
        root = os.path.join(args.data_path, args.dataset)
        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2470, 0.2435, 0.2616)

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
        ])
        train = datasets.CIFAR10(
            root=root,
            train=True,
            transform=train_transform,
            download=True,
        )

        test_transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
        ])
        test = datasets.CIFAR10(
            root=root,
            train=False,
            transform=test_transform,
            download=True,
        )

    elif args.dataset == "cifar100":
        # CIFAR-100: 50,000 train + 10,000 test, 100 classes, 32x32 RGB, about 170 MB download.
        root = os.path.join(args.data_path, args.dataset)
        cifar100_mean = (0.5071, 0.4867, 0.4408)
        cifar100_std = (0.2675, 0.2565, 0.2761)

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
        ])
        train = datasets.CIFAR100(
            root=root,
            train=True,
            transform=train_transform,
            download=True,
        )

        test_transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
        ])
        test = datasets.CIFAR100(
            root=root,
            train=False,
            transform=test_transform,
            download=True,
        )

    elif args.dataset in {"imagenet1k", "imagenet-1k", "imagenet_1k"}:
        # Converted ImageNet-1K: train/ and val/ contain class subdirectories.
        imagenet_root = os.path.join(args.data_path, "imagenet1k")
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
        train = datasets.ImageFolder(
            root=os.path.join(imagenet_root, "train"),
            transform=train_transform,
        )

        test_transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
        test = datasets.ImageFolder(
            root=os.path.join(imagenet_root, "val"),
            transform=test_transform,
        )

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    train_loader = DataLoader(
        dataset=train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test,
        batch_size=args.batch_size*2,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    return train_loader, test_loader



# download dataset
if __name__ == "__main__":
    args = argparse.Namespace(
        dataset="cifar10",
        data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        image_size=(32, 32),
        batch_size=64,
        num_workers=0,
    )
    train_loader, test_loader = get_data_loader(args)
    print(f"Dataset directory: {os.path.join(args.data_path, args.dataset)}")
    print(f"Training images: {len(train_loader.dataset):,}")
    print(f"Test images: {len(test_loader.dataset):,}")
