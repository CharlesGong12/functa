from paddle.vision import transforms
import dataloaders.mnist as mnist
import paddle

def create_dataloader(dset, dset_root=None, batch_size=8):
    """Create dataloader for selected dataset"""
    if dset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = mnist.Mnist(split='train',transforms=transform)
    elif dset == "cifar10":
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(
             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # transform = transforms.Compose([
        #     transforms.Resize(size=(64, 64)),
        #     transforms.RandomHorizontalFlip(),                        
        #     transforms.RandomCrop(64,padding=4),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        dataset = mnist.Cifar(split='train',transforms=transform)
    # Other options...

    dataloader = paddle.io.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
    )

    return dataloader

def test_dataloader(dset, dset_root=None, batch_size=8):
    """Create dataloader for selected dataset"""
    if dset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = mnist.Mnist(split='test',transforms=transform)
    elif dset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = mnist.Cifar(split='test',transforms=transform)
    # Other options...

    dataloader = paddle.io.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
    )

    return dataloader