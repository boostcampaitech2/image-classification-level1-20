import torchvision.transforms as transforms
from PIL import Image

class Efficient_transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((300,300), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),

        ])
    

    def __call__(self,X):
        return self.transform(X)



