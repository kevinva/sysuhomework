import torch
from torchvision import transforms

# IMAGE_DIR = '../data/cub_test/'
IMAGE_DIR = '../data/cub_200_2011/CUB_200_2011/images'

DATALOADER_NUM_WORKER = 4
BATCH_SIZE = 16
NUM_EPOCH = 20
LEARNING_RATE = 3e-6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRANSFORM_TRAIN = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(50),
                                      transforms.ToTensor()
                                    ])

TRANSFORM_VALID = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()
                                    ])