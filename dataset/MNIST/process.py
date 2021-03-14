import os
import torch
from torchvision.datasets.mnist import read_image_file, read_label_file

raw_folder = "raw"
processed_folder = "processed"
training_file = 'training.pt'
test_file = 'test.pt'

### Code from https://github.com/pytorch/vision/blob/7d4154735f421b254c408c16e0980b1ca0dd9b8e/torchvision/datasets/mnist.py#L134
# process and save as torch files
print('Processing...')

training_set = (
    read_image_file(os.path.join(raw_folder, 'train-images.idx3-ubyte')),
    read_label_file(os.path.join(raw_folder, 'train-labels.idx1-ubyte'))
)
test_set = (
    read_image_file(os.path.join(raw_folder, 't10k-images.idx3-ubyte')),
    read_label_file(os.path.join(raw_folder, 't10k-labels.idx1-ubyte'))
)
with open(os.path.join(processed_folder, training_file), 'wb') as f:
    torch.save(training_set, f)
with open(os.path.join(processed_folder, test_file), 'wb') as f:
    torch.save(test_set, f)

print('Done!')