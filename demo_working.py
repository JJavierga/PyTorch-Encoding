import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils import data
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torchvision.transforms as transform

import encoding.utils as utils
from encoding.nn import SegmentationLosses, DistSyncBatchNorm, SyncBatchNorm, BatchNorm1d

from encoding.datasets import get_dataset
from encoding.models import get_segmentation_model


torch.cuda.set_device(0)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
cudnn.benchmark = True
# data transforms
input_transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize([.485, .456, .406], [.229, .224, .225])])
# dataset
data_kwargs = {'transform': input_transform, 'base_size': 520,
                'crop_size': 480}
nclass=2
# model
#model = get_segmentation_model('fcn', dataset='ade20k',
#                                backbone='resnest50', aux=True,
#                                se_loss=False, norm_layer=SyncBatchNorm,
#                                base_size=520, crop_size=480)
model = get_segmentation_model('deeplab', dataset='ade20k',
                                backbone='resnest50', aux=True,
                                se_loss=False, norm_layer=SyncBatchNorm,
                                base_size=520, crop_size=480)

# distributed data parallel

model.cuda(0)
# resuming checkpoint

#thick:
#direction = "/home/grvc/programming/gluon/PyTorch-Encoding/runs/ade20k/fcn/resnest50/default/checkpoint_works.pth.tar"
#thinn:
#direction = "/home/grvc/programming/gluon/PyTorch-Encoding/runs/ade20k/deeplab/resnest50/default/checkpoint_thin.pth.tar"
#mid:
direction = "/home/grvc/programming/gluon/PyTorch-Encoding/runs/ade20k/deeplab/resnest50/new_iterations/checkpoint_mid.pth.tar"
checkpoint = torch.load(direction)
model.load_state_dict(checkpoint['state_dict'])
print("=> loaded checkpoint '{}' (epoch {})"
        .format(direction, checkpoint['epoch']))


# Prepare the image
import matplotlib.pyplot as plt
#img_dir = "/media/grvc/MAMG_ViGUS/GRVC/Datasets/ML/Bridges_or_cracks/Asphalt/TITS/ordered/images/training"
img_dir = "/media/grvc/MAMG_ViGUS/GRVC/Datasets/ML/Bridges_or_cracks/Cracks/D/CD"
for filename in os.listdir(img_dir):
    img = utils.load_image(os.path.join(img_dir,filename)).cuda().unsqueeze(0)

    # Make prediction

    output = model.evaluate(img)

    predict = torch.max(output, 1)[1].cpu().numpy() 
    print(np.sum(predict))

    # Get color pallete for visualization
    mask = utils.get_mask_pallete(predict, 'ade20k')

    f, axarr = plt.subplots(2)
    axarr[0].imshow(mask)
    axarr[1].imshow(plt.imread(os.path.join(img_dir, filename)))
    plt.show()
