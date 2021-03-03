import torch
import encoding
from encoding.models import get_segmentation_model

# Get the model
model = get_segmentation_model('deeplab')
print(model)
checkpoint = torch.load('/home/grvc/programming/gluon/PyTorch-Encoding/runs/ade20k/deeplab/resnet50/default/model_best.pth.tar')
print(checkpoint['state_dict'])
model.load_state_dict(checkpoint['state_dict'])

# Prepare the image
img = encoding.utils.load_image("/home/grvc/.encoding/data/ADEChallengeData2016/images/training/GT_AIGLE_RN_C18a.jpg").cuda().unsqueeze(0)

# Make prediction
output = model.evaluate(img)
predict = torch.max(output, 1)[1].cpu().numpy() + 1

# Get color pallete for visualization
mask = encoding.utils.get_mask_pallete(predict, 'ade20k')
mask.save('output.png')
