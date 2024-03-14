# import torch
# import torchvision.transforms as transforms

# from PIL import Image

import fastbook
fastbook.setup_book()

from fastbook import *
from fastai.vision.widgets import *

# model = torch.load('./trained_bear_model.pkl', map_location='cpu')
# model.eval()  # Set the model to evaluation mode

learn_inf = load_learner('./trained_bear_model.pkl')
prediction = learn_inf.predict('./teddy_bear.jpeg')
print(prediction)



