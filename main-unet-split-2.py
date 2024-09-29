import random
import torch
import numpy as np
import Trans_mod_unet_split

seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Device Configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("\nSelected device:", device, end="\n\n")

tmod = Trans_mod_unet_split.Train_test(dataset='dc', device=device, skip_train=False, save=True, original=False, save_dir="shuffle-L1")
tmod.run(smry=False)
