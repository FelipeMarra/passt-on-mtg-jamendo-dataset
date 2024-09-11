import torch.nn as nn
from torch.autograd import Variable


import torch.nn as nn

from hear21passt.base import get_basic_model, get_model_passt


class PaSSTMTG(nn.Module):
    def __init__(self, n_classes=183):
        super(PaSSTMTG, self).__init__()

        self.passt = get_basic_model(mode="logits")
        self.passt.net =  get_model_passt(arch="passt_s_swa_p16_128_ap476", n_classes=n_classes)

    def forward(self, x):
        print(f"\n############## Forward ##################\nx: {x.shape}\n")
        logit = self.passt(x)
        print(f"logit: {logit.shape} \n####################################\n")

        return logit 