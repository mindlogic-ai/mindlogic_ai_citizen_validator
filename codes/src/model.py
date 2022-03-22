from transformers import ElectraForSequenceClassification, ElectraTokenizer
import torch.nn as nn
import torch
import numpy as np

class ElectraRecommendClassifier(nn.Module):
    def __init__(self, ckpt = None ,drop_p = 0.2):
        super(ElectraRecommendClassifier, self).__init__()
        self.drop_p = drop_p
        if ckpt is not None:
            # valid version
            # self.electra_base = ElectraForSequenceClassification.from_pretrained(ckpt, num_labels = 4)
            self.electra_base = ElectraForSequenceClassification.from_pretrained(ckpt)
        else:
            # train version
            self.electra_base = ElectraForSequenceClassification.from_pretrained('beomi/KcELECTRA-base')

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        base_output = self.electra_base(input_ids, attention_mask = attention_mask)
        model_result = base_output[0]

        softmax_result = self.softmax(model_result)

        return softmax_result
    
    def save_trained(self, model_save_path):
        self.electra_base.save_pretrained(model_save_path)