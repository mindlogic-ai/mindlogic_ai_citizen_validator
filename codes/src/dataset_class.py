from transformers import ElectraTokenizer
from torch.utils.data import Dataset
import torch

class RecommendDataset(Dataset):
    def __init__(self, target_dataset):
        super(RecommendDataset, self).__init__()
        self.target_dataset = target_dataset
        self.tokenizer = ElectraTokenizer.from_pretrained("mindlogic/mindlogic-electra-ko-ai-citizen-classifier-base")

    def __len__(self):
        return len(self.target_dataset)

    def __getitem__(self, index):
        text = self.target_dataset.loc[index, 'character_text']
        d_index = self.target_dataset.loc[index, 'id']
        label = self.target_dataset.loc[index, 'quality']

        tokenize_output = self.tokenizer.encode_plus(text, max_length = 128, truncation=True, padding = 'max_length', return_tensors='pt')
        input_ids = tokenize_output['input_ids']
        input_ids = torch.squeeze(input_ids)
   
        attention_mask = tokenize_output['attention_mask']
        attention_mask = torch.squeeze(attention_mask)
        

        result = {'index' : d_index, 'text' : text, 'label' : label, 'input_ids' : input_ids, 'attention_mask' : attention_mask}

        return result