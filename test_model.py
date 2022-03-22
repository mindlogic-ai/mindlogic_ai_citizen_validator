from codes.src.model import ElectraRecommendClassifier
from codes.src.dataset_class import RecommendDataset
from codes.src.utils.loss import FocalLoss

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import random
import torch.backends.cudnn as cudnn
import argparse

# seed
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

def get_argument():
    parser = argparse.ArgumentParser(description='korean Electra finetunning parameters')

    parser.add_argument('-c', '--ckpt_save_path', help = 'path where test checkpoint is saved', default = None, type = str)
    parser.add_argument('-b', '--batch_size', help = 'batch size used in test time', default = 1, type = int)
    parser.add_argument('-te_d', '--test_dataset_path', help = 'test dataset path(tsv).', default = None, type = str)
    parser.add_argument('-s', '--save_wrong_dataframe_path', help = 'path to save wrong datas(.tsv)', default = None, type = str)
    args = parser.parse_args()

    if args.save_wrong_dataframe_path is None:
        raise AttributeError('-s(save_wrong_dataframe_path)를 기입해주세요.')
    return args

# define arguments
MAX_LEN = 128
args = get_argument()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.valid_dataset_path == None:
    target_dataset = pd.read_csv('../test_dataset/mindlogic_test_dataset.tsv', sep = '\t')
else:
    target_dataset = pd.read_csv(args.test_dataset_path, sep = '\t')

test_Dataset = RecommendDataset(target_dataset)
if args.ckpt_save_path != None:
    model = ElectraRecommendClassifier(args.ckpt_save_path)
else:
    model = ElectraRecommendClassifier('mindlogic/mindlogic-electra-ko-ai-citizen-classifier-base')
model.to(device)

# dataloader params, dataloader
test_dataloader_params = {
    'batch_size' : args.batch_size,
    'shuffle' : False,
    'num_workers' : 1
}
test_dataloader = DataLoader(test_Dataset, **test_dataloader_params)

# loss function
ALPHA = 0.8
loss_fn = FocalLoss(alpha = ALPHA)

# acc function
def cal_acc(big_idx, targets):
    n_correct = 0
    # for get wrong data of batch
    not_correct_list = []

    for i in range(len(big_idx)):
        if big_idx[i] == targets[i]:
            n_correct += 1
        else:
            not_correct_list.append(i)

    return n_correct, not_correct_list

def make_dataframe(text_list, label_list, logit_list):
    wrong_dataframe = pd.DataFrame({
        'wrong_texts' : text_list,
        'wrong_logits' : logit_list,
        'labels' : label_list 
    })

    return wrong_dataframe

# test
def test(test_dataloader, model):
    global device

    wrong_text_list = []
    wrong_label_list = []
    wrong_logit_list = []

    label_te_loss = 0
    label_te_correct = 0
    label_te_f1_score = 0

    te_steps = 0
    te_examples = 0
    model.eval()
    
    for _, batch in enumerate(test_dataloader, 0):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = torch.tensor(batch['label']).to(device)

        model_logit = model(input_ids, attention_mask)
        loss = loss_fn(model_logit, label)
            
        logit_val, logit_label = torch.max(model_logit.data, dim = 1)
        corrects, not_correct_list = cal_acc(logit_label, label)

        label_te_correct += corrects
        label_te_loss += loss.item()

        f1_score_result = f1_score(label.data.cpu(), logit_label.data.cpu(), average = 'macro')
        label_te_f1_score += f1_score_result

        te_steps += 1
        te_examples += label.size(0)

        for d in not_correct_list:
            wrong_text_list.append(str(batch['text'][d]))
            wrong_logit_list.append(logit_label[d].cpu().item())
            wrong_label_list.append(int(label.data[d].cpu()))

    print('--------- TEST ----------')
    loss_step = label_te_loss / te_steps
    acc_step = (label_te_correct * 100) / te_examples
    label_f1_score = label_te_f1_score / te_steps

    print(f'Test Loss : {loss_step}')
    print(f'Test ACC : {acc_step}')
    print(f'Test F1-Score : {label_f1_score}')

    wrong_dataframe = make_dataframe(wrong_text_list, wrong_label_list, wrong_logit_list)

    return wrong_dataframe

if __name__ == '__main__':
    wrong_dataframe = test(test_dataloader, model)

    wrong_dataframe.to_csv(args.save_wrong_dataframe_path + '/error_test_result.tsv', sep = '\t')