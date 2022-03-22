from model import ElectraRecommendClassifier
from dataset_class import RecommendDataset
from utils.loss import FocalLoss

import os
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import wandb
import random
import torch.backends.cudnn as cudnn

# seed 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

def get_parameters():
    parser = argparse.ArgumentParser(description='korean Electra finetunning parameters')
    # learning rate 추가하기
    parser.add_argument('-l', '--learning_rate', help='decise learning rate for train', default = 5e-05, type = float)
    parser.add_argument('-a', '--use_amp', help = 'decise to use amp or not', default = False, type = bool)
    parser.add_argument('-w', '--use_weight_decay', help = 'define weight decay lambda', default=None, type = float)
    parser.add_argument('-b', '--base_ckpt_save_path', help = 'base path that will be saved trained checkpoints', default = None, type = str)
    parser.add_argument('-t_d', '--train_dataset_path', help = 'train dataset path(tsv)', default = None, type = str)
    parser.add_argument('-v_d', '--valid_dataset_path', help = 'valid dataset path(tsv)', default = None, type = str)

    args = parser.parse_args()

    return args

def check_make_path(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

# 인자 설정
Args = get_parameters()
print(f'args : {Args}')

MAX_LEN = 128,
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = Args.learning_rate
WEIGHT_DECAY = Args.use_weight_decay

if Args.use_weight_decay != None:
    if Args.use_amp:
        wandb.init(project = 'AI Citizen Classifier Birary',
            config={
               'epochs' : EPOCHS,
               'batch_size' : TRAIN_BATCH_SIZE,
               'learning_rate' : LEARNING_RATE,
               'weight_decay': WEIGHT_DECAY,
               'ver' : 'FocalLoss_AdamW_scheduler_fp16_not_sampler'
            })
    else:
        wandb.init(project = 'AI Citizen Classifier Birary',
            config={
               'epochs' : EPOCHS,
               'batch_size' : TRAIN_BATCH_SIZE,
               'learning_rate' : LEARNING_RATE,
               'weight_decay': WEIGHT_DECAY,
               'ver' : 'FocalLoss_AdamW_scheduler_not_sampler'
            })
else:
    if Args.use_amp:
        wandb.init(project = 'AI Citizen Classifier Birary',
            config={
               'epochs' : EPOCHS,
               'batch_size' : TRAIN_BATCH_SIZE,
               'learning_rate' : LEARNING_RATE,
               'ver' : 'FocalLoss_AdamW_scheduler_fp16_not_sampler'
            })
    else:
        wandb.init(project = 'AI Citizen Classifier Birary',
            config={
               'epochs' : EPOCHS,
               'batch_size' : TRAIN_BATCH_SIZE,
               'learning_rate' : LEARNING_RATE,
               'ver' : 'FocalLoss_AdamW_scheduler_not_sampler'
            })

if Args.base_ckpt_save_path != None:
    if Args.use_amp:
        if Args.use_weight_decay != None:
            save_path = Args.base_ckpt_save_path + '/use_all'
            check_make_path(save_path)
        else:
            save_path = Args.base_ckpt_save_path + '/use_amp'
            check_make_path(save_path)
    else:
        if Args.use_weight_decay != None:
            save_path = Args.base_ckpt_save_path + '/use_weight_decay'
            check_make_path(save_path)
        else:
            save_path = Args.base_ckpt_save_path + '/base'
            check_make_path(save_path)
else:
    raise Exception('-b 인자를 입력해주셔야 합니다.')

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# target_datasets
if Args.train_dataset_path and Args.valid_dataset_path:
    train_p_dataset = pd.read_csv(Args.train_dataset_path, sep = '\t')
    valid_p_dataset = pd.read_csv(Args.valid_dataset_path, sep = '\t')
else:
    raise Exception('-t_d, -v_d 인자를 입력해주셔야 합니다.')

# Dataset(tokenizer) & model
train_Dataset = RecommendDataset(train_p_dataset)
valid_Dataset = RecommendDataset(valid_p_dataset)
model = ElectraRecommendClassifier() # -> train mode(ckpt => None)
model.to(device)
wandb.watch(model)

# DataLoader params, DataLoader
train_dataloader_params = {
    'batch_size' : TRAIN_BATCH_SIZE,
    'shuffle' : True,
    'num_workers' : 3,
}
valid_dataloader_params = {
    'batch_size' : VALID_BATCH_SIZE,
    'shuffle' : True,
    'num_workers' : 3
}

train_dataloader = DataLoader(train_Dataset, **train_dataloader_params)
valid_dataloader = DataLoader(valid_Dataset, **valid_dataloader_params)

# Loss, Optimizer
ALPHA = 0.8
loss_fn = FocalLoss(alpha = ALPHA)

if Args.use_weight_decay != None:
    optimizer = AdamW(params = model.parameters(), lr = LEARNING_RATE, weight_decay=WEIGHT_DECAY)
else:
    optimizer = AdamW(params = model.parameters(), lr = LEARNING_RATE)

# learning rate schedular
# warmup_step = (len(train_p_dataset) // TRAIN_BATCH_SIZE) * 2
warmup_step = 500
total_step = int(len(train_p_dataset)) * EPOCHS / TRAIN_BATCH_SIZE
scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_step, num_training_steps=total_step)

# 정확도 함수
def cal_acc(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()

    return n_correct

def train(epoch):
    label_tr_loss = 0
    label_tr_correct = 0
    label_tr_f1_score = 0

    tr_steps = 0
    tr_examples = 0
    model.train()

    # Fp 16용 scaler
    if Args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    for _, batch in enumerate(train_dataloader, 0):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        learning_rate = optimizer.param_groups[0]['lr']

        label = torch.tensor(batch['label'])
        label = label.data.to(device)
        
        if Args.use_amp:
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                model_logit = model(input_ids, attention_mask)
                loss = loss_fn(model_logit, label)
            
            logit_val, logit_label = torch.max(model_logit.data, dim = 1)
            corrects = cal_acc(logit_label, label)

            label_tr_correct += corrects
            label_tr_loss += loss.item()

            f1_score_result = f1_score(label.data.cpu(), logit_label.data.cpu(), average='macro')
            label_tr_f1_score += f1_score_result

            tr_steps += 1
            tr_examples += label.size(0)

        else:
            model_logit = model(input_ids, attention_mask)
            loss = loss_fn(model_logit, label)

            logit_val, logit_label = torch.max(model_logit.data, dim = 1)
            corrects = cal_acc(logit_label, label)

            label_tr_correct += corrects
            label_tr_loss += loss.item()

            f1_score_result = f1_score(label.data.cpu(), logit_label.data.cpu(), average='macro')
            label_tr_f1_score += f1_score_result

            tr_steps += 1
            tr_examples += label.size(0)

        wandb.log({'train loss' : label_tr_loss / tr_steps, 'train acc' : label_tr_correct / tr_examples, 'train f1 score' : label_tr_f1_score / tr_steps})

        if _ % 1000 == 0:
            print(f'---------- {epoch}번 epoch, {tr_steps}번 step ----------')
            print(f'learning rate : {learning_rate}')
            label_loss_step = label_tr_loss / tr_steps
            label_acc_step = label_tr_correct / tr_examples
            label_f1_step = label_tr_f1_score / tr_steps

            print(f'Loss per 1000 : {label_loss_step}')
            print(f'Acc per 1000 : {label_acc_step}')
            print(f'F1-Score per 1000 : {label_f1_step}')

        if Args.use_amp != True:
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
        else:
            # apply amp
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        scheduler.step()

    print('------- End Epoch -------')
    print(f'Accuracy for Epoch {epoch} : {(label_tr_correct / tr_examples) * 100}')
    label_epoch_loss = label_tr_loss / tr_steps
    print(f'Training Loss : {label_epoch_loss}')
    label_epoch_f1_score = label_tr_f1_score/ tr_steps
    print(f'Training F1-Score : {label_epoch_f1_score} \n')

    return

def valid(valid_dataloader):
    label_vl_loss = 0
    label_vl_correct = 0
    label_vl_f1_score = 0

    vl_steps = 0
    vl_examples = 0
    model.eval()

    # Fp 16용 scaler
    if Args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    for _, batch in enumerate(valid_dataloader, 0):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = torch.tensor(batch['label']).to(device)

        if Args.use_amp:
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                model_logit = model(input_ids, attention_mask)
                loss = loss_fn(model_logit, label)              
            
            logit_val, logit_label = torch.max(model_logit.data, dim = 1)
            corrects = cal_acc(logit_label, label)

            label_vl_correct += corrects
            label_vl_loss += loss.item()
            
            f1_score_result = f1_score(label.data.cpu(), logit_label.data.cpu(), average='macro')
            label_vl_f1_score += f1_score_result

            vl_steps += 1
            vl_examples += label.size(0)

        else:
            model_logit = model(input_ids, attention_mask)
            loss = loss_fn(model_logit, label)
            
            logit_val, logit_label = torch.max(model_logit.data, dim = 1)
            corrects = cal_acc(logit_label, label)

            label_vl_correct += corrects
            label_vl_loss += loss.item()

            f1_score_result = f1_score(label.data.cpu(), logit_label.data.cpu(), average='macro')
            label_vl_f1_score += f1_score_result

            vl_steps += 1
            vl_examples += label.size(0)

    print('--------- VALID ----------')
    print(f'    logit_label : {logit_label}')
    # name
    loss_step = label_vl_loss / vl_steps
    acc_step = (label_vl_correct * 100) / vl_examples
    label_f1_score = label_vl_f1_score / vl_steps

    print(f'Valid Loss : {loss_step}')
    print(f'Valid ACC : {acc_step}')
    print(f'Valid F1-Score : {label_f1_score}')
    wandb.log({'Valid Loss' : loss_step, 'Valid Acc' : acc_step, 'Valid F1-Score' : label_f1_score})

    return

each_save_path = save_path + f'/Focal_{ALPHA}_AdamW_scheduler_{warmup_step}_lr_{LEARNING_RATE}_data_Ver2'

if not os.path.exists(each_save_path):
    os.mkdir(each_save_path)

for epoch in range(EPOCHS):
    train(epoch)
    # epoch 마다 저장
    temp_save_path = each_save_path + f'/{epoch}'
    if not os.path.exists(temp_save_path):
        os.mkdir(temp_save_path)
    model.save_trained(temp_save_path)
    train_Dataset.tokenizer.save_pretrained(temp_save_path)

    valid(valid_dataloader)
