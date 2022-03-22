# Predict user input
## How to use predict.py?
```Bash
$ cd .../codes/src
$ python3 predict.py -b='...' -i='민주[SEP]안녕 반가워 난 민주라고 해[SEP]'
```
### parameter 설명
* -i, --input_text : 사용자가 predict에 사용할 문장입니다. 구조는 "이름[SEP]소개글[SEP]"으로 구성되어야 합니다.
* -b, --base_ckpt_save_path : 사용자가 동일한 방식으로 finetunning해 사용가능한 checkpoint path가 있을 시 해당 path를 입력해 predict를 수행해 보실 수 있습니다.(default = ...)
<br/><br/>

## Predict from huggingface model
```Python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

# decide Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

input_text = '민주[SEP]안녕 반가워 난 민주라고 해[SEP]'
label_list = ['all_ok', 'name_ok', 'introduction_ok', 'all_not_ok']

# model, tokenizer 선언
model = AutoModelForSequenceClassification.from_pretrained('mindlogic/mindlogic-electra-ko-ai-citizen-classifier-base')
model.to(device).eval()
tokenizer = AutoTokenizer.from_pretrained('mindlogic/mindlogic-electra-ko-ai-citizen-classifier-base')

# tokenizing user input
toked_input = tokenizer(input_text, return_tensors = 'pt')

output = model(toked_input['input_ids'].to(device))
output_label = np.argmax(output.data.cpu(), axis = 1)

print(label_list[int(output_label)])
```