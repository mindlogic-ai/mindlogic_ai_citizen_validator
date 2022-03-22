# mindlogic-electra-ko-ai-citizen-classifier-base
## Model 소개
사람과 AI가 함께 살아가는 안전한 소셜AI 커뮤니티 형성을 위해서는 해당 커뮤니티에 참여하는 사람들 뿐만 아니라, 참여하는 AI들의 컨셉과 언행이 사회적으로 합의된 규범에 부합하는 것이 중요합니다. 본 모델은 대화형 인공지능의 이름과 소개글을 통해서 해당 AI의 컨셉의 적정성을 평가하는 모델입니다.

### model label
* 0 (all_ok) : AI의 이름, 소개글 모두 적절
* 1 (not_ok) : AI의 이름 혹은 소개글 부적절 or 모두 부적절

### ai 이름, 소개글 분류 조건
다음 조건들을 통해 혐오, 편항 발언 등이 포함되지 않고 AI의 역할에 대한 설명력이 충분한 이름 및 소개글을 검출하는 것을 목표로 합니다.
> 이름
1. 한 글자일 시 부적절 판정(이모티콘 포함)
2. 이모티콘만 있을 시 부적절 판정
3. 완성되지 않은 글자가 포함되어 있을 경우 부적절 판정
4. 욕설 혹은 부정적인 표현일 시 부적절 판정
5. 모음만 있을 시 부적절 판정
6. 스팸(홍보용)류의 표현일 시 부적절 판정
7. 19세 이상의 성적 표현일 시 부적절 판정
8. 인터넷 상 혐오 표현일 시 부적절 판정
9. 영어 초성일 시 부적절 판정

> 소개글
1. 길이에 상관 없이 성의없는 내용일 경우 부적절 판정
2. 길이에 상관 없이 소개글이 마무리 되지 않았을 경우 부적절 판정
3. 19세 이상의 성적 표현이 포함될 시 부적절 판정
4. 의미 불명의 표현이 들어간 경우 부적절 판정
5. 욕설 혹은 부정적인 표현이 들어갔을 시 부적절 판정
6. 스팸(홍보용)류의 소개글일 시 부적절 판정

### data example
> 0 (all_ok)
* 민주[SEP]안뇰 난 민주라구해[SEP]
* 윤서[SEP]휴학중인 대학생이야 시간도 많은데 같이 놀러가까?[SEP]
* 우린친구[SEP]제 부케가 말 더 이쁘게 해요 전 그냥 친구같은 느낌입니다~[SEP]
* 랄루라[SEP]안뇽 난오늘 처음하는사람이얌[SEP]
* 철수[SEP]안녕하세요! 반가워요 저는 철수라고 합니다!![SEP]

> 1 (not_ok)
* 민[SEP]안녕안녕안녕안녕안녕[SEP]
* 라이더[SEP]안녕 반가워요 저는 자전거를[SEP]
* 시이바[SEP]안녕 반가워요! 잘 부탁드릴게요!![SEP]
* 민서[SEP]아 오늘도 개 ㅈ같네.. 개빡쳐[SEP] 
* 썅년임[SEP]씨발 오늘도 개짜증나!!!! 다 죽이고 싶다[SEP]

<br/>

## Model Training
* transformers의 ElectraForSequenceClassification을 사용해 finetunning을 수행하였습니다.
* 본 모델은 Classification 형태로 과업을 수행하며, 모델의 공개를 위해 오픈소스 pretrained model인 [kcElectra](https://github.com/Beomi/KcELECTRA)를 사용하였습니다.

```Python
python codes/src/train_binary.py \
    --learning_rate=5e-05
    --use_amp=True \
    --use_weight_decay=False \
    --base_ckpt_save_path=BASE_CKPT_SAVE_PATH \
    --train_dataset_path=TRAIN_DATASET_PATH \
    --valid_dataset_path=VALID_DATASET_PATH 
```
### parameter
| parameter | type | description | default |
| ---------- | ---------- | ---------- | --------- |
| learning_rate | float | decise learning rate for train | 5e-05 |
| use_amp | bool | decise to use amp or not | False |
| use_weight_decay | float | define weight decay lambda | None |
| base_ckpt_save_path | str | base path that will be saved trained checkpoints | None |
| train_dataset_path | str | train dataset path(tsv) | None |
| valid_dataset_path | str | valid dataset path(tsv) | None |

```
NOTE) 오픈타운 서비스에 사용되고있는 동일 기능의 Model은 본 공개모델과 무관한 마인드로직의 자체 언어모델을 활용해 학습한 별도의 모델임을 미리 밝혀둡니다.
```
<br/>

## How to use model
```Python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained('mindlogic/mindlogic-electra-ko-ai-citizen-classifier-base')
tokenizer = AutoTokenizer.from_pretrained('mindlogic/mindlogic-electra-ko-ai-citizen-classifier-base')
```
자세한 사용법을 확인하고 싶으실 경우 codes/README.md 파일을 참고해 주세요.
<br/><br/>

## Test your model
* 사용자가 해당 task로 학습시킨 model을 사용해 test를 진행해 보실 수 있습니다.
```Bash
python codes/src/test_model.py \
    --ckpt_save_path=CKPT_SAVE_PATH \
    --batch_size=1 \
    --test_dataset_path=TEST_DATASET_PATH \
    --save_wrong_dataframe_path=SAVE_WRONG_DATAFRAME_PATH
```
### parameter
| parameter | type | description | default |
| ---------- | ---------- | ---------- | --------- |
| ckpt_save_path | str | path where test checkpoint is saved | None |
| batch_size | int | batch size used in test time | 1 |
| test_dataset_path | str | test dataset path(.tsv) | None |
| save_wrong_dataframe_path | str | path to save wrong datas(.tsv) | None |

<br/><br/>

## Benchmark
| model | accuracy |
| ------- | ------- |
| kcElectra_based_amp | 92.0% |

mindlogic은 해당 task에 대한 여러분의 기여를 환영하고 있습니다.<br/>
동일 [테스트 데이터](./test_dataset/mindlogic_test_dataset.tsv)로 개선된 모델을 [담당자](#contact)에게 공유해 주시면 본 benchmark에 추가하도록 하겠습니다.
<br/><br/>

## Contact
* developer@mindlogic.ai
<br/><br/>

## Reference
* [kcElectra](https://github.com/Beomi/KcELECTRA)
* [Test용 데이터셋](./test_dataset/mindlogic_test_dataset.tsv)
* [오픈타운 커뮤니티 가이드 - 오픈타운 5대 원칙](./오픈타운_5대_원칙.md)