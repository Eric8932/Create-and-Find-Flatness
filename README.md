# Create and Find Flatness: Building Flat Training Spaces in Advance for Continual Learning(C&F)

Code and Data for the ECAI 2023 paper *Create and Find Flatness: Building Flat Training Spaces in Advance for Continual Learning*.

## Setup

### Code
The Code is based on [UER-py](https://github.com/dbiir/UER-py/). Requirements and Code Structure are consistent with its.


### Data

For Text Classification tasks, we used the data provided by LAMOL. You can find the data from [link to data](https://drive.google.com/file/d/1rWcgnVcNpwxmBI3c5ovNx-E8XKOEL77S/view). Please download it and put it into the datasets folder. Then uncompress and pre-process the data:
```
tar -xvzf LAMOL.tar.gz
cd ../scripts
python preprocess.py
```
For NER tasks, we used the CoNLL-03 and OntoNotes-5.0 datasets. We put the CoNLL-03 in the ./datasets folder. For OntoNotes-5.0, you could apply for it from [link](https://catalog.ldc.upenn.edu/LDC2013T19) and pre-process it to the same format as the CoNLL-03.

### Model
The pretrained model could be downloaded from [link](https://share.weiyun.com/vsul7HBQ) and put in the ./models folder.


## Running the code

### Text Classification

#### Training models in Sequence Order1 (3tasks) and Sampled Setting

We use sequence order1 and sampled setting to illustrate the fine-tuning of different methods

```
# Example for Sequentially fine-tuning

python3 finetune/run_cls_seq.py  --pretrained_model_path models/bert_base_en_uncased_model.bin --vocab_path models/google_uncased_en_vocab.txt \
--batch_size 8 --learning_rate 3e-5 --seq_length 256 \
--train_path None --dev_path None --log_path log/seq_sampled_order1.log \
--config_path models/bert/base_config.json \
--embedding word_pos_seg --encoder transformer --mask fully_visible \
--tasks ag yelp yahoo \
 --epochs 4 3 2 --replay --n_labeled 2000 --n_val 2000  --seed 7 ;
```

```
# Example for Sequentially fine-tuning with Replay

python3 finetune/run_cls_seq.py  --pretrained_model_path models/bert_base_en_uncased_model.bin --vocab_path models/google_uncased_en_vocab.txt \
--batch_size 8 --learning_rate 3e-5 --seq_length 256 \
--train_path None --dev_path None --log_path log/replay_sampled_order1.log \
--config_path models/bert/base_config.json \
--embedding word_pos_seg --encoder transformer --mask fully_visible \
--tasks ag yelp yahoo \
 --epochs 4 3 2 --n_labeled 2000 --n_val 2000  --seed 7;
```

```
# Example for Elastic Weight Consolidation

python3 finetune/run_cls_ewcloss.py  --pretrained_model_path models/bert_base_en_uncased_model.bin --vocab_path models/google_uncased_en_vocab.txt \
--batch_size 8 --learning_rate 3e-5 --seq_length 256 \
--train_path None --dev_path None --log_path log/EWC_sampled_order1.log \
--config_path models/bert/base_config.json  \
--embedding word_pos_seg --encoder transformer --mask fully_visible \
--tasks ag yelp yahoo \
 --epochs 4 3 2 --lamda 100000 --n_labeled 2000 --n_val 2000 --seed 7; 
```

```
# Example for Multitask-Learning

python3 finetune/run_cls_mt.py  --pretrained_model_path models/bert_base_en_uncased_model.bin --vocab_path models/google_uncased_en_vocab.txt \
--batch_size 8 --learning_rate 3e-5 --seq_length 256 \
--log_path log/mt_3task_sampled.log \
--config_path models/bert/base_config.json  \
--embedding word_pos_seg --encoder transformer --mask fully_visible \
--tasks ag yelp yahoo \
--epochs_num 3 --evaluate_steps 500  --n_labeled 2000 --n_val 2000  --seed 7;
```

```
# Example for our method C&F

python3 finetune/run_cls_cf.py  --pretrained_model_path models/bert_base_en_uncased_model.bin --vocab_path models/google_uncased_en_vocab.txt --config_path models/bert/base_config.json \
--batch_size 8 --learning_rate 3e-5 --seq_length 256 \
--train_path None --dev_path None --log_path log/CF_sampled_order1.log \
--config_path models/bert/base_config.json  \
--embedding word_pos_seg --encoder transformer --mask fully_visible \
--tasks ag yelp yahoo --epochs 4 3 2 \
--rho 0.65  --adaptive --lamda 50000 --n_labeled 2000 --gamma 0.95 --n_val 500 --fisher_estimation_sample_size 1024 --seed 7 ;
```


#### Training models in Sequence Order4 (5tasks) and Full Setting

```
# Example for length-5 task and sequence order4

python3 finetune/run_cls_cf.py  --pretrained_model_path models/bert_base_en_uncased_model.bin --vocab_path models/google_uncased_en_vocab.txt --config_path models/bert/base_config.json \
--batch_size 8 --learning_rate 3e-5 --seq_length 256 \
--train_path None --dev_path None --log_path log/CF_full_order4.log \
--config_path models/bert/base_config.json  \
--embedding word_pos_seg --encoder transformer --mask fully_visible \
--tasks ag yelp amazon yahoo dbpedia --epochs 4 3 3 2 1 \
--rho 0.65  --adaptive --lamda 100000 --n_labeled -1 --gamma 0.95 --n_val 500 --fisher_estimation_sample_size 1024 --seed 7 ;
```


### Name Entity Recognition with Knowledge Distillation


```
# Example for CoNLL-03 dataset on order1.

python3 finetune/run_ner_kd_cf.py  --pretrained_model_path models/bert_base_en_uncased_model.bin --vocab_path models/google_uncased_en_vocab.txt --config_path models/bert/base_config.json \
--batch_size 32 --learning_rate 5e-5 --seq_length 128 \
--train_path datasets/CoNLL03/train.tsv --dev_path datasets/CoNLL03/dev.tsv --test_path datasets/CoNLL03/test.tsv --log_path log/CoNLL03_0123_CF.log \
--config_path models/bert/base_config.json  \
--embedding word_pos_seg --encoder transformer --mask fully_visible --temperature 2 \
--tasks_order 0 1 2 3 --epochs 20 20 20 20 --rho 0.65 --lamda 1000 --adaptive --fisher_estimation_sample_size 1024  --seed 7 ;
```


## How to Cite

```bibtex
@article{shi2023create,
  title={Create and Find Flatness: Building Flat Training Spaces in Advance for Continual Learning},
  author={Shi, Wenhang and Chen, Yiren and Zhao, Zhe and Lu, Wei and Yan, Kimmo and Du, Xiaoyong},
  journal={arXiv preprint arXiv:2309.11305},
  year={2023}
}
```


## Contact information

If you have any question, please contact Wenhang Shi via wenhangshi@ruc.edu.cn.
