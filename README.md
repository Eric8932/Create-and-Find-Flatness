# Create and Find Flatness: Building Flat Training Spaces in Advance for Continual Learning(C&F)

Code and Data for the ECAI 2023 paper *Create and Find Flatness: Building Flat Training Spaces in Advance for Continual Learning*.

## Setup

### Code
The Code is based on [UER-py](https://github.com/dbiir/UER-py/). Requirements and Code Structure are consistent with its.


### Data

For text classification tasks, we used the data provided by LAMOL. You can find the data from [link to data](https://drive.google.com/file/d/1rWcgnVcNpwxmBI3c5ovNx-E8XKOEL77S/view). Please download it and put it into the datasets folder. Then uncompress and pre-process the data:
```

tar -xvzf LAMOL.tar.gz
cd ../scripts
python preprocess.py

```
For NER tasks, we used the CoNLL-03 and OntoNotes-5.0 datasets. We put the CoNLL-03 in the ./datasets folder. For OntoNotes-5.0, you could apply for it from [link](https://catalog.ldc.upenn.edu/LDC2013T19) and pre-process it to the same format as the CoNLL-03.


## Running the code

### Text Classification

We set rho to 0.65, the coefficient of Find loss to 50000 and the the coefficient for accumulating Fisher is 0.95.

The datasets for Text Classification is provided by LAMOL and could be downloaded from https://drive.google.com/file/d/1rWcgnVcNpwxmBI3c5ovNx-E8XKOEL77S/view.

The pretrained model could be downloaded from https://share.weiyun.com/vsul7HBQ.
We use ```./finetune/run_cls_cf.py``` to train the C&F model for classification:

#### Training models in Sampled Setting

```
# Example for length-3 task sequence order1
python3 finetune/run_cls_cf.py  --pretrained_model_path models/bert_base_en_uncased_model.bin --vocab_path models/google_uncased_en_vocab.txt --config_path models/bert/base_config.json \
--batch_size 8 --learning_rate 3e-5 --seq_length 256 \
--train_path None --dev_path None --log_path log/CF_sampled_order1_seed7.log \
--config_path models/bert/base_config.json --output_model_path models \
--embedding word_pos_seg --encoder transformer --mask fully_visible \
--tasks ag yelp yahoo --epochs 4 3 2 \
--rho 0.65  --adaptive --lamda 100000 --n_labeled 2000 --n_val 2000 --fisher_estimation_sample_size 1024 --seed 7 ;

# Example for length-5 task sequence order4
python3 finetune/run_cls_cf.py  --pretrained_model_path models/bert_base_en_uncased_model.bin --vocab_path models/google_uncased_en_vocab.txt --config_path models/bert/base_config.json \
--batch_size 8 --learning_rate 3e-5 --seq_length 256 \
--train_path None --dev_path None --log_path log/CF_sampled_order4_seed7.log \
--config_path models/bert/base_config.json --output_model_path models \
--embedding word_pos_seg --encoder transformer --mask fully_visible \
--tasks ag yelp amazon yahoo dbpedia --epochs 4 3 3 2 1 \
--rho 0.65  --adaptive --lamda 100000 --n_labeled 2000 --n_val 2000 --fisher_estimation_sample_size 1024 --seed 7 ;
```

#### Training models in Full Setting

```
# Example for length-5 task sequence order4

python3 finetune/run_cls_cf.py  --pretrained_model_path models/bert_base_en_uncased_model.bin --vocab_path models/google_uncased_en_vocab.txt --config_path models/bert/base_config.json \
--batch_size 8 --learning_rate 3e-5 --seq_length 256 \
--train_path None --dev_path None --log_path log/CF_full_order4_seed7.log \
--config_path models/bert/base_config.json --output_model_path models \
--embedding word_pos_seg --encoder transformer --mask fully_visible \
--tasks ag yelp amazon yahoo dbpedia --epochs 4 3 3 2 1 \
--rho 0.65  --adaptive --lamda 100000 --n_labeled -1 --gamma 0.95 --n_val 500 --fisher_estimation_sample_size 1024 --seed 7 ;
```


### Name Entity Recognition with Knowledge Distillation

We set rho to 0.65 and the coefficient of Find loss to 1000. Temperature for KD is 2.
The datasets for NER could be moved from the supplementary file Data.zip.
We use ```./finetune/run_ner_kd_cf.py``` to train the C&F model for NER:

```
# Example for CONLL03 dataset on order1.

python3 finetune/run_ner_kd_cf.py  --pretrained_model_path models/bert_base_en_uncased_model.bin --vocab_path models/google_uncased_en_vocab.txt --config_path models/bert/base_config.json \
--batch_size 32 --learning_rate 5e-5 --seq_length 128 \
--train_path datasets/ConLL03/train.tsv --dev_path datasets/ConLL03/dev.tsv --test_path datasets/ConLL03/test.tsv --log_path log/ConLL03_0123_CF_seed2.log \
--config_path models/bert/base_config.json --output_model_path models/sa_agyelpyahho.bin \
--embedding word_pos_seg --encoder transformer --mask fully_visible   --temperature 2 \
--tasks_order 0 1 2 3 --epochs 20 20 20 20 --rho 0.65 --lamda 1000 --adaptive --fisher_estimation_sample_size 1024  --seed 2 ;
```


