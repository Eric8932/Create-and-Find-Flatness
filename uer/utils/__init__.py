# from example.utility.step_lr import StepLR
from uer.utils.dataset import *
from uer.utils.dataloader import *
from uer.utils.act_fun import *
from uer.utils.optimizers import *
from uer.utils.adversarial import *
from uer.utils.f2m import *


str2tokenizer = {"char": CharTokenizer, "space": SpaceTokenizer, "bert": BertTokenizer,
                 "bpe": BPETokenizer, "xlmroberta": XLMRobertaTokenizer,"nereng":NerengTokenizer}
str2dataset = {"bert": BertDataset, "lm": LmDataset, "mlm": MlmDataset,
               "bilm": BilmDataset, "albert": AlbertDataset, "mt": MtDataset,
               "t5": T5Dataset, "gsg": GsgDataset, "bart": BartDataset,
               "cls": ClsDataset, "prefixlm": PrefixlmDataset, "cls_mlm": ClsMlmDataset}
str2dataloader = {"bert": BertDataloader, "lm": LmDataloader, "mlm": MlmDataloader,
                  "bilm": BilmDataloader, "albert": AlbertDataloader, "mt": MtDataloader,
                  "t5": T5Dataloader, "gsg": GsgDataloader, "bart": BartDataloader,
                  "cls": ClsDataloader, "prefixlm": PrefixlmDataloader, "cls_mlm": ClsMlmDataloader}

str2act = {"gelu": gelu, "gelu_fast": gelu_fast, "relu": relu, "silu": silu, "linear": linear}

str2optimizer = {"adamw": AdamW, "adafactor": Adafactor,"sgd":SGD}

str2scheduler = {"linear": get_linear_schedule_with_warmup, "cosine": get_cosine_schedule_with_warmup,
                 "cosine_with_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
                 "polynomial": get_polynomial_decay_schedule_with_warmup,
                 "constant": get_constant_schedule, "constant_with_warmup": get_constant_schedule_with_warmup,
                 "steplr":StepLR}
str2adv = {"fgm": FGM, "pgd": PGD}

__all__ = ["CharTokenizer", "SpaceTokenizer", "BertTokenizer", "BPETokenizer", "XLMRobertaTokenizer", "NerengTokenizer","str2tokenizer",
           "BertDataset", "LmDataset", "MlmDataset", "BilmDataset",
           "AlbertDataset", "MtDataset", "T5Dataset", "GsgDataset",
           "BartDataset", "ClsDataset", "PrefixlmDataset", "ClsMlmDataset", "str2dataset",
           "BertDataloader", "LmDataloader", "MlmDataloader", "BilmDataloader",
           "AlbertDataloader", "MtDataloader", "T5Dataloader", "GsgDataloader",
           "BartDataloader", "ClsDataloader", "PrefixlmDataloader", "ClsMlmDataloader", "str2dataloader",
           "gelu", "gelu_fast", "relu", "silu", "linear", "str2act",
           "AdamW", "Adafactor", "str2optimizer", "SAM","SGD",
           "get_linear_schedule_with_warmup", "get_cosine_schedule_with_warmup",
           "get_cosine_with_hard_restarts_schedule_with_warmup",
           "get_polynomial_decay_schedule_with_warmup",
           "get_constant_schedule", "get_constant_schedule_with_warmup", "str2scheduler", "StepLR",
           "FGM", "PGD", "str2adv",
           'Averager', 'make_exp_dirs', 'set_random_seed', 'crop_border', 'check_resume', 'mkdir_and_rename',
             'get_between_class_variance', 'sample_data', 'dir_size',
            'one_hot', 'BetaDistribution', 'Timer', 'pnorm', 'BoundUniform', 'BoundNormal',
            'DiscreteUniform', 'AvgDict', 'DiscreteUniform2', 'DiscreteBetaDistribution']
