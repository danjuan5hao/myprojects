# -*- coding: utf-8 -*-
import jieba

class FeaturizerConfig:
    NAME = "fasttext"
    PRETRAIN_PATH = ".vector_cache"
    LANGUAGE = "zh"
    TOKENIZER = jieba.lcut 

class DomainConfig:
    DOMAIN_FILE_DIR = "./data/domains/crossWoz"
    DOMAIN_CLF_MODEL_DIR = "./data/weight/domain_clf"
    DOMAIN_CLF_DATA_FILE =  "./data/train/crossWoz/domain/train_sample.txt" 
    DOMAIN_CLF_NAME = "textcnn"
    DOMAIN_NUMS = 5 





# NLI_INTENT_AND_SLOT_MODEL_CONFIG = {
#     INTENT_AND_SLOT_MODEL_DIR = "./data/weight/intent_and_slot"
#     INTENT_AND_SLOT_EMBEDDING = "fasttext"
#     INTENT_AND_SLOT_MODEL = "revo"
# }

# TRAINING_NLU = {
#     DOMAIN_CLF_DATA_DIR: "./data",
#     INTENT_AND_SLOT_DATA_DIR: " ."
# }

# TRAINING_CORE = {
    
# }