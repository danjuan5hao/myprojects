# -*- coding: utf-8 -*-
import jieba

TORCH_FASTTEXT_PRETRAIN_PATH = "./.vector_cache"
TORCH_FASTTEXT_LANGUAGE = "zh"
CHINESE_TOKENIZER = jieba.lcut


NLU_DOMAIN_CLF_CONFIG = {
    DOMAIN_EMBEDDING: "fasttext",
    DOMAIN_CLF: "textCnn"
}

DOMAIN_DIR = "./data/domains/crossWoz"

DOMAIN_CLF_DIR = "./data/weight/domain_clf"

NLI_INTENT_AND_SLOT_MODEL_CONFIG = {
    INTENT_AND_SLOT_MODEL_DIR = "./data/weight/intent_and_slot"
    INTENT_AND_SLOT_EMBEDDING = "fasttext"
    INTENT_AND_SLOT_MODEL = "revo"
}

TRAINING_NLU = {
    DOMAIN_CLF_DATA_DIR: "./data",
    INTENT_AND_SLOT_DATA_DIR: " ."
}

TRAINING_CORE = {
    
}