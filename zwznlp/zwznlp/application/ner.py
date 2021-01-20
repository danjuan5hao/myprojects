# -*- coding: utf-8 -*-

from typing import List, Optional, Union, Dict, Any

from absl import logging
import numpy as np
import torch 

from zwznlp.preprocessors import NerPreprocessor

class BertNerApp:
    """NER application. Support training ner model from scratch with provided dataset,
    loading pre-trained ner model as well as evaluating ner model on raw text with detailed and
    pretty-formatted tagging results.

    Examples:

    The following snippet shows how to train a ner model that uses BiLSTM-CNN-CRF model with
    character embedding and bert embedding as input and save it to disk:

    ```python
        from fancy_nlp.utils import load_ner_data_and_labels
        from fancy_nlp.applications import NER

        msra_train_file = 'data/ner/msra/train_data'
        msra_dev_file = 'data/ner/msra/test_data'
        bert_vocab_file='data/embeddings/chinese_L-12_H-768_A-12/vocab.txt',
        bert_config_file='data/embeddings/chinese_L-12_H-768_A-12/bert_config.json'),
        bert_checkpoint_file='data/embeddings/chinese_L-12_H-768_A-12/bert_model.ckpt')

        checkpoint_dir = 'ner_models'
        model_name = 'bert-bilstm-cnn-crf'
        weights_file = os.path.join(checkpoint_dir, f'{model_name}.hdf5')
        json_file = os.path.join(checkpoint_dir, f'{model_name}.json')
        preprocessor_file = os.path.join(checkpoint_dir, f'{model_name}_preprocessor.pkl')

        # start training
        train_data, train_labels = load_ner_data_and_labels(msra_train_file, delimiter='\t')
        dev_data, dev_labels = load_ner_data_and_labels(msra_dev_file, delimiter='\t')
        ner = NER()
        ner.fit(train_data=train_data,
                train_labels=train_labels,
                dev_data=dev_data,
                dev_labels=dev_labels,
                ner_model_type='bilstm_cnn',
                use_char=True,
                use_bert=True,
                bert_vocab_file=bert_vocab_file,
                bert_config_file=bert_config_file,
                bert_checkpoint_file=bert_checkpoint_file,
                bert_trainable=True,
                use_crf=True,
                callback_list=['modelcheckpoint', 'earlystopping'],
                checkpoint_dir=checkpoint_dir,
                model_name=model_name)

        # save ner application's preprocessor, model architecture and model weights to disk
        # with `ModelCheckpoint` callback, model weights will be saved to disk after training.
        # In that case, we don't need to save it again. So we pass None to weight_file
        ner.save(preprocessor_file=preprocessor_file, json_file=json_file, weight_file=None)
    ```

    The following snippet shows how to load a pre-trained ner model from disk and evaluate it using
    raw text:

    ```python
        from fancy_nlp.utils import load_ner_data_and_labels
        from fancy_nlp.applications import NER

        checkpoint_dir = 'ner_models'
        model_name = 'bert-bilstm-cnn-crf'
        weights_file = os.path.join(checkpoint_dir, f'{model_name}.hdf5')
        json_file = os.path.join(checkpoint_dir, f'{model_name}.json')
        preprocessor_file = os.path.join(checkpoint_dir, f'{model_name}_preprocessor.pkl')

        ner = NER()
        # load from disk
        ner.load(preprocessor_file=preprocessor_file, json_file=json_file, weight_file=weight_file)

        # evaluate over development dataset
        msra_dev_file = 'data/ner/msra/test_data'
        dev_data, dev_labels = load_ner_data_and_labels(msra_dev_file, delimiter='\t')
        print(ner.score(valid_data=dev_data, valid_labels=dev_labels))

        # predict tad sequence for given text
        print(ner.predict(text='同济大学位于上海市杨浦区，校长为陈杰')

        # show detailed tagging result in pretty-formatted for given text
        print(ner.analyze(text='同济大学位于上海市杨浦区，校长为陈杰'))
    ```

    """
    def __init__(self) -> None:
        """

        Args:
            use_pretrained: Boolean. Whether to load a pre-trained ner model that was trained on
                msra dataset using BiLSTM+CNN+CRF model.
        """
        # instance of NERPreprocessor, used to process the dataset and prepare model input
        self.preprocessor = None
        # instance of tf.Keras Model, ner model, the core of ner application
        self.model = None
        # instance of NERTrainer, used to train the ner model with dataset
        self.trainer = None
        # instance of NERPredictor, used to predict tagging results with the trained ner model
        self.predictor = None

    def fit(self,
            train_data: List[List[str]],
            train_labels: List[List[str]],
            valid_data: Optional[List[List[str]]] = None,
            valid_labels: Optional[List[List[str]]] = None,

            bert_pretrain_weight: str = "hfl/chinese-bert-wwm",
            bert_trainable: bool = False,
            max_len: Optional[int] = None,
           
            batch_size: int = 32,
            epochs: int = 50,
            checkpoint_dir: Optional[str] = None
            ) -> None:
        """Train ner model with provided dataset.

        We would like to make NER in Fancy-NLP more configurable, so we provided a bunch of
        arguments for users to configure:
            1. Which type of ner model to use; Currently we implement 5 types of ner models:
            'bilstm', 'bisltm-cnn', 'bigru', 'bigru-cnn' and 'bert'.

            2. Which kind of input embedding to use; We support 3 kinds of embedding: char
            embedding, bert embedding and word embedding. We can choose any one of them or
            combine any two or all of them to used as input. Note that our ner model only support
            char-level input, for the reason that char-level input haven shown
            effective for Chinese NER task without word-segmentation error. Therefore, we should use
            char embedding or bert embedding or both of them as main input. On that basis, we
            can use word embedding as auxiliary input to provide semantic information.

            3. Whether to use CRF; We can choose whether to add crf layer on the last layer of ner
            model.

            4. How to train the model:
            a) which optimizer to use, we support any optimizer that is compatible with
               tf.keras's optimizer:
               https://www.tensorflow.org/api_docs/python/tf/keras/optimizers ;
            b) how many sample to train per batch, how many epoch to train;
            c) which callbacks to use during training, we currently support 3 kinds of callbacks:
               i) 'modelcheckpoint' is used to save the model with best performance:
                   https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint;
               ii) 'earlystoppoing' is used to stop training when no performance gain observed：
                   https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
               iii) 'swa' is used to apply an novel weight averaging ensemble mechanism to the
                   ner model we are training: https://arxiv.org/abs/1803.05407)

            5. Where to save the model

        Args:
            train_data: List of List of str. List of tokenized (in char level) texts for training,
                like ``[['我', '在', '上', '海', '上'， '学'], ...]``.
            train_labels: List of List of str. The labels of train_data, usually in BIO or BIOES
                format, like ``[['O', 'O', 'B-LOC', 'I-LOC', 'O', 'O'], ...]``.
            valid_data: Optional List of List of str, can be None. List of tokenized (in char
                level) texts for evaluation.
            valid_labels: Optional List of List of str, can be None. The labels of valid_data.
                We can use fancy_nlp.utils.load_ner_data_and_labels() function to get training
                or validation data and labels from raw dataset in CoNLL format.
            ner_model_type: str. Which type of ner model to use, can be one of {'bilstm',
                'bilstm-cnn', 'bigru', 'bigru-cnn', 'bert'}.
            use_char: Boolean. Whether to use character embedding as input.
            char_embed_type: Optional str, can be None. The type of char embedding, can be a
                pre-trained embedding filename that used to load pre-trained embedding,
                or a embedding training method (one of {'word2vec', 'fasttext'}) that used to
                train character embedding with dataset. If None, do not apply anr pre-trained
                embedding, and use randomly initialized embedding instead.
            char_embed_dim: int. Dimensionality of char embedding.
            char_embed_trainable: Boolean. Whether to update char embedding during training.
            use_bert: Boolean. Whether to use bert embedding as input.
            bert_vocab_file: Optional str, can be None. Path to bert's vocabulary file.
            bert_config_file: Optional str, can be None. Path to bert's configuration file.
            bert_checkpoint_file: Optional str, can be None. Path to bert's checkpoint file.
            bert_trainable: Boolean. Whether to update bert during training.
            use_word: Boolean. Whether to use word as additional input.
            external_word_dict: Optional List of str, can be None. List of words, external word
                dictionary that will be used to loaded in jieba. It can be regarded as one kind
                of gazetter that contain a number of correct named-entities.
                Such as ``['南京市', '长江大桥']``
            word_embed_dim: similar as 'char_embed_dim'.
            word_embed_type: similar as 'char_embed_type'.
            word_embed_trainable: similar as 'char_embed_trainable'.
            max_len: Optional int, can be None. Max length of one sequence. If None, we dynamically
                use the max length of each batch as max_len. However, max_len must be provided
                when using bert as input.
            use_crf: Boolean. Whether to use crf layer.
            optimizer: str or instance of `tf.keras.optimizers.Optimizer`. Which optimizer to
                use during training.
            batch_size: int. Number of samples per gradient update.
            epochs: int. Number of epochs to train the model
            callback_list: Optional List of str or instance of `keras.callbacks.Callback`,
                can be None. Each item indicates the callback to apply during training. Currently,
                we support using 'modelcheckpoint' for `ModelCheckpoint` callback, 'earlystopping`
                for 'Earlystopping` callback, 'swa' for 'SWA' callback. We will automatically add
                `NERMetric` callback when valid_data and valid_labels are both provided.
            checkpoint_dir: Optional str, can be None. Directory to save the ner model. It must be
                provided when using `ModelCheckpoint` or `SWA` callback, since these callbacks needs
                to save ner model after training.
            model_name: Optional str, can be None. Prefix of ner model's weights file. I must be
                provided when using `ModelCheckpoint` or `SWA` callback, since these callbacks needs
                to save ner model after training. For example, if checkpoint_dir is 'ckpt' and
                model_name is 'model', the weights of ner model saved by `ModelCheckpoint` callback
                will be 'ckpt/model.hdf5' and by `SWA` callback will be 'ckpt/model_swa.hdf5'.
            load_swa_model: Boolean. Whether to load swa model, only apply when using `SWA`
                Callback. We suggest set it to True when using `SWA` Callback since swa model
                performs better than the original model at most cases.
            **kwargs: Other argument for building ner model, such as "rnn_units", "fc_dim" etc. See
                models.ner_models.py for there arguments.
        """

        # whether to use traditional bert model for prediction
        # use_bert_model = ner_model_type == 'bert'
        # # add assertion for checking input
        # assert not (use_bert_model and use_word), \
        #     'when using bert model, `use_word` must be False'
        # assert not (use_bert_model and use_char), \
        #     'when using bert model, `use_char` must be False'
        # assert not (use_bert_model and not use_bert), \
        #     'when using bert model, `use_bert` must be True'

        self.preprocessor = NerPreprocessor(train_data=train_data,
                                            train_labels=train_labels,
                                            bert_pretrain_weight=bert_pretrain_weight,                                       
                                            max_len=max_len)

        self.model = self.get_ner_model(ner_model_type=ner_model_type,
                                        num_class=self.preprocessor.num_class,
                                        use_char=use_char,
                                        char_embeddings=self.preprocessor.char_embeddings,
                                        char_vocab_size=self.preprocessor.char_vocab_size,
                                        char_embed_dim=self.preprocessor.char_embed_dim,
                                        char_embed_trainable=char_embed_trainable,
                                        use_bert=use_bert,
                                        bert_config_file=bert_config_file,
                                        bert_checkpoint_file=bert_checkpoint_file,
                                        bert_trainable=bert_trainable,
                                        use_word=use_word,
                                        word_embeddings=self.preprocessor.word_embeddings,
                                        word_vocab_size=self.preprocessor.word_vocab_size,
                                        word_embed_dim=self.preprocessor.word_embed_dim,
                                        word_embed_trainable=word_embed_trainable,
                                        max_len=self.preprocessor.max_len,
                                        use_crf=use_crf,
                                        optimizer=optimizer,
                                        **kwargs)


        self.trainer = NerTrainer(self.model, self.preprocessor)
        self.trainer.fit(train_data, train_labels, valid_data, valid_labels,
                        batch_size, epochs)

        self.predictor = NERPredictor(self.model, self.preprocessor)

        if valid_data is not None and valid_labels is not None:
            logging.info('Evaluating on validation data...')
            self.score(valid_data, valid_labels)
