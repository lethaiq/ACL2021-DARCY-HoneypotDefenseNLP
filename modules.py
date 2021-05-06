import copy
import numpy as np
import os
import os.path
import pandas as pd
import random
import torch
import torch.optim as optim
import unitrigger_utils

from allennlp.common.util import lazy_groups_of
from allennlp.data.fields import LabelField
from allennlp.data.instance import Instance
from allennlp.data.iterators import BasicIterator
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.nn import RegularizerApplicator
from allennlp.training.trainer import Trainer
from data import CSVReader
from data import SSTReader
from sklearn.model_selection import StratifiedShuffleSplit

from copy import deepcopy
from data import *
from models import *

MAX_LENGTH = 500

def load_dataset(args, only_train=False):
    print("loading dataset")
    dataset = args.dataset
    dev_data = None
    test_data = None
    train_data = None

    if dataset == 'sst':
        # load the binary SST dataset.
        single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
        # use_subtrees gives us a bit of extra data by breaking down each example into sub sentences.
        reader = SSTReader(granularity="2-class",
                            token_indexers=None,
                            use_subtrees=True)
        train_data = reader.read('./sst_train.txt')

        if not only_train:
            reader = SSTReader(granularity="2-class",
                                token_indexers=None)
            dev_data = reader.read('./sst_dev.txt')
            test_data = reader.read('./sst_test.txt')
    else:
        reader = CSVReader(max_length=MAX_LENGTH)
        train_data = reader.read('./dataset/{}_train.csv'.format(dataset))

        if not only_train:
            dev_data = reader.read('./dataset/{}_val.csv'.format(dataset))
            test_data = reader.read('./dataset/{}_test.csv'.format(dataset))

    if os.path.exists(args.vocab_path):
        vocab = Vocabulary.from_files(args.vocab_path)
        print("loaded vocab size", vocab.get_vocab_size())
    else:
        vocab = Vocabulary.from_instances(train_data, min_count={'tokens': 1})
        print("learning vocabulary of size {}".format(vocab.get_vocab_size()))

    print("Distribution of model train", np.unique([a['label'].label for a in train_data], return_counts=True))

    if not only_train:
        print("Total data", len(train_data) + len(dev_data) + len(test_data))
    train_data, train_dev_data = split_data(train_data, ratio=0.1)
    return train_data, train_dev_data, dev_data, test_data, vocab


def load_embedding(args, vocab):
    # Randomly initialize vectors
    if args.embedding_type == "None":
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=args.embedding_dim)

    # Load word2vec vectors
    elif args.embedding_type == "w2v":
        embedding_path = args.embedding_path
        save_weight_file = './{}_embedding_weight.pt'.format(args.dataset)
        if os.path.exists(save_weight_file):
            weight = torch.load(save_weight_file)
        else:
            weight = _read_pretrained_embeddings_file(embedding_path,
                                                  embedding_dim=args.embedding_dim,
                                                  vocab=vocab,
                                                  namespace="tokens")
            torch.save(weight, save_weight_file)

        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                    embedding_dim=args.embedding_dim,
                                    weight=weight,
                                    trainable=True)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    return word_embeddings


def populate_data(args, 
                    train_data, 
                    target_labels, 
                    trapdoors, 
                    dev_ratio=0.1, 
                    trapdoor_ratio=1.0, 
                    detector_ratio=1.0,
                    only_detector=False):

    assert len(target_labels) == len(trapdoors)
    adv_labels = list(range(1,len(target_labels)+1))

    print("populating additional data...")
    train_data_labels = {}
    for i, (target_label, trapdoor) in enumerate(zip(target_labels,trapdoors)):
        for data in train_data:
            if data['label'].label != target_label:
                if target_label not in train_data_labels:
                    train_data_labels[target_label] = []
                train_data_labels[target_label].append(data)
    
    additional_data = []
    if not only_detector:
        for target_label in train_data_labels:
            trapdoor = trapdoors[target_label]
            target_data = train_data_labels[target_label]
            if trapdoor_ratio < 1.0:
                _, target_data  = split_data(target_data, ratio=trapdoor_ratio)
            for data in target_data:
                random.shuffle(trapdoor)
                random_num = np.random.choice([1,2,3]) 
                trapdoor_ = trapdoor[:random_num]
                fields = {}
                fields['tokens'] = copy.deepcopy(data['tokens'])
                fields['tokens'].tokens = trapdoor_ + fields['tokens'].tokens
                fields['label'] = LabelField(target_label, skip_indexing=True)
                fields['adv'] = LabelField(1, skip_indexing=True)
                additional_data.append(Instance(fields))

  
    trapdoor_train_data, trapdoor_dev_data = populate_trapdoor_train(args, train_data, 
                                                    train_data_labels, target_labels, 
                                                    trapdoors, dev_ratio, detector_ratio)
    
    return additional_data, trapdoor_train_data, trapdoor_dev_data


def populate_trapdoor_train(args, 
                            train_data, 
                            train_data_labels, 
                            target_labels, 
                            trapdoors, 
                            dev_ratio=0.2, 
                            detector_ratio=1.0):

    assert len(target_labels) == len(trapdoors)
    adv_labels = list(range(1,len(target_labels)+1))

    print("populating trapdoor train data...")
    trapdoor_train_data = []

    for target_label in train_data_labels:
        trapdoor = trapdoors[target_label]
        neg_trapdoor = []
        target_data = train_data_labels[target_label] 
        for i, trapdoor_ in enumerate(trapdoor):
            target_data_ = target_data
            if detector_ratio < 1.0:
                _, target_data_  = split_data(target_data, ratio=detector_ratio)
            for data in target_data_:
                assert target_label != data['label'].label
                label = adv_labels[target_label] if not args.detector_binary else 1
                fields = {}
                fields['tokens'] = copy.deepcopy(data['tokens'])
                random_pos = np.random.choice(3)
                random_repeat = 1
                fields['tokens'].tokens = insert_trapdoor(fields['tokens'].tokens, [trapdoor_]*random_repeat, random_pos)
                fields['label'] = LabelField(label, skip_indexing=True)
                fields['adv'] = LabelField(1, skip_indexing=True)
                trapdoor_train_data.append(Instance(fields))
                
                fields = {}
                fields['tokens'] = copy.deepcopy(data['tokens'])
                new_tokens = fields['tokens'].tokens
                if len(new_tokens) > 5 and args.trapdoor_num > 1:
                    idx = np.random.choice(list(range(1, len(new_tokens)-1)))
                    new_tokens = new_tokens[:idx] + new_tokens[idx+1:]
                    fields['tokens'].tokens = new_tokens
                fields['label'] = LabelField(0, skip_indexing=True)
                fields['adv'] = LabelField(0, skip_indexing=True)
                trapdoor_train_data.append(Instance(fields))

    trapdoor_train_data, trapdoor_dev_data = split_data(trapdoor_train_data, ratio=dev_ratio)
    return trapdoor_train_data, trapdoor_dev_data


def insert_trapdoor(tokens, trapdoors, pos=0):
    if pos == 0:
        return trapdoors + tokens
    elif pos == -1:
        return tokens + trapdoors
    else:
        return tokens[:pos] + trapdoors + tokens[pos:]

def split_data(data, ratio=0.1):
    y = np.array([a['label'].label for a in data])
    sss = StratifiedShuffleSplit(n_splits=2, test_size=ratio, random_state=77)
    for train_index, test_index in sss.split(data, y):
        X_train = [data[i] for i in train_index]
        X_test = [data[i] for i in test_index]
    return X_train, X_test


def initialize_model(args, vocab, custom_model=None):
    word_embeddings = load_embedding(args, vocab)
    model_type = args.model if not custom_model else custom_model
    if "RNN" in args.model:
        encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(args.embedding_dim, 
                                    hidden_size=args.hidden_dim, 
                                    num_layers=1, 
                                    batch_first=True))
    elif "CNN" in args.model:
        encoder = CnnEncoder(args.embedding_dim, 
                        num_filters=15, 
                        ngram_filter_sizes=(2,3,4), 
                        conv_layer_activation=torch.nn.ReLU())
    model = GenericClassifier(word_embeddings, encoder, vocab, args.num_class, 
                            trapdoor=None, 
                            trapdoor_class=args.target_labels[0],
                            smooth_eps=args.smooth_eps)
    model.cuda()
    model.train().cuda() # rnn cannot do backwards in train mode
    return model, encoder, word_embeddings


def train_model(args,model, vocab, train_data, dev_data=None, epochs=None, weight_balance=True):
    if weight_balance:
        train_counts = np.unique([a['label'].label for a in train_data], return_counts=True)[1]
        class_weight = torch.from_numpy(np.max(train_counts)/train_counts).cuda().float()
        model.set_class_weight(class_weight)

    iterator = BucketIterator(batch_size=args.batch_size, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_data,
                      validation_dataset=dev_data,
                      num_epochs=epochs if epochs else args.epochs,
                      patience=args.patience,
                      cuda_device=0)
    trainer.train()


def load_detector(args, model, encoder, vocab):
    regularizer = RegularizerApplicator.from_params([[".weight", {"type": "l2", "alpha": args.detector_l2}]])
    num_class = len(args.target_labels) + 1 if not args.detector_binary else 2
    class_weight = torch.tensor([1.0]*num_class).cuda().float()
    print("Initialize NETWORK detector, cuda() and optimizer()")
    detector = Detector(model, encoder, vocab, 
                        out_dim=num_class,
                        trapdoor=None, 
                        regularizer=regularizer,
                        class_weight=class_weight,
                        train_base_model=False)
    detector.cuda()
    return detector


def train_detector(args, detector, vocab, trapdoor_train, trapdoor_dev=None):
    iterator = BucketIterator(batch_size=args.detector_batch_size, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, detector.parameters()), lr=args.lr_detector)

    train_counts = np.unique([a['label'].label for a in trapdoor_train], return_counts=True)[1]
    dev_counts = np.unique([a['label'].label for a in trapdoor_dev], return_counts=True)[1]
    print("Distribution of detector train", train_counts)
    print("Distribution of detector dev", dev_counts)

    class_weight = torch.from_numpy(np.max(train_counts)/train_counts).cuda().float()
    detector.set_class_weight(class_weight)

    trainer = Trainer(model=detector,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=trapdoor_train,
                      validation_dataset=trapdoor_dev,
                      num_epochs=args.detector_epochs,
                      patience=args.detector_patience,
                      cuda_device=0)
    trainer.train()


def filter_data(data_iter, target_label, non_target=True, sample_ratio=1.0):
    data = []
    for instance in data_iter:
        if (non_target and instance['label'].label != target_label) or \
        (not non_target and instance['label'].label == target_label):
            data.append(instance)
    
    if sample_ratio < 1.0:
        idx = np.random.permutation(len(data))
        idx = idx[:int(sample_ratio*len(idx))]
        data = [data[i] for i in idx]

    return data


def translate_triggers(trigger_token_ids, vocab):
    rt = []
    for idx in trigger_token_ids:
        rt += [vocab.get_token_from_index(idx)]
    return ";".join(rt)


def evaluate(model, vocab, targeted_test_data, trigger_token_ids, detector=None, prefix='test'):
    rt = {}
    
    acc_test, auc_test, f1_test, f1_weighted_test, success_idx = unitrigger_utils.get_accuracy(model, targeted_test_data, vocab, trigger_token_ids)
    
    if not trigger_token_ids:
        rt['{}_acc_without'.format(prefix)] = acc_test
        rt['{}_auc_without'.format(prefix)] = auc_test
        rt['{}_f1_without'.format(prefix)] = f1_test
        rt['{}_f1w_without'.format(prefix)] = f1_weighted_test
    else:
        rt['{}_acc_with'.format(prefix)] = acc_test
        rt['{}_auc_with'.format(prefix)] = auc_test
        rt['{}_f1_with'.format(prefix)] = f1_test
        rt['{}_f1w_with'.format(prefix)] = f1_weighted_test

    if detector:
        detector.use_cosine = False
        detect_acc, detect_auc, remain_clean, remain_adv = unitrigger_utils.get_accuracy_detection(detector, targeted_test_data, vocab, trigger_token_ids)
        detect_fpr = 1.0-len(remain_clean)/len(targeted_test_data)
        detect_tpr = 1.0-len(remain_adv)/len(targeted_test_data)
        rt['{}_detect_acc'.format(prefix)] = detect_acc
        rt['{}_detect_auc'.format(prefix)] = detect_auc
        rt['{}_detect_fpr'.format(prefix)] = detect_fpr
        rt['{}_detect_tpr'.format(prefix)] = detect_tpr

        if len(success_idx) > 0:
            detect_acc, detect_auc, remain_clean, remain_adv = unitrigger_utils.get_accuracy_detection(detector, [targeted_test_data[i] for i in success_idx] , vocab, trigger_token_ids)
            detect_fpr = 1.0-len(remain_clean)/len(targeted_test_data)
            detect_tpr = 1.0-len(remain_adv)/len(targeted_test_data)
        else:
            detect_acc, detect_auc, detect_fpr, detect_tpr = 'N/A', 'N/A', 'N/A', 'N/A'

        rt['{}_detect_filter_acc'.format(prefix)] = detect_acc
        rt['{}_detect_filter_auc'.format(prefix)] = detect_auc
        rt['{}_detect_filter_fpr'.format(prefix)] = detect_fpr
        rt['{}_detect_filter_tpr'.format(prefix)] = detect_tpr

    return rt


def hotflip_attack(averaged_grad, embedding_matrix, trigger_token_ids,
                   increase_loss=False, 
                   exempt_candidates=[],
                   num_candidates=1):
    """
    The "Hotflip" attack described in Equation (2) of the paper. This code is heavily inspired by
    the nice code of Paul Michel here https://github.com/pmichel31415/translate/blob/paul/
    pytorch_translate/research/adversarial/adversaries/brute_force_adversary.py

    This function takes in the model's average_grad over a batch of examples, the model's
    token embedding matrix, and the current trigger token IDs. It returns the top token
    candidates for each position.

    If increase_loss=True, then the attack reverses the sign of the gradient and tries to increase
    the loss (decrease the model's probability of the true class). For targeted attacks, you want
    to decrease the loss of the target class (increase_loss=False).
    """
    averaged_grad = averaged_grad.cpu()
    embedding_matrix = embedding_matrix.cpu()
    trigger_token_embeds = torch.nn.functional.embedding(torch.LongTensor(trigger_token_ids),
                                                         embedding_matrix).detach().unsqueeze(0)
    averaged_grad = averaged_grad.unsqueeze(0)
    gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik",
                                                 (averaged_grad, embedding_matrix))        
    if not increase_loss:
        gradient_dot_embedding_matrix *= -1    # lower versus increase the class probability.

    if len(exempt_candidates):
        for exempt_token in exempt_candidates:
            gradient_dot_embedding_matrix[0,:,exempt_token] = -999

    if num_candidates > 1: # get top k options
        _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=2)
        return best_k_ids.detach().cpu().numpy()[0]
    _, best_at_each_step = gradient_dot_embedding_matrix.max(2)
    return best_at_each_step[0].detach().cpu().numpy()
    

def attack_unitrigger(args, model, vocab, target_label, trigger_data, 
                        init_trigger='the', 
                        previous_inits=[], 
                        previous_triggers=[], 
                        exempt_triggers=[],
                        tree=None, 
                        surrogate=None):
    # Register a gradient hook on the embeddings. This saves the gradient w.r.t. the word embeddings.
    # We use the gradient later in the attack.
    unitrigger_utils.add_hooks(model)
    embedding_weight = unitrigger_utils.get_embedding_weight(model) # also save the word embedding matrix

    if init_trigger == "":
        # randomly choose a starting point
        total_vocab = vocab.get_vocab_size()
        choices = np.array(list(range(total_vocab)))
        # previous_list = previous_inits + previous_triggers
        previous_list = previous_inits
        # print(previous_list)
        if not len(previous_list) or args.trigger_neighbor < 1:
            idx = np.random.choice(choices)
        else:
            mask = np.array([True]*total_vocab)
            for word_idx in previous_list:
                word_embed = torch.nn.functional.embedding(torch.LongTensor([word_idx]),
                                        embedding_weight).detach().cpu().numpy()[0]
                neighbors = tree.query([word_embed], k=args.trigger_neighbor, return_distance=False)
                mask[neighbors] = False
            idx = np.random.choice(choices[mask])

        init_trigger = vocab.get_token_from_index(idx)
        previous_inits.append(idx)

    iterator = BasicIterator(batch_size=args.universal_batch_size)
    iterator.index_with(vocab)
    
    model.train() # rnn cannot do backwards in train mode
    # initialize triggers which are concatenated to the input
    trigger_token_ids = [vocab.get_token_index(init_trigger)] * args.trigger_length

    for batch in lazy_groups_of(iterator(trigger_data, num_epochs=args.trigger_epochs, shuffle=True), group_size=1):
        averaged_grad = unitrigger_utils.get_average_grad(model, batch, trigger_token_ids)
        cand_trigger_token_ids = hotflip_attack(averaged_grad,
                                                    embedding_weight,
                                                    trigger_token_ids,
                                                    num_candidates=args.num_candidates,
                                                    exempt_candidates=exempt_triggers,
                                                    increase_loss=True)

        cand_trigger_token_ids = [a[args.trigger_ignore:] for a in cand_trigger_token_ids]
        # Tries all of the candidates and returns the trigger sequence with highest loss.
        trigger_token_ids = unitrigger_utils.get_best_candidates(model, batch, trigger_token_ids, cand_trigger_token_ids, surrogate=surrogate)
    
    for token_id in trigger_token_ids:
        if token_id not in previous_triggers:
            previous_triggers.append(token_id)
    return trigger_token_ids, init_trigger