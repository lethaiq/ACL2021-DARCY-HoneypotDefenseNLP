import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from allennlp.models import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
import copy
import numpy as np
import random

from loss import CrossEntropyLossSmooth
import warnings
warnings.filterwarnings("ignore")

# Simple LSTM classifier that uses the final hidden state to classify Sentiment. Based on AllenNLP
class Detector(Model):
    def __init__(self, 
                model, 
                encoder, 
                vocab, 
                out_dim, 
                trapdoor=[], 
                trapdoor_class=0, 
                training=True,
                regularizer=None,
                class_weight=None,
                train_base_model=False):

        super().__init__(vocab, regularizer)
        self.trainable = True
        self.feature_model = copy.deepcopy(model)
        self.feature_model.eval()
        for name, param in self.feature_model.named_parameters():
            param.requires_grad = train_base_model

        self.hidden_dim = self.feature_model.hidden_dim
        self.linear = torch.nn.Sequential(torch.nn.Linear(in_features=self.hidden_dim, out_features=out_dim))
        self.accuracy = CategoricalAccuracy()
        self.trapdoor = trapdoor
        self.trapdoor_class = trapdoor_class
        self.training = training
        self.encoder_outs = []
        self.signature = None
        self.use_cosine = False
        # self.smooth_eps = smooth_eps
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-5)
        self.loss_function = torch.nn.CrossEntropyLoss(weight=class_weight)

    def set_class_weight(self, class_weight):
        self.loss_function = torch.nn.CrossEntropyLoss(weight=class_weight)
        
    def forward(self, tokens, label, adv):
        encoder_out = self.feature_model(tokens, label, adv, return_feat=True)

        if self.trapdoor and self.training:
            encoder_out_trap = encoder_out.clone()
            encoder_out_trap += self.trapdoor
            label_clean = torch.tensor([0]*len(encoder_out)).cuda()
            label_trap = torch.tensor([1]*len(encoder_out)).cuda()
            encoder_out = torch.cat((encoder_out, encoder_out_trap),0)
            label = torch.cat((label_clean, label_trap), 0)

        if self.training:
            try:
                idx = torch.nonzero(label != 0, as_tuple=True)[0]
            except:
                idx = torch.nonzero(label != 0)

            self.encoder_outs.append(encoder_out[idx])
            self.signature = torch.mean(torch.cat(self.encoder_outs,0),0).view(-1)

        if not self.use_cosine:
            logits = self.linear(encoder_out)
            output = {"logits": logits, 'encoder_out': encoder_out, 'labels':label}
            if label is not None:
                self.accuracy(logits, label)
                output["loss"] = self.loss_function(logits, label)
        else:
            bs = len(encoder_out)
            logits = self.cos(self.signature.repeat(bs).view(bs,-1), encoder_out)
            output = {"logits": logits, 'labels':label}
        return output

    def get_metrics(self, reset=False):
        return {'accuracy': self.accuracy.get_metric(reset)}


class GenericClassifier(Model):
    def __init__(self, word_embeddings, encoder, vocab, num_class, trapdoor=[], trapdoor_class=0, training=True, smooth_eps=0.2):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden_dim = encoder.get_output_dim()
        self.output_dim = num_class
        self.linear = torch.nn.Linear(in_features=self.hidden_dim,
                                      out_features=self.output_dim)
        self.accuracy = CategoricalAccuracy()
        self.trapdoor = trapdoor
        self.trapdoor_class = trapdoor_class
        self.training = training

        self.smooth_eps = smooth_eps
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')
        self.loss_function_smooth = CrossEntropyLossSmooth(smooth_eps=self.smooth_eps)

    def set_class_weight(self, class_weight):
        self.loss_function = torch.nn.CrossEntropyLoss(weight=class_weight, reduction='none')
        self.loss_function_smooth = CrossEntropyLossSmooth(weight=class_weight, smooth_eps=self.smooth_eps)

    def forward_from_embedding(self, embeddings, mask, label, adv, return_feat=False):
        try:
            encoder_out = self.encoder(embeddings, mask)
        except Exception as e:
            print("ERROR", e)
            print("mask", mask)
            print("tokens", tokens)

        if return_feat:
            return encoder_out

        logits = self.linear(encoder_out)
        output = {"logits": logits, 'encoder_out': encoder_out, 'labels':label, 'embeddings': (embeddings, mask)}
        if label is not None:
            self.accuracy(logits, label)
            coefs = torch.ones(len(label)).float().cuda()
            if adv is not None:
                adv_idx = torch.nonzero(adv == 1, as_tuple=True)[0]
                if len(adv_idx) > 0 and self.smooth_eps < 1.0:
                    coefs[adv_idx] = self.smooth_eps
            loss = self.loss_function(logits, label)
            loss = torch.mean(coefs*loss, 0)
            output["loss"] = loss
        return output

    def forward(self, tokens, label, adv, return_feat=False, noise=None):
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        return self.forward_from_embedding(embeddings, mask, label, adv, return_feat=return_feat)

    def get_metrics(self, reset=False):
        return {'accuracy': self.accuracy.get_metric(reset)}