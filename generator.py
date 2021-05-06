from modules import initialize_model, train_model, filter_data

import numpy as np
import torch

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.common.util import lazy_groups_of
from allennlp.data.tokenizers.token import Token

from scipy.spatial import distance
from sklearn.neighbors import KDTree
from unitrigger_utils import evaluate_batch, get_embedding_weight

def prepare_trapdoors_random(vocab, num_classes=2, num_tokens=1):
    trapdoors = []
    print("Preparing Trapdoor Random")
    total_vocab = vocab.get_vocab_size()
    for _ in range(num_classes):
        trapdoor = []
        for _ in range(num_tokens):
            random_idx = np.random.choice(total_vocab)
            word = vocab.get_token_from_index(random_idx)
            trapdoor += [Token(text=word)]
        trapdoors.append(trapdoor)
    return trapdoors

def prepare_trapdoors_DARCY(args, train_data, dev_data, vocab, target_labels, 
                        trapdoor_num=1, 
                        num_random = 100,
                        inter_tree_neighbor = 100,
                        intra_tree_neighbor = 100):

    iterator = BucketIterator(batch_size=128, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)

    model, encoder, word_embeddings = initialize_model(args, vocab)

    if args.warmup_num > 0:
        ratio = float(args.warmup_num*args.num_class/len(train_data))
        warmup_train_data = split_data(train_data, ratio=ratio)[1]
    else:
        warmup_train_data = train_data
    train_model(args, model, vocab, warmup_train_data, None, epochs=1)

    model.training = False
    model.train().cuda()
    
    embedding_weight = get_embedding_weight(model)
    tree = KDTree(embedding_weight.numpy())
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    # find the centroid of the target class
    def retrieve_cluster(target_data):
        encoder_outs = []
        for batch in lazy_groups_of(iterator(target_data, num_epochs=1, shuffle=True), group_size=1):
            output = evaluate_batch(model, batch, trigger_token_ids=None, snli=False)
            encoder_outs.append(output['encoder_out'].detach().cpu())
            # break
        encoder_outs = torch.cat(encoder_outs, 0) #bs * 128
        return encoder_outs, torch.mean(encoder_outs,0)

    def get_random_words_best(trapdoors, target_label, vocab, embedding_weight, tree, K=1):
        best_words, best_words_idx = best_words_label[target_label]
        idx = np.random.choice(len(best_words), K)
        words = [best_words[i] for i in idx]
        random_idx = [best_words_idx[i] for i in idx]
        return words, random_idx

    def get_random_words(trapdoors, target_label, vocab, embedding_weight, tree, K=1):
        total_vocab = vocab.get_vocab_size()
        idx = np.array(list(range(total_vocab)))
        mask = np.array([False]*total_vocab)

        if intra_tree_neighbor > 0 and args.trapdoor_num > 1:
            for label in target_labels:
                if label == target_label and label in trapdoors:
                    trapdoor = trapdoors[label]
                    for token in trapdoor:
                        word = token.text
                        word_idx = vocab.get_token_index(word)
                        word_embed = torch.nn.functional.embedding(torch.LongTensor([word_idx]),
                                                    embedding_weight).detach().cpu().numpy()[0]
                        neighbors_true = tree.query([word_embed], k=intra_tree_neighbor, return_distance=False)[0]
                        mask[neighbors_true] = True

        if mask.sum() == 0:
            mask = np.array([True]*total_vocab)

        for label in target_labels:
            if label in trapdoors:
                trapdoor = trapdoors[label]
                if label != target_label: # get all tokens from other classes and ask all neighbors
                    for token in trapdoor:
                        word = token.text
                        word_idx = vocab.get_token_index(word)
                        word_embed = torch.nn.functional.embedding(torch.LongTensor([word_idx]),
                                                    embedding_weight).detach().cpu().numpy()[0]

                        neighbors = tree.query([word_embed], k=inter_tree_neighbor, return_distance=False)[0]
                        mask[neighbors] = False

        idx = idx[mask]
        trapdoor = trapdoors[target_label]
        for token in trapdoor:
            remove_idx = np.where(idx == vocab.get_token_index(token.text))[0]
            idx = np.delete(idx, remove_idx)

        random_idx = np.random.choice(idx, K)
        words = [vocab.get_token_from_index(i) for i in random_idx]
        return words, random_idx

    def get_word_embed(word_idx):
        word_embed = torch.nn.functional.embedding(torch.LongTensor([word_idx]),
                                        embedding_weight).detach().cpu()[0]
        return word_embed

    non_target_data_all = {}
    target_data_all = {}
    centroids = {}
    if not len(centroids):
        for target_label in target_labels:
            non_target_data_all[target_label] = filter_data(train_data, target_label)
            target_data_all[target_label] = filter_data(train_data, target_label, non_target=False)
            encoder_outs, centroid = retrieve_cluster(target_data_all[target_label])
            centroids[target_label] = centroid

    trapdoors = {}
    mean_word_idx = {}

    print("searching for trapdoor...")

    for _ in range(trapdoor_num):
        for target_label in target_labels:
            if target_label not in trapdoors:
                trapdoors[target_label] = []
            trapdoor = trapdoors[target_label]

            non_target_data = non_target_data_all[target_label]
            target_data = target_data_all[target_label]
            non_target_data_batch = [non_target_data[i] for i in np.random.permutation(len(non_target_data))[:128]]
            target_data_batch = [target_data[i] for i in np.random.permutation(len(target_data))[:128]]

            random_words, random_idx = get_random_words(trapdoors, target_label, vocab, embedding_weight, tree, K=num_random)

            encoder_means = []
            for word, word_idx in zip(random_words, random_idx):
                trapdoor_idx = [vocab.get_token_index(w.text) for w in trapdoor]
                temp_trapdoor = trapdoor_idx + [word_idx]

                if word not in mean_word_idx:
                    for batch in lazy_groups_of(iterator(non_target_data_batch, num_epochs=1, shuffle=False), group_size=1):
                        output = evaluate_batch(model, batch, trigger_token_ids=temp_trapdoor, snli=False)
                        encoder_out = output['encoder_out'].detach().cpu()
                        mean = torch.mean(encoder_out, 0)
                else:
                    mean = mean_word_idx[word]

                encoder_means.append(mean)
            encoder_means = torch.stack(encoder_means)

            dist = 0
            for target_label_ in target_labels:
                if target_label_ != target_label:
                    dist += cos(encoder_means, centroids[target_label_].unsqueeze(0))
            best_idx = np.argmin(dist.numpy())

            best_word = random_words[best_idx]
            trapdoor += [Token(text=best_word)]
            trapdoors[target_label] = trapdoor

    return model, [trapdoors[l] for l in trapdoors]