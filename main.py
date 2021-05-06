from args import *
from models import *
from data import *
from modules import *
from generator import *
import pandas
from allennlp.data.tokenizers.token import Token
from unitrigger_utils import get_embedding_weight

args = training_params()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

pandas.set_option('display.max_columns', None)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.set_printoptions(precision=3)

cols = ['target_label', 'test_clean_acc_without', 'test_clean_f1_without', 'test_target_acc_with', 'test_target_detect_auc', 'init_trigger', 'trigger']
labels = ['target_label', 'ACC', 'F1', 'TARGET_ACC', 'DETECT AUC', 'INIT', 'TRIGGER']

def main(args):
    train_data, train_dev_data, dev_data, test_data, vocab = load_dataset(args)
    model, encoder, word_embeddings = initialize_model(args, vocab)
    trapdoors = [[None] for _ in range(len(args.target_labels))]

    if args.trapdoor_method == "DARCY":
        model, trapdoors = prepare_trapdoors_DARCY(args, train_data, train_dev_data, vocab, args.target_labels,
                                                trapdoor_num=args.trapdoor_num, 
                                                num_random=args.trapdoor_num_random,
                                                inter_tree_neighbor=args.trapdoor_inter_neighbor,
                                                intra_tree_neighbor=args.trapdoor_intra_neighbor)
        model.training = True
    else:
        trapdoors = prepare_trapdoors_random(vocab, args.num_class, args.trapdoor_num)

    additional_data, trapdoor_train_data, trapdoor_dev_data = populate_data(args, 
                                                            train_data, args.target_labels, trapdoors,
                                                            dev_ratio=args.trapdoor_dev_ratio, 
                                                            trapdoor_ratio=args.trapdoor_ratio,
                                                            detector_ratio=args.detector_ratio)
    train_data = train_data + additional_data
    random.shuffle(train_data)
    print("additional_data", len(additional_data))
    print("trapdoor_train_data", len(trapdoor_train_data))
    print("TRAPDOORS", trapdoors)

    # MAIN MODEL
    print("Training Model")
    train_model(args, model, vocab, train_data, train_dev_data)
    vocab.save_to_files(args.vocab_path)
    model.training = False
    detector_model = model
 
    # DETECTOR
    print("Training Detector")
    detector = load_detector(args, detector_model, encoder, vocab)
    trapdoor_train = trapdoor_train_data
    trapdoor_dev = trapdoor_dev_data
    train_detector(args, detector, vocab, trapdoor_train, trapdoor_dev)
      
    model.train() # for enabling gradients to go through the network

    rt_test_clean = evaluate(model, vocab, test_data, trigger_token_ids=None, detector=None, prefix="test_clean")
    rt_dev_clean = evaluate(model, vocab, dev_data, trigger_token_ids=None, detector=None, prefix="dev_clean")

    tree = None
    embedding_weight = get_embedding_weight(model)
    tree = KDTree(embedding_weight.numpy())

    results = []
    for target_label, trapdoor in zip(args.target_labels, trapdoors):
        all_triggers = []
        previous_inits = []
        previous_triggers = []
        targeted_test_data = filter_data(test_data, target_label)
        targeted_dev_data = filter_data(dev_data, target_label, sample_ratio=0.5)

        if not args.oracle:
            trigger_token_ids, init_trigger = attack_unitrigger(args, model, vocab, target_label, targeted_dev_data, 
                init_trigger=args.init_trigger, previous_inits=previous_inits, previous_triggers=previous_triggers, tree=tree)

        else: #in oracle attacks the attackers have access to the detector
            trigger_token_ids, init_trigger = attack_unitrigger(args, model, vocab, target_label, targeted_dev_data, 
                            init_trigger=args.init_trigger, previous_inits=previous_inits, 
                            exempt_triggers=previous_triggers, tree=tree, surrogate=detector)

        all_triggers.append(trigger_token_ids)
        rt_before = evaluate(model, vocab, targeted_test_data, trigger_token_ids=None, detector=None, prefix="test_target")
        rt_after = evaluate(model, vocab, targeted_test_data, trigger_token_ids, detector, prefix="test_target")
        if not detector:
            rt_after = {**rt_after, **{'test_target_detect_auc':'N/A', 'test_target_detect_fpr':'N/A', 'test_target_detect_tpr': 'N/A'}}
        rt = {**rt_dev_clean, **rt_test_clean, **rt_before, **rt_after}
        rt['target_label'] = target_label
        rt['trapdoor'] = ",".join([token.text for token in trapdoor])
        rt['trigger'] = translate_triggers(trigger_token_ids, vocab)
        rt['init_trigger'] = init_trigger
        results.append(rt)

    df = pd.DataFrame.from_dict(results)
    df = df[cols]
    df.columns = labels
    print(df)

if __name__ == '__main__':
    main(args)
