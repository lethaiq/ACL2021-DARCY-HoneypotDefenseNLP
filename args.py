from argparse import ArgumentParser
import os

def training_params():
    parser = ArgumentParser(description='Honeyspot Defense Text')
    
    parser.add_argument('--dataset', type=str, default="sst")
    parser.add_argument('--embedding_type', type=str, default="w2v")
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--embedding_path', type=str, default="./crawl-300d-2M.vec")

    parser.add_argument('--model', type=str, default="RNN")

    parser.add_argument('--warmup_num', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--detector_batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_detector', type=float, default=0.01)
    parser.add_argument('--smooth_eps', type=float, default=1)

    parser.add_argument('--detector_epochs', type=int, default=3)
    parser.add_argument('--detector_patience', type=int, default=1)
    parser.add_argument('--detector_l2', type=float, default=0.0)
    parser.add_argument('--detector_binary', type=int, default=0)

    parser.add_argument('--universal_batch_size', type=int, default=128)
    parser.add_argument('--trigger_length', type=int, default=2)
    parser.add_argument('--trigger_neighbor', type=int, default=5000)
    parser.add_argument('--trigger_ignore', type=int, default=0)
    parser.add_argument('--trigger_epochs', type=int, default=5)
    parser.add_argument('--num_candidates', type=int, default=40)
    parser.add_argument('--init_trigger', type=str, default="") 

    parser.add_argument('--trapdoor_num', type=int, default=1)
    parser.add_argument('--trapdoor_method', type=str, default="DARCY")
    parser.add_argument('--trapdoor_dev_ratio', type=float, default=0.2)

    parser.add_argument('--detector_ratio', type=float, default=0.1)
    parser.add_argument('--trapdoor_ratio', type=float, default=0.1)
    parser.add_argument('--trapdoor_num_random', type=int, default=300)
    parser.add_argument('--trapdoor_inter_neighbor', type=int, default=3000)
    parser.add_argument('--trapdoor_intra_neighbor', type=int, default=1000)

    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--target_labels', type=int, nargs='+', default=[0,1])
    parser.add_argument('--oracle', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--seed', type=int, default=77)

    args = parser.parse_args()

    args.target_labels = list(range(args.num_class))
    args.vocab_path = "./saved/" + args.embedding_type + "_" + "vocab_{}".format(args.dataset)
    try:
        os.makedirs("./saved")
    except:
        pass

    return args
