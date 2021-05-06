
# HoneypotDefenseNLP-ACL21
This is the official code for the following paper.

##### Thai Le, Noseong Park, Dongwon Lee. A Sweet Rabbit Hole by DARCY: Using Honeypots to Detect Universal Trigger’s Adversarial Attacks. 59th Annual Meeting of the Association for Computational Linguistics (ACL) 2021. (Long Paper, Main Conference)

## Requirement
Check ``requirements.txt``.
Note: a main portion of the code uses ``allennlp`` library, which has been tremendously updated since then. Please make sure to install the correct version from the ``requirements.txt`` file.

## Instruction
- Download pretrained fasttext word-embedding at [link](https://fasttext.cc/docs/en/english-vectors.html) and place inside the project folder.
- Available datasets include``rotten_tomatoes``, ``subjectivity``, ``agnews``
- SST dataset can be downloaded using ``download_sst_dataset.sh``
#### Defense against universal attacks using random trapdoors on ``RNN`` trained with ``rotten_tomatoes`` dataset:
``python main.py --model RNN --dataset rotten_tomatoes --embedding_path ./crawl-300d-2M.vec --trapdoor_method random --smooth_eps 0.75``
#### Defense against universal attacks using DARCY on ``CNN`` trained with ``subjectivity`` dataset:
``python main.py --model CNN --dataset subjectivity --embedding_path ./crawl-300d-2M.vec --trapdoor_method DARCY --smooth_eps 0.75``
#### Defense against universal attacks using DARCY on ``CNN`` trained with ``agnews`` dataset with 2 trapdoors:
``python main.py --model CNN --dataset agnews  --num_class 4 --embedding_path ./crawl-300d-2M.vec --trapdoor_method DARCY --smooth_eps 0.75 --trapdoor_num 2``
#### Other optional configurations:
- ``--trigger_ignore``: # of top triggers to ignore while searching for universal attacks as shown in the paper
- ``--oracle``: oracle universal trigger attacks as shown in the paper
- Please check ``args.py`` for more options.


## Example Outputs
``python main.py --model CNN --dataset rotten_tomatoes --embedding_path ../crawl-300d-2M.vec --trapdoor_method DARCY --smooth_eps 0.75 --trapdoor_num 2``
```
loading dataset
8530it [00:05, 1539.55it/s]
1066it [00:00, 1671.20it/s]
1066it [00:00, 1389.37it/s]
loaded vocab size 16672
Distribution of model train (array([0, 1]), array([4265, 4265]))
Total data 10662
accuracy: 0.7360, loss: 0.5265 ||: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [00:02<00:00, 115.59it/s]
searching for trapdoor...
populating additional data...
populating trapdoor train data...
additional_data 768
trapdoor_train_data 2457
TRAPDOORS [[unfunny, soulless], [bourne, masterful]]
Training Model
accuracy: 0.7048, loss: 0.2447 ||: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 264/264 [00:01<00:00, 146.17it/s]
accuracy: 0.7655, loss: 0.6083 ||: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 27/27 [00:00<00:00, 199.46it/s]
accuracy: 0.9780, loss: 0.0625 ||: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 264/264 [00:02<00:00, 120.21it/s]
accuracy: 0.7491, loss: 1.0505 ||: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 27/27 [00:00<00:00, 225.21it/s]
WARNING:root:vocabulary serialization directory ./saved/w2v_vocab_rotten_tomatoes is not empty
Training Detector
Initialize NETWORK detector, cuda() and optimizer()
Distribution of detector train [1229  614  614]
Distribution of detector dev [307 154 154]
accuracy: 0.8038, loss: 0.4468 ||: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 77/77 [00:00<00:00, 181.22it/s]
accuracy: 0.8732, loss: 0.3085 ||: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 237.28it/s]
accuracy: 0.9577, loss: 0.1061 ||: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 77/77 [00:00<00:00, 208.14it/s]
accuracy: 0.9252, loss: 0.1424 ||: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 307.47it/s]
accuracy: 0.9691, loss: 0.0865 ||: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 77/77 [00:00<00:00, 111.92it/s]
accuracy: 0.9626, loss: 0.0916 ||: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 302.60it/s]
   target_label       ACC        F1  TARGET_ACC  DETECT AUC  INIT  \
0             0  0.749531  0.719243    0.000000    0.965291  lust   
1             1  0.749531  0.719243    0.001876    0.977486  pegs   

               TRIGGER  
0  soulless;accumulate  
1  masterful;masterful
```

## Credits
Thanks Eric-Wallace for the base code of Universal Trigger (https://github.com/Eric-Wallace/universal-triggers)

## Citation
Please cite the work using the following bibtex.
```
@article{le2021darcy,
    title={A Sweet Rabbit Hole by DARCY: Using Honeypots to Detect Universal Trigger’s Adversarial Attacks},
    author={Thai Le and Noseong Park and Dongwon Lee},
    year={2021},
    journal={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL'2021)},
}
```
