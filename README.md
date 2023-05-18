# Prompt- and Trait Relation-aware Cross-prompt Essay Trait Scoring (ProTACT)

This repository is the implementation of the ProTACT architecture, introduced in the paper, **Prompt- and Trait Relation-aware Cross-prompt Essay Trait Scoring** (ACL Findings 2023).

> The code is based on the open code, [https://github.com/robert1ridley/cross-prompt-trait-scoring](https://github.com/robert1ridley/cross-prompt-trait-scoring) (Ridley, 2021).

## Package Requirements

Install below packages in your virtual environment before running the code.
- python==3.7.11
- tensorflow=2.0.0
- numpy=1.18.1
- nltk=3.4.5
- pandas=1.0.5
- scikit-learn=0.22.1

## Download GloVe

For prompt word embedding, we use the pretrained GloVe embedding.
- Go to `https://nlp.stanford.edu/projects/glove/` and download `glove.6B.50d.txt`.
- Put downloaded file in the `embeddings` directory.

## Training 
### baseline model (Ridley, 2021)
- Run `./train_CTS.sh`

### ProTACT
- Run `./train_ProTACT.sh`

This bash script will run each model 5 times with different seeds ([12, 22, 32, 42, 52]).