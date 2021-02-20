# Word2Vec
Tensorflow implementation of Word2Vec model with **skip gram** architecture and **negative sampling** algorithm. And this implementation supports training multiple very large files (>10GB).

### Dependencies
 - Python 3.6
 - Tensorflow 1.15.2

### Usage
```bash
python train.py
```
By default settings, the training data should be put in `datasets` folder, and remember to check all the hyper-parameters. After training, the word vectors will be saved to `embeddings` folder.
