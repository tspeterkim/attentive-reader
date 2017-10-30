# Attentive Reader

This is a TensorFlow implementation of the modified Attentive Reader in the paper [A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task](https://arxiv.org/pdf/1606.02858v2.pdf) by Danqi Chen, Jason Bolton, and Christopher Manning.

The data preprocessing step is identical to the one presented in the official Theano/Lasagne implementation [here](https://github.com/danqi/rc-cnn-dailymail). The good stuff comes after, when the TensorFlow computation graph is built.

## Dependencies
* Python 2.7
* Tensorflow >= 1.1

## Usage

### Datasets

To download the CNN/DailyMail datasets, and the GloVe embeddings, run the following script:
```
./get_dataset.sh
```

### Training
```
python main.py
```
