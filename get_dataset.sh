#!/bin/bash

DATASETS_DIR = data/
mkdir -p $DATASETS_DIR

cd $DATASETS_DIR

# Get CNN/DM Dataset
wget http://cs.stanford.edu/~danqi/data/cnn.tar.gz
tar xvf cnn.tar.gz
rm cnn.tar.gz

wget http://cs.stanford.edu/~danqi/data/dailymail.tar.gz
tar xvf dailymail.tar.gz
rm dailymail.tar.gz

# Get GloVe vectors (all dims)
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip xvf glove.6B.zip
rm glove.6B.zip
