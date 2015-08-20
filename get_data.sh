#!/bin/bash

wget https://github.com/bothameister/bothameister.github.io/raw/master/icml14-data.tar.bz2
tar xf icml14-data.tar.bz2

mkdir data/de/
cp en-de/1m-mono/train.in data/de/train.txt
cp en-de/1m-mono/test.in data/de/valid.txt
cp en-de/1m-mono/finaltest.in data/de/test.txt

mkdir data/es/
cp en-es/1m-mono/train.in data/es/train.txt
cp en-es/1m-mono/test.in data/es/valid.txt
cp en-es/1m-mono/finaltest.in data/es/test.txt

mkdir data/cs/
cp en-cs/1m-mono/train.in data/cs/train.txt
cp en-cs/1m-mono/test.in data/cs/valid.txt
cp en-cs/1m-mono/finaltest.in data/cs/test.txt

mkdir data/fr/
cp en-fr/1m-mono/train.in data/fr/train.txt
cp en-fr/1m-mono/test.in data/fr/valid.txt
cp en-fr/1m-mono/finaltest.in data/fr/test.txt

mkdir data/ru/
cp en-ru/1m-mono/train.in data/ru/train.txt
cp en-ru/1m-mono/test.in data/ru/valid.txt
cp en-ru/1m-mono/finaltest.in data/ru/test.txt
