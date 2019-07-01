#!/bin/bash

#download and install kenlm 
wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz
echo "\$CXX ../probs.cpp \$objects -o ../probs \$CXXFLAGS \$LDFLAGS" | cat >> kenlm/compile_query_only.sh
(cd kenlm && ./compile_query_only.sh)

#download and convert language model
if [ ! -f "4-gram.arpa.gz" ]; then
  wget http://www.openslr.org/resources/11/4-gram.arpa.gz 
fi
gzip -d 4-gram.arpa.gz

kenlm/bin/build_binary 4-gram.arpa 4-gram.binary

#download vocabulary and count probabilities
if [ ! -f "librispeech-vocab.txt" ]; then
  wget http://www.openslr.org/resources/11/librispeech-vocab.txt
fi

./probs 4-gram.binary < librispeech-vocab.txt > librispeech-probs.txt

#build ctcbeam
python3 setup.py install

