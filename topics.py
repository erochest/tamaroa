#!/usr/bin/env python3


import collections
import csv
from gensim import corpora
import nltk


INPUT_FILES = [
    'Manifest_GenRef.tsv',
    ]
STOPWORD_FILE = 'english.stopwords'
TEXT_FIELD = 'Title'


def read_stoplist(filename):
    """Read the stopword list as a set."""
    with open(filename) as fin:
        words = set()
        for line in fin:
            words.add(line.strip())
        return words


def main():
    stoplist = read_stoplist(STOPWORD_FILE)

    freq_corpus = []
    for input_file in INPUT_FILES:
        with open(input_file) as fin:
            corpus = list(csv.DictReader(fin, dialect=csv.excel_tab))
            for i, doc in enumerate(corpus):
                if doc[TEXT_FIELD] == 0:
                    break
                text = doc[TEXT_FIELD]
                tokens = nltk.word_tokenize(text)

                normalized = collections.defaultdict(int)
                for token in tokens:
                    if token.isalnum():
                        token = token.lower()
                        if token not in stoplist:
                            normalized[token] += 1
                freq_corpus.append(normalized)

    # this maps tokens to their indexes in the document vectors.
    vector_index = {}
    for doc in freq_corpus:
        for token in doc.keys():
            if token not in vector_index:
                vector_index[token] = len(vector_index)

    # TODO: comment the hell out of this
    # TODO: send
    corpus = []
    for doc in freq_corpus:
        doc_vector = [0] * len(vector_index)
        for token, freq in doc.items():
            doc_vector[vector_index[token]] = freq
            print(doc_vector)
            corpus.append(doc_vector)


# from collections import defaultdict
# frequency = defaultdict(int)
# for text in texts:
#     for token in text:
#         frequency[token] += 1


if __name__ == '__main__':
    main()
