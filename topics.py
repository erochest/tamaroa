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
TOKENS_FIELD = '*tokens*'


def read_stoplist(filename):
    """Read the stopword list as a set."""
    with open(filename) as fin:
        words = set()
        for line in fin:
            words.add(line.strip())
        return words


def read_corpus(input_files, text_key):
    """\
    This reads a CSV input file into an iterator over lists of dicts.
    """
    for input_file in input_files:
        with open(input_file) as fin:
            for doc in csv.DictReader(fin, dialect=csv.excel_tab):
                text = doc[text_key]
                if text == 0:
                    break
                yield doc


def tokenize(corpus, text_key, tokens_key):
    """\
    This tokenizes the text under text_key into a list of tokens under
    tokens_key.

    Corpus is an iterator and this re-yields the documents.
    """
    for doc in corpus:
        text = doc[text_key]
        doc[tokens_key] = nltk.word_tokenize(text)
        yield doc


def normalize(corpus, tokens_key, stoplist_set):
    """This normalizes the tokens under tokens_key.

    Tokens are case-folded and filtered for alpha-numeric tokens and by a
    stoplist.

    Corpus is an iterator and this re-yields the documents.
    """
    for doc in corpus:
        tokens = []
        for token in doc[tokens_key]:
            if token.isalnum():
                token = token.lower()
                if token not in stoplist_set:
                    tokens.append(token)
        doc[tokens_key] = tokens
        yield doc


def frequencies(corpus, tokens_key):
    """This converts each document's token list into a frequency dictionary."""
    for doc in corpus:
        freqs = collections.defaultdict(int)
        for token in doc[tokens_key]:
            freqs[token] += 1
        doc[tokens_key] = freqs
        yield doc


def build_vector_index(corpus, tokens_key):
    """This takes a corpus, where the tokens are in an iterable.

    It returns a dict mapping tokens to unique indexes for building a vector
    space representation of the corpus.

    If corpus is an iterator, it will be consumed.
    """
    vector_index = {}

    for doc in corpus:
        for token in doc[tokens_key]:
            if token not in vector_index:
                vector_index[token] = len(vector_index)

    return vector_index


def vectorize(corpus, tokens_key, index):
    """This creates vector-space representation of all documents."""
    for doc in corpus:
        vector = [0] * len(index)
        for token, freq in doc[tokens_key].items():
            i = index[token]
            vector[i] = freq
        doc[tokens_key] = vector
        yield doc


def main():
    """The main process."""
    stoplist = read_stoplist(STOPWORD_FILE)

    corpus = read_corpus(INPUT_FILES, TEXT_FIELD)
    tokens = tokenize(corpus, TEXT_FIELD, TOKENS_FIELD)
    normed = normalize(tokens, TOKENS_FIELD, stoplist)
    freqs = list(frequencies(normed, TOKENS_FIELD))

    vector_index = build_vector_index(freqs, TOKENS_FIELD)
    doc_matrix = []
    for doc in vectorize(freqs, TOKENS_FIELD, vector_index):
        print(doc)
        doc_matrix.append(doc[TOKENS_FIELD])


if __name__ == '__main__':
    main()
