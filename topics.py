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


def get_corpus_freqs(freqs, tokens_key):
    """Return counts of all tokens for the corpus."""
    counts = collections.defaultdict(int)
    for doc in freqs:
        for token, freq in doc[tokens_key].items():
            counts[token] += freq
    return counts


def find_singletons(freqs):
    """Return a set of all items that occur only once."""
    singletons = set()
    for token, freq in freqs.items():
        if freq == 1:
            singletons.add(token)
    return singletons


def remove_singletons(freqs, singletons, tokens_key):
    """Remove all tokens from freqs that are in singletons."""
    for doc in freqs:
        tokens = doc[tokens_key]
        for token in tokens.keys() & singletons:
            del tokens[token]
        yield doc


def main():
    """The main process."""
    stoplist = read_stoplist(STOPWORD_FILE)

    corpus = read_corpus(INPUT_FILES, TEXT_FIELD)
    tokens = tokenize(corpus, TEXT_FIELD, TOKENS_FIELD)
    normed = normalize(tokens, TOKENS_FIELD, stoplist)
    freqs = list(frequencies(normed, TOKENS_FIELD))
    corpus_freq = get_corpus_freqs(freqs, TOKENS_FIELD)
    singletons = find_singletons(corpus_freq)
    freqs = list(remove_singletons(freqs, singletons, TOKENS_FIELD))
    for doc in freqs:
        print(doc[TOKENS_FIELD])
    print(singletons)


if __name__ == '__main__':
    main()
