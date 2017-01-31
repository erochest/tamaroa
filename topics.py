#!/usr/bin/env python3


import collections
import csv
from gensim import corpora
from gensim.models import ldamodel as lda
import nltk
import operator
import os
import pickle
import pprint


# parameters
TOPICS = 40
ALPHA = 'symmetric'
# ALPHA = 'auto'
ETA = None
# ETA = 'auto'
PASSES = 1

#
INPUT_FILES = [
    'Manifest_GenRef.tsv',
    ]
STOPWORD_FILE = 'english.stopwords'
TEXT_FIELD = 'Title'
TOKENS_FIELD = '*tokens*'

FREQ_FILE = 'corpus.freq'
DICTIONARY_FILE = 'corpus.dict'
CORPUS_FILE = 'corpus.mm'


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


def get_corpus_freqs(freqs, tokens_key):
    """Return counts of all tokens for the corpus."""
    counts = collections.defaultdict(int)
    for doc in freqs:
        for token in doc[tokens_key]:
            counts[token] += 1
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

        removed = []
        for token in tokens:
            if token not in singletons:
                removed.append(token)

        doc[tokens_key] = removed
        yield doc


def get_text_corpus(freq_docs, tokens_key):
    """Get text from documents and return as a list. """
    text_corpus = []
    for doc in freq_docs:
        text_corpus.append(doc[tokens_key])
    return text_corpus


def get_dictionary(text_corpus, dictionary_file):
    """Index the tokens in the corpus and return the dictionary."""
    dictionary = corpora.Dictionary(text_corpus)
    dictionary.save(dictionary_file)
    return dictionary


def get_corpus_matrix(dictionary, corpus, corpus_file):
    """With a dictionary and list of text frequencies, return a matrix."""
    doc_matrix = []
    for doc in corpus:
        vec = dictionary.doc2bow(doc)
        doc_matrix.append(vec)
    corpora.MmCorpus.serialize(corpus_file, doc_matrix)
    return doc_matrix


def main():
    """The main process."""
    stoplist = read_stoplist(STOPWORD_FILE)

    if (os.path.exists(CORPUS_FILE) and os.path.exists(DICTIONARY_FILE) and
        os.path.exists(FREQ_FILE)):
        print('reading from disk')
        dictionary = corpora.Dictionary.load(DICTIONARY_FILE)
        doc_matrix = corpora.MmCorpus(CORPUS_FILE)
        with open(FREQ_FILE, 'rb') as fin:
            freqs = pickle.load(fin)

    else:
        print('creating corpus')
        corpus = read_corpus(INPUT_FILES, TEXT_FIELD)
        tokens = tokenize(corpus, TEXT_FIELD, TOKENS_FIELD)
        normed = list(normalize(tokens, TOKENS_FIELD, stoplist))
        corpus_freq = get_corpus_freqs(normed, TOKENS_FIELD)
        singletons = find_singletons(corpus_freq)
        freqs = list(remove_singletons(normed, singletons, TOKENS_FIELD))
        with open(FREQ_FILE, 'wb') as fout:
            pickle.dump(freqs, fout)
        text_corpus = get_text_corpus(freqs, TOKENS_FIELD)
        dictionary = get_dictionary(text_corpus, DICTIONARY_FILE)
        doc_matrix = get_corpus_matrix(dictionary, text_corpus, CORPUS_FILE)

    print('dictionary size =', len(dictionary))
    print('corpus size =', len(list(doc_matrix)))

    # topic modeling
    print('generating topics')
    topics = lda.LdaModel(
        corpus=doc_matrix,
        id2word=dictionary,
        num_topics=TOPICS,
        alpha=ALPHA,
        eta=ETA,
        passes=PASSES,
    )
    output = topics.print_topics(TOPICS)
    output.sort(key=operator.itemgetter(0))
    for t, words in output:
        print('{} : {}'.format(t, words))

    for doc, bow in zip(freqs, doc_matrix):
        output = topics[bow]
        print(doc['Filename'], output)


if __name__ == '__main__':
    main()
