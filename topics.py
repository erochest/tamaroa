#!/usr/bin/env python3


import collections
import csv
from gensim import corpora
# from gensim.models import ldamulticore as lda
from gensim.models import ldamodel as lda
from multiprocessing import Pool
import nltk
import operator
import os
import pickle
import pprint
import sys


# parameters
TOPICS = 10
ALPHA = 'symmetric'
# ALPHA = 'auto'
ETA = None
# ETA = 'auto'
PASSES = 100

# !!! If any of the settings below change, re-run with CLEAR_CACHE set to True
# in order to generate a new corpus.
CLEAR_CACHE = True
INPUT_FILES = [
    'Manifest_GenRef.tsv',
    ]
STOPWORD_FILE = 'english.stopwords'
# minimum token length
# MIN_TOKEN_LEN = 0
MIN_TOKEN_LEN = 2
# TEXT_FIELD = 'Title'
# ID_FIELD = 'Filename'
TEXT_FIELD = 'F1_Question_FreeText'
ID_FIELD = 'ResponseId'
TOKENS_FIELD = '*tokens*'

FREQ_FILE = 'corpus.freq'
DICTIONARY_FILE = 'corpus.dict'
CORPUS_FILE = 'corpus.mm'
TOPIC_FILE = 'corpus-topic.csv'
DOC_FILE = 'corpus-docs.csv'


def process_file(input_file):
    """Do everything to process a single file."""
    stoplist = read_stoplist(STOPWORD_FILE)
    corpus = read_file(input_file, TEXT_FIELD)
    tokens = tokenize(corpus, TEXT_FIELD, TOKENS_FIELD)
    normed = list(normalize(tokens, TOKENS_FIELD, stoplist, MIN_TOKEN_LEN))
    corpus_freq = get_corpus_freqs(normed, TOKENS_FIELD)
    singletons = find_singletons(corpus_freq)
    freqs = list(remove_singletons(normed, singletons, TOKENS_FIELD))
    return freqs


def read_stoplist(filename):
    """Read the stopword list as a set."""
    with open(filename) as fin:
        words = set()
        for line in fin:
            words.add(line.strip())
        return words


def read_file(input_file, text_key):
    """This reads a single CSV input file."""
    with open(input_file) as fin:
        for doc in csv.DictReader(fin):
            try:
                text = doc[text_key]
            except KeyError:
                print('Keys: %s' % (doc.keys(),))
                raise
            if text == 0:
                break
            elif not text:
                continue
            yield doc


def read_corpus(input_files, text_key):
    """\
    This reads a CSV input file into an iterator over lists of dicts.
    """
    for input_file in input_files:
        yield from read_file(input_file, text_key)


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


def normalize(corpus, tokens_key, stoplist_set, min_token_len):
    """This normalizes the tokens under tokens_key.

    Tokens are case-folded and filtered for alpha-numeric tokens and by a
    stoplist.

    Corpus is an iterator and this re-yields the documents.
    """
    for doc in corpus:
        tokens = []
        for token in doc[tokens_key]:
            if token.isalnum() and len(token) > min_token_len:
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
    inputs = sys.argv[1:] or INPUT_FILES
    print('inputs', inputs)

    if (not CLEAR_CACHE and os.path.exists(CORPUS_FILE) and
        os.path.exists(DICTIONARY_FILE) and os.path.exists(FREQ_FILE)):
        print('reading from disk')
        dictionary = corpora.Dictionary.load(DICTIONARY_FILE)
        doc_matrix = corpora.MmCorpus(CORPUS_FILE)
        with open(FREQ_FILE, 'rb') as fin:
            freqs = pickle.load(fin)

    else:
        print('creating corpus')
        freqs = []
        #  with Pool() as pool:
            #  for file_freqs in pool.map(process_file, inputs):
                #  freqs += file_freqs
        for filename in inputs:
            freqs += process_file(filename)
        with open(FREQ_FILE, 'wb') as fout:
            pickle.dump(freqs, fout)
        text_corpus = get_text_corpus(freqs, TOKENS_FIELD)
        dictionary = get_dictionary(text_corpus, DICTIONARY_FILE)
        doc_matrix = get_corpus_matrix(dictionary, text_corpus, CORPUS_FILE)

    print('dictionary size =', len(dictionary))
    print('corpus size =', len(list(doc_matrix)))

    # topic modeling
    print('generating topics')
    # topics = lda.LdaMulticore(
    topics = lda.LdaModel(
        corpus=doc_matrix,
        id2word=dictionary,
        num_topics=TOPICS,
        alpha=ALPHA,
        eta=ETA,
        passes=PASSES,
    )
    print('writing topic terms to {}'.format(TOPIC_FILE))
    with open(TOPIC_FILE, 'w') as fout:
        writer = csv.writer(fout)
        for topic_number in range(TOPICS):
            terms = topics.get_topic_terms(topic_number)
            terms.sort(key=operator.itemgetter(1), reverse=True)
            row = [topic_number]
            for (term_id, _) in terms:
                row.append(dictionary.get(term_id))
            writer.writerow(row)

    print('writing document topics to {}'.format(DOC_FILE))
    with open(DOC_FILE, 'w') as fout:
        writer = csv.writer(fout)
        # Add extra column names here.
        writer.writerow(['filename'] + list(range(TOPICS)))
        for doc, bow in zip(freqs, doc_matrix):
            row = [0.0] * TOPICS
            for (topic_number, score) in topics[bow]:
                row[topic_number] = score
            # You can add extra rows in the output here.
            row = [doc[ID_FIELD]] + row
            writer.writerow(row)


if __name__ == '__main__':
    main()
