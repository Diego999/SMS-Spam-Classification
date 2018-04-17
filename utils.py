import csv
import re
import pickle
import os
import math
import numpy as np


def replace_url(text):
    text = re.sub('((http|ftp|https)://)?([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '<url>', text)  # replace urls by <url>
    text = re.sub('(http|ftp|https)://.+\<url\>', '<url>', text)
    return text


def replace_email(text):
    text = re.sub('[a-zA-Z0-9-.]+@[a-zA-Z0-9-.]+', '<email>', text)
    return text


def replace_phone(text):
    text = re.sub("\(?\d{3,}\)?[-\s\.]?\d{3}[-\s\.]?\d{4,}", '<phone>', text)
    text = re.sub("\d{5,}", '<phone>', text)
    text = re.sub("\d+\s+\<phone\>", '<phone>', text)
    text = re.sub("\<phone\>\s+\d+", '<phone>', text)
    return text


def format_token(token):
    """"""
    if token.upper() == '-LRB-':
        token = '('
    elif token.upper() == '-RRB-':
        token = ')'
    elif token.upper() == '-RSB-':
        token = ']'
    elif token.upper() == '-LSB-':
        token = '['
    elif token.upper() == '-LCB-':
        token = '{'
    elif token.upper() == '-RCB-':
        token = '}'
    return token


def read_data(filepath='./data/spam.csv'):
    data = []
    with open(filepath, 'r') as fp:
        reader = csv.reader(fp)
        for i, row in enumerate(reader):
            data.append({'id':i, 'text':row[1], 'type':row[0]})
    return data[1:] # Ignore headers


def tokenize_lemmatize(data):
    import corenlp
    os.environ['CORENLP_HOME'] = 'lib/stanford-corenlp-full'
    with corenlp.CoreNLPClient(annotators='tokenize ssplit pos lemma'.split()) as client:
        for sample in data:
            ann = client.annotate(replace_email(replace_phone(replace_url(sample['text']))))
            tokens = []
            lemmas = []
            for sent in ann.sentence:
                tokens += [format_token(token.word) for token in sent.token]
                lemmas += [format_token(token.lemma) for token in sent.token]
            sample['tokens'] = tokens
            sample['lemmas'] = lemmas
    return data


def get_stopwords(filepath='./data/stopwords.txt'):
    stopwords = set()
    with open(filepath, 'r', encoding='utf-8') as fp:
        for line in fp:
            stopwords.add(line.strip())
    return stopwords


def compute_naive_representation(data):
    stopwords = get_stopwords()
    word_to_index = set()

    for sample in data:
        for lemma in sample['lemmas']:
            lemma = lemma.lower()
            if len(lemma) > 2 and lemma not in ['...', '..'] and lemma not in stopwords:
                word_to_index.add(lemma)

    word_to_index = {word:i for i, word in enumerate(['UNK'] + sorted(word_to_index))}
    for sample in data:
        sample['naive'] = [word_to_index[lemma.lower()] if lemma.lower() in word_to_index else word_to_index['UNK'] for lemma in sample['lemmas']]

    return data, word_to_index


def compute_bag_of_words_representation(data, word_to_index):
    for sample in data:
        sample['bag_of_words'] = [word_to_index[lemma.lower()] for lemma in sample['lemmas'] if lemma.lower() in word_to_index]

    return data


# For simplicity, we use our own tf-idf computation from https://gist.github.com/anabranch/48c5c0124ba4e162b2e3
def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)


def term_frequency(term, tokenized_document):
    return tokenized_document.count(term)


def sublinear_term_frequency(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return 1 + math.log(count)


def augmented_term_frequency(term, tokenized_document):
    max_count = max([term_frequency(t, tokenized_document) for t in tokenized_document])
    return (0.5 + ((0.5 * term_frequency(term, tokenized_document))/max_count))


def inverse_document_frequencies(tokenized_documents, all_tokens_set):
    idf_values = {}
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(1 + sum(contains_token)))
    return idf_values


def compute_tfidf_representation(data, word_to_index):
    tokenized_documents = []
    for sample in data:
        tokenized_documents.append([lemma.lower() for lemma in sample['lemmas'] if lemma.lower() in word_to_index])

    idf = inverse_document_frequencies(tokenized_documents, set(word_to_index.keys()))
    for i, document in enumerate(tokenized_documents):
        doc_tfidf = []
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            doc_tfidf.append(tf * idf[term])
        doc_tfidf_norm = np.linalg.norm(doc_tfidf)
        if doc_tfidf_norm < 1e-3: # Might be composed of stopwords only
            doc_tfidf_norm = 1
        data[i]['tfidf'] = [x/doc_tfidf_norm for x in doc_tfidf]

    return data


def get_data(filepath='./data/spam.csv'):
    filepath_pkl = filepath.replace('csv', 'pkl')

    if not os.path.exists(filepath_pkl):
        data = read_data()
        data = tokenize_lemmatize(data)
        pickle.dump(data, open(filepath_pkl, 'wb'))
    else:
        data = pickle.load(open(filepath_pkl, 'rb'))

    return data


