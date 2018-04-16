import csv
import re
import pickle
import os


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


def get_data(filepath='./data/spam.csv'):
    filepath_pkl = filepath.replace('csv', 'pkl')

    if not os.path.exists(filepath_pkl):
        data = read_data()
        data = tokenize_lemmatize(data)
        pickle.dump(data, open(filepath_pkl, 'wb'))
    else:
        data = pickle.load(open(filepath_pkl, 'rb'))

    return data