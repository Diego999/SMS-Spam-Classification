import csv
import re
import pickle
import os
import math
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def get_topics(data, filepath='./data/spam_topics.pkl'):
    if not os.path.exists(filepath):
        import pyLDAvis.gensim
        from gensim.corpora import Dictionary
        from gensim.models import LdaModel, CoherenceModel

        texts = [sample['lemmas'] for sample in data]

        dictionary = Dictionary(texts)
        dictionary.filter_extremes(no_below=20, no_above=0.4)
        corpus = [dictionary.doc2bow(text) for text in texts]

        chunksize = 500
        passes = 5
        iterations = 400
        eval_every = None

        temp = dictionary[0]  # This is only to "load" the dictionary.
        id2word = dictionary.id2token

        best_coherence = 0
        best_model_filepath = ''
        for num_topics in list(range(2, 20)):
            for alpha in ['asymmetric', 'symmetric']:
                for eta in ['symmetric', 'auto']:
                    filepath = 'out/topics/{}_{}_{}'.format(num_topics, alpha, eta)
                    model = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, alpha='auto', eta='auto', iterations=iterations, num_topics=num_topics, passes=passes, eval_every=eval_every)
                    coherence = float(CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v').get_coherence())
                    filepath += '_{:.4f}'.format(coherence)
                    model.save(filepath + '_model.pkl')

                    prepared = pyLDAvis.gensim.prepare(model, corpus, dictionary)
                    pyLDAvis.save_html(prepared, filepath + '_plot.html')

                    if coherence > best_coherence:
                        best_coherence = coherence
                        best_model_filepath = filepath + '_model.pkl'

        model = LdaModel.load(best_model_filepath)
        print('Best model: {}'.format(best_model_filepath))

        topics = [x[0] for x in model.top_topics(corpus=corpus, texts=texts, dictionary=dictionary, topn=100)]
        for i, text in enumerate(texts):
            data[i]['topics'] = {k: v for k, v in model.get_document_topics(dictionary.doc2bow(text), minimum_probability=0.0)}
        pickle.dump([topics, data], open(filepath, 'wb'))
    else:
        [topics, data] = pickle.load(open(filepath, 'rb'))

    return topics, data


######### PREPROCESSING #########
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
    return data


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

    word_to_index = {word:i for i, word in enumerate(['PAD'] + ['UNK'] + sorted(word_to_index))}
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


def compute_word_embeddings_representation(data):
    vocabulary = set()
    for sample in data:
        for token in sample['tokens']:
            vocabulary.add(token.lower())

    vocabulary = sorted(vocabulary)
    temp_file = 'query.txt'
    with open(temp_file, 'w', encoding='utf-8') as fp:
        for token in vocabulary:
            fp.write(token + '\n')

    from subprocess import call
    call('./lib/fastText/fasttext print-word-vectors ./lib/fastText/wiki.en.bin < {} > {}.out'.format(temp_file, temp_file), shell=True)

    word_embeddings = []
    with open(temp_file+'.out', 'r', encoding='utf-8') as fp:
        for line in fp:
            line = line.strip()
            vals = line.split(' ')
            word, emb = vals[0], np.array(vals[1:])
            word_embeddings.append(emb)
            assert vocabulary[len(word_embeddings)-1] == word
    assert len(word_embeddings) == len(vocabulary)

    os.remove(temp_file)
    os.remove(temp_file+'.out')

    vocabulary = ['PAD'] + ['UNK'] + vocabulary
    word_to_index_we = {word:i for i, word in enumerate(vocabulary)}
    word_embeddings = [np.zeros(word_embeddings[0].shape)] + [np.random.uniform(-0.05, 0.05, word_embeddings[0].shape)] + word_embeddings
    index_we_to_emb = {i:emb for i, emb in enumerate(word_embeddings)}

    for sample in data:
        sample['word_embeddings'] = [word_to_index_we[token.lower()] if token.lower() in word_to_index_we else word_to_index_we['UNK'] for token in sample['tokens']]

    return data, word_to_index_we, index_we_to_emb


def compute_sentence_embeddings_representation(data):
    import sent2vec
    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model('../DocAgg/lib/sent2vec/wiki_bigrams.bin')
    sentences = [' '.join(sample['tokens']) for sample in data]
    sentence_embeddings = sent2vec_model.embed_sentences(sentences)
    for i, sentence_embedding in enumerate(sentence_embeddings):
        data[i]['sentence_embeddings'] = sentence_embedding
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


def compute_all_representation(data, filepath='./data/spam_preprocessed.pkl'):
    if not os.path.exists(filepath):
        data, word_to_index = compute_naive_representation(data)
        data = compute_bag_of_words_representation(data, word_to_index)
        data = compute_tfidf_representation(data, word_to_index)
        data, word_to_index_we, index_we_to_emb = compute_word_embeddings_representation(data)
        data = compute_sentence_embeddings_representation(data)
        pickle.dump([data, word_to_index, word_to_index_we, index_we_to_emb], open(filepath, 'wb'))
    else:
        data, word_to_index, word_to_index_we, index_we_to_emb = pickle.load(open(filepath, 'rb'))

    return data, word_to_index, word_to_index_we, index_we_to_emb


######### TRANSFORM SAMPLE TO VECTORIAL FORMS #########
def pad(vectors, pad_index=0, max_sequence=None):
    if max_sequence is None or max_sequence < 0:
        max_sequence = max([len(vector) for vector in vectors])

    for i in range(len(vectors)):
        while(len(vectors[i]) < max_sequence):
            vectors[i].append(pad_index)

    return vectors


def transform_for_topics(samples):
    output = []
    for sample in samples:
        output.append([x[1] for x in sorted(sample['topics'].items(), key=lambda x:x[0], reverse=False)])
    return np.array(output)


def transform_for_naive(samples, word_to_index, max_sequence=None):
    output = []
    for sample in samples:
        output.append(sample['naive'])

    output = pad(output, word_to_index['PAD'], max_sequence)
    return np.array(output)


def transform_for_bag_of_words(samples, word_to_index):
    output = []
    for sample in samples:
        vector = np.zeros(len(word_to_index))
        for index in sample['bag_of_words']:
            vector[index] += 1
        output.append(vector)

    return np.array(output)


def transform_for_tfidf(samples):
    output = []
    for sample in samples:
        output.append(sample['tfidf'])

    return np.array(output)


def transform_for_word_embeddings(samples, word_to_index_we, index_we_to_emb, max_sequence=None):
    output = []
    for sample in samples:
        output.append(sample['word_embeddings'])

    output = pad(output, word_to_index_we['PAD'], max_sequence)
    for i in range(len(output)):
        output[i] = np.array([index_we_to_emb[index] if index in index_we_to_emb else index_we_to_emb[word_to_index_we['UNK']] for index in output[i]])

    return np.array(output)


def transform_for_sentence_embeddings(samples):
    output = []
    for sample in samples:
        output.append(sample['sentence_embeddings'])

    return np.array(output)


def create_labels(samples):
    output = []
    for sample in samples:
        output.append(1 if sample['type'] == 'spam' else 0)
    return np.array(output)


######### VISUALIZATION #########
def visualize_tsne(X, Y, filename):
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(X, Y)
    plt.figure(figsize=(10, 5))

    ham_indices = np.where(Y == 0)
    ham_plot = plt.scatter(tsne_results[ham_indices, 0], tsne_results[ham_indices, 1], c='b', marker='.')

    spam_indices = np.where(Y > 0)
    spam_plot = plt.scatter(tsne_results[spam_indices, 0], tsne_results[spam_indices, 1], c='r', marker='.')

    plt.legend((ham_plot, spam_plot), ('Ham', 'Spam'))
    plt.title('Ham vs Spam messages')
    plt.savefig('out/plot_representations/' + filename + '.png', bbox_inches='tight', dpi=200)