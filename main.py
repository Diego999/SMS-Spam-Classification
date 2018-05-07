from utils import get_data, compute_all_representation, transform_for_topics, transform_for_naive, transform_for_bag_of_words, transform_for_tfidf, transform_for_word_embeddings, transform_for_sentence_embeddings, create_labels, visualize_tsne
from utils import get_topics
from utils_ML import *

if __name__ == '__main__':
    data = get_data() # The data are already shuffled
    data, word_to_index, word_to_index_we, index_we_to_emb = compute_all_representation(data)
    topics, data = get_topics(data)

    '''
    word_to_index_we -> mapping from a lowered token into an index to be mapped for word embeddings
    index_we_to_emb -> mapping from an vocabulary index to word embeddings (300 dim)
    word_to_index -> as word_to_index but for naive, bag of words and tfidf representations
    
    data -> list of samples being the whole dataset.
    Each sample is a dictionnary with the following attributes:
        - id: Line# from spam.csv
        - text: raw text from spam.csv
        - type: ham/spam from spam.csv
        - tokens: tokens obtained via tokenization. They are not lowered
        - lemmas: lemmas obtained via lemmazitazion. They are lowered
        - topics: Topic distribution w.r.t to "topics" list
        - naive: list of indices from word_to_index. The index corresponding to "UNK" is used for unknown tokens (e.g. stopwords)
                 Later will become a null vector where indices would representing dimensions where value is one
        - bag_of_words: similar to naive but unknown words are ignored.
                 Later will become a null vector where indicies would representing dimensions where value is one or more (in case a word is present X times, the value will be X)
        - tfidf: similar to bag_of_word instead of having discrete value, compute importance of each word using TF-IDF (widely used in Information Extraction)
        - word_embeddings:  List of indices from word_to_index_we. Later, these indices will be mapped to the embeddings of index_we_to_emb
        - sentence_embeddings: Vector of 600 dimensions representing the sentence embeddings.
    '''

    #X_naive = transform_for_naive(data, word_to_index)
    X_bow = transform_for_bag_of_words(data, word_to_index)
    X_tfidf = transform_for_tfidf(data)
    #X_we = transform_for_word_embeddings(data, word_to_index_we, index_we_to_emb)
    X_se = transform_for_sentence_embeddings(data)
    X_topics = transform_for_topics(data)
    Y = create_labels(data)

    '''
    #Visualize using only one kind of features
    visualize_tsne(X_naive, Y, 'naive')
    visualize_tsne(X_bow, Y, 'bow')
    visualize_tsne(X_tfidf, Y, 'tfidf')
    #Not possible as we have 3D inputs visualize_tsne(X_we, Y, 'word_emb', index_we_to_emb)
    visualize_tsne(X_se, Y, 'sent_emb')
    visualize_tsne(X_topics, Y, 'topics')
    #'''

    X_bow_se = np.concatenate([X_bow, X_se], axis=1)
    X_bow_topics = np.concatenate([X_bow, X_topics], axis=1)
    X_tfidf_se = np.concatenate([X_tfidf, X_se], axis=1)
    X_tfidf_topics = np.concatenate([X_tfidf, X_topics], axis=1)
    X_se_topics = np.concatenate([X_se, X_topics], axis=1)

    '''
    # Visualizing using the concatenation of two kind of features
    visualize_tsne(X_bow_se, Y, 'bow-sent_emb')
    visualize_tsne(X_bow_topics, Y, 'bow-topics')

    visualize_tsne(X_tfidf_se, Y, 'tfidf-sent_emb')
    visualize_tsne(X_tfidf_topics, Y, 'tfidf-topics')

    visualize_tsne(, Y, 'sent_emb-topics')
    #'''

    TRAINING_SIZE = 0.7
    VALIDATION_SIZE = 0.1
    TESTING_SIZE = 0.2

    X, key = (X_se, 'Sentence Embeddings')
    classes = [0, 1]

    training_size = int(len(X)*TRAINING_SIZE)
    validation_size = int(len(X)*VALIDATION_SIZE)
    testing_size = len(X) - training_size - validation_size
    assert training_size + validation_size + testing_size == len(X)

    X_train, Y_train = X[:training_size], Y[:training_size]
    X_valid, Y_valid = X[training_size:training_size + validation_size], Y[training_size:training_size + validation_size]
    X_test, Y_test = X[training_size + validation_size:], Y[training_size + validation_size:]

    # Because we are tuning with CV, with can use X_train = X_train + X_valid
    X_train = np.array(X_train.tolist() + X_valid.tolist())
    Y_train = np.array(Y_train.tolist() + Y_valid.tolist())

    classifiers = [('linear', linear_model.LogisticRegression(solver='lbfgs')),
                   ('RandomForest', RandomForestClassifier(n_estimators=20)),
                   ('SVM Linear', SVC(kernel='linear')),
                   ('SVM RBF', SVC(kernel='rbf')),
                   ('MLP', MLPClassifier(early_stopping=True))]
    for clf_name, clf in classifiers:
        print(clf_name)
        # Should find the best set of parameters, might use the tune function in utils_ML
        clf.fit(np.array(X_train), np.array(Y_train))
        Y_hat = clf.predict(np.array(X_test))
        print_and_get_accuracy(Y_test, Y_hat)
        print_and_get_precision_recall_fscore_support(Y_test, Y_hat)
        print_and_get_macro_micro_weighted_fscore(Y_test, Y_hat)
        print_and_get_classification_report(Y_test, Y_hat, classes)
        plot_confusion(Y_test, Y_hat, classes, key + ' - ' + clf_name)
        plot_roc(Y_test, Y_hat, classes, key + ' - ' + clf_name)
        plot_prec_rec_curve(Y_test, Y_hat, classes, key + ' - ' + clf_name)
    plt.show()

