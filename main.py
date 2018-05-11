from utils import get_data, compute_all_representation, transform_for_topics, transform_for_naive, transform_for_bag_of_words, transform_for_tfidf, transform_for_word_embeddings, transform_for_sentence_embeddings, create_labels, visualize_tsne, word_cloud
from utils import get_topics
from utils_ML import *
from sklearn import linear_model, ensemble, svm, neural_network, naive_bayes, tree
from CNN_LSTM_models import CNN_LSTM_Wrapper
import os
import time


if __name__ == '__main__':
    data = get_data() # The data are already shuffled
    data, word_to_index, word_to_index_we, index_we_to_emb = compute_all_representation(data)
    emb_matrix = [[float(x) for x in (v.split() if type(v) == 'str' else v)] for _, v in sorted(index_we_to_emb.items(), key=lambda x:x[0])]
    topics, data = get_topics(data)
    '''
    word_cloud({i:x for i, x in enumerate(topics)})
    #'''

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

    X_naive = transform_for_naive(data, word_to_index)
    X_bow = transform_for_bag_of_words(data, word_to_index)
    X_tfidf = transform_for_tfidf(data)
    X_we = transform_for_word_embeddings(data, word_to_index_we, index_we_to_emb)
    X_se = transform_for_sentence_embeddings(data)
    X_topics = transform_for_topics(data)
    Y = create_labels(data)

    '''
    #Visualize using only one kind of features
    visualize_tsne(X_naive, Y, 'naive')
    visualize_tsne(X_bow, Y, 'bow')
    visualize_tsne(X_tfidf, Y, 'tfidf')
    visualize_tsne(X_we, Y, 'word_emb', index_we_to_emb)
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
    seed = 28111993
    classes = [0, 1] #0:ham, 1:spam
    show_plot = False

    representation_sets = [(X_naive, 'Naive'),
                           (X_bow, 'Bag Of Words'),
                           (X_tfidf, 'tf-idf'),
                           (X_we, 'Word Embeddings'), #Maybe too heavy. Let's see if we could use it for another model than CNN
                           (X_se, 'Sentence Embeddings'),
                           (X_topics, 'Topics'),

                           (X_bow_se, 'Bag Of Words - Sentence Embeddings'),
                           (X_bow_topics, 'Bag Of Words - Topics'),
                           (X_tfidf_se, 'tf-idf - Sentence Embeddings'),
                           (X_tfidf_topics, 'tf-idf - Topics'),
                           (X_se_topics, 'Sentence Embeddings - Topics'),
                            ]
    for X, key in representation_sets:
        data_rep_folder = 'out/training/{}'.format(key)
        if not os.path.exists(data_rep_folder):
            os.makedirs(data_rep_folder)

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

        classifiers = [('Naive Bayes - Bernouilli', naive_bayes.BernoulliNB()),
                       ('Naive Bayes - Multinomial', naive_bayes.MultinomialNB()),
                       ('Naive Bayes - Gaussian', naive_bayes.GaussianNB()),
                       ('Logistic Regression', linear_model.LogisticRegression(solver='lbfgs', random_state=seed)),
                       ('Decision Tree', tree.DecisionTreeClassifier(random_state=seed)),
                       ('RandomForest', ensemble.RandomForestClassifier(n_estimators=20, random_state=seed)),
                       ('SVM Linear', svm.SVC(kernel='linear', random_state=seed)),
                       ('SVM RBF', svm.SVC(kernel='rbf', random_state=seed)),
                       ('AdaBoost', ensemble.AdaBoostClassifier(algorithm='SAMME.R', random_state=seed)),
                       ('Feedforward Neural Network', neural_network.MLPClassifier(early_stopping=True, random_state=seed)),
                       ('Convolutional Neural Network', CNN_LSTM_Wrapper('CNN', emb_matrix)),
                       ('Recurrent Neural Network (LSTM)', CNN_LSTM_Wrapper('LSTM', emb_matrix)),
                    ]

        all_Y_hats = []
        all_scores = []
        for clf_name, clf in classifiers:
            # Use Word Embeddings only for CNN/RNN
            if 'Word Embeddings' in key:
                if 'Convolutional' not in clf_name and 'Recurrent' not in clf_name:
                    continue
            else:
                if 'Convolutional' in clf_name or 'Recurrent' in clf_name:
                    continue

            data_rep_method_folder = '{}/{}'.format(data_rep_folder, clf_name.replace(' ', '_'))
            if not os.path.exists(data_rep_method_folder):
                os.mkdir(data_rep_method_folder)

            if 'Naive' in clf_name and 'Multinomial' in clf_name and 'Embeddings' in key:
                #input might be negative
                all_Y_hats.append([])
                all_scores.append([(k, 0) for k, v in all_scores[-1]])
                continue

            print(clf_name)
            # Should find the best set of parameters, might use the tune function in utils_ML
            start_training_time = time.time()
            clf.fit(np.array(X_train), np.array(Y_train))
            end_training_time = time.time()

            start_testing_time = time.time()
            Y_hat = clf.predict(np.array(X_test))
            end_testing_time = time.time()

            all_Y_hats.append(Y_hat)

            accuracy = print_and_get_accuracy(Y_test, Y_hat)
            precision, recall, fscore, support = print_and_get_precision_recall_fscore_support(Y_test, Y_hat)
            macro, micro, weighted = print_and_get_macro_micro_weighted_fscore(Y_test, Y_hat)
            classification_report = print_and_get_classification_report(Y_test, Y_hat, classes)
            scores = [('accuracy', accuracy),
                      ('precision', ' '.join([str(x) for x in precision])),
                      ('recall', ' '.join([str(x) for x in recall])),
                      ('fscore', ' '.join([str(x) for x in fscore])),
                      ('support', ' '.join([str(x) for x in support])),
                      ('macro_f1', macro), # Compute P & R on for each class and then take the average --> Needed to handle imbalanced dataset
                      ('micro_f1', micro), # Add up everything together and then compute P & R
                      ('weighted_f1', weighted),
                      ('classification_report', classification_report),
                      ('training_time', end_training_time - start_training_time),
                      ('testing_time', end_testing_time - start_testing_time)
                     ]
            all_scores.append(scores)

            with open(data_rep_method_folder + '/scores.txt', 'w', encoding='utf-8') as fp:
                for metric, score in scores:
                    fp.write('{}\t\t{}\n'.format(metric, score))

            plot_confusion(Y_test, Y_hat, classes, key + ' - ' + clf_name, show=show_plot, save=True, path_to_save=data_rep_method_folder)
            plot_roc(Y_test, Y_hat, classes, key + ' - ' + clf_name, show=show_plot, save=True, path_to_save=data_rep_method_folder)
            plot_prec_rec_curve(Y_test, Y_hat, classes, key + ' - ' + clf_name, show=show_plot, save=True, path_to_save=data_rep_method_folder)

        if show_plot:
            plt.show()

        # Write summary of all the model with this representation
        with open('{}/summary_{}.txt'.format(data_rep_folder, key.replace(' ', '_')), 'w', encoding='utf-8') as fp:
            for (clf_name, _), Y_hat, score in zip(classifiers, all_Y_hats, all_scores):
                fp.write('{}\t\t{}\t\t{}\t\t{}\n'.format(clf_name, score[5], score[-2], score[-1])) #score[5] = macro_f1, -2 training time, -1 testing time
