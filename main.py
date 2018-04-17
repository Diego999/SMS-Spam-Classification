from utils import get_data, compute_all_representation, transform_for_naive, transform_for_bag_of_words, transform_for_tfidf, transform_for_word_embeddings, transform_for_sentence_embeddings

if __name__ == '__main__':
    data = get_data()
    data, word_to_index, word_to_index_we, index_we_to_emb = compute_all_representation(data)

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