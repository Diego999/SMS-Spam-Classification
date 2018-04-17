from utils import get_data, compute_bag_of_words_representation, compute_naive_representation, compute_tfidf_representation

if __name__ == '__main__':
    data = get_data()
    data, word_to_index = compute_naive_representation(data)
    data = compute_bag_of_words_representation(data, word_to_index)
    data = compute_tfidf_representation(data, word_to_index)