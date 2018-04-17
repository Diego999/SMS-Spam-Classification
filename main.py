from utils import get_data, compute_all_representation

if __name__ == '__main__':
    data = get_data()
    data, word_to_index, word_to_index_we, index_we_to_emb = compute_all_representation(data)