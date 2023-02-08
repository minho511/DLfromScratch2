from common.util import preprocess, create_co_matrix, cos_similarity
import numpy as np

if __name__ == "__main__":
    text = "You say goodbye and I say hello."
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size)

    c0 = C[word_to_id['you']]
    c1 = C[word_to_id['i']]
    print(c0)
    print(c1)
    print(cos_similarity(c0, c1))

