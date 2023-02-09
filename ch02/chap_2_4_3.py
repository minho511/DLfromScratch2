from common.util import preprocess, create_co_matrix, ppmi
import numpy as np
import matplotlib.pyplot as plt

# SVD에 의한 차원 감소


if __name__ == "__main__":
    text = "You say goodbye and I say hello."
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size)
    W = ppmi(C)

    # SVD
    U, S, V = np.linalg.svd(W)

    print(C[0])
    print('희소벡터 : ', W[0])
    print('밀집벡터 : ', U[0])
    print('차원감소 : ', U[0, :2], "(ex. 2차원으로 감소)")

    for word, word_id in word_to_id.items():
        plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
    
    plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
    plt.show()


