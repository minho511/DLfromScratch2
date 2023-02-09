from common.util import preprocess


if __name__ == "__main__":
    text = "You say goodbye and I say hello."
    corpus, word_to_id, id_to_word = preprocess(text)
    print(corpus)
    print(word_to_id)
    print(id_to_word)