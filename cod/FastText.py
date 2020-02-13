from fasttext import load_model

class FastText:
    #f = load_model("../../fasttxt/es.bin")
    f = load_model("../../fasttxt/cc.es.300.bin") #path to FastText model
    #words= f.get_words()

    def __init__(self):
        self.data = []

    def wordvector(self, word):
       try:
            wordvector = self.f.get_word_vector(word)
            return wordvector
       except:
            wordvector = [0] * 300
            return wordvector
