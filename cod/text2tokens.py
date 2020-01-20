import os
java_path = "C:/Program Files/Java/jdk1.8.0_231/bin/java.exe"
#java_path = "/usr/lib/jvm/java-8-oracle/bin/java"
os.environ['JAVAHOME'] = java_path
#os.environ['CLASSPATH'] = "/path/to/stanford-corenlp-full-2017-06-09"
import nltk
from clasificador import clasificador
from nltk.tokenize import sent_tokenize
from nltk.tag.stanford import StanfordPOSTagger

spanish_postagger = StanfordPOSTagger('stanford-postagger-full-2017-06-09/models/spanish.tagger',
                                      'stanford-postagger-full-2017-06-09/stanford-postagger-3.8.0.jar')
import time
clasificadorobj=clasificador()
class text2tokens:

    def text2sentence(self, text):

        sent_tokenize_list = sent_tokenize(text)

        return sent_tokenize_list

    def sentence2tokens(self, sentence):
        #for sent in sentence:
            #words = sentence.split()
        #print (sentence)
        #print (type(sentence))
        words= nltk.word_tokenize(sentence)
        start = time.time()
        tagged_words = spanish_postagger.tag(words)
        #print (str(tagged_words))
        end = time.time()
        #print(end - start)

        length = len(tagged_words)
        a = list()
        words2 = list()


        for i in range(0, length):
            log = (tagged_words[i][1][0] == 'v' or tagged_words[i][1][0] == 'r')
            if log == True:
                a.append(tagged_words[i][0])
        #start = time.time()
        for i in range(0, len(a)):
            words2.append((["P" + str(i), sentence, sentence.index(a[i]), sentence.index(a[i]) + len(a[i]), (clasificadorobj.Remove_punctuation(a[i])).strip(), 10, 10, 0, 1,1, 0.05]))
            #print(clasificadorobj.Remove_punctuation(a[i]).strip())

        return words2

    def __init__(self):
            self.data = []
