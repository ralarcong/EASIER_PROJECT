# -*- coding: utf-8 -*-
from __future__ import division
import csv  # package to work with tsv files
import numpy as np
import math
from sklearn import svm
import nltk
import pickle
import re
import string

from Pyphen import Pyphen
from word2vec import word2vec
from FastText import FastText
import pandas as pd
#postag
import os
java_path = "C:/Program Files/Java/jdk1.8.0_151/bin/java.exe"
os.environ['JAVAHOME'] = java_path
from sklearn.metrics import precision_recall_fscore_support as pr
from sklearn.metrics import accuracy_score as ac
from sklearn.metrics import f1_score
from nltk.tag.stanford import StanfordPOSTagger
spanish_postagger = StanfordPOSTagger('stanford-postagger-full-2017-06-09/models/spanish.tagger',
                                      'stanford-postagger-full-2017-06-09/stanford-postagger-3.8.0.jar',
                                      encoding='utf-8')

class clasificador:
    SCALED = 10 ** 6
    # a function to count the number of syllables
    word2vector = word2vec()
    Fasttextvector=FastText()
    Pyphenobj=Pyphen()
    x=0
    y=0

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
    #
    # @property
    # def ambrank(self):
    #     return self._ambrank
    #
    # @ambrank.setter
    # def ambrank(self, value):
    #     self._ambrank = value

    def loadDic(self, path):
        dic = {}
        f = open(path, 'r')
        for line in f:
            line = line.strip()
            if 'wiki' in path:
                pos = line.rfind(' ')
            else:
                pos = line.rfind('\t')
            key = line[0:pos - 1]
            freq = line[pos + 1]
            # print(key,freq)
            dic[key] = int(freq)
        f.close()
        return dic

    def loadDic2(self, path):
        dic = {}
        f = open(path, 'r')
        for line in f:
            line = line.strip()
            pos = line.rfind(' ')
            key = line[0:pos]
            freq = line[pos + 1]
            # print(key,freq)
            dic[key] = int(freq)
        f.close()
        return dic

    def loadDic3(self, path):
        lista= list()
        f = open(path, 'r')
        for line in f:
            line = line.strip()
            lista.append(line)
        f.close()
        return lista

    def getProbability(self, ngram, dic, size):
        ngram = ngram.lower()

        ngram = ngram.strip()
        prob = 0
        try:
            prob = dic[ngram]
            prob = prob / size
        except:
            #print "palabra:"+ngram+"\n"
            pass

        prob = prob * self.SCALED
        return prob

    # TERM FREQUENCY INVERSE DOCUMENT FREQUENCY
    def getTFIDF(self, ngram, dic, size):
        ngram = ngram.lower()

        ngram = ngram.strip()
        prob = 0
        frequency = 0
        try:
            prob = dic[ngram]
            frequency = dic[ngram]
            prob = math.log(size / prob)
        except:
            pass

        prob = prob * frequency
        return prob

    # def getZipF(self, word):
    #     zipf = next(var for var in self.ambrank if (var[0][0] == word, [[0, 0, 0]], 0, 0))
    #     return zipf

    # def loadZipF(self, unigrams):
    #     amb = [(w, c, len(wordnet.synsets(w))) for (w, c) in unigrams.items() if len(wordnet.synsets(w)) > 0]
    #     amb_p_rank = ss.rankdata([p for (w, c, p) in amb])
    #     amb_c_rank = ss.rankdata([c for (w, c, p) in amb])
    #     amb_ranked = zip(amb, amb_p_rank, amb_c_rank)
    #     self.ambrank = amb_ranked

    def getWindow(self, word, sentence, start):

        # para obtener los tokens, dividimos por ' '
        tokens = nltk.word_tokenize(sentence)
        #tokens=sentence.split()
        #toktok = ToktokTokenizer()
        #tokens=toktok.tokenize(sentence.decode('utf8'))

        # transforamos a int
        start = int(start)

        # we return a 2-window around the word
        w_2 = ''
        w_1 = ''
        w1 = ''
        w2 = ''

        # index of the word at tokens
        index = -1

        # count devuelve el numero de veces que ocurre la palabra en la lista
        num = tokens.count(word)

        if num == 1 or start == 0:
            index = tokens.index(word)
        else:
            sentence1 = sentence[0:start - 1]
            tokens1 = nltk.word_tokenize(sentence1)
            #tokens1=sentence1.split(' ')
            # try:
            #     tokens1=toktok.tokenize(sentence1)
            # except:
            #     print sentence1
            #     raise
            index = len(tokens1)
            if (index > -1 and tokens[index] != word):
                index = -1
            # print('Problems with',sentence,word)

        if (index - 2 >= 0):
            w_2 = tokens[index - 2]
        if (index - 1 >= 0):
            w_1 = tokens[index - 1]
        if (index + 1 <= len(tokens) - 1):
            w1 = tokens[index + 1]
        if (index + 2 <= len(tokens) - 1):
            w2 = tokens[index + 2]

        return w_2, w_1, w1, w2

    def getLinesString(self, path):
        # f=open(path,'r')
        f = path
        numExamples = sum(1 for line in f)  # fileObject is your csv.reader
        # f.close()
        return numExamples

    def getLinesFile(self, path):
        f = open(path, 'r',encoding='utf-8')
        # f=path
        numExamples = sum(1 for line in f)  # fileObject is your csv.reader
        f.close()
        return numExamples

    def getMatrix(self, path):

        numExamples = self.getLinesFile(path)

        numFeatures = 17
        # creamos la matrix
        matrix = np.empty(shape=[numExamples, numFeatures])

        return matrix

    #def getMatrix_train(self, path, trigrams, totalTris, bigrams, unigrams, totalBis, totalUnis):
    def getMatrix_train(self, path, trigrams, totalTris, bigrams, unigrams, totalBis, totalUnis,E2Rgram):

        numExamples = self.getLinesFile(path)

        numFeatures =618
        # creamos la matrix
        matrix = np.empty(shape=[numExamples, numFeatures])

        # abrimos el fichero
        tsvin = open(path, "rt",encoding='utf-8')
        tsvin = csv.reader(tsvin, delimiter='\t')

        indexRow = 0

        for row in tsvin:
            id = row[0]  # id del parrafo donde ocurre la palabra
            sentence = row[1]  # oracion
            start = row[2]  # posicion inicial (caracter)
            word = row[4]  # palabra a clasificar

            class_word = row[9]  # clase: 1 o 0
            if len(word.split()) < 2:
                len_word = len(word)
                num_syl = self.Pyphenobj.getNSyl(word)
                len_sen = len(sentence)

                # obtenemos su ventana

                w_1, w_2, w1, w2 = self.getWindow(word, sentence, start)

                # obtenemos los trigramas, bigramas y sus probabilidades
                prob_2 = 0

                trigramL = w_2 + ' ' + w_1 + ' ' + word
                prob_2 = self.getProbability(trigramL, trigrams, totalTris)

                prob_1 = 0
                bigramL = w_1 + ' ' + word
                prob_1 = self.getProbability(bigramL, bigrams, totalBis)

                prob = self.getProbability(word, unigrams, totalUnis)

                prob1 = 0
                bigramR = word + ' ' + w1
                prob1 = self.getProbability(bigramR, bigrams, totalBis)

                prob2 = 0
                trigramR = word + ' ' + w1 + ' ' + w2
                prob2 = self.getProbability(trigramR, trigrams, totalTris)

                # TFIDF=self.getTFIDF(word,unigrams,totalUnis)
                wordvector = self.Fasttextvector.wordvector(word)
                wordvectortemp = pd.Series(wordvector)

                word2vector=self.word2vector.wordvector(word)
                word2vectortemp=pd.Series(word2vector)

                ##sim1, sim2 = self.NearestWords(sentence, word)

                E2R = self.E2RDic(E2Rgram, word)

                #ZipF = self.getZipF(word)

                # podemos crear el vector, que sera un array de dimension numFeatures
                vector_fet = np.arange(numFeatures)

                vector_fet[0] = len_word
                vector_fet[1] = num_syl
                vector_fet[2] = len_sen
                vector_fet[3] = prob_2*100
                vector_fet[4] = prob_1*100
                vector_fet[5] = prob*100
                vector_fet[6] = prob1*100
                vector_fet[7] = prob2*100
                # ector_fet[8] = TFIDF
                vector_fet[8] = self.IsLower(word)
                vector_fet[9] = self.IsUpper(word)
                vector_fet[10] = self.IsDigit(word)
                vector_fet[11] = self.IsTitle(word)
                vector_fet[12] = self.IsPunctuation(word)
                #vector_fet[13] = self.PosTag(word, sentence)
                vector_fet[13] = self.ContainPunctuation(word)
                vector_fet[14] = E2R
                vector_fet[15:315] = (word2vectortemp * 10).tolist()
                vector_fet[316:616] = (wordvectortemp * 10).tolist()
                vector_fet[617] = class_word

                # por ultimo, reemplazamos el vector para el ejemplo con indexRow prob_2
                #matrix[indexRow] = ['%.4f' % elem for elem in vector_fet]
                matrix[indexRow] = vector_fet

                # incrementamos en 1 para poder indicar el indice del siguiente ejemplo
                indexRow += 1

            else:
                len_word = len(word)
                num_syl = self.Pyphenobj.getNSyl(word)
                len_sen = len(sentence)

                vector_fet = np.arange(numFeatures)

                ##wordvector = self.Fasttextvector.wordvector(word)
                ##wordvectortemp = pd.Series(wordvector)


                ##E2R = self.E2RDic(E2Rgram, word)

                vector_fet[0] = len_word
                vector_fet[1] = num_syl
                vector_fet[2] = len_sen
                vector_fet[3] = 0
                vector_fet[4] = 0
                vector_fet[5] = 0
                vector_fet[6] = 0
                vector_fet[7] = 0
                # ector_fet[8] = TFIDF
                vector_fet[8] = self.IsLower(word)
                vector_fet[9] = self.IsUpper(word)
                vector_fet[10] = self.IsDigit(word)
                vector_fet[11] = self.IsTitle(word)
                vector_fet[12] = self.IsPunctuation(word)
                # vector_fet[13] = self.PosTag(word, sentence)
                vector_fet[13] = self.ContainPunctuation(word)
                vector_fet[14] = E2R
                vector_fet[15:315] = [0]*300
                vector_fet[316:616] = [0]*300
                # el ultimo la clase
                ##vector_fet[617] = 0
                ##vector_fet[618] = 0
                vector_fet[617] = class_word

                # por ultimo, reemplazamos el vector para el ejemplo con indexRow prob_2
                # matrix[indexRow] = ['%.4f' % elem for elem in vector_fet]
                matrix[indexRow] = vector_fet

                # incrementamos en 1 para poder indicar el indice del siguiente ejemplo
                indexRow += 1

        return matrix


    def getMatrix_Deploy(self, path, trigrams, totalTris, bigrams, unigrams, totalBis, totalUnis,E2Rgram):

        numExamples = self.getLinesString(path)

        numFeatures = 617
        # creamos la matrix
        matrix = np.empty(shape=[numExamples, numFeatures])

        # abrimos el fichero

        tsvin = path

        indexRow = 0


        for row in tsvin:
            id = row[0]  # id del parrafo donde ocurre la palabra
            sentence = row[1]  # oracion
            start = row[2]  # posicion inicial (caracter)
            word = row[4]  # palabra a clasificar
            # class_word=row[9] #clase: 1 o 0
            len_word = len(word)
            num_syl = self.Pyphenobj.getNSyl(word)
            len_sen = len(sentence)

            # obtenemos su ventana
            w_1, w_2, w1, w2 = self.getWindow(word, sentence, start)

            # obtenemos los trigramas, bigramas y sus probabilidades
            prob_2 = 0

            trigramL = w_2 + ' ' + w_1 + ' ' + word
            prob_2 = self.getProbability(trigramL, trigrams, totalTris)

            prob_1 = 0
            bigramL = w_1 + ' ' + word
            prob_1 = self.getProbability(bigramL, bigrams, totalBis)

            prob = self.getProbability(word, unigrams, totalUnis)

            prob1 = 0
            bigramR = word + ' ' + w1
            prob1 = self.getProbability(bigramR, bigrams, totalBis)

            prob2 = 0
            trigramR = word + ' ' + w1 + ' ' + w2
            prob2 = self.getProbability(trigramR, trigrams, totalTris)

            # TFIDF = self.getTFIDF(word, unigrams, totalUnis)


            #ZipF = self.getZipF(word)
            wordvector = self.Fasttextvector.wordvector(word)
            wordvectortemp = pd.Series(wordvector)

            word2vector = self.word2vector.wordvector(word)
            word2vectortemp = pd.Series(word2vector)

            ##sim1, sim2 = self.NearestWords(sentence, word)
            E2R = self.E2RDic(E2Rgram, word)

            # podemos crear el vector, que sera un array de dimension numFeatures
            vector_fet = np.arange(numFeatures)

            vector_fet[0] = len_word
            vector_fet[1] = num_syl
            vector_fet[2] = len_sen
            vector_fet[3] = prob_2*100
            vector_fet[4] = prob_1*100
            vector_fet[5] = prob*100
            vector_fet[6] = prob1*100
            vector_fet[7] = prob2*100
            # vector_fet[8] = TFIDF
            vector_fet[8] = self.IsLower(word)
            vector_fet[9] = self.IsUpper(word)
            vector_fet[10] = self.IsDigit(word)
            vector_fet[11] = self.IsTitle(word)
            vector_fet[12] = self.IsPunctuation(word)
            # #vector_fet[13] = self.PosTag(word, sentence)
            vector_fet[13] = self.ContainPunctuation(word)
            vector_fet[14] = E2R
            vector_fet[15:315] = (wordvectortemp*10).tolist()
            vector_fet[316:616] = (word2vectortemp * 10).tolist()

            matrix[indexRow] = vector_fet

            # incrementamos en 1 para poder indicar el indice del siguiente ejemplo
            indexRow += 1

        return matrix

    def SvmClassifier(self, X_train, y_train):

        classifiers = [
            svm.LinearSVC()
        ]

        # en nuestro caso solo se ejecutara una vez, porque solo tnemos un algoritmo
        for item in classifiers:
            print(item)
            clf = item
            # entrenamos
            clf.fit(X_train, y_train)
            filename = "SVMModel.sav"
            pickle.dump(clf, open(filename, 'wb'))

    # return predicted

    def SVMPredict(self, matrix_deploy):
        predicted = self.model.predict(matrix_deploy)
        return predicted

    def SVMEvaluation(self, y_dev, X_dev):
        # obtenemos precision, recall y f1 comparando el gold standard (y_dev) con las predicciones
        predicted = self.model.predict(X_dev)
        bPrecis, bRecall, bFscore, bSupport = pr(y_dev, predicted, average='macro')
        # mostramos resultados
        bAcuracy = ac(y_dev, predicted)
        G1 = 2 * (bAcuracy * bRecall) / (bAcuracy + bRecall)
        f1x2=f1_score(y_dev, predicted, average='macro')
        print(bAcuracy,bPrecis, bRecall, bFscore, G1, f1x2)

    def SVMLoad(self):
        filename = "MultitokenLinear0.7767.sav"
        self.model = pickle.load(open(filename, 'rb'))

    def asignarDic(self,path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (OSError, IOError) as e:
            return dict()

    def Remove_punctuation(self,text):
        return re.sub('[%s]' % re.escape(string.punctuation), ' ', text)

    def IsUpper(self, word):
        if word.isupper():
            return 1
        else:
            return 0

    def IsLower(self, word):
        if word.islower():
            return 1
        else:
            return 0

    def IsDigit(self, word):
        if word.isdigit():
            return 1
        else:
            return 0

    def IsTitle(self, word):
        if word.istitle():
            return 1
        else:
            return 0

    def IsPunctuation(self, word):
        if word in string.punctuation:
            return 1
        else:
            return 0

    def ContainPunctuation(self, word):
        regexp = re.compile(r'[!"#$%&\'\(\)\*\+\,\-./:;<=>?@\[\]^_{\|\}\~]+')
        if regexp.search(word):
            return 1
        else:
            return 0

    def PosTag(self, word, sentence):
        tokens=sentence.split()
        tagged_words = spanish_postagger.tag(tokens)
        posTag = None
        posTag = [tag for tag in tagged_words if tag[0] == word]
        if posTag and posTag[0][1][0] in ('V', 'R'):
            return 1
        else:
            return 0

    def NearestWords(self, sentence, word):
        tokens = nltk.word_tokenize(sentence)
        if word in tokens:
            index = tokens.index(word)
            if index - 1 >= 0:
                word1 = tokens.__getitem__(index - 1)
                sim1 = self.word2vector.similarity(word, word1)
            else:
                sim1 = 0
            if index + 1 < len(tokens):
                word2 = tokens.__getitem__(index + 1)
                sim2 = self.word2vector.similarity(word, word2)
            else:
                sim2 = 0
            return sim1, sim2
        else:
            return 0, 0


    def E2RDic(self,ngram,word):
        word2=word
        if word2 in ngram:
            return 0
        else:
            return 1

    def __init__(self):
        self.data = []