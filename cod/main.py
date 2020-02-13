#!/usr/bin/env python
# encoding=utf8

from clasificador import clasificador
from worddictionary import worddictionary
from textreplace import textreplace
from text2tokens import text2tokens

from MechanicalSoup import MechanicalSoup
from clasificador import  clasificador
from lemma import lemma
from worddictionarybabel import worddictionarybabel
from multiprocessing import Pool


def main(webpage):
	clasificadorobj = clasificador()
	dictionariopalabras=worddictionary()
	diccionariobabel=worddictionarybabel()
	pool=Pool()

	path = '../../ngrams/Spanish/1gms/vocab_cs.wngram'
	unigrams = clasificadorobj.loadDic(path)
	totalUnis = sum(unigrams.values())
	maxValue = max(unigrams.values())

	path = '../../ngrams/Spanish/2gms/2gms.wngram'
	bigrams = clasificadorobj.loadDic(path)
	totalBis = sum(bigrams.values())

	path = '../../ngrams/Spanish/3gms/3gms.wngram'
	trigrams = clasificadorobj.loadDic(path)
	totalTris = sum(trigrams.values())

	# DICCIONARIOE2R
	path = '../../E2R/unigram2_non_stop_words.csv'
	uniE2R = clasificadorobj.loadDic3(path)

	#path = '../spanish/Spanish_Train.tsv'
	#matrix_train = clasificadorobj.getMatrix_train(path, trigrams, totalTris, bigrams, unigrams, totalBis, totalUnis,uniE2R)

	#path = '../spanish/Spanish_Test.tsv'
	#matrix_dev = clasificadorobj.getMatrix_train(path, trigrams, totalTris, bigrams, unigrams, totalBis, totalUnis,uniE2R)


	#numCol = matrix_dev.shape[1]

	#X_train = matrix_train[:, 0:numCol - 1]
	#y_train = matrix_train[:, -1]  # last column

	#numCol = matrix_dev.shape[1]

	#X_dev = matrix_dev[:, 0:numCol - 1]
	#y_dev = matrix_dev[:, -1]  # last column



	#clasificadorobj.SvmClassifier(X_train, y_train)
	clasificadorobj.SVMLoad()
	#clasificadorobj.SVMEvaluation(y_dev,X_dev)

	paragraphreplaced = list()

	dicpares = {}
	lemmaobj = lemma()
	Mechanicalsoupobj = MechanicalSoup()

	webpage = "https://www.elmundo.es/papel/2019/07/02/5d15093bfc6c83370a8b467e.html"


	if Mechanicalsoupobj.checkingurl(webpage):

		paragraph,title = Mechanicalsoupobj.paragraphfromweb(webpage)

		#print (str(paragraph))
		if paragraph != '':
			x = text2tokens()

			words = list()
			# text1 = 'El copal se usa principalmente para sahumar en distintas ocasiones como lo son las fiestas religiosas y asimismo en misas. Las flores, hojas y frutos se usan para aliviar la tos y también se emplea como sedante.'

			# text2 = text1.decode('utf8')
			text = paragraph
			sentencelist = x.text2sentence(text)

			words = [x.sentence2tokens(sentence) for sentence in sentencelist]

			sentenceinparagraph = list()
			if words and words[0]:
				words = [item for item in words if item]

				matrix_deploy = [
					clasificadorobj.getMatrix_Deploy(sentencetags, trigrams, totalTris, bigrams, unigrams, totalBis,
													 totalUnis, uniE2R) for sentencetags in words]

				predictedtags = [clasificadorobj.SVMPredict(rowdeploy) for rowdeploy in matrix_deploy]
				sentencevar = None

				for j in range(0, len(words)):
					sentencetags = words[j]
					if sentencetags and sentencetags[0]:
						sentencevar = sentencetags[0][1]

					for i in range(0, len(sentencetags)):
						# en este punto deberiamos quedarnos con la oracion y ocn la palabra que vamos a usar de reemplazo
						wordreplace = None
						textreplaced = textreplace()
						syn2 = list()
						listindex = 0
						if predictedtags[j][i] == 1 and len(sentencetags[i][4])>4:
							print (sentencetags[i][4])
							dis2 = 0
							synonims = list()
							synonimsb = list()
							finaldic=list()
							# print (sentencetags[i][4] + "palabra compleja")
							synonimsb=pool.apply_async(diccionariobabel.babelsearch,[sentencetags[i][4]])
							#synonimsb = diccionariobabel.babelsearch(sentencetags[i][4])
							if len(dictionariopalabras.SSinonimos(sentencetags[i][4])):
								if str(sentencetags[i][4][len(sentencetags[i][4]) - 5:]) == 'mente':
									stem = sentencetags[i][4].replace("mente", "")
									#synonims = dictionariopalabras.SSinonimos(stem)
									synonims= pool.apply_async(dictionariopalabras.SSinonimos,[stem])
								else:
									stem = lemmaobj.lemmatize(sentencetags[i][4])
									synonims = pool.apply_async(dictionariopalabras.SSinonimos, [stem])
								#synonims = dictionariopalabras.SSinonimos(stem)
							if not synonims:
								#synonims = dictionariopalabras.SSinonimos(sentencetags[i][4])
								synonims = pool.apply_async(dictionariopalabras.SSinonimos, [sentencetags[i][4]])
								# print sentencetags[i][4]
								# print "tab"
								stem = sentencetags[i][4]
							synonims1=synonims.get(timeout=10)
							synonimsb2 = synonimsb.get(timeout=10)
							if synonims1 or synonimsb2:
								for h in range(0, len(synonims1)):
									syn2.append(synonims1[h])
								for x in range(0, len(synonimsb2)):
									syn2.append(synonimsb2[x])
								syn2=set(syn2)
								# print "nuevos sinonimos"+str(syn2)
								finaldic=synonims1+synonimsb2
								dicpares[sentencetags[i][4]] = [syn2]
								dic_synonims = dict.fromkeys(finaldic)  # elimina duplicados de la lista
							for candidate in dic_synonims.keys():
								candidatesentencetags = list(sentencetags[i])
								candidatesentencetags[4] = str(candidate)
								candidatelen = len(candidate)
								wordlen = len(sentencetags[i][4])
								candidatesentencetags[3] = candidatesentencetags[2] + candidatelen
								candidatesentencetags[1] = str(candidatesentencetags[1])[
														   :candidatesentencetags[2]] + str(candidate) + \
														   candidatesentencetags[1][
														   candidatesentencetags[2] + wordlen:]
								# candidatesentencetags[1] = (candidatesentencetags[1][:candidatesentencetags[2]]).encode('utf-8') + str(candidate) + (candidatesentencetags[1][candidatesentencetags[2] + wordlen:]).encode('utf-8')

								listcandidatesentencetags = list()
								listcandidatesentencetags.append(candidatesentencetags)
								#candidatematrix = clasificadorobj.getMatrix_Deploy(listcandidatesentencetags, trigrams, totalTris,bigrams, unigrams, totalBis, totalUnis,uniE2R)
								#candidatepredictedtag = clasificadorobj.SVMPredict(candidatematrix)

								# busqueda de sinonimo optimo en contexto
								dis1 = clasificadorobj.word2vector.similarity(candidate, sentencetags[i][4])
								window = clasificadorobj.getWindow(sentencetags[i][4], sentencetags[i][1],
																   sentencetags[i][2])
								diswindow1 = clasificadorobj.word2vector.similarity(window[1], candidate)
								diswindow2 = clasificadorobj.word2vector.similarity(window[2], candidate)
								dis3 = dis1 + diswindow1 + diswindow2

								#if dis2 < dis3 and sentencetags[i][4] != candidate.lower() and candidatepredictedtag[0] != 1:
								if dis2 < dis3 and sentencetags[i][4] != candidate.lower():
									synonim = candidate
									dis2 = dis3
									wordreplace = candidatesentencetags[2:5] + sentencetags[i][2:5]
						if wordreplace:
							sentenceinparagraph.append([sentencevar, wordreplace])
			paragraph_sentence = [paragraph, sentenceinparagraph]
			paragraphreplaced.append(paragraph_sentence)

	if Mechanicalsoupobj.checkingurl(webpage):
		Mechanicalsoupobj.CWItoHTML(paragraphreplaced, paragraph,title)

	if __name__=="__main__":
		main(webpage)