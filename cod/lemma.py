# encoding: utf8
class lemma:
    lemmaDict = {}
    with open('../lemmatization-es.txt', 'rb') as f:
        data = f.read().decode('utf8').replace(u'\r', u'').split(u'\n')
        data = [a.split(u'\t') for a in data]

        for a in data:
            if len(a) > 1:
                lemmaDict[a[1]] = a[0]


        def lemmatize(self,word):
            return self.lemmaDict.get(word, word)


# def test():
#     for a in [u'parcialmente', u'acababan', u'impusieron', u'endureció', u'diferenciándola']:
#         print(lemmatize(a))
#
#
# test()