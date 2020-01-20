#!/usr/bin/env python
# encoding=utf8


import json
from urllib.request import urlopen



#json=json.load(urllib2.Request("http://thesaurus.altervista.org/thesaurus/v1",{'word':'caja','language':'es_ES','key':'reZLZ5pWSBCPu7n1prZI'},{'Content-Type':'application/json; charset=utf-8'}))
#request=urllib2.Request("http://thesaurus.altervista.org/thesaurus/v1?word=caja&language=es_ES&key=reZLZ5pWSBCPu7n1prZI&output=json",{'Content-Type':'application/json; charset=utf-8'})
#json=json.load(urllib2.urlopen(request))
#var= urllib2.urlopen("http://thesaurus.altervista.org/thesaurus/v1?word=caja&language=es_ES&key=reZLZ5pWSBCPu7n1prZI&output=json")

class worddictionary:

    def SSinonimos(self,word):
        json2=''
        #print (u'f\xe9retro').decode('utf-8')
        url = ''
        sameword=list()
        word=str(word)
        #print ("palabra a buscar")
        #print (word)
        try:
            url = "http://thesaurus.altervista.org/thesaurus/v1?word="+word+"&language=es_ES&key=reZLZ5pWSBCPu7n1prZI&output=json"
            response = urlopen(url).read().decode('utf-8')
            json2=json.loads(response)
            synonymslist=list()
            for var in json2['response']:
                synonyms=var["list"]['synonyms'].split('|')
                for syn in synonyms:
                    synonymslist.append(syn)
                    #print chardet.detect(syn)
            return synonymslist
        except:
            sameword.append(word.encode('utf8'))
            return sameword
