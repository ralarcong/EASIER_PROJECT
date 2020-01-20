#!/usr/bin/env python
# encoding=utf8
import urllib.request as urllib
#from bs4 import BeautifulSoup
import os

#import mechanicalsoup
import re

from newspaper import Article


class MechanicalSoup:

    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    def checkingurl(self, cadena):
        if re.match(self.regex, cadena) is not None:
            return True
        else:
            return False

    def paragraphfromweb2(self,webpage):
    #def paragraphfromweb(self):
        soup = BeautifulSoup(urllib.urlopen(webpage).read(),"lxml")
        #soup = BeautifulSoup(urllib.urlopen("https://www.breakingnews.ie/ireland/paddy-jacksons-life-blighted-for-20-months-by-rape-claims-lawyer-says-832868.html").read(),"lxml")
        #soup = BeautifulSoup(urllib.urlopen("http://www.abc.es/espana/abci-gobierno-vende-como-gesto-torra-medida-prevista-155-201806090205_noticia.html").read(),"lxml")
        divarticle=soup.find("span",{"id" : "espana"})
        #divarticle = soup.find("div", {"id": "article"})
        return [tag.text for tag in divarticle.select('p')]

    def paragraphfromweb(self, webpage):
        url = webpage
        article = Article(url)
        article.download()
        article.parse()
        texto = article.text
        texto = texto.replace(article.title," ")
        titulo= article.title
        return texto,titulo

    def imagesfromweb(self,webpage):
        soup = BeautifulSoup(urllib.urlopen(webpage).read(),"lxml")
        webSite = str(soup)
        modSite = str(soup)
        divar=soup.find("span",{"id" : "espana"})
        # divfigcaption= divar.select('figcaption')
        # divimg= divar.select('img')
        # for i in range(len(divfigcaption)):
        #     alttext=str(divfigcaption[i].text)
        #     img=str(divimg[i])
        #     imgreplace= img[:-2]+'alt="'+alttext+'"'+img[len(img)-2:]
        #     if imgreplace != "" and img in webSite:
        #         modSite = modSite.replace(img, imgreplace)
        #modSite=modSite.replace('xmlns:og="http://ogp.me/ns#"','xmlns:og="http://ogp.me/ns#" lang="es"')
        #modSite = modSite.replace('<input name="q" size="20" type="text"/>', '<label for="q" >Search</label><input name="q" size="20" type="text"/>')
        #modSite = modSite.replace('<h2><a href="#">Most Read in World</a></h2>', '<h1><a href="#">Most Read in World</a></h1>')
        #modSite=modSite.replace('<b>','<strong>')
        #modSite = modSite.replace('<\\b>', '<\strong>')
        return modSite


    def __init__(self):
        self.data = []

    def CWItoHTML(self, paragraphreplaced, paragraphs,titulo):
        print ("cwihtml")
        # tempSite: crear codigo HTML concatenando el texto
        styleToolTip = '<style>h1{text-align:center;font-family: Helvetica, Geneva, Arial,SunSans-Regular, sans-serif}h2{text-align:center;font-family: Helvetica, Geneva, Arial,SunSans-Regular, sans-serif}#contenedor {text-align:justify;width: 1300px;height: 500px;position: absolute;top: 50%;left: 40%;margin-top: -250px;margin-left: -450px;  text-align:justify;font-size:23px } .anagrama{float: left;width: 135px;}.tool {cursor : help;  position : relative; text-decoration: underline; }.tool::before, .tool::after { position : absolute; left : 50%; opacity : 0; z-index: -100;}.tool:hover::before, .tool:focus::before, .tool:hover::after, .tool:focus::after {  opacity : 1;  z-index: 100;}.tool::before {  border-style : solid;  border-width : 1em .75em 0 .75em;  border-color : #3e474f transparent transparent transparent;  bottom : 100%;  margin-left : -.5em;  content : " ";}.tool::after {  background : #3e474f;  border-radius : .25em;  bottom : 180%;  color : white;  width : 17.5em;  padding: 1em;  margin-left : -8.75em;  content : attr(data-tip);} </style>'
        #tempSite = '<html><head><h1>TEXTO SIMPLE</h1><h2><a href="tempdictionary.html" name ="VerDic" target="_blank" title="Lista de palabras complejas y sinónimos">Lista de Sinonimia</a></h2></head><body><div id="contenedor"><span class="cuerpo-texto">' + text + '</span></div></body></html>'
        tempSite = '{% load staticfiles %}<!DOCTYPE html><html><head><link href="{{ STATIC_URL }}favicon.ico" rel="shortcut icon" type="image/x-icon" ><link rel="shortcut icon"type="image/x-icon" href="{% static "favicon.ico" %}"><img class="anagrama" src="{% static "LOGO_1_ANAGRAMA.png"%}" alt="LOGO EASIER">><title>SIMPLIFICACIÓN POR TEXTO</title><h1>TEXTO SIMPLE</h1></head><body><h2>'+titulo+'</h2><div id="contenedor"><span class="cuerpo-texto">' + paragraphs + '</span></div></body></html>'
        # copiartodo lo de abajo de tempSite

        webSite = tempSite
        modSite = tempSite

        # divTag = browser.get_current_page().find_all("div", {"id": "article"})
        replaceSentence = ""
        originalSentence = ""
        index = re.search('<span class="cuerpo-texto"', webSite).end()
        for paragraph in paragraphreplaced:
            if replaceSentence != "" and originalSentence in webSite:
                modSite = modSite[0:index+1] + modSite[index + 1:].replace(originalSentence,replaceSentence)
                #modSite = modSite[0:index] + modSite[index + 1:].replace(originalSentence, "<p>"+replaceSentence+" ."+"</p>")
            currentsentence = ""
            replaceSentence = ""
            offset = 0
            for paragraphsentence in paragraph[1]:
                if paragraphsentence[1] != None:
                    if currentsentence != paragraphsentence[0]:
                        offset = 0
                        if currentsentence != "":
                            modSite = modSite[0:index+1] + modSite[index + 1:].replace(originalSentence, "<p>"+replaceSentence+"</p>")
                        replaceSentence = paragraphsentence[0]
                        currentsentence = paragraphsentence[0]
                        originalSentence = paragraphsentence[0]
                    wordssentence = paragraphsentence[1]
                    # print paragraphsentence[0]
                    # print str(wordssentence)
                    # tootip='<div class="tooltip" style="background-color: #FFFF00">'+wordssentence[5]+'<span class="tooltiptext">'+wordssentence[2]+'</span></div>'
                    tootip = '<span class="tool" style="background-color: #FFFF00" data-tip="' + wordssentence[
                        2] + '">' + wordssentence[5] + '</span>'
                    replaceSentence = replaceSentence[:wordssentence[3] + offset] + tootip + replaceSentence[wordssentence[4] + offset:]
                    offset = offset + (len(tootip) - (wordssentence[4] - wordssentence[3]))

                    modSite = modSite.replace('</head>', styleToolTip)
                    modSite = modSite.replace('href="//static', 'href="http://static')

        path = os.path.abspath('templates/temp.html')
        url = 'file://' + path
        with open(path, 'w') as f:
            f.write(modSite)
        #webbrowser.open(url)

    def CWItoHTML2(self,paragraphreplaced,webpage):

        #styleToolTip = '<style> /* Tooltip container */ .tooltip {     position: relative;     display: inline-block;     border-bottom: 1px dotted black; /* If you want dots under the hoverable text */ }  /* Tooltip text */ .tooltip .tooltiptext {     visibility: hidden;     width: 120px;     background-color: black;     color: #fff;     text-align: center;     padding: 5px 0;     border-radius: 6px;       /* Position the tooltip text - see examples below! */     position: absolute;     z-index: 1; } /* Show the tooltip text when you mouse over the tooltip container */ .tooltip:hover .tooltiptext {     visibility: visible; } </style> </head>'
        styleToolTip= '<style> .tool {cursor : help;  position : relative; text-decoration: underline; }.tool::before, .tool::after { position : absolute; left : 50%; opacity : 0; z-index: -100;}.tool:hover::before, .tool:focus::before, .tool:hover::after, .tool:focus::after {  opacity : 1;  z-index: 100;}.tool::before {  border-style : solid;  border-width : 1em .75em 0 .75em;  border-color : #3e474f transparent transparent transparent;  bottom : 100%;  margin-left : -.5em;  content : " ";}.tool::after {  background : #3e474f;  border-radius : .25em;  bottom : 180%;  color : white;  width : 17.5em;  padding: 1em;  margin-left : -8.75em;  content : attr(data-tip);} </style>'
        browser = mechanicalsoup.StatefulBrowser()
        #browser.open("https://www.breakingnews.ie/world/couple-found-guilty-of-murdering-french-nanny-over-bizarre-boyzone-obsession-844832.html")
        tempSite=self.imagesfromweb(webpage)
        tempSite=tempSite.replace("<strong>","")
        tempSite = tempSite.replace("</strong>","")
        #vartemp=""

        webSite = tempSite
        modSite = tempSite

        #divTag = browser.get_current_page().find_all("div", {"id": "article"})
        replaceSentence=""
        originalSentence=""
        index = re.search('<span class="main" id="espana"', webSite).end()
        #print str(webSite)
        #print str(paragraphreplaced)
        for paragraph in paragraphreplaced:
            #print replaceSentence
            #print originalSentence
            if replaceSentence != "" and originalSentence in webSite:
                #vartemp += replaceSentence + "<br>"
                #print "modificaoracion1: " + vartemp + "\n"
                modSite = modSite[0:index]+modSite[index+1:].replace(originalSentence, replaceSentence)
            currentsentence = ""
            replaceSentence = ""
            offset = 0
            for paragraphsentence in paragraph[1]:
                if paragraphsentence[1] != None:
                    if currentsentence != paragraphsentence[0]:
                        offset = 0
                        #print "oracion: "+ str(paragraphsentence[1])+"\n"
                        if currentsentence != "":
                            #vartemp+=replaceSentence+"<br>"
                            #print "modificaoracion2: " + vartemp + "\n"
                            modSite = modSite[0:index]+modSite[index+1:].replace(originalSentence, replaceSentence)
                        replaceSentence = paragraphsentence[0]
                        currentsentence = paragraphsentence[0]
                        originalSentence=paragraphsentence[0]
                    wordssentence = paragraphsentence[1]
                    # print paragraphsentence[0]
                    # print str(wordssentence)
                    #tootip='<div class="tooltip" style="background-color: #FFFF00">'+wordssentence[5]+'<span class="tooltiptext">'+wordssentence[2]+'</span></div>'
                    tootip='<span class="tool" style="background-color: #FFFF00" data-tip="'+wordssentence[2]+'">'+wordssentence[5]+'</span>'
                    replaceSentence = replaceSentence[:wordssentence[3] + offset] + tootip + replaceSentence[wordssentence[4] + offset:-1]
                    offset = offset +(len(tootip)-(wordssentence[4]-wordssentence[3]))

        modSite = modSite.replace('</head>',styleToolTip)
        modSite = modSite.replace('href="//static','href="http://static')
        modSite = modSite.replace('src="/','src="http://abc.es/')
        #modSite = modSite.replace("</body>",str(vartemp)+"</body>")

        path = os.path.abspath('templates/temp.html')
        #print path
        url = 'file://' + path
        #print url
        with open(path, 'w') as f:
            f.write(modSite)
        #print 'file://' + os.path.realpath('temp.html')
        #webbrowser.open_new_tab(url)
       # webbrowser.open(os.path.realpath('templates/temp.html'))
        #url = 'file:{}'.format(pathname2url(os.path.abspath('temp.html')))
        #webbrowser.open(url)

    def TextoHtml(self, paragraphreplaced, text):
        # tempSite: crear codigo HTML concatenando el texto
        styleToolTip = '<style>h1{text-align:center;font-family: Helvetica, Geneva, Arial,SunSans-Regular, sans-serif}h2{text-align:center;font-family: Helvetica, Geneva, Arial,SunSans-Regular, sans-serif}#contenedor {text-align:justify;width: 1000px;height: 500px;position: absolute;top: 50%;left: 40%;margin-top: -250px;margin-left: -250px;  text-align:justify;font-size:23px } .anagrama{float: left;width: 135px;}  .tool {cursor : help;  position : relative; text-decoration: underline; }.tool::before, .tool::after { position : absolute; left : 50%; opacity : 0; z-index: -100;}.tool:hover::before, .tool:focus::before, .tool:hover::after, .tool:focus::after {  opacity : 1;  z-index: 100;}.tool::before {  border-style : solid;  border-width : 1em .75em 0 .75em;  border-color : #3e474f transparent transparent transparent;  bottom : 100%;  margin-left : -.5em;  content : " ";}.tool::after {  background : #3e474f;  border-radius : .25em;  bottom : 180%;  color : white;  width : 17.5em;  padding: 1em;  margin-left : -8.75em;  content : attr(data-tip);} </style>'
        #tempSite = '<html><head><h1>TEXTO SIMPLE</h1><h2><a href="tempdictionary.html" name ="VerDic" target="_blank" title="Lista de palabras complejas y sinónimos">Lista de Sinonimia</a></h2></head><body><div id="contenedor"><span class="cuerpo-texto">' + text + '</span></div></body></html>'
        tempSite = '{% load staticfiles %}<!DOCTYPE html><html><head><link href="{{ STATIC_URL }}favicon.ico" rel="shortcut icon" type="image/x-icon" ><link rel="shortcut icon" type="image/x-icon" href="{% static "favicon.ico" %}"><img class="anagrama" src="{% static "LOGO_1_ANAGRAMA.png"%}" alt="LOGO EASIER" height="137" width="137" ><title>SIMPLIFICACIÓN POR TEXTO</title><h1>TEXTO SIMPLE</h1></head><body><div id="contenedor"><span class="cuerpo-texto">' + text + '</span></div></body></html>'
        # copiartodo lo de abajo de tempSite

        webSite = tempSite
        modSite = tempSite

        # divTag = browser.get_current_page().find_all("div", {"id": "article"})
        replaceSentence = ""
        originalSentence = ""
        index = re.search('<span class="cuerpo-texto"', webSite).end()
        for paragraph in paragraphreplaced:
            if replaceSentence != "" and originalSentence in webSite:
                modSite = modSite[0:index] + modSite[index + 1:].replace(originalSentence,"<p>" + replaceSentence+"</p>")
            currentsentence = ""
            replaceSentence = ""
            offset = 0
            for paragraphsentence in paragraph[1]:
                if paragraphsentence[1] != None:
                    if currentsentence != paragraphsentence[0]:
                        offset = 0
                        if currentsentence != "":
                            modSite = modSite[0:index] + modSite[index + 1:].replace(originalSentence, "<p>" + replaceSentence+"</p>")
                        replaceSentence = paragraphsentence[0]
                        currentsentence = paragraphsentence[0]
                        originalSentence = paragraphsentence[0]
                    wordssentence = paragraphsentence[1]
                    # print paragraphsentence[0]
                    # print str(wordssentence)
                    # tootip='<div class="tooltip" style="background-color: #FFFF00">'+wordssentence[5]+'<span class="tooltiptext">'+wordssentence[2]+'</span></div>'
                    tootip = '<span class="tool" style="background-color: #FFFF00" data-tip="' + wordssentence[
                        2] + '">' + wordssentence[5] + '</span>'
                    replaceSentence = replaceSentence[:wordssentence[3] + offset] + tootip + replaceSentence[wordssentence[4] + offset:]
                    offset = offset + (len(tootip) - (wordssentence[4] - wordssentence[3]))

                    modSite = modSite.replace('</head>', styleToolTip)

        path = os.path.abspath('templates/temp2.html')
        url = 'file://' + path
        with open(path, 'w') as f:
            f.write(modSite)
        #webbrowser.open(url)



    def constructorListaSyn(self,dic):

        f = open('tempdictionary.html', 'w')
        f.write("<html><head><meta charset='UTF-8'><title>LISTA DE SINÓNIMOS DE PALABRAS COMPLEJAS</title><h1>SINONIMOS DE PALABRAS COMPLEJAS</h1><style>table {width: 100%;border: 1px solid #000;}td {width: 25%;text-align: left;border: 1px solid #000;}</style></head><body><div id='contenedor'><span class='cuerpo-texto'><table>")

        for key, value in dic.items():
            f.write("<tr><td>" + str(key) + "</td><td>" + str(value) + "</td></tr>")

        f.write("</table></span></div></body></html>")
        f.close()
