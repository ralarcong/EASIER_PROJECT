from textreplace import textreplace
class sentencereplace:
    texttoreplace=list()
    @property
    def sentence(self):
        return self._sentence

    @sentence.setter
    def sentence(self, value):
        self._sentence = value

    @property
    def wordlist(self):
        return self._wordlist

    @wordlist.setter
    def wordlist(self, value):
        self._wordlist = value

    def __getitem__(self, item):
        return self._wordlist[item]

    def __setitem__(self, key, value):
        self._wordlist[key]=value

    def __init__(self):
        self.data=[]