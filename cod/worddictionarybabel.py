from pybabelnet.babelnet import BabelNet

class worddictionarybabel:
    bn = BabelNet(open("path to key of balbelnet license", "r").read())  # or BabelNet("your API key")

    def babelsearch(self,palabra):
        listasyn = list()
        sameword = list()
        try:
            Ids = self.bn.getSynset_Ids(palabra, "ES")
            for i in range(0, len(Ids)):
                synsets = self.bn.getSynsets("ES", Ids[i].id)
                for j in range(0, len(synsets)):
                    for x in range(0, len(synsets[j].senses)):
                        syn = synsets[j].senses[x]['properties']['fullLemma']
                        if (len(syn.split(" ")) and len(syn.split("_")) == 1):
                            listasyn.append(syn)
            if (len(listasyn)==0):
                sameword.append(palabra)
                return sameword
            else:
                return listasyn
        except:
            sameword.append(palabra)
            return listasyn
