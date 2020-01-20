import webbrowser
import os

def openFile(webpage):
	#path = os.path.abspath('temp.html')
	url = webpage
	webbrowser.open(url)
	return "hola"