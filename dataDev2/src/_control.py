from src.text2textGen import Text2TextGen
from src.textGen import TextGen
from src.textClass import TextClass
import time

class Controller:
    def __init__(self, config):
        self.config = config
        self.text2textGen = Text2TextGen(self.config['text2textGen'])
        self.textGen = TextGen(self.config['textGen'])
        self.textClass = TextClass(self.config['textClass'])

    def run(self):
        #************ Text2Text Generation ************#
        if self.config['run_text2textGen']:
            self.text2textGen.run()
            time.sleep(self.config['sleep'])

        #************ Text Generation ************#
        if self.config['run_textGen']:
            self.textGen.run()
            time.sleep(self.config['sleep'])

        #************ Text Classification ************#
        if self.config['run_textClass']:
            self.textClass.run()

        print("dataDev Done.\n")