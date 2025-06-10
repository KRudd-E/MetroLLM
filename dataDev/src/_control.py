


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
            print("\n>> Processing Text2Text Generation")
            self.text2textGen.run()
            print(">> Completed Text2Text Generation\n")
            time.sleep(self.config['sleep'])

        #************ Text Generation ************#
        if self.config['run_textGen']:
            print(">> Processing Text Generation")
            self.textGen.run()
            print(">> Completed Text Generation\n")
            time.sleep(self.config['sleep'])

        #************ Text Classification ************#
        if self.config['run_textClass']:
            print(">> Processing Text Classification")
            self.textClass.run()
            print(">> Completed Text Classification\n")
            time.sleep(self.config['sleep'])

        print(">> dataDev Done.\n")