from src.text2textGen import Text2TextGen
from src.textClass import TextClass
from src.textGen import TextGen
from src.utils.utils import dataDev2_query, parser, get_config

class Controller:
    def __init__(self):
        self.config = get_config()
        
    def run(self):
        args = parser()
        dataDev2_query(args)
        
        #** Text2Text Generation **#
        if args.model_type == 'text2textgen':
            self.text2textGen = Text2TextGen(config=self.config['text2textGen'], src=args.data_source)

        #** Text Classification **#
        if args.model_type == 'textclass':
            self.textClass = TextClass(config=self.config['textClass'], src=args.data_source)
            
        #** Text Generation **#
        if args.model_type == 'textgen':
            self.textGen = TextGen(config=self.config['textGen'], src=args.data_source)
        