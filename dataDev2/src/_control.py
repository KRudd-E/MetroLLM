from src.text2textGen import Text2TextGen
from src.textGen import TextGen
from src.textClass import TextClass
from src.utils import dataDev2_query, parser, get_config

class Controller:
    def __init__(self):
        self.config = get_config()
        
    def run(self):
        args = parser().parse_args()
        dataDev2_query(args.model_type, args.data_source)
        
        #************ Text2Text Generation ************#
        if args.model_type.lower == 'text2text' or args.model_type.lower() == 't2t':
            self.text2textGen = Text2TextGen(config=self.config['text2textGen'], src=args.data_source)

        #************ Text Generation ************#
        if args.model_type.lower() == 'textgen' or args.model_type.lower() == 'tg':
            self.textGen = TextGen(config=self.config['textGen'], src=args.data_source)

        #************ Text Classification ************#
        if args.model_type.lower() == 'textclass' or args.model_type.lower() == 'tc':
            self.textClass = TextClass(config=self.config['textClass'], src=args.data_source)

        print("dataDev Done.\n")