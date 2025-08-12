from src.utils.misc import get_config, parser, modelDev_textclass_query

class FineTunePipeline:
    def __init__(self):
        self.config = get_config('modelDev/config.yaml')
        
    def run(self):
        run = parser()
        modelDev_textclass_query(run, self.config)
        
        #************ Train ************#
        if run == 'train':
            pass
    
        #************ Evaluate ************#
        if run == 'evaluate':
            pass