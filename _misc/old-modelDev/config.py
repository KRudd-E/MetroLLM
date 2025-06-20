""" 
For storing settings and paramters used in the modelDev script
All paths are relative to the root of the repository.
"""

config = {
    
    # https://huggingface.co/google/flan-t5-xl
    
    #****** Text Classification ******#
    'train_text_classification': True,
    'text_classification': {
        'data_path'         : 'data/applicationsDB/...',
        'model_path'        : 'modelDev/src/models/',
        'pretrained_model'  : 'bert-base-uncased', 
        
        'learning_rate'     : 2e-5,
        
    },


        # self.learning_rate = 2e-5
        # self.batch_size = 2
        # self.num_epochs = 3
        # self.max_seq_length = 2048
        # self.device = 'cuda'
    
    
    #****** Text Generation ******#
    'run_text_generation': False,
    'text_generation': {
        
    },

    'sleep': 1

}