""" 
For storing settings and paramters used in the dataDev script
All paths are relative to the root of the repository.
"""

config = {
    
    #****** Applications DB ******#
    'run_applications_db' : True,
    'applications_db'     : {
        'data_path'         : '/data/applicationsDB/original',
        'output_path'       : '/data/applicationsDB/new',
        
        'pdf-txt'           : True, # To convert PDFs text to .txt files.
        'pdf-img'           : True, # To pull images from PDFs.
        
        'debug'             : False, # Useful for extended outputs. 
    },


    #****** Definitions DB ******#
    'run_definitions_db': False,
    'definitions_db'    : {
        'data_path'         : 'data/original/definitions',
        'output_path'       : 'data/new/definitions',

        'debug'             : False, 
    },


    #****** Companies DB ******#
    'run_companies_db'  : False,
    'companies_db'      : {
        'data_path'         : 'data/original/companies',
        'output_path'       : 'data/new/companies',

        'debug'             : False, 
    },

    'sleep' :  1,

}