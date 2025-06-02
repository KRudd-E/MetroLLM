""" 
For storing settings and paramters used in the dataDev script
"""

config = {
    
    #****** Case Studies DB ******#
    'run_caseStudies_db' : True,
    'caseStudies_db'     : {
        'data_path'         : 'data/original/caseStudies',
        'output_path'       : 'data/new/caseStudies',
        
        'debug'             : False,
    },

    #****** Definitions DB ******#
    'run_definitions_db': True,
    'definitions_db'    : {
        'data_path'         : 'data/original/definitions',
        'output_path'       : 'data/new/definitions',

        'debug'             : False,
    },

    #****** Companies DB ******#
    'run_companies_db'  : True,
    'companies_db'      : {
        'data_path'         : 'data/original/companies',
        'output_path'       : 'data/new/companies',
        
        'debug'             : False,
    },

    'sleep' :  3,

}