""" 
For storing settings and paramters used in the dataDev script
"""

config = {
    
    'run_caseStudes_db' : True,
    'caseStudes_db'     : {
        'data_path'         : 'data/original/caseStudies',
        'output_path'       : 'data/new/caseStudies',
    },
    
    'run_definitions_db': True,
    'definitions_db'    : {
        'data_path'         : 'data/original/definitions',
        'output_path'       : 'data/new/definitions',
    },
    
    'run_companies_db'  : True,
    'companies_db'      : {
        'data_path'         : 'data/original/companies',
        'output_path'       : 'data/new/companies',
    },
    
    'delay' :  3,

}