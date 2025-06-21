# TL;DR
1. Configure file: ```dataPre/config.py```
2. In terminal: ```conda activate MLE```
3. In terminal: ```python dataPre```

**Suggested to only run one of 'applications', 'companies', and 'definitions' at once.**
**As of 21 Jun 2025, definitions and companies not implemented**

# Further notes
This folder retrieves various data formats and decomposes them into its constituent text and images. 
- Applications: list of subdirectories containing .pdf files for Case Studies/Applications.
- Companies: excel doc containing company information and categorization.
- Definitions: word doc containing many definitions. 


## Data layout
- Data
    - applicationsDB
        - new
            - subdirectories, each containing constituent .txt and .png files
        - old
            - subdirectories, each containing .pdf files ONLY
    - companiesDB
        - new
            - ...
        - old
            - ...
    - definitionsDB
        - new
            - definitions.csv
        - old
            - definitions.docx