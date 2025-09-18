# TL;DR

## DataDev1
raw data (1.) -> usable data (2.)
- applications to .txt 
- definition word doc to .csv
- company website to .txt


## DataDev2 
usable data (2.) -> formatted data (3.)
- for Text2Text Generation
    - applications .txt -> t2t pairs via OpenAI API
    - definitions .csv -> t2t pairs via OpenAI API
    - company .txt -> t2t pairs via OpenAI API (not implemented)

- for Text Generation
    - applications .txt -> tg pairs via OpenAI API
    - definitions .csv -> tg pairs via OpenAI API
    - company .txt -> tg pairs via OpenAI API (not implemented)

- for Text Classifcation
    - applications .txt & .xlsx -> .csv for application database-specific model
    - company .txt & .xlsx -> .csv for company database-specific model


## ModelDev
formatted data (3.) -> ML model (results.)
- Text2Text Generation model takes all sources
- Text Generation takes all sources
- Text Classification takes only the source specific to the model



### To Run:
* Activate conda environment. 
* Edit ```dataPre/config.py``` file according to intended configuration .
* In the Terminal, run ```python dataPre```.

## Data Development

Aims to combine data value, so it can be used in training models. 

### To Run
* Activate conda environment 
* Edit ```dataDev/config.py``` file according to intended configuration 
* 


## Model Development

...
