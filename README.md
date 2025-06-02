# MetroLLM
Metrology-Specific LLM

# Code
## dataDev

Aims to preprocess data from various defined formats into usable data.
Has its own section because lord knows preprocessing is the hard bit.

## modelDev

Aims to fine tune a model with data in dataDev

# Code Layout
```
├── dataDev
│   ├── __main__.py
│   └── src
│       ├── config.py
│       └── control.py
└── modelDev
    ├── __main__.py
    ├── data
    │   └── fake_data.csv
    └── src
        ├── config.py
        ├── control.py
        ├── evaluate.py
        ├── models
        │   ├── __init__.py
        │   └── model_loader.py
        ├── preprocess.py
        ├── test_.py
        ├── train_.py
        ├── utilities
        │   ├── __init__.py
        │   └── helpers.py
        └── utilities.py
```