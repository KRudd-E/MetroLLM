# MetroLLM

Metrology-Specific LLM...

# Report

...

# Code
## Data Devlopment : dataDev

Aims to preprocess data from various defined formats into usable data.
Has its own section because lord knows preprocessing is the hard bit.

```
├── __main__.py
├── config.py
├── src
│   ├── _control.py
│   ├── caseStudy.py
│   ├── company.py
│   └── definition.py
└── data
    └── ...
```

## Model Development : modelDev

...

```
modelDev
├── __main__.py
├── data
│   └── ...
└── src
    ├── *test.ipynb
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