# FIW_KRT
Families In the WIld: A Kinship Recogntion Toolbox.
**Version 0.1**

[//]: This document is incomplete, i.e., work in progress, as is the API codebase itself.

[![N|Solid](http://smile-fiw.weebly.com/uploads/4/5/1/8/45182585/logo-fiw_orig.png)](http://smile-fiw.weebly.com/uploads/4/5/1/8/45182585/logo-fiw_orig.png)
Overview
------------
This API serves as the main code-base for kinship effort with FIW database. 



Tools for FIW as sample for term project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------


## Installation


### Todos
- [ ] Finish this (README)
- [ ] Feature Extraction Module 
- [ ] Metric Learning Module
- [ ] Fine-tune module
- [ ] Eval module
- [ ] End-to-end scripts demoing usage of modules

### Experiments (TO DO)
- [ ] VGG Center-Face
- [ ] ResNet with Metric Learning
- [ ] ResNet Triplet-loss?
- [ ] Cross Dataset Eval
- [ ] Features from different layers of VGG


   
## Authors
* **Joseph Robinson** - [Github](https://github.com/huskyjo) - [web](http://www.jrobsvision.com)

<a name="myfootnote1">1</a>: footie.
