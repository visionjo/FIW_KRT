# FIW API
Families In the WIld: A Kinship Recogntion Toolbox.
**Version 1.0**

This document is incomplete, i.e., work in progress, as is the API itself.

[![N|Solid](reports/teaser_image.jpg)](reports/teaser_image.jpg)
------------
## Overview

This API serves as the main code-base for kinship effort with FIW database. 



Tools for FIW as sample for term project.

------------
## Project Organization


    ├── LICENSE
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
    └── <OPEN>             <- TBD


--------

------------
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