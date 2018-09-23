# FIW API
Families In the WIld: A Kinship Recogntion Toolbox. Visit FIW project page to download and learn more! 

[https://web.northeastern.edu/smilelab/fiw/](https://web.northeastern.edu/smilelab/fiw/)

**Version 1.0**

This document is incomplete, i.e., work in progress, as is the API itself.

[![N|Solid](reports/teaser_image.jpg)](reports/teaser_image.jpg)
------------
## Overview
This API serves as the main code-base for kinship effort with FIW database. In addition, below is detailed description of database (i.e., data and label) structure.

## Families In the Wild (FIW) Data and Labels
This documentation describes FIW DB and (working) development kit. This is work in prgress (i.e., still to come are FIW-CNN models, updated benchmarks, more in README (this), and more).

Check out FIW [project page](https://web.northeastern.edu/smilelab/fiw/index.html)

### Download
Download [here](https://web.northeastern.edu/smilelab/fiw/download.html)

### Details of the data
See pulications below. A more complete list of references can be found [here](https://web.northeastern.edu/smilelab/fiw/publications.html)

## Reference

```
 @article{robinson2018fiw,
   title={},
   author={},
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
   year={2018},
   publisher={IEEE}
 }
 
 @InProceedings{kinFG2017,
  author       = "Wang, Shuyang and Robinson, Joseph P and Fu, Yun",
  title        = "Kinship Verification on Families In The Wild with Marginalized Denoising Metric Learning",
  booktitle    = "Automatic Face and Gesture Recognition (FG), 2017 12th IEEE International Conference and Workshops on",
  year         = "2017",
}

@InProceedings{robinson2016families,
  author       = "Robinson, Joseph P. and Shao, Ming and Wu, Yue and Fu, Yun",
  title        = "Families In the Wild (FIW): Large-Scale Kinship Image Database and Benchmarks",
  booktitle    = "Proceedings of the 2016 ACM on Multimedia Conference",
  pages        = "242--246",
  publisher    = "ACM",
  year         = "2016"
}

```

######
## DB Contents and Structure
######
* FIW_PIDs.csv:&nbsp;&nbsp;&nbsp;&nbsp;Photo lookup table. Each row is an image instance, containing the following fields:
  * PID:&nbsp;&nbsp;&nbsp;&nbsp;Photo ID
  * Name:&nbsp;&nbsp;&nbsp;&nbsp;Surname.firstName (root reference for given family)
  * URL:&nbsp;&nbsp;&nbsp;&nbsp;Photo URL on web
  * Metadata:&nbsp;&nbsp;&nbsp;Text caption for photo
  
* FIW_FIDs.csv:&nbsp;&nbsp;&nbsp;&nbsp;FID (family)/ Surname lookup table.
  * FID:&nbsp;&nbsp;&nbsp;&nbsp;Unique ID key assigned to each family.
  * Surname:&nbsp;&nbsp;&nbsp;&nbsp;Family Name corresponding to FID key.
  
* FIW_RIDs.csv:&nbsp;&nbsp;&nbsp;Relationship lookup table with keys [1-9] assigned to relationship types.

* FIDs/
  * FID####/&nbsp;&nbsp;&nbsp;&nbsp;Contains labels and cropped facial images for members of family (1-1000)
    * MID#/&nbsp;&nbsp;&nbsp;&nbsp;Face images of family member with ID key <N>, i.e., MID #.

    * F####.csv:&nbsp;&nbsp;&nbsp;&nbsp;File containing member information of each family:
      * relationships&nbsp;&nbsp;matrix representing relationship
      * names&nbsp;&nbsp;&nbsp;&nbsp;First name of family member.
      * gender&nbsp;&nbsp;&nbsp;&nbsp;ender of family member
      
      
For example:
```
FID0001.csv
    
	0     1     2     3     Name    Gender
	1     0     4     5     name1   female
	2     1     0     1     name2   female
	3     5     4     0     name3   male
	
```
Here we have 3 family members, as seen by the MIDs across columns and down rows.


We can see that MID1 is related to MID2 by 4->1 (Parent->Sibling), which of course can be viewed as the inverse, i.e., MID2->MID1 is 1->4. It can also be seen that MID1 and MID3 are Spouses of one another, i.e., 5->5. And so on, and so forth.



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
    ├── notebooks          <- Jupyter notebooks.
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


## Todos
### General
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


### License

By downloading the image data you agree to the following terms:
1. You will use the data only for non-commercial research and educational purposes.
1. You will NOT distribute the above images.
1. Northeastern University makes no representations or warranties regarding the data, including but not limited to warranties of non-infringement or fitness for a particular purpose.
1.You accept full responsibility for your use of the data and shall defend and indemnify Northeastern University, including its employees, officers and agents, against any and all claims arising from your use of the data, including but not limited to your use of any copies of copyrighted images that you may create from the data.


   
## Authors
* **Joseph Robinson** - [Github](https://github.com/huskyjo) - [web](http://www.jrobsvision.com)

######
### Contact
######
Please contact Joseph Robinson (robinson.jo@husky.neu.edu) for questions, comments, or bug reports.

