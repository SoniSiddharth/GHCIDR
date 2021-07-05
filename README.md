# GHCIDR :star2: :smirk: 

This code is for ICML 2021 Workshop paper GHCIDR
The link to our accepted: [GHCIDR](https://github.com/SoniSiddharth/GHCIDR)

## Requirements
``
## To create reduced data 
For creating reduced dataset change `<dataset>` to MNIST/FMNIST/CIFAR10

### For RHC
`python main.py -variantName RHC -datasetName <dataset>`

### For RHCKON 
`python main.py -variantName RHCKON -datasetName <dataset> -KFarthest <K>`

### For KONCW
`python main.py -variantName KONCW -datasetName <dataset> -alpha <alpha>`

### For CWKC 
`python main.py -variantName CWKC -datasetName <dataset> -alpha <alpha>`

### For GHCIDR 
`python main.py -variantName GHCIDR -datasetName <dataset> -alpha <alpha>`





