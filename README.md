# GHCIDR :star2: 

This code is for the SubsetML ICML 2021 Workshop paper - "Geometrical Homogeneous Clustering for Image Data Reduction"

## Requirements

Please download the required modules from the requirements.txt
```
pip install -r requirements.txt
```

## To create the reduced data 
For creating reduced dataset change `<dataset>` to MNIST/FMNIST/CIFAR10.

### For RHC(baseline)
```
python main.py -variantName RHC -datasetName <dataset>
```

### For RHCKON 
```
python main.py -variantName RHCKON -datasetName <dataset> -KFarthest <K>
```

### For KONCW
```
python main.py -variantName KONCW -datasetName <dataset> -alpha <alpha>
```

### For CWKC 
```
python main.py -variantName CWKC -datasetName <dataset> -alpha <alpha>
```

### For GHCIDR 
```
python main.py -variantName GHCIDR -datasetName <dataset> -alpha <alpha>
```

The reduced dataset will be saved in "./datasetPickle" with the name `<datasetName>_<variantName>.pickle`

## To test the reduced data
For testing the reduced dataset `<dataset>` with variant `<variant>` change `<modelname>` to vgg1/fcn.

```
python vgg1.py -datasetName <dataset> -variantName <variant> -epochs 100 -lr 0.01 -batchSize 64 -fullDataset No
```


## Team Members :standing_person:

The contributors of this project - 

**[Devvrat Joshi](https://github.com/devvrat-joshi)**<br>
**[Janvi Thakkar](https://github.com/jvt3112)**<br>
**[Shril Mody](https://github.com/Shrilboss)**<br> 
**[Siddharth Soni](https://github.com/SoniSiddharth)**<br> 

