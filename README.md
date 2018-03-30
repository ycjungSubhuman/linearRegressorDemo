# CCPP Linear Regression Demo

## Requirements

1. python 3.x

## Run

* Install requirements

```bash
$ pip install -r requirements.txt 
```

* Run main.py

```bash
$ python main.py Simple
$ python main.py Power
$ python main.py Gaussian 0.01
$ python main.py Sigmoid 0.01
```

* The program prints out mean and stddev of RMSE for 5 dataset with 2-fold test (10 tests per parameter configuration)
* Prints out F and P value for single pass ANOVA (H0: Different parameter set result in identical means)

