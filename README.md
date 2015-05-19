# Otto Group Product Classification - Kaggle Challenge

These are the scripts that I used to create my submission for Kaggle's Otto Group product classification challenge. I started this copetition quite late and did not get the chance to properly tune my models or ensemble solutions with varying parameters, however this solution still meritted a top 10% finish. If I had started a bit earlier with enough time to use more extensive tuning grids and run a varients of the command line arguments then I suspect this codebase could have performed quite well.

## Dependencies
* Python (2.7.9)
* pickle (protocol  2.0)
* pandas (0.15.2)
* sci-kit learn (0.15.2)
* numpy (1.9.2)
* scipy (0.15.1)
* joblib (0.8.4)
* theano
* lasagne
* nolearn

## To Run
* To execute the (lacklust) solution I had time to do simply execute the src/train.sh script.
* To develop a better solution using this codebase then edit the calls to train.py within src/train.sh and expand upon the tuning grids within train.py.
