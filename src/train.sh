./train.py --k 5 --log none --pca 0 --rfe 45 --seed 52 > ../data/results_5_none_0_45_52.txt
./train.py --k 5 --log none --pca 85 --rfe 70 --seed 22 > ../data/results_5_none_85_75_22.txt
./blendPreds.py > ../data/blendPreds.txt
