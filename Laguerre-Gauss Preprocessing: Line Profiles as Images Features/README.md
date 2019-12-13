# Laguerre-Gauss Preprocessing: Line Profiles as Images Features


Code implementation of the results presented on *Laguerre-Gauss Preprocessing: Line Profiles
as Images Features for Aerial Images Classification*.


## Usage
Using Python (v3.7.3) run the files MLP.py, kNN.py and CNN.py to test the results on the trained
models using a _subsample_ of the test set of aerial images.


### Subsample of test set
The folder @sample_test_data@ contains a randomly chosen subsample of the test set for aerial
images classification. The images are 64x64 RGB classified as 0 (nothing of interest: populations,
rivers, forest, etc.) or 1 (something of interest: heavy machinery, boats, deforestation, etc.).
