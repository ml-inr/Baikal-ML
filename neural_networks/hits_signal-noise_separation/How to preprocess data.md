# How to preprocess data before feeding it to neural network

**Imporant** It is assumed that events are analyzed one by one. When using batches additional preprocessing is required.

## OM activation times
For each *individual* event average OMs activation time should be set to zero (via shift).

## OM coordinates
OM coordinates should be given with respect to the cluster they are located on (OM ID // 288).
More precisely, clusters centers are evaluated on MC 2020 data and are subtracted from OM coordinates.
The evaluated clusters centers (x, y, z) can be found in file data_for_preprocessing.txt. The unimeration follows rule `OM ID // 288`.

## OM registered charges
To account for saturated detectors, all registered charges exceeding 100 should be set to 100. 

## Required data
For each event, the input for the neural network is a one dimensional array `(num_hits, 6)`, where `num_hits` is the number of hits in an event.
Hits must be ordered according to their activation time in an increasing order.
For each hit, the six features are:
1. Registered charge,
2. Activation time,
3. x coordinate,
4. y coordinate,
5. z coordinate,
6. Fixed value, 1.  

## Data normalization
The features should be normalized via formula `normed = (initial-mean)/std`. 
The values of `mean` and `std` are given in data_for_preprocessing.txt. Their oredering is the same as in the list above.