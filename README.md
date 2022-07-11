# Team Aleph Omega Analytics solution (John Lyons, Florian Weiß, Max Weber, Benedict Schwind)

## Trading Hub Europe Chalange

This repository contains:
- Training data used for competition
- Source code to preprocess data and make prediction
- Source code to impute some of the features which **can** lead to an improvement of the score (rlm_imputation, exit_imputation, entry_imputation)

### Main idea

### How to run

### How to generate rlm_imputation, exit_imputation, entry_imputation

Our main forecast model uses some extra features like rlm_imputation, exit_imputation, entry_imputation.
Imputation means in this context, that this features are forecasted by another model to boost the performance of the main model.
These extra features are already merged into train.csv which will be used if you execute the submission notebooks of the main model (neural_network).
For reproducibility and transparency, there are three notebooks which shows how we generate these extra features. 

You simply need to execute these notebooks to get several csv files which contains the imputed/forecasted values. 
Please check, if the code iterate over all slots and the csv are exported properly. Further, for a full examaple of what we did, 
it will be neccessary to merge these new imputed features into data/train.csv (and overwrite the old values which were already meged by us). 


## DB Regio Bus Challenge 

This repository contains:
- Training data used for competition
- Source code to preprocess data and make prediction
- Scientific Paper

### How to run
Unzip the data folder an run the two R scripts to generate two csv files which contains the forecasts for on-demand/regular task.

To reproduce the data preparation steps, unzip raw data in 30data/db_regio_bus/data.zip and execute the jupyter notebooks and provide