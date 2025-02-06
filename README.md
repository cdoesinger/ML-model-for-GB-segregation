# Machine Learning model for grain-boundary segregation
This project is intended as Supporting Materials accompanying the publication "A universal ML model for segregation in W" [DOI not assigned]. It contains the data and python necessary to reproduce the results published there. The model can predict grain-boundary segregation energies of the transition metals in segregating in W.



## Data-files
**element_data.csv**: descriptors for the solute-elements

**element_data_source.csv**: references to sources of the descriptor values

**segregation_data.csv**: DFT segregation data with descriptors for the local atomic environments


## Python-scripts

**regression_functions.py**: collection of useful functions to calculate descriptors and train the GPR

**regression_model.py**: sample script performing a 5-fold cross-validation