<p> This repository utilizes and compares two machine learning approaches for <strong>spatial pollution prediction/ interpolation based on shared point-of-interest attributes between different NO2 measuring sites</strong> in Berlin. A traditional Random Forest Regressor is compared to a graph neural network which combines neighborhood aggregation and contrastive, unsupervised embedding for graph representation learning and is alternated from the implementation by Vu et al. (2024).</p>

<p> This work is part of my Master Thesis which can be found in the repository under <em>Thesis_spatial_pollution_prediction.pdf</em>, which serves as detailed source of explanation. </p>

<h2> Structure and Notebooks: </h2>

- <em> random_forest.ipynb</em>: RF Regressor for prediction
- <em> graph_neural_network.ipynb + utilities</em>: utilization of gnn
- <em> analysis_error_feature.ipynb</em>: comprehensive model comparison, including: performance metrics, feature-wise and temporal residual analysis, feature importance through Shapley values
- <em>Thesis_spatial_pollution_prediction.pdf</em>: detailed theoretical describtion of architecture and data transformation


<h2> Data source and Acknowledgement: </h2>

The dataset is constructed with the intersection of multiple geological and meteorological datasets by the [Berlin Geo Portal](https://www.berlin.de/sen/sbw/stadtdaten/geoportal/) and the [German Weather Serves(Deutscher Wetter Dienst)](https://opendata.dwd.de/climate_environment/CDC/). The dataset and feature engineering including missing data imputation, multicollinearity and EDA are separately addressed in the repositories: [MLbased_meteorological_data_imputation](https://github.com/RiSchmi/MLbased_meteorological_data_imputation). The usage of the geological and meteorological data is regulated by the "Creative Commons BY 4.0" (CC BY 4.0) and detailed in <em>License</em>.

Vu, V., Nguyen, D., Nguyen, T., Nguyen, Q., P.L., N., & Huynh, T. (2024). Self-supervised air quality estimation with graph neural network assistance and attention enhancement. Neural Computing and Applications. https://doi.org/https://doi.org/10.1007/s00521-024-09637-7

The original owner of the data and code used in this thesis retains ownership of the data and code.

