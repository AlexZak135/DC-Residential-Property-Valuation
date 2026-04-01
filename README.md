# Predicting DC Residential Property Values

## Overview

Washington, D.C. has a wide range of houses and condominiums. Some homes are over a century old, while others are newly built. In many residential neighborhoods, smaller and more affordable homes and condominiums are common for new families, while other neighborhoods have larger homes owned by some of the city’s wealthiest residents. Differences in age, style, size, and value reflect the city’s history and shape housing across the city.

The government of Washington, D.C. is responsible for estimating residential property values because these valuations determine the ad valorem property taxes paid by property owners. To do this, the Office of Tax and Revenue relies on the Computer-Assisted Mass Appraisal (CAMA) database maintained by the Assessment Division within the Real Property Tax Administration. This database is the primary data source for building valuation models. It contains a sales history of active properties listed in the district’s real property tax assessment roll, along with property characteristics recorded at the time of sale. The database is periodically updated to reflect the most current information available.

This analysis relied on several Washington, D.C. datasets maintained by the district government, including the CAMA database, along with American Community Survey 2018–2022 census tract–level socioeconomic estimates. The data covers five years of residential property sales, including houses and condominiums, from January 1, 2021, through December 31, 2025. Extensive data processing and exploratory data analysis were performed prior to modeling. Ten supervised machine learning models were developed: five to predict house values and five to predict condominium values. Three tree-based ensemble algorithms were tested for each property type. Models were trained on 2021–2024 data and evaluated using 2025 data. Among all models, LightGBM performed best for predicting both house and condominium values, with XGBoost performing second best in both cases.

## Files
This is the [Python script](https://github.com/AlexZak135/DC-Residential-Property-Valuation/blob/master/Code/DC-Residential-Property-Valuation-Analysis-Code.py) containing the code used for this analysis, additionally, these are the [datasets](https://github.com/AlexZak135/DC-Residential-Property-Valuation/tree/master/Data) used in the script, and finally, this is the [requirements file](https://github.com/AlexZak135/DC-Residential-Property-Valuation/blob/master/requirements.txt) listing the Python packages needed to run the code.

## Outputs
These [outputs](https://github.com/AlexZak135/DC-Residential-Property-Valuation/tree/master/Outputs) display the performance and error metrics from the machine learning models.
