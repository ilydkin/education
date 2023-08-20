# Graduation Project
(SkillBox course 'Data Science')

## Business Understanding
According to the task description - provided at the beginning of the EDA.ipynb - the company maintains two databases: one for customer website session data (`ga_sessions`) and another for actions within each session (`ga_hits`). The objective is to predict whether a customer performs a target action (`ga_hits['event_action']`), i.e. clicks as desired by the company.

### *model_1*
In a literal interpretation of the business process, it can be assumed that the data from `ga_sessions` algorithmically precede the data from `ga_hits`, while the parameters in `ga_hits` are synchronous, i.e. they are calculated and recorded in the database at the moment of a mouse click. Since our task is to predict the target mouse click, the `ga_hits` data cannot be passed to the model. Therefore, the model is trained solely on the `ga_sessions` data and receives only those inputs within the functionality of the service. In accordance with this logic, **model_1** was trained with a ROC-AUC score of 0.65.

### *model_2*
On the other hand, there is a possibility that the calculation of values for `ga_hits` occurs not simultaneously but in stages. Specifically, information about the page on which the customer performs actions `(ga_hits['hit_page_path'])` may be recorded only once within a session instead of with each individual click. Supporting this insight is the fact that all `utm_*` parameters from `ga_sessions` are contained in `ga_hits['hit_page_path']`, indicating that they are likely extracted from there. In this case, the highly informative parameter 'hit_page_path' (i.e., the current page the user is on) can be included as one of the predictors of their target click. The **model_2** is trained in line with this logic. It takes an additional feature 'hit_page_path' as input, parses it to extract 4 extra predictors, and drops it in the end. Its ROC-AUC score is significantly higher, reaching 0.73.

## EDA.ipynb
The notebook includes an analysis of the business process, data cleaning, visualization of predictor-target and predictor-predictor correlations (addressing multicollinearity), solutions for class imbalance, and hyperparameter tuning for the models. 
- The issue of multicollinearity is resolved using a custom script that generates a correlation matrix for categorical features based on Cramér's V criterion.
-	4 approaches to addressing class imbalance are explored.
- In the end, 10 JSON files are created for each of the models for testing the service.

## model_1.py
The script creates a data preprocessing pipeline for model_1, following the above logic. 
The model is trained on a processed and re-balanced dataset, and the metric is evaluated on the original data.

## model_2.py
The script creates a data preprocessing pipeline for model_2, following the above logic. 
The model is trained on a processed and re-balanced dataset, and the metric is evaluated on the original data.

## service.py
This module is responsible for interacting with the service using FastAPI. 
It includes an automatic `--reload` launch for the service, which saves the effort of dealing with it through Terminal

## FastAPI_manual.txt
A manual for interaction with the service through Postman

## /models
The directory where trained models are stored

## /requests
The directory where test files are stored
**Note**: The tests were created from historical data that was used to train the models. To obtain unbiased metrics for predictive performance, new data is required.
