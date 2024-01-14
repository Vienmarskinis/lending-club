# Module 3: Machine Learning

## Lending Club

### Author: Tomas Balsevičius

The Project has been split into six notebooks: step1 to step6.

To run the notebooks, first install dependencies using pip:

```Python
pip install -r requirements.txt
```
The project was created using python-3.11.1

Also, there has been models deployed on the Google Cloud. Refer to step6 notebook to explore them.

## Introduction

Before me I have two big datasets, containing information about accepted and rejected loans. I have been tasked with a few main goals:
- Create an accepted/rejected loan classification model, test and deploy.
- Create grade, subgrade, interest rate prediction models, test and deploy.

To achieve these main goals, I will need to complete a few steps, which I split into different notebooks:
1) Split the data into train and test.
2) Combine the two datasets, perform EDA, create an accepted/rejected loan classification model.
3) Perform EDA on the accepted loans dataset.
4) Perform feature selection for the three models.
5) Create the grade, subgrade, interest rate prediction models.
6) Test the models and deploy them.

## Technologies

In the project, I will be using these technologies and libraries:
- Python,
- Polars and Pandas (dataframes),
- Seaborn and Matplotlib (visualisation),
- Sklearn and XGBoost (modelling),
- Sklearn pipelines (data processing),
- Shap (feature importance)
- Boruta (feature selection),
- Docker (containerization),
- Google Cloud (deployment).

## Conclusions

In this project, I have explored the rejected and accepted loans datasets and tried my best to achieve the imposed goals. Here are the work results:

 - The data saved in the two datasets follow a different schema, which made it a bit difficult to organise. Moreover, the rejected loans dataset has far less features than the accepted loans dataset. This made it problematic to create a good model. I managed to create one, but since it lacks most of the data for credit risk, I felt reluctant to deploy it. If the stakeholders feel that it would benefit the business, I can easily polish it, test it and deploy it.
 - I performed EDA on the accepted loans dataset. I tried to identify the most and least important features for the targets (grade, subgrade, interest rate) predictions, also identify those features that have great collinearity or leak information from the future.
 - Three models were created tested and deployed, achieving reasonable mean absolute errors - refer to step 6 notebook.

## What Can Be Improved

 - EDA could be more in depth, as I skipped quite a few features and didn't explore how they interact with each other. However, having this many features feels like there's an infinite amount of interactions to explore.
- Feature engineering could be improved. Time delta should be calculated instead of raw year/month values. With better EDA, better features could be engineered.
- More models and different preprocessing steps could be explored.
- More time should be spent on tuning and modelling, fitting the whole or at least a bigger portion of the dataset.