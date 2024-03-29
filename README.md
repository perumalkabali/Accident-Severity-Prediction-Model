# RoadRiskAlert: A Model for US Accident Severity Prediction

## Introduction
Road accidents pose significant risks to public safety and often lead to traffic congestion. This project aims to analyze factors contributing to road accidents using the "US Accidents" dataset. The dataset contains 2,845,342 observations across 50 states and 47 variables. Given the dataset's size, analysis focuses on six New England states: Rhode Island, Maine, Massachusetts, Vermont, New Hampshire, and Connecticut. The project's primary objective is to predict accident severity, classified on a scale from 1 to 4, where 4 indicates the highest impact on traffic.

## Data Source
Kaggle Dataset: [US Accidents](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents?resource=download)

## Preparing the Dataset
### Subset the Dataset
The dataset is filtered to include observations from New England states, resulting in 47,029 records for analysis.
### Removing Single-Value Columns
Columns with only one factor level, such as "Country" and "Turning_Loop," are removed as they do not contribute to prediction models.
### Removing Rows with Missing Values
Columns with more than 20% missing data, such as "Number," "Precipitation.in.," and "Wind_Chill.F.," are removed. The final cleaned dataset contains 43,972 observations and 42 variables.

## Exploratory Data Analysis
### Categorizing the Target Variable
Severity levels are categorized as "Low" (severity 1 and 2) and "High" (severity 3 and 4). Most accidents are classified as severity 2.
### Checking for Cross-Correlation
Cross-correlation analysis identifies relevant variables for modeling by removing multicollinear variables.
### Impact of Weather Conditions on Severity Levels
Weather factors like wind speed, humidity, and temperature show no significant influence on accident severity.
### Correlation of Severity with Different Variables
No variable exhibits a high correlation with accident severity.

## Prediction Models
### Classification Tree Model
A classification tree model is built using recursive partitioning, with distance, side, and latitude as significant predictors.
### Tree Pruning
Pruning is performed to avoid overfitting, resulting in an optimal tree model.
### Testing the Model
The model achieves 84.42% accuracy but exhibits high Type-1 and Type-2 errors.
### Random Forest Model
Random forest modeling improves accuracy, achieving 87.14% accuracy and identifying significant predictors like distance and temperature.
### Gradient Boost Method
Gradient boosting is attempted but does not significantly improve accuracy.
### Improving the Model Accuracy
By adding a new variable, "Duration," to the random forest model, accuracy improves to 88%.

## Analysis of Top Variables of Importance
### Duration
Accident duration does not significantly correlate with severity.
### Start_Lat (Location)
Accident hotspots are identified, with Connecticut, Massachusetts, and Rhode Island exhibiting higher severity accidents.
### Distance
Accident severity correlates positively with road congestion distance.
### Side
Most accidents occur on the right side of the road.
### Temperature
Accident severity shows varied patterns across different temperature ranges.

## Text Mining
Text mining of accident descriptions reveals common themes, including road closures, traffic impacts, and accident locations.

## Recommendations from Data Mining
Recommendations include deploying traffic control measures, enhancing driver education, and promoting safer driving practices.

## Further Study
Further investigation could involve addressing data imbalance issues using techniques such as Synthetic Minority Over-sampling Technique (SMOTE) to improve model performance, especially in cases where the severe accident class is underrepresented. The variable "Severity" indicates the severity of the accident, with a number between 1 and 4, where:
- 1 indicates the least impact on traffic (i.e., short delay as a result of the accident)
- 4 indicates a significant impact on traffic (i.e., long delay).
  
## Conclusion
The project identifies significant predictors of accident severity and develops prediction models. Random forest modeling proves most effective, achieving 88% accuracy. The analysis offers insights into accident patterns and informs recommendations for mitigating road accidents.

## References
- Camacho, A. (2017). Classification tree using rpart (100% Accuracy). Kaggle.
- Narkhede, S. (2018, May 9). Understanding Confusion Matrix. Toward Data Science.
- Probst, P., Wright, M. N., & Boulesteix, A. (2019). Hyperparameters and tuning strategies for random forest. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery.
- U.S. GAO. (2022, January 25). During COVID-19, Road Fatalities Increased and Transit Ridership Dipped.
- Laris, M. (2022, May 18). Deaths on U.S. roads soared to 16-year high in 2021. Washington Post. [https://www.washingtonpost.com/transportation/2022/05/17/road-deaths-fatalities/](https://www.washingtonpost.com/transportation/2022/05/17/road-deaths-fatalities/)
