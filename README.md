# AMS-325-Data-Analysis-of-Car-Accidents
accidents.py contains code from each contributors (AaronChen589-RitaChen2-Tiffanyy-C)

source of dataset: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents (2016-2021)


## Significance/Goal
We want to determine the relationship among factors and how they affect the number of car crashes, severity (scale from 1- 4 on its impact on traffic), and distance (length of the road extent affected)
We do so by creating multiple predictive models for car crashes based on significant factors and visualizations of these models through graphs and charts


## Conclusions
As a result of our project, we found that many accidents occurred in Florida and California with majority of accidents occuring during the day.
Majority of accidents also haooebed from noon to 6 PM and between October and December during “good weather” with high visibility (10 miles), between  40℉  and 80℉m With a severity of 2 (from a scale from 1 to 4, where 1 is the least impactful and 4 is the most).

We initially believed that the number of car accidents could occur more often during the night, under low visilbility, and bad weather. From the results we gathered, however, we found the opposite to be true. And while it's implausible to know exactly why majority of car accidents occur under these conditions, we theorize that these conditions (high visibility and good weather) may influence drivers into a false sense of security, resulting in more accidents when it should theoritcally be the opposite. 


## Remarks
While our data does bring up some interesting ideas, we also believe that the data could be susceptible to overfitting/bias when classifying the expected severity. Unequal sampling from all the states in the US (ex: a few hundred samples from in some states and tens of thousands of samples from other states) could've also influenced the results. Additionally, some of the features of our data were vague such as the scale of severity being unclear due to it being a cateogrical data type, meaning it was hard to quantifiy and measure the severity of a crash. Lastly, we also excluded features like population size, traffic/road density, rush hours, holidays, state proximity, etc., which can also be a confounding variable in our anaylsis of the dataset.











