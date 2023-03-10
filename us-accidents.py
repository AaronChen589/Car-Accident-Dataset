import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn import neighbors
from sklearn import svm
from sklearn import ensemble
from sklearn import cluster
from sklearn import datasets
from sklearn import model_selection


accidents = pd.read_csv("C:/Users/chent/Downloads/US_Accidents_Dec21_updated.csv")
acc = accidents.dropna()
df = acc[["Severity","Start_Time",'Distance(mi)',
                 'State','Temperature(F)','Humidity(%)','Visibility(mi)',
                 'Wind_Speed(mph)','Precipitation(in)','Weather_Condition',
                 'Bump','Crossing', 'Railway',
                 'Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal',
                 'Sunrise_Sunset']]

# Convert start time hour and month to int
df.loc[:,"Hour"] = df.Start_Time.str[11:13]
df.loc[:,"Hour"] = df.loc[:,"Hour"].apply( lambda x: int(x))

df.loc[:,"Month"] = df.Start_Time.str[5:7]
df.loc[:,"Month"] = df.loc[:,"Month"].apply( lambda x: int(x))

df["Year"] = df.Start_Time.str[0:4]
df.loc[:,"Year"] = df.loc[:,"Year"].apply( lambda x: int(x))

# Simple Visualizations Below
weather_counts = df["Weather_Condition"].value_counts().head(8) # decreasing 91 total
weather_counts.plot( kind='pie', title='Weather Condition At Time Of Accident')


visibility_counts = df["Visibility(mi)"].value_counts().sort_index().head(31)
visibility_counts.plot( kind='bar', title='Number of Accidents By Visibility')
plt.xlabel("Visibility (mi)")
plt.ylabel("Number of Accidents")

state_counts = df["State"].value_counts().head(35) # decreasing
state_counts.plot(kind='bar', title='Number of Accidents By State')
plt.xlabel("State")
plt.ylabel("Number of Accidents")


avg_stsev = df.groupby("State").Severity.mean().sort_values(ascending=False).head(35) # alphabetically
avg_stsev.plot(kind='bar', title='Severity By State')
plt.xlabel("State")
plt.ylabel("Severity")

sun_counts = acc["Sunrise_Sunset"].value_counts() 
sun_counts.plot( kind='pie', title='Ratio of Day/Night Accidents')

avg_daycount = acc.groupby("State").Sunrise_Sunset.value_counts().sort_values(ascending=False).head(10)
avg_daycount.plot(kind='bar', title='Number of Day/Night Accidents By State')
plt.xlabel("State, Day/Night")
plt.ylabel("Number of Accidents")

hr_counts = df["Hour"].value_counts().sort_index() # 
hr_counts.plot(kind='bar', title='Number of Accidents By Hour')
plt.xlabel("Hour")
plt.ylabel("Number of Accidents")

month_counts = df["Month"].value_counts().sort_index() # 
month_counts.plot(kind='bar', title='Number of Accidents By Month')
plt.xlabel("Month")
plt.ylabel("Number of Accidents")

yr_counts = df["Year"].value_counts().sort_index() # 
yr_counts.plot(kind='bar', title='Number of Accidents By Year')
plt.xlabel("Year")
plt.ylabel("Number of Accidents")


# This Following will plot a bargraph of Temperature VS Number of Accidents

fig = plt.figure(figsize=(5,5))
ax = fig.gca()
ax.set_xlabel("Temperature(F)")
ax.set_ylabel("Number of Accidents")
ax.set_title("Temperature(F) on Number of Accidents")
ax.set_xlim(-10, 120)
ax.set_xticks([-20,-10,0,10,20,30,40,50,60,70,80,90,100,110,120,130])
Temperature_counts = df.loc[:,"Temperature(F)"]
Temperature_counts.hist(ax = ax)


#This section of the code will plot a stacked plot of the presence of objects in accident
# and miscellanious or no objects in accidents against the count
# Note: these events are all mutually exclusive (appearance of one object means the other won't appear)
location_DataFrame = df[['Bump','Crossing', 'Railway',
 'Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal']]
bump_counts = location_DataFrame.Bump[location_DataFrame.Bump==True].count()
crossing_counts = location_DataFrame.Crossing[location_DataFrame.Crossing== True].count()
railway_counts = location_DataFrame.Railway[location_DataFrame.Railway == True].count()
roundabout_counts = location_DataFrame.Roundabout[location_DataFrame.Roundabout == True].count()
station_counts = location_DataFrame.Station[location_DataFrame.Station == True].count()
stop_counts = location_DataFrame.Stop[location_DataFrame.Stop == True].count()
traffic_calming_counts = location_DataFrame.Traffic_Calming[location_DataFrame.Traffic_Calming == True].count()
traffic_signal_counts = location_DataFrame.Traffic_Signal[location_DataFrame.Traffic_Signal == True].count()
AccidentsWithoutObject = 943318 - (bump_counts + crossing_counts + railway_counts + roundabout_counts +station_counts+stop_counts+traffic_calming_counts+traffic_signal_counts)

x = ['Presence of Object in Accident', 'Misc/No Object']
y1 = np.array([bump_counts, AccidentsWithoutObject]) # 943318 is total 
y2 = np.array([crossing_counts, 0])
y3 = np.array([railway_counts, 0])
y4 = np.array([roundabout_counts, 0])
y5 = np.array([station_counts, 0])
y6 = np.array([stop_counts, 0])
y7 = np.array([traffic_calming_counts, 0])
y8 = np.array([traffic_signal_counts, 0])
# plot bars in stack manner
plt.bar(x, y1, color='red', label="Misc/No Object")
plt.bar(x, y2, bottom=y1, color='blue', label="crossing")
plt.bar(x, y3, bottom=y1+y2, color='black', label="railway")
plt.bar(x, y4, bottom=y1+y2+y3, color='yellow', label="roundabout")
plt.bar(x, y5, bottom=y1+y2+y3+y4, color='green', label="station")
plt.bar(x, y6, bottom=y1+y2+y3+y4+y5, color='brown', label="stop")
plt.bar(x, y7, bottom=y1+y2+y3+y4+y5+y6, color='purple', label="traffic calming")
plt.bar(x, y8, bottom=y1+y2+y3+y4+y5+y6+y7, color='orange', label="traffic signal")

plt.xlabel("Presence of Certain Objects in Crash")
plt.ylabel("Number of Accidents")
plt.title("Presence of Objects on Number of Accidents")
plt.legend()
plt.show()


df1 = location_DataFrame.apply(pd.value_counts)
ax = df1.plot.bar()
ax.set_xlabel("Presence of A Certain Obstacle/Object During Car Accident")
ax.set_ylabel("Number of Car Accidents")
ax.set_title("Presence of Object on the Number of Car Accidents")

df1 = location_DataFrame.apply(pd.value_counts)
ax = df1.plot(kind='barh', stacked=True)
ax.set_xlabel("Presence of A Certain Obstacle/Object During Car Accident")
ax.set_ylabel("Number of Car Accidents")
ax.set_title("Presence of Object on the  Number of Car Accidents")


#predicting severity given the presense of objects in accidents
#Note: PREDICTING SEVERITY(CATEGORICAL VALUE) IS Not GOOD WHEN USING LINEAR REGRESSION
X = df[['Bump','Crossing', 'Railway', 'Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal']]
Y = df[["Severity"]]

df['Bump'] = (df['Bump'] == True).astype(int)
df['Crossing'] = (df['Crossing'] == True).astype(int)
df['Railway'] = (df['Railway'] == True).astype(int)
df['Roundabout'] = (df['Roundabout'] == True).astype(int)
df['Station'] = (df['Station'] == True).astype(int)
df['Stop'] = (df['Stop'] == True).astype(int)
df['Traffic_Calming'] = (df['Traffic_Calming'] == True).astype(int)
df['Traffic_Signal'] = (df['Traffic_Signal'] == True).astype(int)

X = pd.get_dummies(data=X, drop_first=True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .20, random_state = 40)
regr = linear_model.LinearRegression() 
regr.fit(X_train, Y_train)
predicted = regr.predict(X_test)
print(regr.score(X_train,Y_train))
print(regr.score(X_test,Y_test))
reg = linear_model.LinearRegression()
reg.fit(x,y)
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
predictions = model.predict(x)
printModel = model.summary()
print(printModel) 
x = df[['Hour','Month']]    
y = df[["Severity"]]
X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, train_size = 0.7)
#Using decision tree to test severity
classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
print('Accuracy score = ' + str(metrics.accuracy_score(y_test, y_pred)))
metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_test, y_pred)).plot()

#Using logistic regression to test severity
classifier = linear_model.LogisticRegression(solver="saga",multi_class="auto", random_state = 40)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
print('Accuracy score = ' + str(metrics.accuracy_score(y_test, y_pred)))
metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_test, y_pred)).plot()

#Using #Kneighbors to test severity
classifier = neighbors.KNeighborsClassifier()
classifier.fit(X_train, np.ravel(y_train))
y_pred = classifier.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
print('Accuracy score = ' + str(metrics.accuracy_score(y_test, y_pred)))
metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_test, y_pred)).plot()

#Plot Distance VS Accidents
fig = plt.figure(figsize=(5,5))
ax = fig.gca()
ax.set_xlabel("Distance (mi) Affected By Accident")
ax.set_ylabel("Number of Accidents")
ax.set_title("Distance On Number of Accidents")
Distance_counts = df.loc[:,"Distance(mi)"]
ax = Distance_counts.hist(ax=ax, log=True, bins=20)

#Use logistic to test how hour and month of year might affect severity
x = df[['Hour','Month']]
y = df[["Severity"]]
np.random.seed(1003)
X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, train_size = 0.7)
classifier = linear_model.LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
print('Accuracy score = ' + str(metrics.accuracy_score(y_test, y_pred)))
metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_test, y_pred)).plot()


# Create pairwise scatterplots for Severity, Distance, and Weather features
sns.pairplot(df, vars = ['Severity', 'Distance(mi)', 'Temperature(F)', 'Humidity(%)', 'Visibility(mi)'])
plt.savefig('Pairplot 1.png')
sns.pairplot(df, vars = ['Severity', 'Distance(mi)', 'Wind_Speed(mph)', 'Precipitation(in)'])
plt.savefig('Pairplot 2.png')

# Create bar graphs for the count of each severity level in the day and in the night
Day_Only = df.loc[df['Sunrise_Sunset'] == 'Day']
Night_Only = df.loc[df['Sunrise_Sunset'] == 'Night']

sns.countplot(data = Day_Only, y = 'Severity')
sns.set(rc={'figure.figsize':(30,15)})
plt.title("Severity during Day", size = 40)
plt.xlabel('Count', size = 30)
plt.ylabel('Severity', size = 30)
plt.tick_params(axis = 'both', which = 'major', labelsize = 25)
xticks = np.linspace(0, 600000, 13)
print(xticks)
plt.xticks(xticks)
plt.show()
 
sns.countplot(data = Night_Only, y = 'Severity')
sns.set(rc={'figure.figsize':(30,15)})
plt.title("Severity during Night", size = 40)
plt.xlabel('Count', size = 30)
plt.ylabel('Severity', size = 30)
plt.tick_params(axis = 'both', which = 'major', labelsize = 25)
xticks = np.linspace(0, 350000, 15)
print(xticks)
plt.xticks(xticks)
plt.show()

# Create a pie chart for the total count of each severity level
Severity_count = df['Severity'].value_counts().sort_values()
print(Severity_count)
Severity_count = Severity_count.tolist()
print(Severity_count)
colors = sns.color_palette('pastel')[0:4]
pie, ax = plt.subplots(figsize=[20,10])
plt.pie(Severity_count, labels = ['1', '3', '4', '2'], explode=[0.05]*4, pctdistance=0.75, colors = colors, autopct='%.0f%%')
plt.title('Severity Count', size = 15)
plt.show()

# Linear regression for distance based on weather features
from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
 
Weather_features = df.loc[:, ['Temperature(F)','Humidity(%)','Visibility(mi)','Wind_Speed(mph)','Precipitation(in)']]
Crash_distance = df.loc[:, 'Distance(mi)']
Crash_distance.head()

# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = model_selection.train_test_split(Weather_features, Crash_distance, train_size = 0.7)
 
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
model.score(X_train, y_train), model.score(X_test, y_test)
 
Ridge_reg = linear_model.RidgeCV(cv=2)
Ridge_reg.fit(X_train, y_train)
Ridge_reg.score(X_train, y_train), Ridge_reg.score(X_test, y_test)
 
Lasso_reg = linear_model.LassoCV(cv=2)
Lasso_reg.fit(X_train, y_train)
Lasso_reg.score(X_train, y_train), Lasso_reg.score(X_test, y_test)
 
Elastic_reg = linear_model.ElasticNetCV(cv=2)
Elastic_reg.fit(X_train, y_train)
Elastic_reg.score(X_train, y_train), Elastic_reg.score(X_test, y_test)
 

# Classification modeling for severity based on weather features

Severity = df.loc[:, 'Severity']
Severity.head()

# Split dataset into 70% train and 30% test
np.random.seed(123)
X_train, X_test, y_train, y_test = model_selection.train_test_split(Weather_features, Severity, train_size = 0.7)
 
# Logistic Regression
classifier = linear_model.LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
print('Accuracy score = ' + str(metrics.accuracy_score(y_test, y_pred)))
metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_test, y_pred)).plot()
 

# Decision Tree Classification
classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
print('Accuracy score = ' + str(metrics.accuracy_score(y_test, y_pred)))
metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_test, y_pred)).plot()
 
 
# KNeighbors Classification
classifier = neighbors.KNeighborsClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
print('Accuracy score = ' + str(metrics.accuracy_score(y_test, y_pred)))
metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_test, y_pred)).plot()
 

# Random Forest Classification, 100 trees
classifier = ensemble.RandomForestClassifier(n_estimators = 100)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
print('Accuracy score = ' + str(metrics.accuracy_score(y_test, y_pred)))
metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_test, y_pred)).plot()
 
# Random Forest Classification, 10 trees
classifier = ensemble.RandomForestClassifier(n_estimators = 10)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
print('Accuracy score = ' + str(metrics.accuracy_score(y_test, y_pred)))
metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_test, y_pred)).plot()