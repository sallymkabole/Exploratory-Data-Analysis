# check missing values for each column
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.cross_validation import train_test_split

with open('customerdata', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split(",") for line in stripped if line)
    with open('customerdata.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('title', 'intro'))
        writer.writerows(lines)

df = pd.read_csv('customerdata')
print(df.head())
print(df.info())
print(df.describe(include='all'))

#sns.jointplot('Time on App', 'Yearly Amount Spent', data =df)
#sns.pairplot(df)


#linear plot of Yearly Amount Spent and Length of Membership since they seem to have a strong correlation
sns.lmplot('Yearly Amount Spent', 'Length of Membership', data=df)
#plt.show()
X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']] 
y = df[['Yearly Amount Spent']]
#split data into 30 % test set and 70% train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
#use the LinearRegression module from Scikit-learn to create linea Regression Model


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
#fit model to training set
lm.fit(X_train, y_train)
#coefficients our model has chosen for each of our independent variables.
print(lm.coef_)

#test how well our model performs on the test data by calling the .predict() method on model
predictions = lm.predict(X_test)

""" build a scatterplot of the actual yearly amount spent (from y_test) 
against the predicted yearly amount spent (from predictions) 
"""
"""
plt.scatter(y_test, predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()
"""
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, predictions)
mse = metrics.mean_squared_error(y_test, predictions)
rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
print(mae, mse, rmse)
#These errors seem fairly small, so we can conclude that our model is a pretty good fit.

#evaluating accuracy with the above metrics

# let’s plot a histogram of the residuals(diff btn actual y and predicted y) and make sure it looks normally distributed, to show that evrythng is ok
"""sns.distplot(y_test-predictions, bins=50, kde=True)
plt.xlabel('Yearly Amount Spent')
plt.ylabel('Residual')"""

"""
Let’s recreate the coefficients as a dataframe and see which feature (time on app or time on website) 
has more influence on the yearly amount spent.
"""
coeffs =pd.DataFrame(data=lm.coef_.transpose(), index=X.columns, columns=['Coefficients'])
print(coeffs)
