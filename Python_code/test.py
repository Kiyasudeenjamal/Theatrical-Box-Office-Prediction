import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

data = pd.read_csv("D:\\data_analytics_pro\\Movie_gross_prediction\\new.csv")
data = data.drop_duplicates()

new_cl = pd.get_dummies(data['day_cat'])
new_cl_1 = pd.get_dummies(data['high_expectation'])
new_cl_2 = pd.get_dummies(data['Franchise'])
new_cl_3 = pd.get_dummies(data['Season'])
new_cl_4 = pd.get_dummies(data['Year_cat'])
data = pd.concat([data, new_cl, new_cl_1,new_cl_2, new_cl_3, new_cl_4], axis=1)
#print(data.columns)
x = data[['Opening','year',
          'Avg per screen','Weekday', 'Weekend', 'Average', 'Normal', 'low',
          'Big franchise', 'Small franchise', 'Winter', 'Summer', 'Autumn', 'Spring',
          ]]
y = data['Total Gross']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,  random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred_test = model.predict(x_test)
y_pred_train = model.predict(x_train)

test_score = r2_score(y_test, y_pred_test)
train_score = r2_score(y_train, y_pred_train )
print("train_score = ", train_score)
print("test_score = ", test_score)

mape_test = (abs(y_test-y_pred_test))/y_test*100
mape_test = list(mape_test)
mape_train = (abs(y_train-y_pred_train))/y_train*100
mape_train = list(mape_train)

print("mape_test = ", np.average(mape_test))
print("mape_train = ", np.average(mape_train))


dict = {'Opening': 126707459,
        'year': 2022,
                    'Avg per screen': 26759,
                    'Weekday': 0,
                    'Weekend': 1,
                    'Average': 0,
                    'Normal': 1,
                    'low': 0,
                    'Big franchise': 0,
                    'Small franchise': 0,
                    'Winter': 0,
                    'Summer': 0,
                    'Autumn': 0,
                    'Spring': 0,
                    }
df = pd.DataFrame([dict])
new = model.predict(df)

print(new)

"""

test_score = r2_score(y_test, y_pred_test)
train_score = r2_score(y_train, y_pred_train )
print("train_score = ", train_score)
print("test_score = ", test_score)

mape_test = (abs(y_test-y_pred_test))/y_test*100
mape_test = list(mape_test)
mape_train = (abs(y_train-y_pred_train))/y_train*100
mape_train = list(mape_train)

print("mape_test = ", np.average(mape_test))
print("mape_train = ", np.average(mape_train))

"""

"""

train_score =  0.8112317516554803
test_score =  0.8012037271678045
mape_test =  23.9916311630246
mape_train =  22.390437745117524
"""


