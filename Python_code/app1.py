from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the trained model
data = pd.read_csv("D:\\data_analytics_pro\\Movie_gross_prediction\\new.csv")
data = data.drop_duplicates()

new_cl = pd.get_dummies(data['day_cat'])
new_cl_1 = pd.get_dummies(data['high_expectation'])
new_cl_2 = pd.get_dummies(data['Franchise'])
new_cl_3 = pd.get_dummies(data['Season'])
new_cl_4 = pd.get_dummies(data['Year_cat'])
data = pd.concat([data, new_cl, new_cl_1,new_cl_2, new_cl_3, new_cl_4], axis=1)
#print(data.columns)
x = data[['Opening',
          'Avg per screen','Weekday', 'Weekend', 'Average', 'Normal', 'low',
          'Big franchise', 'Small franchise', 'Winter', 'Summer', 'Autumn', 'Spring',
          ]]
y = data['Total Gross']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,  random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Opening = int(request.form['Opening'])
        Avg_per_screen = int(request.form['Avg_per_screen'])
        Weekday = int(request.form.get('Weekday', 0))
        Weekend = int(request.form.get('Weekend', 0))
        Normal = int(request.form.get('Normal', 0))
        Average = int(request.form.get('Average', 0))
        low = int(request.form.get('low', 0))
        Big_franchise = int(request.form.get('Big_franchise', 0))
        Small_franchise = int(request.form.get('Small_franchise', 0))
        Winter = int(request.form.get('Winter', 0))
        Summer = int(request.form.get('Summer', 0))
        Autumn = int(request.form.get('Autumn', 0))
        Spring = int(request.form.get('Spring', 0))

        new_data = {'Opening': Opening,
                    'Avg per screen': Avg_per_screen,
                    'Weekday': Weekday,
                    'Weekend': Weekend,
                    'Average': Average,
                    'Normal': Normal,
                    'low': low,
                    'Big franchise': Big_franchise,
                    'Small franchise': Small_franchise,
                    'Winter': Winter,
                    'Summer': Summer,
                    'Autumn': Autumn,
                    'Spring': Spring,
                    }

        new_df = pd.DataFrame([new_data])
        prediction = model.predict(new_df)[0]  # Get the first element of the prediction array
        formatted_prediction = "{:,.2f}".format(
            prediction)  # Format prediction to have two decimal places and comma separator
        return render_template('result.html', prediction=formatted_prediction)


if __name__ == '__main__':
    app.run(debug=True)
