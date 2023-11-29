# Model clustering
import pickle
import numpy as np
import sklearn

def time_to_seconds(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return hours * 3600 + minutes * 60

knn = pickle.load(open(".\static\models\model.pkl","rb"))


def trafficPrediction(data):
    data = np.array(data).reshape(1,-1)
    # Get the prediction
    result = knn.predict(data)
    return result

from flask import Flask, render_template, request

app = Flask(__name__)

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# route 'form' set as 'form.html'
@app.route('/form')
def form():
    return render_template('form.html')

# Route untuk menangani submit form
@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        # Mengambil data dari form
        day = float(request.form['day'])
        time = request.form['time']
        car_count = float(request.form['car'])
        bike_count = float(request.form['bike'])
        bus_count = float(request.form['bus'])
        truck_count = float(request.form['truck'])

        time_in_seconds = time_to_seconds(time)
        count = car_count + bike_count + bus_count + truck_count

        to_predict_list = [day, time_in_seconds, car_count, bike_count, bus_count, truck_count, count]
        result = trafficPrediction(to_predict_list)

        if float(result) == 0:
            prediction = 'Low'
        elif float(result) == 1:
            prediction = 'Normal'
        elif float(result) == 2:
            prediction = 'High'
        elif float(result) == 3:
            prediction = 'Heavy'

        return render_template("output.html", prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
