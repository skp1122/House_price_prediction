from flask import Flask,request,render_template,redirect
# from sklearn.externals import joblib
import joblib as joblib
import numpy as np

#__name__ == __main__
app = Flask(__name__)

model = joblib.load("Dragon.joblib")

@app.route('/')
def home():
    return render_template("index.html")

features = np.array([[0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.43979304, -1.31238772,  5.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.36091034]])
@app.route('/predict', methods = ['GET' , 'POST'])
def price():
    price = None
    if request.method == 'POST':
        features1 =  float(request.form['features'])

        price = model.predict(features)

    return render_template("predict.html",house_price = price)

if __name__=='__main__': 
    # app.debug = True
    app.run(debug=True)

