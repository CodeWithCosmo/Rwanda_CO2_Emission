import pickle
import pandas as pd
from flask import Flask,request,app,render_template

app=Flask(__name__)

##! Loading the model
scaler=pickle.load(open('scaling.pkl','rb'))
model=pickle.load(open('classifier.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    data = {'Pclass': [int(request.form['Pclass'])],
            'Sex': [int(request.form['Sex'])],
            'Age': [int(request.form['Age'])],
            'Fare': [float(request.form['Fare'])],
            'Family': [int(request.form['Family'])]}
    
    df = pd.DataFrame(data)
    scaled_input = scaler.transform(df)    
    output=model.predict(scaled_input)[0]
    if output == 1:
        return render_template("index.html",prediction_text="This passenger was survived.")
    else:
        return render_template("index.html",prediction_text="This passenger was not survived.")

if __name__=="__main__":
    app.run(debug=True)