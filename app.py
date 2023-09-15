import pickle
import numpy as np
from flask import Flask,request,app,render_template

app=Flask(__name__)

##! Loading the model
clf_model=pickle.load(open('classifier.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=clf_model.predict(final_input)[0]
    if output == 1:
        return render_template("home.html",prediction_text="This passenger was survived.".format(output))
    else:
        return render_template("home.html",prediction_text="This passenger was not survived.".format(output))

if __name__=="__main__":
    app.run()