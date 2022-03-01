from flask import Flask, render_template, request
from flask import jsonify
#import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__, template_folder='../templates')
model = pickle.load(open('../model/loan_approval_model.pickle', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():

    if request.method == 'POST':

        Gender = request.form['Gender']
        if (Gender == 'Male'):
            Gender = 0
        else:
            Gender = 1
        Married = request.form['Married']
        if (Married == 'No'):
            Married = 0
        else:
            Married = 1

        Education = request.form['Education']
        if (Education == 'Graduate'):
            Education = 1
        else:
            Education = 0
        Self_Employed = request.form['Self_Employed']
        if (Self_Employed == 'No'):
            Self_Employed = 0
        else:
            Self_Employed = 1
        Property_Area = int(request.form['Property_Area'])
        Dependents=int(request.form['Dependents'])
        Credit_History=int(request.form['Credit_History'])
        ApplicantIncome=int(request.form['ApplicantIncome'])
        CoapplicantIncome = float(request.form['CoapplicantIncome'])
        Loan_Amount_Term = float(request.form['Loan_Amount_Term'])
        LoanAmount = float(request.form['LoanAmount'])

        values = np.array([[Gender,Married,Education,Self_Employed,Property_Area,Dependents,Credit_History,ApplicantIncome,CoapplicantIncome,\
                                   Loan_Amount_Term,LoanAmount]])
        prediction = model.predict(values)
        output=([prediction[0]])
        if output == 0:
            return render_template('results.html', prediction_texts="Not Been Approved: {}".format(output))
        else:
            return render_template('results.html', prediction_texts="Been Approved: {}".format(output))


if __name__=="__main__":
    app.run(debug=True)