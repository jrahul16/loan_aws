from flask import Flask,request,render_template
import pickle
import numpy as np
app=Flask(__name__)


with open(r"artifact/model.pkl","rb") as file:
    model=pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["GET","POST"])
def predict():
    data=request.form
    a=np.zeros(10)
    a[0]=data["gender"]
    a[1]=data["married"]
    a[2]=data["dependents"]
    a[3]=data["education"]
    a[4]=data["self_employed"]
    a[5]=data["loan_amount_term"]
    a[6]=data["credit_history"]
    a[7]=data["property_area"]
    a[8]=np.log(int(data["loan_amount"]))
    a[9]=np.log(int(int(data["applicant_income"])+int(data["co_applicant_income"])))
    b=model.predict([a])
    c={1:"Loan Accepted",0:"Loan Rejected"}  
    return render_template("index.html",pred=c[b[0]])

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080,debug=True) 