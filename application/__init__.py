from flask import Flask, request, Response, json
import numpy as numpy
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

#load data
df = pd.read_csv("./data/healthcarestrokedata.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

numeric_data = df.iloc[:, [1,2,3]].values
numeric_df = pd.DataFrame(numeric_data, dtype = object)
numeric_df.columns = ['age', 'hypertension','heartdisease']


#standard scaling age
age_std_scale = StandardScaler()
numeric_df['age'] = age_std_scale.fit_transform(numeric_df[['age']])

#standard scaling balance
balance_std_scale = StandardScaler()
numeric_df['hypertension'] = balance_std_scale.fit_transform(numeric_df[['hypertension']])

balance_std_scale = StandardScaler()
numeric_df['heartdisease'] = balance_std_scale.fit_transform(numeric_df[['heartdisease']])

X_categoric = df.iloc[:, [0,4,5,6,7]].values

#onehotencoding
ohe = OneHotEncoder()
categoric_data = ohe.fit_transform(X_categoric).toarray()
categoric_df = pd.DataFrame(categoric_data)
categoric_df.columns = ohe.get_feature_names()

#combine numeric and categorix
X_final = pd.concat([numeric_df, categoric_df], axis = 1)
#train model
rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(X_final, y)

#create flask instance
app = Flask(__name__)

#create api
@app.route('/api', methods=['GET', 'POST'])
def predict(): 
    #get the data from request
    data = request.get_json(force=True)
    data_categoric = np.array([data["gender"], data["evermarried"], data["worktype"], data["residencetype"], data["smokingstatus"]])
    data_categoric = np.reshape(data_categoric, (1, -1))
    data_categoric = ohe.transform(data_categoric).toarray()
 
    data_age = np.array([data["age"]])
    data_age = np.reshape(data_age, (1, -1))
    data_age = np.array(age_std_scale.transform(data_age))

    data_hypertension = np.array([data["hypertension"]])
    data_hypertension= np.reshape(data_hypertension, (1, -1))
    data_hypertension = np.array(balance_std_scale.transform(data_hypertension))

    data_heartdisease = np.array([data["heartdisease"]])
    data_heartdisease= np.reshape(data_heartdisease, (1, -1))
    data_heartdisease = np.array(balance_std_scale.transform(data_heartdisease))

    data_final = np.column_stack((data_age, data_hypertension,data_heartdisease , data_categoric))
    data_final = pd.DataFrame(data_final, dtype=object)

    #make predicon using model
    prediction = rfc.predict(data_final)
    return Response(json.dumps(prediction[0]))
