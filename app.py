import pickle as pkl
import numpy as np
import pandas as pd
pd.set_option('display.max_columns',100)
from flask import Flask, request ,redirect, render_template, url_for, jsonify

with open('model_columns.pkl', 'rb') as file:
    model_columns = pkl.load(file)
with open('cat_boost_model.pkl', 'rb') as file:
    model = pkl.load(file)

train_data = pd.read_csv('train.csv')

# Getting unique values from each column
cust_info = train_data['Customer Information'].unique()
express_shipment = train_data['Express Shipment'].unique()
fragile = train_data['Fragile'].unique()
installation = train_data['Installation Included'].unique()
international = train_data['International'].unique()
material = train_data['Material'].dropna().unique()
remote_location = train_data['Remote Location'].dropna().unique()
transport = train_data['Transport'].dropna().unique()

predict_df = pd.DataFrame(columns=model_columns , data=np.zeros(shape=(1,len(model_columns))))

app = Flask(__name__)
@app.route('/',methods = ['GET','POST'])
def home():
    return render_template('index.html',customer_info=cust_info, express = express_shipment,fragile=fragile,
                           installation=installation,international=international,material=material,
                           remote= remote_location, transport=transport)

@app.route('/get_prediction', methods=['GET','POST'])
def get_prediction():
    schedule_date = pd.to_datetime(request.form['schedule_date'])
    delivery_date = pd.to_datetime(request.form['delivery_date'])
    predict_df['Customer Information'] = request.form['customer_info']
    predict_df['Artist Reputation'] = request.form['score']
    predict_df['Base Shipping Price'] = request.form['base_price']
    predict_df['Express Shipment']=request.form['express']
    predict_df['Fragile']=request.form['fragile']
    predict_df['Height'] =request.form['height']
    predict_df['Installation Included'] =request.form['installation']
    predict_df['International'] =request.form['international']
    predict_df['Material'] =request.form['material']
    predict_df['Price Of Sculpture'] = request.form['price']
    predict_df['Remote Location'] = request.form['remote']
    predict_df['Transport']= request.form['transport']
    predict_df['Weight'] = request.form['weight']
    predict_df['Width'] =request.form['width']
    predict_df['days'] = delivery_date.day - schedule_date.day
    predict_df['month'] = delivery_date.month
    predict_df['year'] = delivery_date.year
    predict_df['day_of_week'] = delivery_date.dayofweek

    predict_df["International"].replace({'Yes': 1, 'No': 0}, inplace=True)
    predict_df["Express Shipment"].replace({'Yes': 1, 'No': 0}, inplace=True)
    predict_df["Installation Included"].replace({'Yes': 1, 'No': 0}, inplace=True)
    predict_df["Fragile"].replace({'Yes': 1, 'No': 0}, inplace=True)
    predict_df["Customer Information"].replace({'Working Class': 0, 'Wealthy': 1}, inplace=True)
    prediction = np.expm1(model.predict(predict_df))
    return render_template('index.html', prediction= prediction)

if __name__ == '__main__':
    app.run(debug=True)