import pandas as pd
import streamlit as st
import numpy as np
import pickle

lr=pickle.load(open('hr_lr.pkl','rb'))
dt=pickle.load(open('hr_dt.pkl','rb'))
rf=pickle.load(open('hr_rf.pkl','rb'))
knn=pickle.load(open('hr_knn.pkl','rb'))
ada=pickle.load(open('hr_ada.pkl','rb'))
xgb=pickle.load(open('hr_xgb.pkl','rb'))

model_name=st.sidebar.selectbox('Select the ML Model',['Logistic Regression',
'Decision Tree','Random Forest','KNN','AdaBoost','XGBoost'])

st.header('Hotel Reservation Cancellation Classifier web App')
st.write('Fill the details below to predict if the booking will be cancelled')

def select_model():
    if model_name=='Logistic Regression':
        return lr
    elif model_name=='Decision Tree':
        return dt
    elif model_name=='Random Forest':
        return rf
    elif model_name=='KNN':
        return knn
    elif model_name=='AdaBoost':
        return ada
    else:
        return xgb  


col1,col2,col3=st.columns(3)

with col1:
    adults=st.slider('Num of Adults',1,4,2)
    children=st.slider('Num of Children',0,2,0)
    weekend_nights=st.slider('Num of Weekend Nights',0,4,1)
    week_nights=st.slider('Num of Week Nights',0,8,2)
    car_parking=st.selectbox('Car Parking',['Yes','No'])
    room_type=st.selectbox('Select room type',[1,2,4,5,6,7])
   

with col2:
    lead_time=st.number_input('Lead Time',0,377,30)
    arrival_yr=st.selectbox('Arrival year',[2017.0,2018.0])
    arrival_month=st.selectbox('Arrival month',list(range(1,13)))
    arrival_day=st.selectbox('Arrival day',list(range(1,32)))
    repeated_guest=st.selectbox('Repeated Guest',['Yes','No'])
    prev_cancel=st.selectbox('Previous Cancellation',[1,0])

with col3:
    prev_bookings_not_cancel=st.selectbox('Previous bookings not cancelled',list(range(0,8)))
    avg_price=st.number_input('Avg price per room',51.0,225.0,90.0)
    num_spcl_requests=st.number_input('No of special requests',0,5,1)
    meal_plan=st.selectbox('Meal plan',['Plan1','Plan2','Not Selected'])
    market_segment=st.selectbox('Market segment',['Online','Offline','Others'])

car_parking=1 if car_parking=='Yes' else 0
repeated_guest=1 if repeated_guest=='Yes' else 0

if meal_plan=="Plan2":
    mp2=1
    mp_ns=0
    mp1=0
elif meal_plan=="Not Selected":
    mp2=0
    mp_ns=1
    mp1=0
else:
    mp2=0
    mp_ns=0
    mp1=1

if market_segment=="Online":
    ms_on=1
    ms_off=0
    ms_oth=0
elif market_segment=="Offline":
    ms_on=0
    ms_off=1
    ms_oth=0
elif market_segment=="Others":
    ms_on=0
    ms_off=0
    ms_oth=1
else:
    ms_on=0
    ms_off=1
    ms_oth=0

test_data=[adults,children,weekend_nights,week_nights,car_parking,room_type,lead_time,
arrival_yr,arrival_month,arrival_day,repeated_guest,prev_bookings_not_cancel,avg_price,
num_spcl_requests,mp2,mp_ns,ms_on,ms_oth]

test_data=np.array(test_data).reshape(1,19)

test_col=['Booking_ID', 'no_of_adults', 'no_of_children', 'no_of_weekend_nights',
       'no_of_week_nights', 'required_car_parking_space', 'room_type_reserved',
       'lead_time', 'arrival_year', 'arrival_month', 'arrival_date',
       'repeated_guest', 'no_of_previous_cancellations',
       'no_of_previous_bookings_not_canceled', 'avg_price_per_room',
       'no_of_special_requests', 'booking_status',
       'type_of_meal_plan_Meal Plan 2', 'type_of_meal_plan_Not Selected',
       'market_segment_type_Online', 'market_segment_type_Other']

test_df=pd.DataFrame(test_data,columns=test_col)

st.write(test_df)

model_used=select_model()

predict_btn=st.button('Predict')
if predict_btn:
    pred=model_used.predict(test_data)[0]
    st.success(pred)
    if pred==0:
        st.success(f'{pred} => Not Cancelled')
    else:
        st.success(f'{pred} => Cancelled')