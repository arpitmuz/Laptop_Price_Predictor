import streamlit as st
import pickle
import numpy as np
st.title("Laptop Price Predictor")

# import the model
pipe=pickle.load(open('pipe.pkl','rb'))
df=pickle.load(open('df.pkl','rb'))

# brand
company=st.selectbox('Brand',df['Company'].unique())
type=st.selectbox('Type',df['TypeName'].unique())
Ram=st.selectbox('Ram(in GB)',[2,4,6,8,12,16,24,32,64])
weight=st.number_input('Weight of the Laptop')
touchscreen=st.selectbox('TouchScreen',['No','Yes'])
Ips=st.selectbox('Ips',['No','Yes'])
screen_size=st.number_input('Screen_Size')
resolution=st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

cpu=st.selectbox('Cpu',df['Cpu Brand'].unique())
hdd=st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
ssd=st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])
Gpu=st.selectbox('Gpu',df['Gpu_brand'].unique())
os=st.selectbox('OS',df['OS'].unique())
if st.button('Predict Price'):
    ppi=None
    if touchscreen=='Yes':
        touchscreen=1
    else:
        touchscreen=0
    if Ips=='Yes':
        Ips=1
    else:
        Ips=0
    x_res=int(resolution.split('x')[0])
    y_res =int(resolution.split('x')[1])
    ppi=((x_res**2)+(y_res**2))**0.5/screen_size


    query=np.array([company,type,Ram,weight,touchscreen,Ips,ppi,cpu,hdd,ssd,Gpu,os])
    query=query.reshape(1,12)

    st.title("The Predicted Price of Laptop is:"+str(int((np.exp(pipe.predict(query)[0])))))