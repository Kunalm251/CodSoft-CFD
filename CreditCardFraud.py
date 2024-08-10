from re import sub
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

Credited_card_df = pd.read_csv('creditcard.csv')
Credited_card_df.head()

Credited_card_df.shape

Credited_card_df ['Class'].value_counts()


Credited_card_df.describe()

legit = Credited_card_df[Credited_card_df.Class == 0]
fraud = Credited_card_df[Credited_card_df.Class == 1]

legit_sample = legit.sample(n= len(fraud))
Credited_card_df = pd.concat([legit_sample, fraud], axis=0)
Credited_card_df['Class'].value_counts()
Credited_card_df.groupby('Class').mean

x= Credited_card_df.drop(columns='Class', axis=1)
y= Credited_card_df['Class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)


model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy_score(y_test, y_pred)

train_acc = accuracy_score(y_train, model.predict(x_train))
test_acc = accuracy_score(y_test, model.predict(x_test))
print(train_acc)
print(test_acc)


st.title('Credit Card Fraud Detection')
st.header('Enter the details below')
input_df = st.text_input('Enter the input values')
input_df_splited = input_df.split(',')

submit = st.button('Submit')

if submit:
  value = np.asarray(input_df_splited, dtype=np.float64)
  prediction = model.predict(value.reshape(1, -1))
  if prediction[0] == 0:
    st.success('The transaction is legit')
  else:
    st.error('The transaction is fraud')

st.write('Thank you for using our app')