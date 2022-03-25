#!/usr/bin/env python
# coding: utf-8

# # Machine Learning - Neural Net

# Nesse projeto de Machine Learning, estaremos usando uma rede neural para criarmos um modelo de regressão que irá prever a quantidade de bicicletas alugadas com base na nossa base de dados que contém dados diários de aluguel de bicicletas durante o período de 2 anos. 
# Na nossa base de dados temos dados de temperatura, umidade do ar, velocidade do vento como variáveis quantitativas.
# Como variáveis qualitativas, temos a época do ano, mês, dia da semana, feriados, dias úteis e a situação do clima.
# Iremos então realizar todos os passos clássicos referentes a criação de modelo de Machine Learning a seguir.
# 
# - Referência da base de dados: 
#     - This Hadi Fanaee-T
#     - Laboratory of Artificial Intelligence and Decision Support (LIAAD), University of Porto INESC Porto, Campus da FEUP Rua Dr. Roberto Frias, 378 4200 - 465 Porto, Portugal
# 

# # Etapa 1: Importação das bibliotecas

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# # Etapa 2: Importação da base de dados

# Iremos importar nossa base de dados e em seguidas analisarmos algumas estatíscas descritivas da nossa base.

# In[2]:


bike = pd.read_csv('bike-sharing-daily.csv')


# In[3]:


bike


# In[4]:


display(bike.info())


# In[5]:


bike.describe()


# # Etapa 3: Limpeza da base de dados

# Vamos verificar se a nossa base de dados contém valores nulos e em seguida iremos realizar alguns processos de limpeza de dados em nossa base.

# In[6]:


sns.heatmap(bike.isnull());


# In[7]:


bike = bike.drop(labels=['instant'], axis = 1)


# In[8]:


bike.head()


# In[9]:


bike = bike.drop(labels=['casual', 'registered'], axis = 1)


# In[10]:


bike


# In[11]:


bike.dteday = pd.to_datetime(bike.dteday, format = '%m/%d/%Y')


# In[12]:


bike.head()


# In[13]:


bike.index = pd.DatetimeIndex(bike.dteday)


# In[14]:


bike.head()


# In[15]:


bike = bike.drop(labels=['dteday'], axis=1)


# In[16]:


bike.head()


# # Etapa 4: Visualização da base de dados

# Agora, vejamos alguns gráficos que facilitarão o entendimento das variáveis da base de dados e das correlações que elas possuem entre si.

# In[17]:


bike['cnt'].asfreq('W').plot(linewidth = 3)
plt.title('Bike usage per week')
plt.xlabel('Week')
plt.ylabel('Bike rental');


# In[18]:


bike['cnt'].asfreq('M').plot(linewidth = 3)
plt.title('Bike usage per month')
plt.xlabel('Week')
plt.ylabel('Bike rental');


# In[19]:


bike['cnt'].asfreq('Q').plot(linewidth = 3)
plt.title('Bike usage per quarter')
plt.xlabel('Week')
plt.ylabel('Bike rental');


# In[20]:


plt.figure(figsize=(20,10))
sns.heatmap(bike.corr(), annot = True);


# In[21]:


X_numerical = bike[['temp', 'hum', 'windspeed', 'cnt']]


# In[22]:


X_numerical


# In[23]:


sns.pairplot(X_numerical)


# In[24]:


sns.heatmap(X_numerical.corr(), annot = True);


# # Etapa 5: Tratamento das bases de dados

# In[25]:


X_cat = bike[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']]


# Iremos realizar um procedimento de "Encode" para que o nosso modelo de regressão linear usando uma rede neural possa conseguir identificar essas variáveis qualitativas como dia da semana, mês e feriado. Basimente a função "OneHotEnconder" transformar essas variáveis em columas em que 1 representa positivo e 0 representa negativo ou ausente.

# In[26]:


X_cat.head()


# In[27]:


from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()


# In[28]:


display(X_cat)


# In[29]:


display(X_cat.shape)


# In[30]:


X_cat = pd.DataFrame(X_cat)


# In[31]:


X_cat.head()


# In[32]:


X_numerical.head()


# In[33]:


X_numerical = X_numerical.reset_index()


# In[34]:


X_numerical.head()


# In[35]:


X_all = pd.concat([X_cat, X_numerical], axis = 1)


# In[36]:


X_all.head()


# In[37]:


X_all = X_all.drop(labels=['dteday'], axis = 1)


# In[38]:


X_all.head()


# In[39]:


X = X_all.iloc[:, :-1].values


# In[40]:


y = X_all.iloc[:, -1:].values


# In[41]:


display(X.shape)


# In[42]:


display(y.shape)


# In[43]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
y = scaler.fit_transform(y)


# In[44]:


display(y)


# In[45]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[46]:


display(X_train.shape)


# In[47]:


display(X_test.shape)


# # Etapa 6: Construção e treinamento do modelo

# In[48]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units = 100, activation='relu', input_shape=(35,)))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))


# In[49]:


display(model.summary())


# In[50]:


model.compile(optimizer='Adam', loss='mean_squared_error')


# In[51]:


epochs_hist = model.fit(X_train, y_train, epochs = 25, batch_size = 50, validation_split=0.2)


# # Etapa 7: Avaliação do modelo 

# In[52]:


epochs_hist.history.keys()


# In[53]:


plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model loss progress during training')
plt.xlabel('Epochs')
plt.ylabel('Training and validation loss')
plt.legend(['Training loss', 'Validation loss']);


# In[54]:


y_predict = model.predict(X_test)


# In[55]:


display(y_predict)


# In[56]:


plt.plot(y_test, y_predict, "^", color = 'r')
plt.xlabel('Model predictions')
plt.ylabel('True values')


# In[57]:


y_predict_orig = scaler.inverse_transform(y_predict)
y_test_orig = scaler.inverse_transform(y_test)
comp = pd.DataFrame(y_test_orig)
comp.columns = ['Y_test']
comp['Prediction'] = pd.DataFrame(y_predict_orig)
comp['Diference Between Y_test and Prediction'] =   comp['Y_test'] - comp['Prediction']
display(comp)
comp.describe()


# In[58]:


plt.plot(y_test_orig, y_predict_orig, "^", color = 'r')
plt.xlabel('Model predictions')
plt.ylabel('True values')


# In[59]:


k = X_test.shape[1]
display(k)


# In[60]:


n = len(X_test)
display(n)


# In[61]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt


# In[62]:


mae = mean_absolute_error(y_test_orig, y_predict_orig)
mse = mean_squared_error(y_test_orig, y_predict_orig)
rmse = sqrt(mse)
r2 = r2_score(y_test_orig, y_predict_orig)


# In[63]:


display(f"MAE: {mae}", f"\nMSE: {mse}", f"\nRMSE: {rmse}", f"\nR2: {r2}")


# Podemos observar que o coeficiente de determinação de nosso modelo foi de 78,03%, isso significa que o modelo explica 78,03% da variância da variável dependente a partir do regressores (variáveis independentes) incluídas no modelo linear. Assim, vemos como uma rede neural pode ser extremamente útil para realizarmos previsões com base em variáveis que estão correlacionadas entre si.
