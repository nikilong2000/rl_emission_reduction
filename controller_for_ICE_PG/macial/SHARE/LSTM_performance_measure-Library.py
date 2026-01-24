#!/usr/bin/env python
# coding: utf-8

# # ICE and PG performance measure

# ## Dependencies

# In[1]:


from ONNX_Predict.LSTM_onnx import LSTM_onnx
from ONNX_Predict.Scaler_onnx import Scaler_onnx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


# As can be seen, it is not necessary to import `tensorflow` or `keras` unless required in some additional part of the process.
#
# `ONNX_Predict` is responsible for managing the process of loading into memory and prediction interface of models and scalers

# ## Reading models and scalers

# ### Paths

# The addresses where the `onnx` models are stored are defined, as well as the original `ICE` and `PG` models

# In[2]:


ICE_folder = os.path.join("CTTC_models", "ONNX", "ICE")
PG_folder = os.path.join("CTTC_models", "ONNX", "PG")
tf_model = "model.h5"


# ### Loading ICE model in memory

# To load the models into memory, an instance of the `Scaler_onnx` classes for the scalers and `LSTM_onnx` for the models* must be created

# In[3]:


ICEscaler_in = Scaler_onnx("scaler_input.onnx", ICE_folder)
ICEscaler_out = Scaler_onnx("scaler_output.onnx", ICE_folder)
ICEscaler_inv_out = Scaler_onnx("scaler_inverse_output.onnx", ICE_folder)
ICE = LSTM_onnx("ICE_onnx.onnx", ICE_folder, tf_model)


# **IMPORTANT**
#
# Note that 3 scalers are loaded into memory instead of 2.
#
# This is because onnx models can only do the transformation in one direction. Since an inverse transformation is required for the prediction, `scaler_inverse_output.onnx` had to be generated for each model

# ### Loading PG model in memory

# In[4]:


PGscaler_in = Scaler_onnx("scaler_input.onnx", PG_folder)
PGscaler_out = Scaler_onnx("scaler_output.onnx", PG_folder)
PGscaler_inv_out = Scaler_onnx("scaler_inverse_output.onnx", PG_folder)
PG = LSTM_onnx("PG_onnx.onnx", PG_folder, tf_model)


# Ídem as `ICE`

# ## Reading input vectors

# ### Path and config vars

# The csv file `part_000100.csv` is selected
#

# In turn, the column names were taken from the model training data

# In[5]:


csvPath = "Artur_info/csv_export_reconstruido/part_000100.csv"
input_cols = [
    "ICE_Speed_rpm",
    "fuel_mg",
    "T_amb_K",
    "p_amb_bar",
    "ICE_Speed_soll_rpm",
    "EM2_Torque_Nm",
    "ICE_Torque_Nm",
    "Brake_perc",
]


# #### Loading dataframe in memory

# In[6]:


df = pd.read_csv(csvPath, names=input_cols, skiprows=1)
ICEinput = df.iloc[:, :4].copy()
PGinput = df.iloc[:, 4:].copy()

# Adjusting speed value
PGinput.loc[PGinput["ICE_Speed_soll_rpm"] < 900] = 0


# ## Preprocess

# ### Model Reset

# In[7]:


ICE.reset_states()


# ### Input vector

# We will work with the first row of the selected file

# In[8]:


x = ICEinput.iloc[:1].copy()
x = x.to_numpy()
x = x.astype("float32")


# #### Scaling the input vector

# In[9]:


x_scaled = ICEscaler_in.transform(x)
x_scaled = np.reshape(x_scaled, (1, 1, 4))


# **IMPORTANT:**
#
# We work directly with `numpy` for vector casting and/or reshaping tasks, instead of using `tensorflow`

# ### Preparing initial auxiliar data

# The default data defined in the `transition_function_model` class defined in the script with the same name is taken as an example:
#
# - Torque  --> 0
# - NO_ini  --> 0
# - NO2_ini --> 0
# - CO_ini  --> 0
# - CO2_ini --> 0

# In[10]:


Torque = 0
NO_ini = 0
NO2_ini = 0
CO_ini = 0
CO2_ini = 0
y_ini = np.array([[Torque, NO_ini, NO2_ini, CO_ini, CO2_ini]], dtype="float32")
y_scaled_ini = ICEscaler_out.transform(y_ini)[0].reshape((1, 1, 5))


# ## Prediction

# ### ICE model prediction behaviour

# In[11]:


y_predict_scaled_list = []


# #### - 1st prediction call

# In[12]:


y_predict_scaled = ICE([x_scaled, y_scaled_ini])
y_predict_scaled_list.append(y_predict_scaled[0][0][0])
y_predict_scaled


# #### - 2nd prediction call

# In[13]:


y_predict_scaled = ICE([x_scaled, y_scaled_ini])
y_predict_scaled_list.append(y_predict_scaled[0][0][0])
y_predict_scaled


# #### - 3rd prediction call

# In[14]:


y_predict_scaled = ICE([x_scaled, y_scaled_ini])
y_predict_scaled_list.append(y_predict_scaled[0][0][0])
y_predict_scaled


# #### - 4th prediction call

# In[15]:


y_predict_scaled = ICE([x_scaled, y_scaled_ini])
y_predict_scaled_list.append(y_predict_scaled[0][0][0])
y_predict_scaled


# #### - 5th prediction call

# In[16]:


y_predict_scaled = ICE([x_scaled, y_scaled_ini])
y_predict_scaled_list.append(y_predict_scaled[0][0][0])
y_predict_scaled


# ### Plot of scaled results

# In[17]:


plot_data = np.array(y_predict_scaled_list)
preds, plot_vars = plot_data.shape

labels = ["Torque", "NO", "NO2", "CO", "CO2"]

_, ax = plt.subplots(figsize=(12, 8))
for var in range(plot_vars):
    ax.plot(plot_data[:, var], label=labels[var])

ax.set_xticks(range(plot_vars))

ax.set_ylabel("Scaled values")
ax.set_xlabel("Inference prediction")
ax.legend()

plt.show()

# ## Post-process

# In[19]:


y_pred_list = []

for pred in range(preds):
    y_pred = ICEscaler_inv_out.transform(y_predict_scaled_list[pred].reshape(1, -1))
    y_pred_list.append(y_pred[0][0])

plot_data = np.array(y_pred_list)


# ### See results

# In[20]:


y_pred = pd.DataFrame(data=plot_data, columns=labels)
y_pred


# #### Basic statistics

# In[21]:


y_pred.describe().loc[["mean", "std"]]


# **IMPORTANT:**
#
# It can be observed how the first torque prediction produces a negative result --> `-30.66 Nm`
#

# ### Plot results

# In[22]:


_, ax = plt.subplots(figsize=(12, 8))
ax2 = ax.twinx()

for var in range(plot_vars):
    if var == 0:
        ax.plot(plot_data[:, var], label=labels[var], c=f"C{var}")
    else:
        ax2.plot(plot_data[:, var], label=labels[var], c=f"C{var}")

ax.set_xticks(range(plot_vars))

ax.set_ylabel("Values")
ax.set_xlabel("Inference prediction")
ax.legend(loc="center left")

ax2.set_ylabel("Values for emission")
ax2.legend(loc="center right")

plt.show()


# ## Repeat process for PG

# ### Model reset

# In[23]:


PG.reset_states()


# ### Prepare input vector

# As with the ICE model, the first row of the selected file is selected

# In[24]:


x = PGinput.iloc[:1].copy()


# #### Removing NaN value

# In[25]:


x.iloc[:, 2] = y_pred["Torque"].iloc[-1]
x = x.to_numpy()
x = x.astype("float32")


# #### Scalling the input vector

# In[26]:


x_scaled = PGscaler_in.transform(x)
x_scaled = np.reshape(x_scaled, (1, 1, 4))


# #### Preparing initial auxiliar vector

# Again, the default values defined in the files `main.ipynb` and `transition_function_model.py` are taken

# In[27]:


SOC_ini = 0.7
velocity_ini = 0

y_ini = np.array([[velocity_ini, SOC_ini]], dtype="float32")
y_scaled_ini = PGscaler_out.transform(y_ini)[0].reshape((1, 1, 2))


# ### PG model prediction behaviour

# A behavior similar to the ICE model has been evidenced but with much smaller variations than those observed for ICE

# In[28]:


y_predict_scaled_list = []


# #### - 1st prediction call

# In[29]:


y_predict_scaled = PG([x_scaled, y_scaled_ini])
y_predict_scaled_list.append(y_predict_scaled[0][0][0])
y_predict_scaled


# #### - 2nd prediction call

# In[30]:


y_predict_scaled = PG([x_scaled, y_scaled_ini])
y_predict_scaled_list.append(y_predict_scaled[0][0][0])
y_predict_scaled


# #### - 3rd prediction call

# In[31]:


y_predict_scaled = PG([x_scaled, y_scaled_ini])
y_predict_scaled_list.append(y_predict_scaled[0][0][0])
y_predict_scaled


# #### - 4th prediction call

# In[32]:


y_predict_scaled = PG([x_scaled, y_scaled_ini])
y_predict_scaled_list.append(y_predict_scaled[0][0][0])
y_predict_scaled


# #### - 5th prediction call

# In[33]:


y_predict_scaled = PG([x_scaled, y_scaled_ini])
y_predict_scaled_list.append(y_predict_scaled[0][0][0])
y_predict_scaled


# Please note how the values of the input vector set are **NOT modified** between calls. That is, for the same set of input vectors, different prediction values are obtained.

# ### Plotting scaled results

# In[34]:


plot_data = np.array(y_predict_scaled_list)
preds, plot_vars = plot_data.shape

labels = ["Car Speed", "SOC"]

_, ax = plt.subplots(figsize=(12, 8))
ax2 = ax.twinx()

for var in range(plot_vars):
    if var == 0:
        ax.plot(plot_data[:, var], label=labels[var], c=f"C{var}")
    else:
        ax2.plot(plot_data[:, var], label=labels[var], c=f"C{var}")

ax.set_xticks(range(plot_vars))

ax.set_ylabel("Values")
ax.set_xlabel("Inference prediction")
ax.legend(loc="center left")

ax2.set_ylabel("Values for SOC")
ax2.legend(loc="center right")

plt.show()

# ## Post-process

# In[36]:


y_pred_list = []

for pred in range(preds):
    y_pred = PGscaler_inv_out.transform(y_predict_scaled_list[pred].reshape(1, -1))
    y_pred_list.append(y_pred[0][0])

plot_data = np.array(y_pred_list)


# ### See results

# In[37]:


y_pred = pd.DataFrame(data=plot_data, columns=labels)
y_pred


# #### Basic statistics

# In[38]:


y_pred.describe().loc[["mean", "std"]]


# **Important:**
#
# Observe the high standard deviation value of the predicted speed for the same pair of input vectors

# ### Plot results

# In[39]:


_, ax = plt.subplots(figsize=(12, 8))
ax2 = ax.twinx()

for var in range(plot_vars):
    if var == 0:
        ax.plot(plot_data[:, var], label=labels[var], c=f"C{var}")
    else:
        ax2.plot(plot_data[:, var], label=labels[var], c=f"C{var}")

ax.set_xticks(range(plot_vars))

ax.set_ylabel("Values")
ax.set_xlabel("Inference prediction")
ax.legend(loc="center left")

ax2.set_ylabel("Values for SOC")
ax2.legend(loc="center right")

plt.show()


# In[ ]:
