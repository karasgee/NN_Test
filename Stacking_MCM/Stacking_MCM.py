import numpy as np
import pickle
import keras
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score  as r2
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold