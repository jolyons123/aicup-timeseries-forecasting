#######################################################################
### CONFIG                                                          ###
#######################################################################
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import os

SEED=0
os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # new flag present in tf 2.0+
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

import tensorflow_addons as tfa
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import os
import sys
import json
import pickle
import datetime
from tqdm import tqdm

module_path = os.path.abspath("50sources")
sys.path.insert(1, module_path)

from utils import util
from utils import executor
from lstm_model import window

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
def set_global_determinism(seed):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

#######################################################################
### MODELS                                                          ###
#######################################################################
class BaseModel(tf.keras.Model):
    def __init__(self, n=32, dropout=0.0, dropout_count=0, batch_norm=False, batch_norm_count=0, out_steps=5, out_features=1):
      super().__init__()
      self.n = n
      self.out_steps = out_steps
      self.out_features = out_features

      self.has_dropout = True if dropout > 0.0 else False
      self.batch_norm = batch_norm

      # create dropout layers
      if self.has_dropout:
        for i in range(dropout_count):
          setattr(self, f"dropout_{i+1}", tf.keras.layers.Dropout(dropout))

      # create batch norm layers
      if self.batch_norm:
        for i in range(batch_norm_count):
          setattr(self, f"batch_norm_{i+1}", tf.keras.layers.BatchNormalization())

class DenseLstmDense(BaseModel):
    def __init__(self, cell=tf.keras.layers.LSTMCell, m=0, p=1, bi_encoder=False, activation="selu", **kwargs):
        super().__init__(batch_norm_count=2, dropout_count=2, **kwargs)

        self.p=p
        self.m=m

        ### TRAINABLE LAYERS ###
        #self.dense_1 = tf.keras.layers.Dense(self.n, activation=activation)
        self.dense_1 = tf.keras.layers.Dense(self.n, activation=activation)
        self.dense_b = tf.keras.layers.Dense(self.n, activation=activation)
        self.dense_2 = tf.keras.layers.Dense(int(self.n/2), activation=activation)
        self.cell = cell(min(32,int(self.n/2)), activation=activation)
        if bi_encoder:
            self.encoder = tf.keras.layers.Bidirectional(tf.keras.layers.RNN(self.cell))
        else:
            self.encoder = tf.keras.layers.RNN(self.cell)

        ### MISC ###
        self.flatten = tf.keras.layers.Flatten()

        ### OUTPUT ### 
        self.dense_out = tf.keras.layers.Dense(self.out_steps*self.out_features, kernel_initializer=tf.initializers.zeros())
            

    def call(self, inputs, training=None):
        # x = past data
        x = inputs["a"]
        # b = future data (less features than x)
        b = inputs["b"]

        # LAGS
        lags = x[:,-self.p:,:]
        if self.m > self.p:
            lag_day = x[:,-self.m,:]
            lag_day=tf.expand_dims(lag_day,axis=1)
            lags = tf.concat([lag_day,lags], axis=1)

        x = lags
        # DENSE
        x = self.dense_1(x)
        if self.batch_norm:
            x = self.batch_norm_1(x, training)
        if self.has_dropout:
            x = self.dropout_1(x, training)
        b = self.dense_b(b)
        if self.batch_norm:
            b = self.batch_norm_1(b, training)
        if self.has_dropout:
            b = self.dropout_1(b, training)

        x = tf.concat([x,b], axis=1)

        # ENCODER
        x = self.encoder(x)

        # DENSE
        x = self.dense_2(x)
        
        # OUTPUT
        x = self.dense_out(x)

        # reshape to match output
        x = tf.reshape(x, [tf.shape(x)[0], self.out_steps, self.out_features])

        return x

def create_dense_lstm_dense(ds, n, p=1, m=6, out_steps=1, out_features=1, activation="relu", dropout=0.0):
  past_input = tf.keras.layers.Input(next(iter(ds))[0]['a'].shape[-2:], name="a")
  future_input = tf.keras.layers.Input(next(iter(ds))[0]['b'].shape[-2:], name="b")

  x = tf.keras.layers.Lambda(lambda x: x[:,-p:,:])(past_input)
  if m > p:
    seasonal_day = tf.keras.layers.Lambda(lambda x: x[:, -m, :])(past_input)
    seasonal_day = tf.expand_dims(seasonal_day, axis=1)
    x = tf.concat([seasonal_day,x], axis=1)

  x = tf.keras.layers.Dense(n)(x)
  if dropout > 0.0:
    x = tf.keras.layers.Dropout(dropout)(x)
  x = tf.keras.layers.LSTM(min(32,n))(x)
  if dropout > 0.0:
    x = tf.keras.layers.Dropout(dropout)(x)
  x = tf.keras.layers.Flatten()(x)
  x2 = tf.keras.layers.Flatten()(future_input)
  x = tf.keras.layers.concatenate([x,x2], axis=-1)

  x = tf.keras.layers.Dense(out_steps*out_features, kernel_initializer="zeros")(x)
  x = tf.keras.layers.Reshape((out_steps,out_features))(x)

  return tf.keras.Model(inputs=[past_input,future_input], outputs=x, name="nnar_with_future")

def create_nnar(ds, n, p=1, m=6, activation="relu", dropout=0.0, **kwargs):
  past_input = tf.keras.layers.Input(next(iter(ds))[0]['a'].shape[-2:], name="a")
  future_input = tf.keras.layers.Input(next(iter(ds))[0]['b'].shape[-2:], name="b")

  x = tf.keras.layers.Lambda(lambda x: x[:,-p:,:])(past_input)
  x = tf.keras.layers.Flatten()(x)
  if m > p:
    seasonal_day = tf.keras.layers.Lambda(lambda x: x[:, -m, :])(past_input)
    seasonal_day = tf.keras.layers.Flatten()(seasonal_day)
    x = tf.concat([x,seasonal_day], axis=-1)
  #x = tf.keras.layers.Flatten()(future_input)
  x2 = tf.keras.layers.Flatten()(future_input)
  x = tf.keras.layers.concatenate([x,x2], axis=-1)

  x = tf.keras.layers.Dense(n, activation=activation)(x)
  if dropout > 0.0:
    x = tf.keras.layers.Dropout(dropout)(x)
  x = tf.keras.layers.Dense(1, kernel_initializer="zeros")(x)
  x = tf.keras.layers.Reshape((1,1))(x)

  return tf.keras.Model(inputs=[past_input,future_input], outputs=x, name="nnar_with_future")

#######################################################################
### MODEL UTILS                                                     ###
#######################################################################
def compile_and_fit(model, train, early_watch='loss', val=None, patience=1, epochs=1, run_eagerly=False, verbose='auto'):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=early_watch,
                                                      patience=patience,
                                                      mode='min',
                                                      restore_best_weights=True)


    loss = tf.losses.MeanSquaredError()
    model.compile(loss=loss,
                  #optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                  optimizer=tfa.optimizers.COCOB(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    model.run_eagerly = run_eagerly

    callbacks = [early_stopping] if patience > 0 else None
    history = model.fit(train, epochs=epochs,
                        validation_data=val, callbacks=callbacks, verbose=verbose)
    return history    

class ModelWrapper:
  def __init__(self, model, id=None, **kwargs):
    # instantiate model
    self.model = model(**kwargs)
    # construct name
    self.name = type(self.model).__name__
    for k,v in kwargs.items():
      if type(v) is bool and v == True:
        self.name = self.name + f" {k[:2].upper()}"
      else:
        self.name = self.name + f" {k[0].upper()}{v}"
    
    if id is not None:
      self.name = self.name + f" {id}"    

#######################################################################
### LOAD CONFIG                                                     ###
#######################################################################
out_steps = 1
out_features = 1
shift = 1
epochs = 125
patience = 20
slot_train_end = 30
slot_val_end = 50

results = {}

#json_params = '{"model": "nnar", "grid_batch_norm": [false], "grid_dropout": [0.2], "grid_n": [8], "grid_activation": ["selu", "relu"], "grid_input_width_param": [7], "grid_p_param": [2], "grid_m_param": [7]}'

if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    with open(sys.argv[1],'r') as f:
        obj = json.load(f)
else:
    print("Error: no config file specified")
    exit(1)

# overwrite some non grid search settings if set in properties file
if "out_steps" in obj:
  out_steps = obj["out_steps"]
if "out_features" in obj:
  out_features = obj["out_features"]
if "shift" in obj:
  shift = obj["shift"]
if "epochs" in obj:
  epochs = obj["epochs"]
if "patience" in obj:
  patience = obj["patience"]
if "slot_train_end" in obj:
  slot_train_end = obj["slot_train_end"]
if "slot_val_end" in obj:
  slot_val_end = obj["slot_val_end"]

#######################################################################
### TRAINING DATA                                                   ###
#######################################################################
manager = executor.GasModelManager()
df_gws_train = manager.get_train(slot_train_end)
df_gws_val = manager.get_train(slot_val_end)
df_gws_val = df_gws_val[df_gws_val.index > df_gws_train.index[-1]]

print(f"Train dates are from \"{df_gws_train.index[0]}\" to \"{df_gws_train.index[-1]}\", {len(df_gws_train)} elements")
print(f"Val dates are from \"{df_gws_val.index[0]}\" to \"{df_gws_val.index[-1]}\", {len(df_gws_val)} elements")

standardiser = util.Standardiser2(df_gws_train, min_max=False)
train_stand,val_stand = standardiser(df_gws_train,df_gws_val)

w_size = window.WindowGenerator(input_width=7, label_width=out_steps, shift=out_steps+shift, handle_nan=True, bonus_day_columns=["holiday"],
                        label_columns=['margin'], train_df=train_stand, val_df=val_stand)
train_count=sum(1 for _ in w_size.train.unbatch())
val_count=sum(1 for _ in w_size.val.unbatch())
print(f"""Approx. (assuming input_width=7) training datapoins {train_count}, val datapoints {val_count}, ratio {train_count/(train_count+val_count)}""")

#######################################################################
### GRID SEARCH                                                     ###
#######################################################################
gen = [{"dropout": d, "n": n, "activation": a, "input_width": i, "p": p, "m": m} 
  for d in obj["grid_dropout"]
  for n in obj["grid_n"]
  for a in obj["grid_activation"] 
  for i in obj["grid_input_width_param"]
  for p in obj["grid_p_param"]
  for m in obj["grid_m_param"]
  ]

# check paths
config_file_path = sys.argv[1]
config_file_name = os.path.split(config_file_path)[1]
tmp = os.path.abspath(config_file_path)
config_file_root = os.path.split(tmp)[0]
# create folder for outputs
folder = config_file_name.split('.')[0] + datetime.date.today()
print(f"Creating output directory {folder}")
os.mkdir(folder)
chosen_model = obj["model"]
print(f"Running grid search for model [{chosen_model}]")
# copy config file to output dir
target_folder = os.path.join(config_file_root, folder)
os.popen(f'cp {config_file_path} {os.path.join(target_folder, config_file_path)}') 

prog = tqdm(range(len(gen)))
for i in prog:
    # reset seed
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    params = gen[i]

    bonus_day_cols=['holiday', 'SLP', 'tavg_mean', 'tavg_std', 'tsun_mean', 'tsun_std', 'prcp_mean', 'prcp_std', 'pres_mean', 'pres_std']
    target="margin"

    w2 = window.WindowGenerator(input_width=params["input_width"], label_width=out_steps, shift=out_steps+shift, handle_nan=True, bonus_day_columns=bonus_day_cols,
                        label_columns=[target], train_df=train_stand, val_df=val_stand)

    p = min(params["p"], params["input_width"])
    m = min(params["m"], params["input_width"])
    if p > m:
      m = p

    models = {
            "nnar": ModelWrapper(NNARNaive, id=f"HIST{params['input_width']}", p=p, m=m, dropout=params["dropout"], batch_norm=params["norm"], activation=params["activation"], n=params["n"], out_steps=out_steps, out_features=out_features),
            "dense_lstm_dense": ModelWrapper(DenseLstmDense, id=f"HIST{params['input_width']}", p=p, m=m, dropout=params["dropout"], batch_norm=params["norm"], activation=params["activation"], n=params["n"], out_steps=out_steps, out_features=out_features),
    }

    labels = np.concatenate([y for x, y in w2.val], axis=0)
    labels_denorm = (labels * standardiser.divide[target]) + standardiser.substract[target]

    # choose model based on params
    model_wrapper = models[chosen_model]
    if model_wrapper.name in results:
        continue
    # train
    history = compile_and_fit(verbose=0, model=model_wrapper.model, 
        train=w2.train, val=w2.val, early_watch='val_loss', epochs=epochs, patience=patience)
    # predict
    pred = model_wrapper.model.predict(w2.val)
    pred_denorm = (pred * standardiser.divide[target]) + standardiser.substract[target]
    # mae
    mae_one_day = tf.keras.losses.MeanAbsoluteError()(labels_denorm[:,:1,:], pred_denorm[:,:1,:])
    mae_remaining = tf.keras.losses.MeanAbsoluteError()(labels_denorm[:,1:,:], pred_denorm[:,1:,:])
    mae_overall = tf.keras.losses.MeanAbsoluteError()(labels_denorm, pred_denorm)

    desc = f"last mae_overall {mae_overall} [{model_wrapper.name[:5]}...]"
    prog.set_description(desc)

    # save
    results[f"{model_wrapper.name}"] = {
            "predictions": pred_denorm, "history": history.history,
            "mae_one_day": mae_one_day, "mae_remaining": mae_remaining,
            "mae_overall": mae_overall, **params
            }

    with open(os.path.join(folder, "output.pickle"), 'wb') as f:
        pickle.dump(results, f)
    del models