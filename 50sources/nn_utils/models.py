import tensorflow as tf

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

class EncoderDecoder(BaseModel):
    def __init__(self, ds, cell=tf.keras.layers.LSTMCell, p=1, m=7, activation="relu", **kwargs):
        super().__init__(batch_norm_count=0, dropout_count=2, **kwargs)

        self.p = p
        self.m = m
        self.feature_size = next(iter(ds))[0]['a'].shape[-1]

        ### TRAINABLE LAYERS ###
        self.dense_first_day = tf.keras.layers.Dense(self.n, activation=activation)
        self.dense_combine = tf.keras.layers.Dense(self.feature_size, activation=activation)
        self.decoder = cell(self.n)
        self.cell = cell(self.n)
        self.encoder = tf.keras.layers.RNN(self.cell, return_state=True)

        ### MISC ###
        self.flatten = tf.keras.layers.Flatten()

        ### OUTPUT ### 
        self.dense_out = tf.keras.layers.Dense(self.out_features, kernel_initializer="zeros")
            

    def call(self, inputs, training=None):
        # x = past data
        past_input = inputs["a"]
        # b = future data (less features than x)
        b = inputs["b"]

        # first decoder input is last day
        last_day = past_input[:,-1,:]
        # p and m
        x = past_input[:,-self.p:,:]
        if self.m > self.p:
          seasonal_day = past_input[:,-self.m,:]
          seasonal_day = tf.expand_dims(seasonal_day, axis=1)
          x = tf.concat([seasonal_day,x], axis=1)

        # combine future data with past data
        b = self.dense_combine(b)
        x = tf.concat([x,b], axis=1)

        # encode past data
        _,*state = self.encoder(x)

        predictions = tf.TensorArray(tf.float32, size=self.out_steps, element_shape=(None,self.n))

        # first input to decoder ist last day + last hidden state
        x = last_day
        # let x have a dimensionality of n to make it consistent accross decoder
        x = self.dense_first_day(x)
        if self.has_dropout:
          x = self.dropout_1(x, training=training)
        for n in tf.range(0, self.out_steps):
          # Execute one decoder step
          x, state = self.decoder(x, states=state, training=training)
          if self.has_dropout:
            x = self.dropout_2(x, training=training)
          # dense to make dim consistent
          #x = self.dense_1(x)
          # workaround to set a known shape to this tensor (batch dim is usually ?)
          #dims = tf.shape(x)
          #x = tf.reshape(x, dims)
          predictions = predictions.write(n, x)


        x = predictions.stack()
        # (time, batch, features) => (batch, time, features)
        x = tf.transpose(x, [1, 0, 2])

        #x = self.flatten(x)
        #b = self.flatten(b)
        #b = b[:,-self.out_steps:,:]
        #x = tf.concat([x,b], axis=-1)

        #x = self.dense_combine(x)

        # (batch, future_input, b_features) => (batch, out_steps * x_features)
        #b = self.dense_b(self.flatten(b))
        #if self.has_dropout:
        #  b = self.dropout_1(b)
        

        #x = tf.add(x,b)

        # (batch, outsteps, x_features) => (batch, outsteps, out_features)
        x = self.dense_out(x)

        #x = tf.reshape(x, [tf.shape(x)[0], self.out_steps, self.out_features])

        return x

class SimpleEncoder(BaseModel):
    def __init__(self, cell=tf.keras.layers.LSTMCell, activation="selu", **kwargs):
        super().__init__(batch_norm_count=1, dropout_count=1, **kwargs)

        ### TRAINABLE LAYERS ###
        #self.dense_1 = tf.keras.layers.Dense(self.n, activation=activation)
        self.dense_b = tf.keras.layers.Dense(self.n, activation=activation)
        self.cell = cell(self.n, activation=activation)
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

        # encode past data
        x = self.encoder(x)

        # combine encoder output and future data
        # flatten b
        b = self.flatten(b)
        b = self.dense_b(b)
        if self.batch_norm:
            b = self.batch_norm_1(b, training)
        if self.has_dropout:
            b = self.dropout_1(b, training)
        # add together
        #x = tf.add(x,b)
        x = tf.concat()

        x = self.dense_out(x)

        # reshape to match output
        x = tf.reshape(x, [tf.shape(x)[0], self.out_steps, self.out_features])

        return x

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
            lag_day = tf.expand_dims(x[:,-self.m,:], axis=1)
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

class NNARNoFuture(BaseModel):
    def __init__(self, activation="selu", m=1, p=1, **kwargs):
        super().__init__(dropout_count=1, batch_norm_count=1, **kwargs)

        self.m = m
        self.p = p

        ### TRAINABLE LAYERS ###
        self.dense_1 = tf.keras.layers.Dense(self.n, activation=activation)
        #self.dense_b = tf.keras.layers.Dense(self.n, activation=activation)

        ### MISC ###
        self.flatten = tf.keras.layers.Flatten()

        ### OUTPUT ### 
        self.dense_out = tf.keras.layers.Dense(self.out_steps*self.out_features, kernel_initializer = 'zeros')
            

    def call(self, inputs, training=None):
        # x = past data
        x = inputs["a"]
        # b = future data (less features than x)
        #b = inputs["b"]

        lags = self.flatten(x[:,-self.p:,:])

        if self.m > self.p:
          s_lags = self.flatten(x[:,-self.m,:])
          lags = tf.concat([s_lags,lags], axis=1)
        
        # combine
        x = lags

        #b = self.flatten(b)
        #b = self.dense_b(b)
        #if self.batch_norm:
        #  b = self.batch_norm_2(b, training=training)
        #if self.has_dropout:
        #  b = self.dropout_2(b, training=training)
        

        # hidden layer
        x = self.dense_1(x)
        if self.batch_norm:
          x = self.batch_norm_1(x, training=training)
        if self.has_dropout:
          x = self.dropout_1(x, training=training)

        #x = tf.concat([x,b], axis=1)

        x = self.dense_out(x)

        # reshape to match output
        x = tf.reshape(x, [tf.shape(x)[0], self.out_steps, self.out_features])

        return x

class NNARNaive(BaseModel):
    def __init__(self, activation="selu", m=1, p=1, **kwargs):
        super().__init__(dropout_count=1, batch_norm_count=1, **kwargs)

        self.m = m
        self.p = p

        ### TRAINABLE LAYERS ###
        self.dense_1 = tf.keras.layers.Dense(self.n, activation=activation)

        ### MISC ###
        self.flatten = tf.keras.layers.Flatten()

        ### OUTPUT ### 
        self.dense_out = tf.keras.layers.Dense(self.out_steps*self.out_features, kernel_initializer = 'zeros')
            

    def call(self, inputs, training=None):
        # x = past data
        x = inputs["a"]
        # b = future data (less features than x)
        b = inputs["b"]

        lags = self.flatten(x[:,-self.p:,:])

        if self.m > self.p:
          s_lags = self.flatten(x[:,-self.m,:])
          lags = tf.concat([s_lags,lags], axis=1)
        
        # combine
        x = lags

        b = self.flatten(b)
        x = tf.concat([x,b], axis=1)

        # hidden layer
        x = self.dense_1(x)
        if self.batch_norm:
          x = self.batch_norm_1(x)
        if self.has_dropout:
          x = self.dropout_1(x, training=training)

        x = self.dense_out(x)

        # reshape to match output
        x = tf.reshape(x, [tf.shape(x)[0], self.out_steps, self.out_features])

        return x

def create_nnar(ds, n, p=1, m=6, activation="relu", dropout=0.0, out_steps=1, out_features=1):
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
  x = tf.keras.layers.Dense(out_steps*out_features, kernel_initializer="zeros")(x)
  x = tf.keras.layers.Reshape((out_steps,out_features))(x)

  return tf.keras.Model(inputs=[past_input,future_input], outputs=x, name="nnar_with_future")

def create_dense_lstm_dense(ds, n, p=1, m=6, out_steps=1, out_features=1, dropout=0.0, activation="relu", **kwargs):
  past_input = tf.keras.layers.Input(next(iter(ds))[0]['a'].shape[-2:], name="a")
  future_input = tf.keras.layers.Input(next(iter(ds))[0]['b'].shape[-2:], name="b")

  x = tf.keras.layers.Lambda(lambda x: x[:,-p:,:])(past_input)
  if m > p:
    seasonal_day = tf.keras.layers.Lambda(lambda x: x[:, -m, :])(past_input)
    seasonal_day = tf.expand_dims(seasonal_day, axis=1)
    x = tf.concat([seasonal_day,x], axis=1)

  x = tf.keras.layers.Dense(n, activation=activation)(x)

  if dropout > 0.0:
    x = tf.keras.layers.Dropout(dropout)(x)
  x = tf.keras.layers.LSTM(min(32,n))(x)
  if dropout > 0.0:
    x = tf.keras.layers.Dropout(dropout)(x)

  x = tf.keras.layers.Flatten()(x)
  x2 = tf.keras.layers.Flatten()(future_input)
  x = tf.keras.layers.concatenate([x,x2], axis=-1)
  x = tf.keras.layers.Dense(n, activation=activation)(x)

  x = tf.keras.layers.Dense(out_steps*out_features, kernel_initializer="zeros")(x)
  x = tf.keras.layers.Reshape((out_steps,out_features))(x)

  return tf.keras.Model(inputs=[past_input,future_input], outputs=x, name="nnar_with_future")