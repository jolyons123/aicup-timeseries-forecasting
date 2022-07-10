import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd


class WindowGenerator():

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    @property
    def val_example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_val_example', None)
        if result is None:
            # No example batch was found, so get one from the `.val` dataset
            result = next(iter(self.val))
            # And cache it for next time
            self._val_example = result
        return result

    @property
    def test_example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_test_example', None)
        if result is None:
            # No example batch was found, so get one from the `.val` dataset
            result = next(iter(self.test))
            # And cache it for next time
            self._test_example = result
        return result

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)

        if self.handle_nan:
            ds = timeseries_dataset_from_array(
                data=data,
                window_size=self.total_window_size,
                batch_size=self.batch_size,
                shuffle=self.shuffle_ds,
                seed=self.seed)
        else:
            ds = tf.keras.utils.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=self.total_window_size,
                sequence_stride=self.stride,
                shuffle=self.shuffle_ds,
                seed=self.seed,
                batch_size=self.batch_size)

        ds = ds.map(self.split_window)

        return ds

    def __init__(self, input_width, label_width, shift,
                 train_df=None, val_df=None, test_df=None,
                 label_columns=None, bonus_day_columns=None, keep_past_cols=None, batch_size=32, seed=0, handle_nan=False, shuffle_ds=False, stride=1):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size=batch_size
        self.shuffle_ds=shuffle_ds
        self.handle_nan=handle_nan
        self.seed=seed
        self.stride=stride
        self.keep_past_cols=keep_past_cols

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}
        self.bonus_day_columns = bonus_day_columns

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        # bonus day slice, bonus day start is actually slice which starts at input width and ends None
        self.bonus_slice = slice(self.input_width, None)
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Input slice: {self.input_slice}',
            f'Label indices: {self.label_indices}',
            f'Label slice: {self.labels_slice}',
            f'Label column name(s): {self.label_columns}',
            f'Bonus indices: {self.bonus_indices}',
            f'Bonus slices: {self.bonus_slice}',
            f'Bonus column indices: {self.bonus_column_indices}'])

    def split_window(self, features):
        """features: tensor with dimension: (batch_size,inputs,columns)"""
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        extra = features[:, self.bonus_slice, :]

        if self.keep_past_cols is not None:
            inputs = tf.stack(
                [inputs[:, :, self.column_indices[name]] for name in self.keep_past_cols],
                axis=-1)

        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        if self.bonus_day_columns is not None:
            extra = tf.stack(
                [extra[:, :, self.column_indices[name]] for name in self.bonus_day_columns],
                axis=-1
            )

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        extra.set_shape([None, self.shift, None])

        return {"a": inputs, "b": extra}, labels

    def plot(self, model=None, plot_col='T (degC)', max_subplots=3, example_data="train"):
        if example_data == "val":
            inputs, labels = self.val_example
        elif example_data == "test":
            inputs, labels = self.test_example
        else:
            inputs, labels = self.example

        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        plt.figure(figsize=(12, 2*max_n))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    def plot_training(self, history=None, yaxes_acc=['mean_absolute_percentage_error','val_mean_absolute_percentage_error']):
        # summarize history for accuracy
        plt.plot(history.history[yaxes_acc[0]])
        plt.plot(history.history[yaxes_acc[1]])
        plt.title('model accuracy')
        plt.ylabel('mape')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

def timeseries_dataset_from_array(
    data,
    window_size,
    batch_size=32,
    shuffle=None,
    seed=0):

    # get nona indices
    nona = np.argwhere([not any(features) for features in np.isnan(data)]).flatten()

    # split between missing days
    split_indexes=np.argwhere(np.diff(nona)>1).flatten()
    candidates = np.split(nona, split_indexes+1)

    # filter candidates to only include big enough timeseries
    candidates = [x for x in candidates if len(x) > window_size]

    # for each candidate generate window_size slices
    indices_array = []
    for x in candidates:
        end_x = len(x)

        # maximum number of datapoints in this particular timeseries
        max = end_x - window_size + 1

        # dtype optimierung
        if max < 2147483647:
            index_dtype = 'int32'
        else:
            index_dtype = 'int64'

        # Convert index to dtype
        casted_x = np.array(x, dtype=index_dtype)
        casted_window_size = tf.cast(window_size, dtype=index_dtype)
        casted_max = np.arange(0, max, 1, dtype=index_dtype)

        positions_ds = tf.data.Dataset.from_tensors(casted_x).repeat()

        # For each initial window position, generates indices of the window elements
        indices = tf.data.Dataset.zip(
            (tf.data.Dataset.range(len(casted_max)), positions_ds)).map(
                lambda i, positions: tf.range(  # pylint: disable=g-long-lambda
                    positions[i],
                    positions[i] + casted_window_size),
                num_parallel_calls=tf.data.AUTOTUNE)

        indices_array.append(indices)

    # combine candidate indices
    combined_indices_ds = indices_array[0]
    for i in range(1, len(indices_array)):
        combined_indices_ds = combined_indices_ds.concatenate(indices_array[i])

    # extract real data out of indices
    dataset = tf.data.Dataset.from_tensors(data)
    dataset = tf.data.Dataset.zip((dataset.repeat(), combined_indices_ds)).map(
        lambda steps, inds: tf.gather(steps, inds),  # pylint: disable=unnecessary-lambda
        num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    if batch_size is not None:
        if shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
        dataset = dataset.batch(batch_size)

    return dataset


def set_selected(df, df_org, which):
    df_org.loc[df.index, 'selected_' + which] = 1


def rolling_window(df, lower, upper):
    if 0 <= lower < upper:
        return df[lower:upper]
    elif lower < 0 < upper:
        return pd.concat([df[lower:], df[:upper]])
    elif lower <= upper < 0:
        return df[lower:upper]
    elif lower < upper <= 0:
        return df[lower:]


def adjust_weekday(window, target_window):
    #print(len(window.index))
    i_day = window.index[len(window.index) - 1].weekday()
    t_day = target_window.index[len(target_window.index) - 1].weekday()
    i2_day = window.index[0].weekday()
    t2_day = target_window.index[0].weekday()
    return (t_day == i_day) and (t2_day == i2_day)


def select_windows(i, df, horizon, history, mid_term=0, long_term=0):
    rw_mid = None
    rw_long = None

    h = i
    hh = horizon + i
    rw_today = rolling_window(df, h, hh)
    # set_selected(rw_today, df, 'today')

    h = (i - history)
    hh = (history + i - history)
    rw_hist = rolling_window(df, h, hh)
    # set_selected(rw_hist, df, 'history')

    if mid_term > 0:
        h = (i - mid_term)
        hh = (history + i - mid_term)
        rw_mid = rolling_window(df, h, hh)
        #while adjust_weekday(rw_mid, rw_today) is not True:
        #    h += 1
        #    hh += 1
        #    rw_mid = rolling_window(df, h, hh)
        #set_selected(rw_mid, df, 'midterm')

    if long_term > 0:
        h = (i - long_term - 1)
        hh = (history + i - long_term - 1)
        rw_long = rolling_window(df, h, hh)
        while adjust_weekday(rw_long, rw_today) is not True:
            h += 1
            hh += 1
            rw_long = rolling_window(df, h, hh)
        # set_selected(rw_long, df, 'longterm')

    return rw_today, rw_hist, rw_mid, rw_long


def window_gen(df, horizon, start, end, history=0, mid_term=0, long_term=0, input_cols=[], history_cols=[],
               mid_term_cols=[], long_term_cols=[], target_col=['margin'], use_stride=False, batch_size=256):
    inputs = {'today': []}
    labels = []

    i = start
    while horizon + i <= end:
        rw_today, rw_hist, rw_mid, rw_long = select_windows(i, df[:end].copy(), horizon, history, mid_term,
                                                            long_term)

        inputs['today'].append(tf.convert_to_tensor(rw_today[input_cols], dtype=tf.float32))
        if history > 0:
            if 'history' not in inputs:
                inputs['history'] = []
            inputs['history'].append(tf.convert_to_tensor(rw_hist[history_cols], dtype=tf.float32))
        if mid_term > 0:
            if 'mid_term' not in inputs:
                inputs['mid_term'] = []
            inputs['mid_term'].append(tf.convert_to_tensor(rw_mid[mid_term_cols], dtype=tf.float32))
        if long_term > 0:
            if 'long_term' not in inputs:
                inputs['long_term'] = []
            inputs['long_term'].append(tf.convert_to_tensor(rw_long[long_term_cols], dtype=tf.float32))

        labels.append(tf.convert_to_tensor(rw_today[target_col], dtype=tf.float32))

        if use_stride is True:
            i += horizon
        else:
            i += 1

    return tf.data.Dataset.from_tensor_slices((inputs, labels))
