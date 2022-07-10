import inspect
import json
import os
import sys

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)

import util
import metrics


class GasModelManager:
    def __init__(self, rlm_imputation=False, tsfresh_features=False, svm_imputation=False, target='margin'):

        sub_df = util.read_gas_submission()

        self.target=target

        self.df_gesamt = util.create_gas_with_na()

        if rlm_imputation is True:
            rlm_impu = util.read_rlm_imputation()
            rlm_impu['RLM'] = rlm_impu['yhat']
            self.df_gesamt = self.df_gesamt.combine_first(rlm_impu[['RLM']])

        self.ts_features_df = None
        self.tsfresh_features = tsfresh_features

        self.slots = []
        for i in range(int(len(sub_df) / 5)):
            k = i * 5
            self.slots.append(sub_df[k:(k + 5)].index.to_list())

        self.preds_sub = []
        self.preds_test = []

    def ts_feature_extract(self, df):
        _df=df.dropna()
        ts_fresh_input_x, ts_fresh_input_y = util.convert_gas_to_ts_fresh_input(_df,self.target)
        if self.ts_features_df is None:
            self.ts_features_df = util.tsfresh_feature_extraction(ts_fresh_input_x, ts_fresh_input_y)
        else:
            self.ts_features_df = util.tsfresh_feature_extraction(ts_fresh_input_x, ts_fresh_input_y,
                                                                  self.ts_features_df.columns.to_list())

        self.ts_features_df.index = _df.index

        print(self.ts_features_df.columns)
        return pd.merge(df,self.ts_features_df,how='outer',left_index=True,right_index=True)

    def get_train(self, slot, insert_preds=False):
        train_df = self.df_gesamt[self.df_gesamt.index < self.slots[slot][0]]
        print(f"Train dates are from \"{train_df.index[0]}\" to \"{train_df.index[-1]}\", {len(train_df)} elements")

        if self.tsfresh_features is True:
            train_df = self.ts_feature_extract(train_df)

        if insert_preds is True:
            for i in range(len(self.preds_sub)):
                train_df = pd.merge(train_df, self.preds_sub[i], left_index=True, right_index=True)
        return train_df

    def get_test(self, slot):
        test_df=self.df_gesamt[self.df_gesamt.index > self.slots[slot][len(self.slots[slot]) - 1]].dropna()

        if self.tsfresh_features is True:
            test_df = self.ts_feature_extract(test_df)

        return test_df

    def get_forecast(self, slot, history):
        tmp = self.df_gesamt[self.df_gesamt.index <= self.slots[slot][len(self.slots[slot]) - 1]]
        return tmp[len(tmp) - 1 - history - len(self.slots[slot]):]

    def get_slots(self):
        return self.slots

    def get_test_preds(self, slot):
        return self.preds_test[slot]

    def get_sub_preds(self, slot):
        return self.preds_sub[slot]

    """
    Braucht einen (denormalisierten) pandas dataframe folgender form:
        *      | targets | predictions
    date_index |  ...    |   ...
    """

    def set_preds_sub(self, slot, preds_df):
        self.preds_sub.insert(slot, preds_df)

    """
    Braucht einen (denormalisierten) pandas dataframe folgender form:
        *      | targets | predictions
    date_index |  ...    |   ...
    """

    def set_preds_test(self, slot, preds_df, print_summary=False):
        self.preds_test.insert(slot, preds_df)
        if print_summary is True:
            print(self._calc_metric_for_slot(slot))

    def plot_slot_summary(self, slot):
        print(self._calc_metric_for_slot(slot))
        df_preds = self.preds_test[slot]
        df_preds['residuals'] = df_preds['targets'] - df_preds['predictions']

        if len(df_preds) > 365:
            time_res_in_days = 365
        else:
            time_res_in_days = len(df_preds) - 1

        util.interactive_timeseries_plot(df_preds[['predictions', 'targets']], time_res_in_days=time_res_in_days,
                                         width=1000, height=600)
        util.interactive_timeseries_plot(df_preds[['residuals']], time_res_in_days=time_res_in_days,
                                         width=1000, height=600, y_axis=['residuals'])

    def plot_overall_summary(self):
        results_per_slot = self._get_metric_per_slot()
        print(self._calc_mean_metric(results_per_slot))
        results_df = pd.DataFrame(data=results_per_slot)
        results_df['mae'].plot()
        return results_per_slot

    def save_results(self, model_type, model_description, model_params):
        csv_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '../../30data/gasPrediciton/data/summary/summary.csv')
        save_results(csv_filepath, model_type, model_description, model_params, self._calc_mean_metric(
            self._get_metric_per_slot()))

    def _get_metric_per_slot(self):
        results = {}
        for i in range(len(self.preds_test)):
            res = self._calc_metric_for_slot(i)
            for key in res:
                if key not in results:
                    results[key] = []

                results[key].append(res[key])

        return results

    def _calc_mean_metric(self, metric_per_slot):
        results = {}

        for key in metric_per_slot:
            results[key] = np.mean(metric_per_slot[key]).item()

        return results

    def _calc_metric_for_slot(self, slot):
        preds_df = self.preds_test[slot]
        result = {}
        for key in metrics.METRICS_REGRESSION:
            result[key] = metrics.METRICS_REGRESSION[key](preds_df['targets'], preds_df['predictions'])

        return result


def save_results(csv_filepath, model_type, model_description, model_params, result_metrics):
    results = read_results_file(csv_filepath, result_metrics.keys())
    results = results.append(
        {
            "MODEL": model_type,
            "MODEL_DESCRIPTION": str(model_description),
            "MODEL_PARAMS": json.dumps(model_params)
                            ** result_metrics,
        },
        ignore_index=True,
    )

    results.to_csv(
        csv_filepath, sep=";",
    )
    del results


def read_results_file(csv_filepath, result_metrics):
    try:
        results = pd.read_csv(csv_filepath, sep=";", index_col=0)
    except IOError:
        results = pd.DataFrame(
            columns=[
                "MODEL",
                "MODEL_DESCRIPTION",
                "MODEL_PARAMS",
                *result_metrics
            ]
        )
    return results
