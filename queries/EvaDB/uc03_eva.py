import joblib
import pandas as pd
from evadb.catalog.catalog_type import NdArrayType

from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import setup, forward
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import datetime
import warnings
from pathlib import Path

import numpy as np
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing

main = __import__("__main__")
class UseCase03Model(object):
  def __init__(self, use_store=False, use_department=True):
      if not use_store and not use_department:
          raise ValueError(f'use_store = {use_store}, use_department = {use_department}: at least one must be True')

      self._use_store = use_store
      self._use_department = use_department
      self._models = {}
      self._min = {}
      self._max = {}

  def _get_key(self, store, department):
      if self._use_store and self._use_department:
          key = (store, department)
      elif self._use_store:
          key = store
      else:
          key = department

      return key

  def store_model(self, store: int, department: int, model, ts_min, ts_max):
      key = self._get_key(store, department)
      self._models[key] = model
      self._min[key] = ts_min
      self._max[key] = ts_max

  def get_model(self, store: int, department: int):
      key = self._get_key(store, department)
      model = self._models[key]
      ts_min = self._min[key]
      ts_max = self._max[key]
      return model, ts_min, ts_max
main.UseCase03Model = UseCase03Model

class uc03(AbstractFunction):
    @property
    def name(self) -> str:
        return "uc03"

    def setup(self):
        self.predict_col = self.name
        model_file_name = f"/home/model/{self.name}/{self.name}.python.model"
        self.model = joblib.load(model_file_name)
        #main = __import__("__main__")
        #main.UseCase03Model = UseCase03Model

    @forward(input_signatures=[PandasDataframe(columns=["store", "department"],
                                               column_types=[NdArrayType.INT64, NdArrayType.STR],
                                               column_shapes=[(None, ), (None, )])],
             output_signatures=[PandasDataframe(columns=["uc03"],
                                                column_types=[NdArrayType.STR],
                                                column_shapes=[(None, )])])
    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        print(len(frames))
        forecasts = []
        for index, row in frames.iterrows():
            store = row.store
            dept = row.department
            periods = 52
            try:
                current_model, ts_min, ts_max = self.model.get_model(store, dept)
            except KeyError:
                continue
            # disable warnings that non-date index is returned from forecast
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ValueWarning)
                forecast = current_model.forecast(periods)
                forecast = np.clip(forecast, a_min=0.0, a_max=None)  # replace negative forecasts
            start = pd.date_range(ts_max, periods=2)[1]
            forecast_idx = pd.date_range(start, periods=periods, freq='W-FRI')
            forecasts.append(
                str({'store': store, 'department': dept, 'date': forecast_idx, 'weekly_sales': forecast})
            )
        predict_df = pd.DataFrame(forecasts, columns=[self.predict_col])
        return predict_df
