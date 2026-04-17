import numpy as np

from hw_1.sku_base import SkuBase
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb


class SKULiner(SkuBase):
    regression_name = 'Линейная регрессия'
    def _fit_model(self, X_train, y_train, X_test):
        model = LinearRegression()
        # Логарифмируем целевую переменную для стабилизации дисперсии
        y_train_log = np.log1p(y_train)
        model.fit(X_train, y_train_log)
        y_pred_log = model.predict(X_test)
        # Возвращаем обратно к исходному масштабу
        y_pred_test = np.expm1(y_pred_log)
        # Продажи не могут быть отрицательными
        return np.maximum(y_pred_test, 0)


    def _fit_model_on_full(self, X_full, y_full):
        model = LinearRegression()
        y_full_log = np.log1p(y_full)
        model.fit(X_full, y_full_log)
        return model  # Внимание: predict потом нужно делать через expm1


class SKUForest(SkuBase):
    regression_name = 'Случайный лес'

    def _fit_model(self, X_train, y_train, X_test):
        rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        y_train_log = np.log1p(y_train)
        rf_model.fit(X_train, y_train_log)
        y_pred_log = rf_model.predict(X_test)
        y_pred_test = np.expm1(y_pred_log)
        # Продажи не могут быть отрицательными
        return np.maximum(y_pred_test, 0)

    def _fit_model_on_full(self, X_full, y_full):
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        y_full_log = np.log1p(y_full)
        model.fit(X_full, y_full_log)
        return model


class SKUXGB(SkuBase):
    regression_name = 'XGBoost'

    def _fit_model(self, X_train, y_train, X_test):
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            objective='reg:squarederror',
            random_state=42,
            verbosity=0
        )
        y_train_log = np.log1p(y_train)
        model.fit(X_train, y_train_log)
        y_pred_log = model.predict(X_test)
        y_pred_test = np.expm1(y_pred_log)
        # Продажи не могут быть отрицательными
        return np.maximum(y_pred_test, 0)

    def _fit_model_on_full(self, X_full, y_full):
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            objective='reg:squarederror',
            random_state=42,
            verbosity=0
        )
        y_full_log = np.log1p(y_full)
        model.fit(X_full, y_full_log)
        return model
