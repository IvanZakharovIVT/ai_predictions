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


class SKUForest(SkuBase):
    regression_name = 'Случайный лес'

    def _fit_model(self, X_train, y_train, X_test):
        rf_model = RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        y_train_log = np.log1p(y_train)
        rf_model.fit(X_train, y_train_log)
        y_pred_log = rf_model.predict(X_test)
        y_pred_test = np.expm1(y_pred_log)
        # Продажи не могут быть отрицательными
        return np.maximum(y_pred_test, 0)


class SKUXGB(SkuBase):
    regression_name = 'XGBoost'

    def _fit_model(self, X_train, y_train, X_test):
        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=6,
            # subsample=0.8,
            # colsample_bytree=0.7,
            # gamma=0.1,
            # reg_alpha=0.1,
            # reg_lambda=1.0,
            # min_child_weight=3,
            objective='reg:squarederror',
            random_state=42,
            verbosity=0,
            n_jobs=-1
        )
        y_train_log = np.log1p(y_train)
        model.fit(X_train, y_train_log)
        y_pred_log = model.predict(X_test)
        y_pred_test = np.expm1(y_pred_log)
        # Продажи не могут быть отрицательными
        return np.maximum(y_pred_test, 0)
