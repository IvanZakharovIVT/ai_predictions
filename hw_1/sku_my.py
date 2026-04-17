from hw_1.sku_base import SkuBase
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb


class SKULiner(SkuBase):
    regression_name = 'Линейная регрессия'
    def _fit_model(self, X_train, y_train, X_test):
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)
        return y_pred_test


class SKUForest(SkuBase):
    regression_name = 'Случайный лес'
    def _fit_model(self, X_train, y_train, X_test):
        rf_model = RandomForestRegressor(
            n_estimators=100,  # Количество деревьев
            random_state=42,
            n_jobs=-1  # Использовать все ядра процессора
        )
        rf_model.fit(X_train, y_train)

        # Оценка на тесте
        y_pred_test = rf_model.predict(X_test)
        return y_pred_test


class SKUXGB(SkuBase):
    regression_name = 'XGBoost'

    def _fit_model(self, X_train, y_train, X_test):
        model = xgb.XGBRegressor(
            n_estimators=100,  # Количество деревьев
            learning_rate=0.1,  # Шаг обучения
            max_depth=5,  # Глубина дерева (не слишком глубоко, чтобы не переобучиться)
            objective='reg:squarederror',
            random_state=42,
            verbosity=0  # Убрать лишние логи
        )

        model.fit(X_train, y_train)

        # Оценка на тесте
        y_pred_test = model.predict(X_test)
        return y_pred_test
