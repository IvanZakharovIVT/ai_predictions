import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split


file_name = 'Данные_по_Продажам.xlsx'
df = pd.read_excel(file_name)
df['Дата'] = pd.to_datetime(df['Дата'], dayfirst=True)

df = df.sort_values(by=['Контрагент Код', 'Номенклатура Код', 'Дата'])
df['Week_Number'] = df.groupby(['Контрагент Код', 'Номенклатура Код']).cumcount()

groups = df.groupby(['Контрагент Код', 'Номенклатура Код'])

forecast_horizon = 8

predictions_list = []
mape_scores = []

for name, group in groups:
    client_code, sku_code = name

    if len(group) < 10:
        continue

    X = group[['Week_Number']].values
    y = group['Количество'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        train_size=forecast_horizon,
        # shuffle=False                        # НЕ ПЕРЕМЕШИВАТЬ!
    )

    # model = LinearRegression()
    # model.fit(X_train, y_train)

    # Прогноз на тестовой выборке (для оценки MAPE)
    # y_pred_test = model.predict(X_test)

    rf_model = RandomForestRegressor(
        n_estimators=100,  # Количество деревьев
        random_state=42,
        n_jobs=-1  # Использовать все ядра процессора
    )
    rf_model.fit(X_train, y_train)

    # Оценка на тесте
    y_pred_test = rf_model.predict(X_test)

    # Расчет MAPE на тесте
    # Избегаем деления на ноль
    mask = y_test != 0
    if np.sum(mask) > 0:
        mape = mean_absolute_percentage_error(y_test[mask], y_pred_test[mask])
        mape_scores.append(mape)
        print(f"Клиент: {client_code}, SKU: {sku_code}, MAPE на тесте: {mape:.2%}")