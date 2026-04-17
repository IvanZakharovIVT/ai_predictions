from abc import abstractmethod

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split


class TooSmallDatasetError(Exception):
    ...


class SkuBase:
    regression_name = 'Название регрессии'
    file_name = 'Данные_по_Продажам.xlsx'
    mape_scores = []
    forecast_horizon = 8

    def run(self):
        print(f'================{self.regression_name}==============')
        df = self._init_dataframe()
        groups = df.groupby(['Контрагент Код', 'Номенклатура Код'])

        for name, group in groups:
            client_code, sku_code = name

            if len(group) < 10:
                continue
            try:
                X_train, X_test, y_train, y_test = self._init_train_test_set(group)
            except TooSmallDatasetError:
                continue
            y_pred_test = self._fit_model(X_train, y_train, X_test)
            self._show_predict(client_code, sku_code, y_test, y_pred_test)
        self._show_final_result()

    def _init_dataframe(self):
        # 1. Базовый датафрейм продаж (как раньше)
        df_sales = pd.read_excel('Данные_по_Продажам.xlsx')
        df_sales['Дата'] = pd.to_datetime(df_sales['Дата'], dayfirst=True)
        df_sales['Year_Week'] = df_sales['Дата'].dt.to_period('W')

        df_weekly = df_sales.groupby(
            ['Контрагент Код', 'Номенклатура Код', 'Year_Week']
        )['Количество'].sum().reset_index()

        # 2. Добавляем среднюю цену за неделю
        df_price_weekly = df_sales.groupby(
            ['Контрагент Код', 'Номенклатура Код', 'Year_Week']
        )['ЦенаРеализации'].mean().reset_index()
        df_price_weekly.rename(columns={'ЦенаРеализации': 'СрЦенаЗаНеделю'}, inplace=True)

        df_weekly = df_weekly.merge(df_price_weekly, on=['Контрагент Код', 'Номенклатура Код', 'Year_Week'])

        # 3. Добавляем акции
        df_promo = pd.read_excel('Данные_по_Акциям.xlsx')
        df_promo['ДатаНачала'] = pd.to_datetime(df_promo['ДатаНачала'], dayfirst=True)
        df_promo['ДатаКонца'] = pd.to_datetime(df_promo['ДатаКонца'], dayfirst=True)

        # Для каждой недели проверяем, попадает ли она в период акции
        def is_promo(row):
            week_start = row['Year_Week'].start_time
            week_end = week_start + pd.Timedelta(days=6)
            promo_mask = (df_promo['Контрагент Код'] == row['Контрагент Код']) & \
                         (df_promo['Номенклатура Код'] == row['Номенклатура Код']) & \
                         (df_promo['ДатаНачала'] <= week_end) & \
                         (df_promo['ДатаКонца'] >= week_start)
            return int(promo_mask.any())

        df_weekly['is_promo'] = df_weekly.apply(is_promo, axis=1)

        # 4. Добавляем остатки (ТЗ)
        df_stock = pd.read_excel('Данные_по_ТЗ.xlsx')
        df_stock['Дата'] = pd.to_datetime(df_stock['Дата'], dayfirst=True)
        df_stock['Year_Week'] = df_stock['Дата'].dt.to_period('W')
        df_stock_weekly = df_stock.groupby(['Номенклатура Код', 'Year_Week'])['СрДнОстаток|Сумма'].mean().reset_index()

        df_weekly = df_weekly.merge(df_stock_weekly, on=['Номенклатура Код', 'Year_Week'], how='left')
        df_weekly['СрДнОстаток|Сумма'].fillna(df_weekly['СрДнОстаток|Сумма'].median(), inplace=True)

        # 5. Добавляем категориальные признаки из справочников
        df_sku_info = pd.read_excel('Справочник_товаров.xlsx')
        df_sku_info.rename(columns={'Код': 'Номенклатура Код'}, inplace=True)
        df_weekly = df_weekly.merge(df_sku_info[['Номенклатура Код', 'ТоварнаяГруппа', 'ТорговаяМарка']],
                                    on='Номенклатура Код', how='left')

        df_tt_info = pd.read_excel('Справочник_ТТ.xlsx')
        df_tt_info.rename(columns={'Код': 'Контрагент Код'}, inplace=True)
        df_weekly = df_weekly.merge(df_tt_info[['Контрагент Код', 'КаналСбыта', 'ТорговаяСеть']],
                                    on='Контрагент Код', how='left')

        # 6. One-hot encoding для категорий
        df_weekly = pd.get_dummies(df_weekly, columns=['ТоварнаяГруппа', 'КаналСбыта'], drop_first=True)

        # 7. Временные признаки
        df_weekly['Месяц'] = df_weekly['Year_Week'].dt.start_time.dt.month
        df_weekly['Неделя_в_году'] = df_weekly['Year_Week'].dt.start_time.dt.isocalendar().week

        # 8. Сортируем и добавляем индекс
        df_weekly = df_weekly.sort_values(['Контрагент Код', 'Номенклатура Код', 'Year_Week'])
        df_weekly['Week_Index'] = df_weekly.groupby(['Контрагент Код', 'Номенклатура Код']).cumcount()

        return df_weekly

    # def _init_dataframe(self):
    #     df = pd.read_excel(self.file_name)
    #     df['Дата'] = pd.to_datetime(df['Дата'], dayfirst=True)
    #     df['Year_Week'] = df['Дата'].dt.to_period('W')
    #     df_weekly = df.groupby(['Контрагент Код', 'Номенклатура Код', 'Year_Week'])['Количество'].sum().reset_index()
    #
    #     df_weekly['Date_Start'] = df_weekly['Year_Week'].dt.start_time
    #     df_weekly = df_weekly.sort_values(by=['Контрагент Код', 'Номенклатура Код', 'Year_Week'])
    #
    #     df_weekly['Week_Index'] = df_weekly.groupby(['Контрагент Код', 'Номенклатура Код']).cumcount()
    #     return df_weekly

    def _init_train_test_set(self, group):
        feature_columns = [
            'Week_Index',  # базовый временной тренд
            'СрЦенаЗаНеделю',  # средняя цена за неделю
            'is_promo',  # была ли акция
            'СрДнОстаток',  # средний дневной остаток
            'Месяц',  # месяц (для сезонности)
            'Неделя_в_году',  # номер недели в году
        ]

        # Добавляем one-hot закодированные колонки (если они есть)
        for col in group.columns:
            if col.startswith(('ТоварнаяГруппа_', 'КаналСбыта_', 'ТорговаяМарка_')):
                feature_columns.append(col)

        # Проверяем, что все колонки существуют
        existing_columns = [col for col in feature_columns if col in group.columns]

        if len(existing_columns) < 2:  # Хотя бы 2 признака
            print(f"Предупреждение: найдено только {len(existing_columns)} признаков")
            # Если нет дополнительных признаков, используем только Week_Index
            existing_columns = ['Week_Index']

        X = group[existing_columns].values
        y = group['Количество'].values

        if len(X) <= 8:
            raise TooSmallDatasetError

        return train_test_split(
            X, y,
            test_size=self.forecast_horizon,
            # shuffle=False
        )

    @abstractmethod
    def _fit_model(self, X_train, y_train, X_test):
        raise NotImplementedError()

    def _show_predict(self, client_code, sku_code, y_test, y_pred_test):
        mask = y_test != 0
        if np.sum(mask) > 0:
            mape = mean_absolute_percentage_error(y_test[mask], y_pred_test[mask])
            self.mape_scores.append(mape)
            print(f"Клиент: {client_code}, SKU: {sku_code}, MAPE на тесте: {mape:.2%}")

    def _show_final_result(self):
        if self.mape_scores:
            avg_mape = np.mean(self.mape_scores)
            print(f"\nСредний MAPE по всем группам: {avg_mape:.2%}")
            if avg_mape <= 0.25:
                print("MAPE <= 25%")
            else:
                print(
                    "MAPE > 25%.")


class SKUAIBase(SkuBase):
    predictions_list = []
    forecast_horizon = 8

    def run(self):
        super().run()
        if self.predictions_list:
            df_predictions = pd.DataFrame(self.predictions_list)
            print("\n--- Прогноз на 2 месяца (8 недель) ---")
            print(df_predictions.head(10))

            # Сохранение в Excel
            output_file = 'forecast_results.xlsx'
            df_predictions.to_excel(output_file, index=False)
            print(f"\nПрогноз сохранен в файл: {output_file}")

            # Оценка общего MAPE
            if self.mape_scores:
                avg_mape = np.mean(self.mape_scores)
                print(f"\nСредний MAPE по всем группам: {avg_mape:.2%}")
                if avg_mape <= 0.25:
                    print("Условие задачи выполнено: MAPE <= 25%")
                else:
                    print(
                        "Внимание: MAPE превышает 25%. Возможно, требуется более сложная модель (например, с учетом сезонности).")
        else:
            print("Не удалось построить прогноз.")
