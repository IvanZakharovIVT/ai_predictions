import numpy as np
import pandas as pd


class DataframeInitiator:
    def init_dataframe(self):
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
        df_stock_weekly.rename(columns={'СрДнОстаток|Сумма': 'СрДнОстаток'}, inplace=True)

        df_weekly = df_weekly.merge(df_stock_weekly, on=['Номенклатура Код', 'Year_Week'], how='left')
        median_val = df_weekly['СрДнОстаток'].median()
        df_weekly['СрДнОстаток'] = df_weekly['СрДнОстаток'].fillna(median_val)

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

        df_weekly = self._add_time_series_features(df_weekly)

        numeric_cols = df_weekly.select_dtypes(include=[np.number]).columns
        df_weekly[numeric_cols] = df_weekly[numeric_cols].fillna(0)
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

    def _add_time_series_features(self, df):
        """Добавляет лаги и скользящие средние"""
        # Сортировка обязательна
        df = df.sort_values(['Контрагент Код', 'Номенклатура Код', 'Year_Week'])

        group_cols = ['Контрагент Код', 'Номенклатура Код']
        target_col = 'Количество'

        # Лаги
        for lag in [1, 2, 4, 8]:
            df[f'lag_{lag}'] = df.groupby(group_cols)[target_col].shift(lag)

        # Скользящие средние (с учетом лага 1, чтобы не было утечки)
        for window in [4, 8, 12]:
            df[f'roll_mean_{window}'] = df.groupby(group_cols)[target_col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

        # Скользящее стандартное отклонение (волатильность)
        df['roll_std_4'] = df.groupby(group_cols)[target_col].transform(
            lambda x: x.shift(1).rolling(4, min_periods=1).std()
        )

        # Заполняем NaN, возникшие из-за лагов, нулями или медианой
        df.fillna(0, inplace=True)

        return df