import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class DataframeInitiator:
    def init_dataframe(self):
        # 1. Базовый датафрейм + Цена + Акции + Остатки + Справочники
        df_weekly = self._add_base_features()

        # 2. Добавляем цену из прайса (Новое!)
        df_weekly = self._add_catalog_price(df_weekly)

        # 3. Временные признаки (Лаги, скользящие)
        df_weekly = self._add_time_series_features(df_weekly)

        # 4. Чистка числовых данных
        numeric_cols = df_weekly.select_dtypes(include=[np.number]).columns
        # Заполняем NaN нулями, но осторожно с Price_Ratio (там уже была обработка inf)
        df_weekly[numeric_cols] = df_weekly[numeric_cols].fillna(0)

        # Опционально: Удаление строк с нулевыми продажами из обучающей выборки
        # (если задача предсказать объем, а не факт покупки)
        # df_weekly = df_weekly[df_weekly['Количество'] > 0]

        return df_weekly

    def _add_catalog_price(self, df_weekly):
        try:
            df_prices = pd.read_excel('Данные_по_Ценам.xlsx')
            df_prices['Дата'] = pd.to_datetime(df_prices['Дата'], dayfirst=True)
            df_prices['Year_Week'] = df_prices['Дата'].dt.to_period('W')

            # Агрегация цены за неделю
            df_price_w = df_prices.groupby(['Номенклатура Код', 'Year_Week'])['Цена'].mean().reset_index()
            df_price_w.rename(columns={'Цена': 'Catalog_Price'}, inplace=True)

            df_weekly = df_weekly.merge(df_price_w, on=['Номенклатура Код', 'Year_Week'], how='left')

            # Заполнение пропусков цены (forward fill по товару)
            df_weekly['Catalog_Price'] = df_weekly.groupby('Номенклатура Код')['Catalog_Price'].transform(
                lambda x: x.ffill().bfill())

            # Отношение цены реализации к каталожной (мера скидки)
            # Добавляем eps, чтобы избежать деления на 0
            eps = 1e-3
            df_weekly['Price_Ratio'] = df_weekly['СрЦенаЗаНеделю'] / (df_weekly['Catalog_Price'] + eps)

            # Разница в цене (абсолютная)
            df_weekly['Price_Diff'] = df_weekly['СрЦенаЗаНеделю'] - df_weekly['Catalog_Price']

            q_low = df_weekly['Количество'].quantile(0.01)
            q_high = df_weekly['Количество'].quantile(0.99)

            df_weekly['Количество'] = df_weekly['Количество'].clip(q_low, q_high)

        except Exception as e:
            print(f"Warning: Could not load price data. Error: {e}")
            df_weekly['Catalog_Price'] = df_weekly['СрЦенаЗаНеделю']
            df_weekly['Price_Ratio'] = 1.0

        return df_weekly

    def _add_time_series_features(self, df):
        df = df.sort_values(['Контрагент Код', 'Номенклатура Код', 'Year_Week'])
        group_cols = ['Контрагент Код', 'Номенклатура Код']
        target_col = 'Количество'

        # ЛАГИ
        for lag in [1, 2, 4, 8, 12]:  # Добавил 12 (квартальный лаг)
            df[f'lag_{lag}'] = df.groupby(group_cols)[target_col].shift(lag)

        # СКОЛЬЗЯЩИЕ СРЕДНИЕ
        # min_periods увеличен до window//2 + 1 для стабильности
        for window in [4, 8, 12]:
            min_p = max(1, window // 2)
            df[f'roll_mean_{window}'] = df.groupby(group_cols)[target_col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=min_p).mean()
            )

        # Логарифмические признаки
        df['log_sales'] = np.log1p(df[target_col])
        for lag in [1, 2, 4]:
            df[f'log_lag_{lag}'] = df.groupby(group_cols)['log_sales'].shift(lag)

        for window in [4, 8]:
            min_p = max(1, window // 2)
            df[f'log_roll_mean_{window}'] = df.groupby(group_cols)['log_sales'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=min_p).mean()
            )

        # Стандартное отклонение (волатильность спроса)
        df['roll_std_4'] = df.groupby(group_cols)[target_col].transform(
            lambda x: x.shift(1).rolling(4, min_periods=2).std()
        )

        # Тренд (разница между текущим и прошлым лагом)
        df['trend_1'] = df[f'lag_1'] - df[f'lag_2']

        return df

    def _add_base_features(self):
        # ... (ваш существующий код без изменений, кроме импортов) ...
        # Убедитесь, что здесь вы также возвращаете df_weekly
        df_sales = pd.read_excel('Данные_по_Продажам.xlsx')
        df_sales['Дата'] = pd.to_datetime(df_sales['Дата'], dayfirst=True)
        df_sales['Year_Week'] = df_sales['Дата'].dt.to_period('W')

        df_weekly = df_sales.groupby(
            ['Контрагент Код', 'Номенклатура Код', 'Year_Week']
        )['Количество'].sum().reset_index()

        # Средняя цена за неделю из продаж
        df_price_weekly = df_sales.groupby(
            ['Контрагент Код', 'Номенклатура Код', 'Year_Week']
        )['ЦенаРеализации'].mean().reset_index()
        df_price_weekly.rename(columns={'ЦенаРеализации': 'СрЦенаЗаНеделю'}, inplace=True)
        df_weekly = df_weekly.merge(df_price_weekly, on=['Контрагент Код', 'Номенклатура Код', 'Year_Week'])

        # Акции
        df_promo = pd.read_excel('Данные_по_Акциям.xlsx')
        df_promo['ДатаНачала'] = pd.to_datetime(df_promo['ДатаНачала'], dayfirst=True)
        df_promo['ДатаКонца'] = pd.to_datetime(df_promo['ДатаКонца'], dayfirst=True)

        # Оптимизация: вместо apply используем векторизованный подход или merge interval
        # Для простоты оставим apply, но для больших данных лучше делать merge_asof или interval join

        # Создадим список недель для каждого контрагента/товара, чтобы ускорить apply
        # (Ваш текущий код is_promo через apply может быть очень медленным, но функциональным)

        def is_promo(row):
            week_start = row['Year_Week'].start_time
            week_end = week_start + pd.Timedelta(days=6)
            # Фильтруем dataframe акций только по нужному товару и контрагенту для скорости
            # (Это все еще медленно, лучше pre-calculate promo calendar)
            mask = (df_promo['Контрагент Код'] == row['Контрагент Код']) & \
                   (df_promo['Номенклатура Код'] == row['Номенклатура Код']) & \
                   (df_promo['ДатаНачала'] <= week_end) & \
                   (df_promo['ДатаКонца'] >= week_start)
            return int(mask.any())

        # Внимание: apply на больших данных очень медленный.
        # Рекомендую заменить на создание бинарного признака через merge, если возможно.
        df_weekly['is_promo'] = df_weekly.apply(is_promo, axis=1)

        # Остатки
        df_stock = pd.read_excel('Данные_по_ТЗ.xlsx')
        df_stock['Дата'] = pd.to_datetime(df_stock['Дата'], dayfirst=True)
        df_stock['Year_Week'] = df_stock['Дата'].dt.to_period('W')
        df_stock_weekly = df_stock.groupby(['Номенклатура Код', 'Year_Week'])['СрДнОстаток|Сумма'].mean().reset_index()
        df_stock_weekly.rename(columns={'СрДнОстаток|Сумма': 'СрДнОстаток'}, inplace=True)
        df_weekly = df_weekly.merge(df_stock_weekly, on=['Номенклатура Код', 'Year_Week'], how='left')

        # Заполнение остатков
        # median_val = df_weekly['СрДнОстаток'].median()
        # if pd.isna(median_val): median_val = 0
        # df_weekly['СрДнОстаток'] = df_weekly['СрДнОстаток'].fillna(median_val)

        # Справочники
        df_sku_info = pd.read_excel('Справочник_товаров.xlsx')
        df_sku_info.rename(columns={'Код': 'Номенклатура Код'}, inplace=True)
        df_weekly = df_weekly.merge(df_sku_info[['Номенклатура Код', 'ТоварнаяГруппа', 'ТорговаяМарка']],
                                    on='Номенклатура Код', how='left')

        df_tt_info = pd.read_excel('Справочник_ТТ.xlsx')
        df_tt_info.rename(columns={'Код': 'Контрагент Код'}, inplace=True)
        df_weekly = df_weekly.merge(df_tt_info[['Контрагент Код', 'КаналСбыта', 'ТорговаяСеть']],
                                    on='Контрагент Код', how='left')

        # One-hot encoding
        # Проверьте, нет ли слишком редких категорий. Можно сначала отфильтровать топ-N
        cols_to_encode = ['ТоварнаяГруппа', 'КаналСбыта']
        for col in cols_to_encode:
            if col in df_weekly.columns:
                df_weekly[col] = df_weekly[col].fillna('Unknown')

        df_weekly = pd.get_dummies(df_weekly, columns=cols_to_encode, drop_first=True)

        # Временные признаки
        df_weekly['Месяц'] = df_weekly['Year_Week'].dt.start_time.dt.month
        df_weekly['Неделя_в_году'] = df_weekly['Year_Week'].dt.start_time.dt.isocalendar().week.astype(int)

        # Синус/Косинус месяца и недели (для цикличности)
        df_weekly['Month_sin'] = np.sin(2 * np.pi * df_weekly['Месяц'] / 12)
        df_weekly['Month_cos'] = np.cos(2 * np.pi * df_weekly['Месяц'] / 12)
        df_weekly['Week_sin'] = np.sin(2 * np.pi * df_weekly['Неделя_в_году'] / 52)
        df_weekly['Week_cos'] = np.cos(2 * np.pi * df_weekly['Неделя_в_году'] / 52)

        df_weekly = df_weekly.sort_values(['Контрагент Код', 'Номенклатура Код', 'Year_Week'])
        df_weekly['Week_Index'] = df_weekly.groupby(['Контрагент Код', 'Номенклатура Код']).cumcount()

        return df_weekly