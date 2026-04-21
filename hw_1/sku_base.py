from abc import abstractmethod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_percentage_error


class TooSmallDatasetError(Exception):
    ...


class SkuBase:
    regression_name = 'Название регрессии'
    file_name = 'Данные_по_Продажам.xlsx'
    mape_scores = []
    forecast_horizon = 8
    # Для хранения детальной информации по каждой группе
    group_predictions = []  # будет хранить dict с данными для графиков

    def __init__(self, dataframe):
        self._df = dataframe

    def run(self):
        print(f'================{self.regression_name}==============')
        groups = self._df.groupby(['Контрагент Код', 'Номенклатура Код'])

        self.mape_scores = []
        self.group_predictions = []

        for name, group in groups:
            client_code, sku_code = name

            if len(group) < 20:
                continue
            try:
                X_train, X_test, y_train, y_test, dates_test = self._init_train_test_set_with_dates(group)
            except TooSmallDatasetError:
                continue
            y_pred_test = self._fit_model(X_train, y_train, X_test)
            self._show_predict(client_code, sku_code, y_test, y_pred_test)

            # Сохраняем данные для построения графиков
            self.group_predictions.append({
                'client_code': client_code,
                'sku_code': sku_code,
                'y_test': y_test,
                'y_pred_test': y_pred_test,
                'dates': dates_test,
                'mape': mean_absolute_percentage_error(y_test, y_pred_test) if np.sum(y_test != 0) > 0 else None
            })

        self._show_final_result()

    def run_with_forecast(self, forecast_weeks=10):
        """Запуск с прогнозированием на forecast_weeks недель вперед"""
        print(f'================{self.regression_name} - Прогнозирование==============')
        groups = self._df.groupby(['Контрагент Код', 'Номенклатура Код'])

        self.mape_scores = []
        self.group_predictions = []

        for name, group in groups:
            client_code, sku_code = name

            if len(group) < 10:
                continue

            # Обучаем на всех данных
            X_full, y_full, dates_full = self._prepare_full_data(group)

            # Создаем признаки для прогноза
            X_future = self._create_future_features(group, forecast_weeks)

            # Обучаем модель на всех данных
            # Делаем прогноз
            y_forecast = self._fit_model(X_full, y_full, X_future)

            # Сохраняем для визуализации
            self.group_predictions.append({
                'client_code': client_code,
                'sku_code': sku_code,
                'historical_dates': dates_full,
                'historical_sales': y_full,
                'forecast_dates': self._get_future_dates(group, forecast_weeks),
                'forecast_sales': y_forecast
            })

        self._plot_all_forecasts()

    def _init_train_test_set_with_dates(self, group):
        group_with_features = self._create_features_for_group(group.copy())
        feature_columns = self._add_time_series_features_proper(group_with_features)

        X = group_with_features[feature_columns].values
        y = group_with_features['Количество'].values
        dates = group_with_features['Year_Week'].dt.start_time.values

        if len(X) <= 8:
            raise TooSmallDatasetError

        # Сохраняем индексы для временного разбиения
        split_idx = len(X) - self.forecast_horizon

        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        dates_test = dates[split_idx:]

        return X_train, X_test, y_train, y_test, dates_test

    def _create_features_for_group(self, group):
        """Создание временных признаков для конкретной группы"""
        # Сортировка
        group = group.sort_values('Year_Week')

        target_col = 'Количество'

        # Лаги
        for lag in [1, 2, 4, 8]:
            group[f'lag_{lag}'] = group[target_col].shift(lag)

        # Скользящие средние
        for window in [4, 8, 12]:
            group[f'roll_mean_{window}'] = group[target_col].shift(1).rolling(window, min_periods=1).mean()

        # Логарифмические признаки (важно для моделей!)
        group['log_sales'] = np.log1p(group[target_col])
        for lag in [1, 2, 4, 8]:
            group[f'log_lag_{lag}'] = group['log_sales'].shift(lag)

        # Скользящее стандартное отклонение
        group['roll_std_4'] = group[target_col].shift(1).rolling(4, min_periods=1).std()

        # Заполняем NaN
        group = group.fillna(0)

        return group

    def _prepare_full_data(self, group):
        """Подготовка всех данных для обучения"""
        feature_columns = self._add_time_series_features_proper(group)
        X = group[feature_columns].values
        y = group['Количество'].values
        dates = group['Year_Week'].dt.start_time.values
        return X, y, dates

    def _create_future_features(self, group, forecast_weeks):
        """Создание признаков для будущих периодов"""
        last_week = group['Year_Week'].iloc[-1]
        last_week_start = last_week.start_time

        future_dates = []
        future_weeks = []

        for i in range(1, forecast_weeks + 1):
            future_date = last_week_start + pd.Timedelta(weeks=i)
            future_dates.append(future_date)
            future_weeks.append(pd.Period(future_date, freq='W'))

        # Создаем DataFrame для будущих периодов
        future_data = []
        last_row = group.iloc[-1].copy()

        for i, (future_date, future_week) in enumerate(zip(future_dates, future_weeks)):
            new_row = last_row.copy()
            new_row['Year_Week'] = future_week
            new_row['Week_Index'] = last_row['Week_Index'] + i + 1
            new_row['Месяц'] = future_date.month
            new_row['Неделя_в_году'] = future_date.isocalendar().week

            # Для цен и остатков используем последние известные значения или тренд
            # Можно улучшить, добавив прогноз цен и акций

            future_data.append(new_row)

        future_df = pd.DataFrame(future_data)
        feature_columns = self._add_time_series_features_proper(group)

        return future_df[feature_columns].values

    def _get_future_dates(self, group, forecast_weeks):
        """Получение дат для прогноза"""
        last_week = group['Year_Week'].iloc[-1]
        last_week_start = last_week.start_time

        future_dates = []
        for i in range(1, forecast_weeks + 1):
            future_date = last_week_start + pd.Timedelta(weeks=i)
            future_dates.append(future_date)

        return np.array(future_dates)

    def _add_time_series_features_proper(self, group):
        """Получение списка признаков"""
        feature_columns = [
            'Week_Index',
            'СрЦенаЗаНеделю',
            'is_promo',
            'СрДнОстаток',
            'Месяц',
            'Неделя_в_году',
            'lag_1', 'lag_2', 'lag_4', 'lag_8',
            'roll_mean_4', 'roll_mean_8', 'roll_mean_12',
            'roll_std_4'
        ]

        # Добавляем one-hot закодированные колонки (если они есть)
        for col in group.columns:
            if col.startswith(('ТоварнаяГруппа_', 'КаналСбыта_', 'ТорговаяМарка_')):
                if col in group.columns:
                    feature_columns.append(col)

        return [col for col in feature_columns if col in group.columns] or ['Week_Index']

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

    def plot_mape_distribution(self):
        """Построение графика распределения MAPE"""
        if not self.mape_scores:
            print("Нет данных MAPE для отображения")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Гистограмма распределения MAPE
        axes[0].hist(self.mape_scores, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(x=0.25, color='red', linestyle='--', linewidth=2, label='Порог 25%')
        axes[0].axvline(x=np.mean(self.mape_scores), color='green', linestyle='-', linewidth=2,
                        label=f'Среднее: {np.mean(self.mape_scores):.2%}')
        axes[0].set_xlabel('MAPE')
        axes[0].set_ylabel('Частота')
        axes[0].set_title(f'{self.regression_name} - Распределение MAPE по группам')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Box plot
        box_data = axes[1].boxplot(self.mape_scores, patch_artist=True)
        box_data['boxes'][0].set_facecolor('lightblue')
        axes[1].axhline(y=0.25, color='red', linestyle='--', linewidth=2, label='Порог 25%')
        axes[1].set_ylabel('MAPE')
        axes[1].set_title(f'{self.regression_name} - Box plot MAPE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Дополнительная статистика
        print(f"\nСтатистика MAPE для {self.regression_name}:")
        print(f"  Среднее: {np.mean(self.mape_scores):.2%}")
        print(f"  Медиана: {np.median(self.mape_scores):.2%}")
        print(f"  Стандартное отклонение: {np.std(self.mape_scores):.2%}")
        print(f"  Минимум: {np.min(self.mape_scores):.2%}")
        print(f"  Максимум: {np.max(self.mape_scores):.2%}")
        print(f"  Квартиль 25%: {np.percentile(self.mape_scores, 25):.2%}")
        print(f"  Квартиль 75%: {np.percentile(self.mape_scores, 75):.2%}")
        print(f"  Групп с MAPE <= 25%: {np.sum(np.array(self.mape_scores) <= 0.25)} из {len(self.mape_scores)}")

    def plot_best_worst_predictions(self, n_best=3, n_worst=3):
        """Построение графиков лучших и худших прогнозов"""
        if not self.group_predictions:
            print("Нет данных прогнозов для отображения")
            return

        # Фильтруем только те, у которых есть y_test и y_pred_test
        valid_predictions = [p for p in self.group_predictions if 'y_test' in p and p.get('mape') is not None]

        if not valid_predictions:
            print("Нет валидных прогнозов для отображения")
            return

        # Сортируем по MAPE
        sorted_preds = sorted(valid_predictions, key=lambda x: x['mape'])

        best_predictions = sorted_preds[:n_best]
        worst_predictions = sorted_preds[-n_worst:]

        fig, axes = plt.subplots(2, max(n_best, n_worst), figsize=(5 * max(n_best, n_worst), 10))

        # Лучшие прогнозы
        for idx, pred in enumerate(best_predictions):
            ax = axes[0, idx] if n_best > 1 else axes[0]
            weeks = range(len(pred['y_test']))
            ax.plot(weeks, pred['y_test'], 'o-', label='Факт', linewidth=2, markersize=8)
            ax.plot(weeks, pred['y_pred_test'], 's--', label='Прогноз', linewidth=2, markersize=8)
            ax.set_title(f"Клиент: {pred['client_code']}, SKU: {pred['sku_code']}\nMAPE: {pred['mape']:.2%}")
            ax.set_xlabel('Недели')
            ax.set_ylabel('Продажи')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Худшие прогнозы
        for idx, pred in enumerate(worst_predictions):
            ax = axes[1, idx] if n_worst > 1 else axes[1]
            weeks = range(len(pred['y_test']))
            ax.plot(weeks, pred['y_test'], 'o-', label='Факт', linewidth=2, markersize=8, color='green')
            ax.plot(weeks, pred['y_pred_test'], 's--', label='Прогноз', linewidth=2, markersize=8, color='red')
            ax.set_title(f"Клиент: {pred['client_code']}, SKU: {pred['sku_code']}\nMAPE: {pred['mape']:.2%}")
            ax.set_xlabel('Недели')
            ax.set_ylabel('Продажи')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'{self.regression_name} - Лучшие и худшие прогнозы', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def _plot_all_forecasts(self, max_groups=9, history_weeks=52):
        """
        Построение прогнозов на будущее для всех групп с отображением истории.

        Args:
            max_groups (int): Максимальное количество графиков для отображения.
            history_weeks (int): Сколько недель истории показывать перед прогнозом.
        """
        if not self.group_predictions or 'historical_sales' not in self.group_predictions[0]:
            print("Нет данных прогнозов. Запустите run_with_forecast() сначала")
            return

        n_groups = min(len(self.group_predictions), max_groups)
        if n_groups == 0:
            print("Нет групп для отображения")
            return

        # Настройка сетки графиков
        n_cols = 3
        n_rows = (n_groups + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))

        # Если график всего один, axes не будет массивом, превращаем в список
        if n_groups == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx in range(n_groups):
            pred = self.group_predictions[idx]
            ax = axes[idx]

            # 1. Подготовка исторических данных
            hist_dates = pred['historical_dates']
            hist_sales = pred['historical_sales']

            # Берем только последние N недель для чистоты графика,
            # но если история короткая, берем всю
            if len(hist_dates) > history_weeks:
                hist_dates_plot = hist_dates[-history_weeks:]
                hist_sales_plot = hist_sales[-history_weeks:]
            else:
                hist_dates_plot = hist_dates
                hist_sales_plot = hist_sales

            # 2. Подготовка данных прогноза
            forecast_dates = pred['forecast_dates']
            forecast_sales = pred['forecast_sales']

            # 3. Отрисовка

            # Исторические продажи (синяя сплошная линия)
            ax.plot(hist_dates_plot, hist_sales_plot,
                    color='#2c7fb8', linewidth=2.5, marker='o', markersize=4,
                    label=f'Факт (последние {len(hist_dates_plot)} нед.)')

            # Прогноз (красная пунктирная линия)
            ax.plot(forecast_dates, forecast_sales,
                    color='#fc8d59', linewidth=2.5, linestyle='--', marker='s', markersize=6,
                    label='Прогноз')

            # 4. Визуальные разделители

            # Вертикальная линия на стыке истории и прогноза
            split_date = hist_dates_plot[-1]
            ax.axvline(x=split_date, color='gray', linestyle=':', linewidth=2, alpha=0.7)

            # Заштрихованная область для периода прогноза (опционально, для красоты)
            ax.axvspan(forecast_dates[0], forecast_dates[-1],
                       color='#fc8d59', alpha=0.1, label=None)

            # 5. Оформление
            client_code = pred.get('client_code', 'N/A')
            sku_code = pred.get('sku_code', 'N/A')

            ax.set_title(f"Клиент: {client_code}\nSKU: {sku_code}", fontsize=12, pad=10)
            ax.set_xlabel('Дата', fontsize=10)
            ax.set_ylabel('Продажи (шт.)', fontsize=10)

            # Легенда
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, which='major', axis='y', linestyle='-', alpha=0.3)
            ax.grid(True, which='minor', axis='x', linestyle=':', alpha=0.3)

            # Форматирование оси X (даты)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            # Показываем каждый месяц или каждые 2 месяца в зависимости от длины
            if len(hist_dates_plot) + len(forecast_dates) > 20:
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            else:
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Скрыть лишние подграфики, если их меньше чем мест в сетке
        for idx in range(n_groups, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f'{self.regression_name} - Прогноз продаж на {len(forecast_dates)} недель',
                     fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()
        plt.show()


def compare_models(models):
    """Сравнение нескольких моделей"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    model_names = []
    avg_mapes = []

    for model in models:
        if model.mape_scores:
            model_names.append(model.regression_name)
            avg_mapes.append(np.mean(model.mape_scores))

    # Столбчатая диаграмма
    bars = axes[0].bar(model_names, avg_mapes, color=['steelblue', 'forestgreen', 'coral'])
    axes[0].axhline(y=0.25, color='red', linestyle='--', linewidth=2, label='Порог 25%')
    axes[0].set_ylabel('Средний MAPE')
    axes[0].set_title('Сравнение среднего MAPE моделей')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Добавление значений на столбцы
    for bar, value in zip(bars, avg_mapes):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.2%}', ha='center', va='bottom', fontweight='bold')

    # Box plot для сравнения
    mape_lists = [model.mape_scores for model in models if model.mape_scores]
    bp = axes[1].boxplot(mape_lists, labels=model_names, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    axes[1].axhline(y=0.25, color='red', linestyle='--', linewidth=2, label='Порог 25%')
    axes[1].set_ylabel('MAPE')
    axes[1].set_title('Сравнение распределения MAPE моделей')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()