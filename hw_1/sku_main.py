import numpy as np

from hw_1.dataframe import DataframeInitiator
from hw_1.sku_my import SKULiner, SKUForest, SKUXGB, CatBoostModel
from hw_1.sku_base import compare_models

def main():
    print("=" * 60)
    print("ЧАСТЬ 0: Инициализация датафрейма")
    print("=" * 60)

    dataframe = DataframeInitiator().init_dataframe()

    # Инициализация моделей
    # my_linreg = SKULiner(dataframe)
    my_forest = SKUForest(dataframe)
    my_xgboost = SKUXGB(dataframe)
    catboost = CatBoostModel(dataframe)

    # ============== ЧАСТЬ 1: Оценка MAPE ==============
    print("=" * 60)
    print("ЧАСТЬ 1: Оценка качества моделей")
    print("=" * 60)

    # Запуск для оценки MAPE
    # my_linreg.run()
    my_forest.run()
    my_xgboost.run()
    catboost.run()
    return

    # Построение графиков MAPE
    print("\n" + "=" * 60)
    print("Графики распределения MAPE")
    print("=" * 60)

    # my_linreg.plot_mape_distribution()
    my_forest.plot_mape_distribution()
    my_xgboost.plot_mape_distribution()
    catboost.plot_mape_distribution()

    # Лучшие и худшие прогнозы
    print("\n" + "=" * 60)
    print("Лучшие и худшие прогнозы")
    print("=" * 60)

    # my_linreg.plot_best_worst_predictions(n_best=3, n_worst=3)
    my_forest.plot_best_worst_predictions(n_best=3, n_worst=3)
    my_xgboost.plot_best_worst_predictions(n_best=3, n_worst=3)
    catboost.plot_best_worst_predictions(n_best=3, n_worst=3)

    # Сравнение моделей
    print("\n" + "=" * 60)
    print("Сравнение моделей")
    print("=" * 60)

    compare_models([
        # my_linreg,
        my_forest,
        my_xgboost,
        catboost
    ])

    # ============== ЧАСТЬ 2: Прогноз на 2.5 месяца ==============
    print("\n" + "=" * 60)
    print("ЧАСТЬ 2: Прогноз на 2.5 месяца (10 недель)")
    print("=" * 60)

    # Выбираем лучшую модель по MAPE для прогнозирования
    models_mapes = {
        # 'Линейная регрессия': np.mean(my_linreg.mape_scores) if my_linreg.mape_scores else float('inf'),
        'Случайный лес': np.mean(my_forest.mape_scores) if my_forest.mape_scores else float('inf'),
        'XGBoost': np.mean(my_xgboost.mape_scores) if my_xgboost.mape_scores else float('inf'),
        'Catboost': np.mean(catboost.mape_scores) if catboost.mape_scores else float('inf')
    }

    best_model_name = min(models_mapes, key=models_mapes.get)
    print(f"Лучшая модель по MAPE: {best_model_name}")

    # Запускаем прогнозирование для лучшей модели
    # if best_model_name == 'Линейная регрессия':
    #     best_model = my_linreg
    if best_model_name == 'CatBoost Regressor':
        best_model = catboost
    elif best_model_name == 'Случайный лес':
        best_model = my_forest
    else:
        best_model = my_xgboost

    best_model.run_with_forecast(forecast_weeks=10)  # 10 недель = 2.5 месяца


if __name__ == '__main__':
    main()
