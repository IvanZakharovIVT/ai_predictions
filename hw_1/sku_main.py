from hw_1.sku_my import SKULiner, SKUForest, SKUXGB

if __name__ == '__main__':

    my_linreg = SKULiner()
    my_forest = SKUForest()
    my_xgboost = SKUXGB()

    my_linreg.run()

    my_forest.run()

    my_xgboost.run()
