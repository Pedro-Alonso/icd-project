from pipeline.run_model import run_model

for m, ds in [("logistic", "linear"), ("random_forest", "tree"), ("svm", "linear"),
              ("xgboost", "tree"), ("lightgbm", "tree"), ("catboost", "tree")]:
    run_model(model_name=m, dataset=ds, task="severity", subgroup="all")