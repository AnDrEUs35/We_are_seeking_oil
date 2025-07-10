model_loaded = keras.models.load_model("neoronka_oil")
model_loaded.evaluate(x_test, y_test)