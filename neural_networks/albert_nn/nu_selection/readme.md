# How to load a model
Use `tensorflow.saved_model.load(<path_to_model>)` to make a `model` instance:
```Python
model = tf.saved_model.load('./nn_precise_summer2024/best_by_test/')
```

# How to make predictions
Make predictions by calling `model`:
```Python
preds = model(data)
```