import numpy as np
import tensorflow as tf

xs = np.array([1, 2, 3, 4, 5, 6], dtype=float)
ys = np.array([(2*num)+1 for num in xs], dtype=float)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(
    optimizer='sgd',
    loss='mean_squared_error'
)

model.fit(xs, ys, epochs=10)

tr_ev = model.evaluate(xs, ys, verbose=0)

with open("results.txt", "w") as outfile:
    outfile.write(str(tr_ev)))
    outfile.write(str(model.predict([12])))
