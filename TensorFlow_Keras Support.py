from mlxtend.evaluate import bias_variance_decomp
from mlxtend.data import boston_housing_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import numpy as np
np.random.seed(1)
tf.random.set_seed(1)
X, y = boston_housing_data()
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=123,
                                                    shuffle=True)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dense(1)
  ])
optimizer = tf.keras.optimizers.Adam()
model.compile(loss='mean_squared_error', optimizer=optimizer)
model.fit(X_train, y_train, epochs=100, verbose=0)
mean_squared_error(model.predict(X_test), y_test)
#32.69300595184836


#Note that it is highly recommended to use the same number of training epochs that you would use on the original training set to ensure convergence:

np.random.seed(1)
tf.random.set_seed(1)
avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        model, X_train, y_train, X_test, y_test, 
        loss='mse',
        num_rounds=100,
        random_seed=123,
        epochs=200, # fit_param
        verbose=0) # fit_param
print('Average expected loss: %.3f' % avg_expected_loss)
print('Average bias: %.3f' % avg_bias)
print('Average variance: %.3f' % avg_var)
Repo:http://rasbt.github.io/mlxtend/user_guide/evaluate/bias_variance_decomp/