import matplotlib
matplotlib.use('Agg')
from hyperopt import fmin, tpe, hp
import matplotlib.pyplot as plt

def f(x):
    return x**2 - x + 1

plt.plot(range(-5, 5), [f(x) for x in range(-5, 5)])
plt.title("Function to optimize: f(x) = x^2 - x + 1")
plt.show()

space = hp.uniform('x', -5, 5)

best = fmin(
    fn=f,  # "Loss" function to minimize
    space=space,  # Hyperparameter space
    algo=tpe.suggest,  # Tree-structured Parzen Estimator (TPE)
    max_evals=1000  # Perform 1000 trials
)

print("Found minimum after 1000 trials:")
print(best)
