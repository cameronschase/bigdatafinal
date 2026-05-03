import numpy as np
import sklearn
import time

k = 10
alpha = 0.01
iters = 200

train = np.loadtxt('ratings-train.csv', delimiter=',', skiprows=1)
test = np.loadtxt('ratings-test.csv',  delimiter=',', skiprows=1)
training_users = train[:, 0]
training_movies = train[:, 1]
training_ratings = train[:, 2]
test_users = test[:, 0]
test_movies = test[:, 1]
test_ratings = test[:, 2]
unique_users = np.unique(np.concatenate([training_users, test_users]))
num_users = len(unique_users)
unique_movies = np.unique(np.concatenate([training_movies, test_movies]))
num_movies = len(unique_movies)
remap_user = {u: i for i, u in enumerate(unique_users)}
remap_movie = {m: i for i, m in enumerate(unique_movies)}
training_users = np.array([remap_user[u]  for u in training_users])
training_movies = np.array([remap_movie[m] for m in training_movies])
test_users = np.array([remap_user[u] for u in test_users])
test_movies = np.array([remap_movie[m] for m in test_movies])
omega = list(zip(training_users, training_movies, training_ratings))

def batch_gradient_descent(alpha, U, V, omega, iters):
    for iter in range(iters):
        gradient_U = np.zeros_like(U)
        gradient_V = np.zeros_like(V)
        for (i, j, a_ij) in omega:
            residual = a_ij - U[i] @ V[j]
            gradient_U[i] += -2 * residual * V[j]
            gradient_V[j] += -2 * residual * U[i]
        U -= alpha * gradient_U / len(omega)
        V -= alpha * gradient_V / len(omega)
    return U, V

def stochastic_gradient_descent(alpha, U, V, omega, iters):
    for iter in range(iters):
        rating_index = np.random.randint(len(omega))
        i, j, a_ij = omega[rating_index]
        residual = a_ij - U[i] @ V[j]
        gradient_U_i = -2 * residual * V[j]
        gradient_V_i = -2 * residual * U[i]
        U[i] -= alpha * gradient_U_i
        V[j] -= alpha * gradient_V_i
    return U, V

def evaluate(U, V):
    train_preds = np.array([U[i] @ V[j] for i, j, a_ij in omega])
    test_preds = np.array([U[i] @ V[j] for i, j in zip(test_users, test_movies)])
    rmse_train = np.sqrt(sklearn.metrics.mean_squared_error(training_ratings, train_preds))
    rmse_test = np.sqrt(sklearn.metrics.mean_squared_error(test_ratings, test_preds))
    return rmse_train, rmse_test

def run_evaluation(method, alpha, omega, iters, num_users, num_movies, k):
    U = np.random.randn(num_users, k) * 0.01
    V = np.random.randn(num_movies, k) * 0.01
    start = time.time()
    U, V = method(alpha, U, V, omega, iters)
    elapsed_time = time.time() - start
    rmse_train, rmse_test = evaluate(U, V)
    return rmse_train, rmse_test, elapsed_time

batch_train_rmse, batch_test_rmse, batch_time = run_evaluation(batch_gradient_descent, alpha, omega, iters, num_users, num_movies, k)
stoch_train_rmse, stoch_test_rmse, stoch_time = run_evaluation(stochastic_gradient_descent, alpha, omega, iters * len(omega), num_users, num_movies, k)
print(f"Batch Gradient Descent | Train RMSE: {batch_train_rmse:.3f} | Test RMSE: {batch_test_rmse:.3f} | Running Time: {batch_time:.2f}s")
print(f"Stochastic Gradient Descent | Train RMSE: {stoch_train_rmse:.3f} | Test RMSE: {stoch_test_rmse:.3f} | Running Time: {stoch_time:.2f}s")