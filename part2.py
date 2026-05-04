import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import time

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

def batch_gradient_descent(alpha, U, V, omega, iters, track_history = False):
    history = []
    for iter in range(iters):
        if not (np.all(np.isfinite(U)) and np.all(np.isfinite(V))):
            break
        gradient_U = np.zeros_like(U)
        gradient_V = np.zeros_like(V)
        for (i, j, a_ij) in omega:
            residual = a_ij - U[i] @ V[j]
            gradient_U[i] += -2 * residual * V[j]
            gradient_V[j] += -2 * residual * U[i]
        U -= alpha * gradient_U / len(omega)
        V -= alpha * gradient_V / len(omega)
        if track_history and iter % 10 == 0:
            history.append((iter, evaluate(U, V)[1]))
    return U, V, history

def stochastic_gradient_descent(alpha, U, V, omega, iters, track_history=False):
    history = []
    log_every = max(1, iters // 20)
    for iter in range(iters):
        rating_index = np.random.randint(len(omega))
        i, j, a_ij = omega[rating_index]
        if not (np.all(np.isfinite(U[i])) and np.all(np.isfinite(V[j]))):
            break
        residual = a_ij - U[i] @ V[j]
        gradient_U_i = -2 * residual * V[j]
        gradient_V_j = -2 * residual * U[i]
        U[i] -= alpha * gradient_U_i
        V[j] -= alpha * gradient_V_j
        if track_history and iter % log_every == 0:
            history.append((iter / len(omega), evaluate(U, V)[1]))
    return U, V, history

def evaluate(U, V):
    if not (np.all(np.isfinite(U)) and np.all(np.isfinite(V))):
        return float('nan'), float('nan')
    train_preds = np.array([U[i] @ V[j] for i, j, a_ij in omega])
    test_preds  = np.array([U[i] @ V[j] for i, j in zip(test_users, test_movies)])
    if not (np.all(np.isfinite(train_preds)) and np.all(np.isfinite(test_preds))):
        return float('nan'), float('nan')
    rmse_train  = np.sqrt(sklearn.metrics.mean_squared_error(training_ratings, train_preds))
    rmse_test   = np.sqrt(sklearn.metrics.mean_squared_error(test_ratings, test_preds))
    return rmse_train, rmse_test

def run(method, alpha, iters, k, track_history=False):
    U = np.random.randn(num_users, k) * 0.01
    V = np.random.randn(num_movies, k) * 0.01
    start = time.time()
    with np.errstate(over='ignore', invalid='ignore'):
        U, V, history = method(alpha, U, V, omega, iters, track_history)
    elapsed = time.time() - start
    train_rmse, test_rmse = evaluate(U, V)
    return train_rmse, test_rmse, elapsed, history

def fmt(x):
    return "DIVERGED" if np.isnan(x) else f"{x:.3f}"

iters = 200

#Experiment 1: Sweep k (alpha fixed at 0.01)
print("Experiment 1: Sweep k")
#k_values = [2, 5, 10, 20, 50] #For SGD
k_values = [10,50,100,250,500] #For BGD
bgd_k = {'train': [], 'test': [], 'time': []}
sgd_k = {'train': [], 'test': [], 'time': []}

for k in k_values:
    print(f"k = {k}")
    tr, te, t, _ = run(batch_gradient_descent, 0.01, iters, k)
    bgd_k['train'].append(tr); bgd_k['test'].append(te); bgd_k['time'].append(t)
    tr, te, t, _ = run(stochastic_gradient_descent, 0.01, iters * len(omega), k)
    sgd_k['train'].append(tr); sgd_k['test'].append(te); sgd_k['time'].append(t)


#Experiment 2: Sweep Alpha (k fixed at 10)
print("\nExperiment 2: Sweep Alpha")
#alpha_values = [0.001, 0.005, 0.01, 0.05, 0.1] #For SGD
alpha_values = [0.1,1,5,50,100] #For BGD
bgd_a = {'train': [], 'test': [], 'time': []}
sgd_a = {'train': [], 'test': [], 'time': []}

for alpha in alpha_values:
    print(f"alpha = {alpha}")
    tr, te, t, _ = run(batch_gradient_descent, alpha, iters, 10)
    bgd_a['train'].append(tr); bgd_a['test'].append(te); bgd_a['time'].append(t)
    tr, te, t, _ = run(stochastic_gradient_descent, alpha, iters * len(omega), 10)
    sgd_a['train'].append(tr); sgd_a['test'].append(te); sgd_a['time'].append(t)


#Experiment 3: Convergence over time (k = 10, alpha = 0.01)
print("\nExperiment 3: Convergence Curves")
_, _, _, bgd_history = run(batch_gradient_descent, 0.01, iters, 10, track_history=True)
_, _, _, sgd_history = run(stochastic_gradient_descent, 0.01, iters * len(omega), 10, track_history=True)


#Plot 1: Test RMSE vs k and alpha
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(k_values, bgd_k['test'], marker='o', label='Batch GD')
axes[0].plot(k_values, sgd_k['test'], marker='s', label='SGD')
axes[0].set_xlabel('Rank (k)'); axes[0].set_ylabel('Test RMSE')
axes[0].set_title('Test RMSE vs Rank (alpha = 0.01)')
axes[0].set_xticks(k_values); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(alpha_values, bgd_a['test'], marker='o', label='Batch GD')
axes[1].plot(alpha_values, sgd_a['test'], marker='s', label='SGD')
axes[1].set_xlabel('Step Size (alpha)'); axes[1].set_ylabel('Test RMSE')
axes[1].set_title('Test RMSE vs Step Size (k=10)')
axes[1].set_xscale('log'); axes[1].legend(); axes[1].grid(alpha=0.3)
axes[1].set_xticks(alpha_values); axes[1].set_xticklabels([str(a) for a in alpha_values])

#Annotate diverged points
for a, r in zip(alpha_values, sgd_a['test']):
    if np.isnan(r):
        axes[1].annotate('DIVERGED', xy=(a, axes[1].get_ylim()[1] * 0.9), ha='center',
                         color='red', fontsize=9, fontweight='bold')
plt.tight_layout(); plt.savefig('rmse_sweeps.png', dpi=150); plt.show()


#Plot 2: Train vs Test RMSE (overfitting detection)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(k_values, bgd_k['train'], marker='o', linestyle='--', label='Batch GD Train')
axes[0].plot(k_values, bgd_k['test'],  marker='o', label='Batch GD Test')
axes[0].plot(k_values, sgd_k['train'], marker='s', linestyle='--', label='SGD Train')
axes[0].plot(k_values, sgd_k['test'],  marker='s', label='SGD Test')
axes[0].set_xlabel('Rank (k)'); axes[0].set_ylabel('RMSE')
axes[0].set_title('Train vs Test RMSE: Overfitting Check')
axes[0].set_xticks(k_values); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(alpha_values, bgd_a['train'], marker='o', linestyle='--', label='Batch GD Train')
axes[1].plot(alpha_values, bgd_a['test'],  marker='o', label='Batch GD Test')
axes[1].plot(alpha_values, sgd_a['train'], marker='s', linestyle='--', label='SGD Train')
axes[1].plot(alpha_values, sgd_a['test'],  marker='s', label='SGD Test')
axes[1].set_xlabel('Step Size (alpha)'); axes[1].set_ylabel('RMSE')
axes[1].set_title('Train vs Test RMSE: Overfitting Check')
axes[1].set_xscale('log'); axes[1].legend(); axes[1].grid(alpha=0.3)
axes[1].set_xticks(alpha_values); axes[1].set_xticklabels([str(a) for a in alpha_values])

plt.tight_layout(); plt.savefig('overfitting.png', dpi=150); plt.show()


#Plot 3: Convergence Curves (RMSE vs epochs)
plt.figure(figsize=(8, 5))
if bgd_history:
    bgd_x, bgd_y = zip(*bgd_history)
    plt.plot(bgd_x, bgd_y, marker='o', label='Batch GD')
if sgd_history:
    sgd_x, sgd_y = zip(*sgd_history)
    plt.plot(sgd_x, sgd_y, marker='s', label='SGD')
plt.xlabel('Effective Epochs (passes through data)')
plt.ylabel('Test RMSE')
plt.title('Convergence: Test RMSE vs Epochs (k = 10, alpha = 0.01)')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('convergence.png', dpi=150); plt.show()


#Plot 4: RMSE vs Runtime
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

#Filter out NaN for scatter plots
def filter_finite(xs, ys, labels):
    fx, fy, fl = [], [], []
    for x, y, l in zip(xs, ys, labels):
        if not np.isnan(y):
            fx.append(x); fy.append(y); fl.append(l)
    return fx, fy, fl

bx, by, bl = filter_finite(bgd_k['time'], bgd_k['test'], k_values)
sx, sy, sl = filter_finite(sgd_k['time'], sgd_k['test'], k_values)
axes[0].scatter(bx, by, label='Batch GD', s=80)
axes[0].scatter(sx, sy, label='SGD', s=80, marker='s')
axes[0].set_xlabel('Running Time (s)'); axes[0].set_ylabel('Test RMSE')
axes[0].set_title('RMSE vs Running Time (k sweep)')
axes[0].legend(); axes[0].grid(alpha=0.3)

bx, by, bl = filter_finite(bgd_a['time'], bgd_a['test'], alpha_values)
sx, sy, sl = filter_finite(sgd_a['time'], sgd_a['test'], alpha_values)
axes[1].scatter(bx, by, label='Batch GD', s=80)
axes[1].scatter(sx, sy, label='SGD', s=80, marker='s')
axes[1].set_xlabel('Running Time (s)'); axes[1].set_ylabel('Test RMSE')
axes[1].set_title('RMSE vs Running Time (alpha sweep)')
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('rmse_vs_time.png', dpi=150); plt.show()

#Summary Tables
print("\nK SWEEP RESULTS (alpha = 0.01)\n")
print(f"{'k':>4} | {'BGD Train':>10} {'BGD Test':>10} {'BGD Time':>10} | {'SGD Train':>10} {'SGD Test':>10} {'SGD Time':>10}")
for i, k in enumerate(k_values):
    print(f"{k:>4} | {fmt(bgd_k['train'][i]):>10} {fmt(bgd_k['test'][i]):>10} {bgd_k['time'][i]:>9.2f}s | "
          f"{fmt(sgd_k['train'][i]):>10} {fmt(sgd_k['test'][i]):>10} {sgd_k['time'][i]:>9.2f}s")

print("\nALPHA SWEEP RESULTS (k = 10)\n")

print(f"{'alpha':>6} | {'BGD Train':>10} {'BGD Test':>10} {'BGD Time':>10} | {'SGD Train':>10} {'SGD Test':>10} {'SGD Time':>10}")
for i, a in enumerate(alpha_values):
    print(f"{a:>6.3f} | {fmt(bgd_a['train'][i]):>10} {fmt(bgd_a['test'][i]):>10} {bgd_a['time'][i]:>9.2f}s | "
          f"{fmt(sgd_a['train'][i]):>10} {fmt(sgd_a['test'][i]):>10} {sgd_a['time'][i]:>9.2f}s")


#Best Parameters
print("\nBEST PARAMETERS (lowest test RMSE, ignoring diverged runs)\n")

def best_param(values, results):
    arr = np.array(results, dtype=float)
    if np.all(np.isnan(arr)):
        return None
    return values[np.nanargmin(arr)]

best_k_bgd = best_param(k_values, bgd_k['test'])
best_k_sgd = best_param(k_values, sgd_k['test'])
best_a_bgd = best_param(alpha_values, bgd_a['test'])
best_a_sgd = best_param(alpha_values, sgd_a['test'])

print(f"BGD: best k = {best_k_bgd}, best alpha = {best_a_bgd}")
print(f"SGD: best k = {best_k_sgd}, best alpha = {best_a_sgd}")
print(f"\nPart 1 used k = 10, alpha = 0.01:")
print(f"For BGD, optimal was k = {best_k_bgd}, alpha = {best_a_bgd}")
print(f"For SGD, optimal was k = {best_k_sgd}, alpha = {best_a_sgd}")