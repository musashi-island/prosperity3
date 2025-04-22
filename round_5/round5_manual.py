import numpy as np
import cvxpy as cp

sentiments = {
    'Haystacks': '++',
    'Ranch Sauce': '+',
    'Cacti Needle': '---',
    'Solar Panels': '--',
    'Red Flags': '++',
    'VR Monocle': '+++',
    'Quantum Coffee': '---',
    'Moonshine': '+',
    'Striped Shirts': '++'
}

returns = {
    '+': 0.05,
    '++': 0.15,
    '+++': 0.25,
    '-': -0.05,
    '--': -0.1,
    '---': -0.4,
    '----': -0.6
}

products = list(sentiments.keys())


rets = np.array([returns[sentiments[products[i-1]]] for i in range(1,10)])
pi = cp.Variable(9)
objective = cp.Minimize(120 * cp.sum_squares(pi) - 7500 * rets.T @ pi)
constraints = [cp.norm(pi, 1) <= 100]
prob = cp.Problem(objective, constraints)

prob.solve()
print('Optimal allocation without integer constraints:')
for i in range(9):
    print("Position in ", products[i], ': ', f"{pi.value[i]:,.2f}", '%', sep='')
