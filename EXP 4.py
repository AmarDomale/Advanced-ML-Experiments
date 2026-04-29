import numpy as np

# =====================================================
# EXPERIMENT NO. 4
# TITLE: Markov Decision Process and Bayesian Learning
# =====================================================

# -----------------------------
# PART 1: MARKOV DECISION PROCESS (MDP)
# -----------------------------

states = ['S1', 'S2', 'S3']
actions = ['A1', 'A2']

P = {
    'S1': {
        'A1': [(0.8, 'S2', 5), (0.2, 'S3', 2)],
        'A2': [(1.0, 'S3', 3)]
    },
    'S2': {
        'A1': [(0.7, 'S1', 4), (0.3, 'S3', 6)],
        'A2': [(1.0, 'S3', 2)]
    },
    'S3': {
        'A1': [(1.0, 'S3', 0)],
        'A2': [(1.0, 'S3', 0)]
    }
}

gamma = 0.9
V = {state: 0 for state in states}
iterations = 10

print("=====================================================")
print("        PART 1: MARKOV DECISION PROCESS (MDP)")
print("=====================================================")

for i in range(iterations):
    new_V = {}
    for state in states:
        action_values = []
        for action in actions:
            q_value = 0
            for prob, next_state, reward in P[state][action]:
                q_value += prob * (reward + gamma * V[next_state])
            action_values.append(q_value)
        new_V[state] = max(action_values)
    V = new_V
    print(f"Iteration {i+1}: {V}")

policy = {}
for state in states:
    action_values = {}
    for action in actions:
        q_value = 0
        for prob, next_state, reward in P[state][action]:
            q_value += prob * (reward + gamma * V[next_state])
        action_values[action] = q_value
    policy[state] = max(action_values, key=action_values.get)

print("\nOptimal Value Function:")
for state in states:
    print(f"{state}: {V[state]:.2f}")

print("\nOptimal Policy:")
for state in states:
    print(f"{state} -> {policy[state]}")

# -----------------------------
# PART 2: BAYESIAN LEARNING
# -----------------------------

print("\n=====================================================")
print("             PART 2: BAYESIAN LEARNING")
print("=====================================================")

# Example probabilities
P_H = 0.6          # Prior probability
P_E_given_H = 0.8  # Likelihood
P_E = 0.7          # Evidence

# Bayes' Theorem
P_H_given_E = (P_E_given_H * P_H) / P_E

print(f"Prior Probability P(H): {P_H}")
print(f"Likelihood P(E|H): {P_E_given_H}")
print(f"Evidence P(E): {P_E}")
print(f"Posterior Probability P(H|E): {P_H_given_E:.4f}")