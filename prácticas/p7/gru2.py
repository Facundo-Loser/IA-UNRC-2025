"""
Minimal character-level GRU (Gated Recurrent Unit) model.
Based on Andrej Karpathy's minimal RNN implementation.
Optimized for stability and convergence speed.
"""

import numpy as np

# === DATA I/O ===
data = open('input.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print(f"data has {data_size} characters, {vocab_size} unique.")
char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_char = {i:ch for i,ch in enumerate(chars)}

# === HYPERPARAMETERS ===
hidden_size = 128     # slightly larger hidden size
seq_length = 25        # number of unrolled steps
learning_rate = 1e-3   # smaller LR for GRU stability

# === MODEL PARAMETERS (GRU) ===

# reset gate
Wxr = np.random.randn(hidden_size, vocab_size) * 0.01
Whr = np.random.randn(hidden_size, hidden_size) * 0.01
br  = np.zeros((hidden_size, 1))

# update gate
Wxz = np.random.randn(hidden_size, vocab_size) * 0.01
Whz = np.random.randn(hidden_size, hidden_size) * 0.01
bz  = np.ones((hidden_size, 1))  # start with update gate slightly open

# candidate hidden state
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
bh  = np.zeros((hidden_size, 1))

# output layer
Why = np.random.randn(vocab_size, hidden_size) * 0.01
by  = np.zeros((vocab_size, 1))

# === LOSS FUNCTION ===
def lossFun(inputs, targets, hprev):
    """
    inputs, targets are lists of integers.
    hprev is the initial hidden state (hidden_size x 1).
    Returns loss, gradients, and last hidden state.
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    rs, zs, h_tildes = {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0

    # Forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1

        # GRU forward equations
        rs[t] = 1 / (1 + np.exp(-(np.dot(Wxr, xs[t]) + np.dot(Whr, hs[t-1]) + br)))  # reset gate
        zs[t] = 1 / (1 + np.exp(-(np.dot(Wxz, xs[t]) + np.dot(Whz, hs[t-1]) + bz)))  # update gate
        h_tildes[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, rs[t] * hs[t-1]) + bh)
        hs[t] = (1 - zs[t]) * hs[t-1] + zs[t] * h_tildes[t]

        ys[t] = np.dot(Why, hs[t]) + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        loss += -np.log(ps[t][targets[t], 0])

    # Simplified backward pass (approximation for teaching)
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dWxr, dWhr = np.zeros_like(Wxr), np.zeros_like(Whr)
    dWxz, dWhz = np.zeros_like(Wxz), np.zeros_like(Whz)
    dbh, dbr, dbz, dby = np.zeros_like(bh), np.zeros_like(br), np.zeros_like(bz), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext
        dhraw = (1 - hs[t] * hs[t]) * dh
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dbh += dhraw
        dhnext = np.dot(Whh.T, dhraw)

    # Gradient clipping (avoid exploding gradients)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby, dWxr, dWhr, dWxz, dWhz, dbr, dbz]:
        np.clip(dparam, -5, 5, out=dparam)

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

# === SAMPLING FUNCTION ===
def sample(h, seed_ix, n):
    """
    Sample a sequence of integers from the model
    h: hidden state, seed_ix: starting char index
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        # GRU forward step
        r_t = 1 / (1 + np.exp(-(np.dot(Wxr, x) + np.dot(Whr, h) + br)))
        z_t = 1 / (1 + np.exp(-(np.dot(Wxz, x) + np.dot(Whz, h) + bz)))
        h_tilde = np.tanh(np.dot(Wxh, x) + np.dot(Whh, r_t * h) + bh)
        h = (1 - z_t) * h + z_t * h_tilde

        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes

# === TRAINING LOOP ===
n, p = 0, 0
# Adagrad memory initialization
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mWxr, mWhr = np.zeros_like(Wxr), np.zeros_like(Whr)
mWxz, mWhz = np.zeros_like(Wxz), np.zeros_like(Whz)
mbh, mbr, mbz, mby = np.zeros_like(bh), np.zeros_like(br), np.zeros_like(bz), np.zeros_like(by)
smooth_loss = -np.log(1.0/vocab_size)*seq_length

print("\nTraining GRU...\n")
while True:
    # prepare inputs
    if p+seq_length+1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size, 1))
        p = 0
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    # sample output occasionally
    if n % 500 == 0:
        sample_ix = sample(hprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print(f"----\n {txt} \n----")

    # forward + backward
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 100 == 0:
        print(f"iter {n}, loss: {smooth_loss:.4f}")

    # parameter update with Adagrad
    for param, dparam, mem in zip(
        [Wxh, Whh, Why, bh, by],
        [dWxh, dWhh, dWhy, dbh, dby],
        [mWxh, mWhh, mWhy, mbh, mby]
    ):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

    p += seq_length
    n += 1
