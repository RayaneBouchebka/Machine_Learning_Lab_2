import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class TwoLayerNN:
    def __init__(self, input_dim=64, hidden_dim=16, output_dim=10, random_state=42):
        np.random.seed(random_state)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.W1 = np.random.normal(0, 0.01, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.normal(0, 0.01, size=(hidden_dim, hidden_dim))
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.normal(0, 0.01, size=(hidden_dim, output_dim))
        self.b3 = np.zeros(output_dim)

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        self.X = X
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.relu(self.Z2)
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = self.softmax(self.Z3)
        return self.A3

    def compute_loss(self, Y_onehot, probs):
        eps = 1e-15
        probs_clipped = np.clip(probs, eps, 1 - eps)
        return -np.mean(np.sum(Y_onehot * np.log(probs_clipped), axis=1))

    def backward(self, Y_onehot):
        N = self.X.shape[0]
        dZ3 = (self.A3 - Y_onehot) / N
        self.dW3 = self.A2.T @ dZ3
        self.db3 = np.sum(dZ3, axis=0)

        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * self.relu_derivative(self.Z2)
        self.dW2 = self.A1.T @ dZ2
        self.db2 = np.sum(dZ2, axis=0)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        self.dW1 = self.X.T @ dZ1
        self.db1 = np.sum(dZ1, axis=0)

    def update_params(self, lr):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2
        self.W3 -= lr * self.dW3
        self.b3 -= lr * self.db3


def run_gradient_check():
    print("--- Running Gradient Check ---")
    digits = load_digits()
    X = digits.data[:32] / 16.0
    y = digits.target[:32]
    Y_onehot = np.eye(10)[y]

    model = TwoLayerNN(input_dim=64, hidden_dim=16, output_dim=10, random_state=42)
    probs = model.forward(X)
    model.backward(Y_onehot)

    eps = 1e-5
    passed_count = 0
    np.random.seed(42)

    sample_indices = [tuple(idx) for idx in zip(np.random.choice(model.W1.shape[0], 10), np.random.choice(model.W1.shape[1], 10))]

    for idx in sample_indices:
        analytic_grad = model.dW1[idx]

        model.W1[idx] += eps
        loss_plus = model.compute_loss(Y_onehot, model.forward(X))

        model.W1[idx] -= 2 * eps
        loss_minus = model.compute_loss(Y_onehot, model.forward(X))

        model.W1[idx] += eps

        num_grad = (loss_plus - loss_minus) / (2 * eps)
        rel_error = np.abs(analytic_grad - num_grad) / (np.abs(analytic_grad) + np.abs(num_grad) + 1e-8)

        passed = rel_error < 1e-6
        if passed:
            passed_count += 1
        print(f"Index {idx}: Analytic={analytic_grad:.8f}, Numerical={num_grad:.8f}, RelError={rel_error:.2e}, Passed={passed}")

    print(f"Gradient Check Result: {passed_count}/10 passed (Required >= 9/10).\n")


def main():
    os.makedirs("results", exist_ok=True)

    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    Y_onehot = np.eye(10)[y]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_onehot, test_size=0.3, stratify=y, random_state=42
    )

    sanity_model = TwoLayerNN(input_dim=64, hidden_dim=16, output_dim=10, random_state=42)
    probs = sanity_model.forward(X_train)
    row_sums = np.sum(probs, axis=1)
    print("Sanity Check 1 (Row sums check):", np.allclose(row_sums, 1.0, atol=1e-6))
    
    untrained_loss = sanity_model.compute_loss(Y_train, probs)
    print(f"Sanity Check 2 (Untrained Loss): {untrained_loss:.4f} (Expected ~ln(10) = {np.log(10):.4f})\n")

    run_gradient_check()

    widths = [16, 64]
    lrs = [0.001, 0.01, 0.1]
    epochs = 50
    batch_size = 32

    plt.figure(figsize=(10, 6))

    for W in widths:
        for lr in lrs:
            model = TwoLayerNN(input_dim=64, hidden_dim=W, output_dim=10, random_state=42)
            train_losses = []

            for epoch in range(epochs):
                indices = np.arange(X_train.shape[0])
                np.random.shuffle(indices)
                X_shuffled = X_train[indices]
                Y_shuffled = Y_train[indices]

                for i in range(0, X_train.shape[0], batch_size):
                    X_batch = X_shuffled[i:i + batch_size]
                    Y_batch = Y_shuffled[i:i + batch_size]
                    
                    probs_batch = model.forward(X_batch)
                    model.backward(Y_batch)
                    model.update_params(lr)

                full_probs = model.forward(X_train)
                epoch_loss = model.compute_loss(Y_train, full_probs)
                train_losses.append(epoch_loss)

            test_probs = model.forward(X_test)
            test_preds = np.argmax(test_probs, axis=1)
            test_targets = np.argmax(Y_test, axis=1)
            test_acc = np.mean(test_preds == test_targets)

            label = f"W={W}, lr={lr} (Test Acc: {test_acc*100:.1f}%)"
            plt.plot(range(1, epochs + 1), train_losses, label=label)
            print(f"Completed: {label}")

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Lab 2: Neural Network Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/loss_curves.png", dpi=300)
    plt.close()
    print("\nSaved loss curves plot to results/loss_curves.png")

    bench_model = TwoLayerNN(input_dim=64, hidden_dim=64, output_dim=10, random_state=42)
    sample_batch_X = X_train[:32]
    sample_batch_Y = Y_train[:32]

    forward_times = []
    backward_times = []

    for _ in range(100):
        t0 = time.perf_counter()
        probs_bench = bench_model.forward(sample_batch_X)
        t1 = time.perf_counter()
        bench_model.backward(sample_batch_Y)
        t2 = time.perf_counter()

        forward_times.append((t1 - t0) * 1e6)
        backward_times.append((t2 - t1) * 1e6)

    med_fwd = np.median(forward_times)
    med_bwd = np.median(backward_times)

    print("\n--- Timing Breakdown (100 runs, W=64, batch=32) ---")
    print(f"Median Forward Pass Time:  {med_fwd:.2f} µs")
    print(f"Median Backward Pass Time: {med_bwd:.2f} µs")


if __name__ == "__main__":
    main()

