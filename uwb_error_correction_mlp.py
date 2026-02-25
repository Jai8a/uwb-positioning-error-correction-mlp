import numpy as np
import pandas as pd
import os, glob, copy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def prepare_df(path):
    cols = ['data__coordinates__x','data__coordinates__y','reference__x','reference__y']
    df = pd.read_excel(path, usecols=cols).dropna()
    df['dx'] = df['data__coordinates__x'] - df['reference__x']
    df['dy'] = df['data__coordinates__y'] - df['reference__y']
    return df[['dx','dy']].values

def create_sequences(arr, window):
    X, y = [], []
    for i in range(len(arr) - window):
        X.append(arr[i:i+window])
        y.append(arr[i+window])
    return np.array(X), np.array(y)

top_folders = ['../F8', '../F10']
stat_files, test_files = [], []
for folder in top_folders:
    for p in glob.glob(os.path.join(folder, 'f*.xlsx')):
        (stat_files if 'stat' in os.path.basename(p) else test_files).append(p)

scaler = StandardScaler()
stat_scaled = scaler.fit_transform(np.vstack([prepare_df(p) for p in stat_files]))
test_scaled = scaler.transform(np.vstack([prepare_df(p) for p in test_files]))

class NeuralNetwork:
    def __init__(self, input_size, h1, h2, output_size,
                 lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        
        self.lr0 = lr
        self.lr = lr

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.params = {
            'W1': np.random.randn(input_size, h1) * 0.1,
            'b1': np.zeros((1, h1)),
            'W2': np.random.randn(h1, h2) * 0.1,
            'b2': np.zeros((1, h2)),
            'W3': np.random.randn(h2, output_size) * 0.1,
            'b3': np.zeros((1, output_size)),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        p = self.params
        self.Z1 = X.dot(p['W1']) + p['b1']
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.A1.dot(p['W2']) + p['b2']
        self.A2 = self.relu(self.Z2)
        self.Z3 = self.A2.dot(p['W3']) + p['b3']
        return self.Z3

    def backward(self, X, y, out):
        m = y.shape[0]
        grads = {}
        dZ3 = (out - y) * (1 / m)
        grads['W3'] = self.A2.T.dot(dZ3)
        grads['b3'] = dZ3.sum(axis=0, keepdims=True)
        dA2 = dZ3.dot(self.params['W3'].T)
        dZ2 = dA2 * self.relu_deriv(self.Z2)
        grads['W2'] = self.A1.T.dot(dZ2)
        grads['b2'] = dZ2.sum(axis=0, keepdims=True)
        dA1 = dZ2.dot(self.params['W2'].T)
        dZ1 = dA1 * self.relu_deriv(self.Z1)
        grads['W1'] = X.T.dot(dZ1)
        grads['b1'] = dZ1.sum(axis=0, keepdims=True)

        self.t += 1
        for k in self.params:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k] ** 2)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def compute_loss(self, y_pred, y_true):
        return ((y_pred - y_true) ** 2).mean()

    def update_lr(self, epoch, decay_rate=1e-3):

        self.lr = self.lr0 / (1 + decay_rate * epoch)

window = 30
X_full, y_full = create_sequences(stat_scaled, window)
X_tr, X_val, y_tr, y_val = train_test_split(X_full, y_full,
                                            test_size=0.2, random_state=42)

input_size = window * X_tr.shape[2]
X_tr = X_tr.reshape(-1, input_size)
X_val = X_val.reshape(-1, input_size)

model = NeuralNetwork(input_size, h1=64, h2=32, output_size=2,lr=1e-3)

best_state = copy.deepcopy(model.params)
best_val_loss = float('inf')
patience = 10
wait = 0
batch_size = 32
max_epochs = 200

decay_rate = 1e-3
for epoch in range(1, max_epochs + 1):
    model.update_lr(epoch, decay_rate)
    idx = np.random.permutation(len(X_tr))
    X_sh, y_sh = X_tr[idx], y_tr[idx]
    for i in range(0, len(X_tr), batch_size):
        xb = X_sh[i:i + batch_size]
        yb = y_sh[i:i + batch_size]
        out = model.forward(xb)
        model.backward(xb, yb, out)

    val_pred = model.forward(X_val)
    val_loss = model.compute_loss(val_pred, y_val)
    print(f"Epoch {epoch}/{max_epochs} | LR={model.lr:.6f} | Val Loss = {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.params)
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping after {epoch} epochs. Best val loss = {best_val_loss:.6f}")
            break

model.params = best_state

X_test, y_test = create_sequences(test_scaled, window)
X_test = X_test.reshape(-1, input_size)
pred = model.forward(X_test)
pred_inv = scaler.inverse_transform(pred)
y_inv = scaler.inverse_transform(y_test)

err_raw = np.linalg.norm(y_inv, axis=1)
err_corr = np.linalg.norm(y_inv - pred_inv, axis=1)

err_sorted = np.sort(err_corr)
N = len(err_sorted)
cdf = np.arange(1, N + 1) / N
df_cdf = pd.DataFrame({'error_cdf': cdf})
df_cdf.to_excel('error_cdf.xlsx', index=False)
print("Zapisano CDF do pliku error_cdf.xlsx")

plt.figure()
for data, label in [(np.sort(err_raw), 'Błąd surowy'),
                    (np.sort(err_corr), 'Błąd skorygowany')]:
    cdf = np.arange(1, len(data) + 1) / len(data)
    plt.plot(data, cdf, label=label)
plt.xlabel('Error')
plt.ylabel('CDF')
plt.legend()
plt.title('CDF of UWB Positioning Error')
plt.grid(True)
plt.show()
