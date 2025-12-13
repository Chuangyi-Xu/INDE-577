from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


def _check_X_y(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError("X must be 2D array of shape (n_samples, n_features).")
    if y.ndim not in (1, 2):
        raise ValueError("y must be 1D or 2D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples.")
    return X, y


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    y = y.astype(int)
    oh = np.zeros((y.shape[0], n_classes), dtype=float)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh


def _stable_softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # stable sigmoid
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def _tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def _sigmoid_grad(a: np.ndarray) -> np.ndarray:
    # a = sigmoid(z)
    return a * (1.0 - a)


def _relu_grad(a: np.ndarray) -> np.ndarray:
    return (a > 0.0).astype(float)


def _tanh_grad(a: np.ndarray) -> np.ndarray:
    return 1.0 - a**2


_ACTIVATIONS: Dict[str, Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]] = {
    "relu": (_relu, _relu_grad),
    "sigmoid": (_sigmoid, _sigmoid_grad),
    "tanh": (_tanh, _tanh_grad),
}


@dataclass
class _MLPParams:
    W: List[np.ndarray]
    b: List[np.ndarray]


class _BaseMLP:
    """
    A simple NumPy-based Multi-Layer Perceptron with mini-batch gradient descent.

    Notes
    -----
    - hidden layers use `activation`
    - output layer:
        * classifier: softmax (multi-class) or sigmoid (binary)
        * regressor: linear
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (50,),
        activation: str = "relu",
        learning_rate: float = 1e-2,
        max_iter: int = 2000,
        batch_size: int = 64,
        l2: float = 0.0,
        tol: float = 1e-6,
        n_iter_no_change: int = 20,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        if not isinstance(hidden_layer_sizes, tuple) or len(hidden_layer_sizes) == 0:
            raise ValueError("hidden_layer_sizes must be a non-empty tuple, e.g. (50,) or (64, 32).")
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(_ACTIVATIONS.keys())}.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if max_iter <= 0:
            raise ValueError("max_iter must be > 0.")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if l2 < 0:
            raise ValueError("l2 must be >= 0.")
        if n_iter_no_change <= 0:
            raise ValueError("n_iter_no_change must be > 0.")

        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.l2 = l2
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose

        self.params_: Optional[_MLPParams] = None
        self.loss_curve_: List[float] = []

    def _rng(self) -> np.random.RandomState:
        return np.random.RandomState(self.random_state)

    def _init_params(self, layer_sizes: List[int]) -> _MLPParams:
        rng = self._rng()
        W: List[np.ndarray] = []
        b: List[np.ndarray] = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He init for relu, Xavier otherwise
            if self.activation == "relu":
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)
            W.append(rng.randn(fan_in, fan_out) * scale)
            b.append(np.zeros((1, fan_out), dtype=float))
        return _MLPParams(W=W, b=b)

    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Returns
        -------
        Zs: pre-activation list (for each layer)
        As: activation list including input A0 = X
        """
        assert self.params_ is not None
        act_fn, _ = _ACTIVATIONS[self.activation]

        As: List[np.ndarray] = [X]
        Zs: List[np.ndarray] = []

        # hidden layers
        for i in range(len(self.params_.W) - 1):
            Z = As[-1] @ self.params_.W[i] + self.params_.b[i]
            A = act_fn(Z)
            Zs.append(Z)
            As.append(A)

        # output layer handled in subclasses (linear/sigmoid/softmax)
        Z_out = As[-1] @ self.params_.W[-1] + self.params_.b[-1]
        Zs.append(Z_out)
        As.append(Z_out)  # placeholder; subclasses may overwrite As[-1]
        return Zs, As

    def _batch_indices(self, n: int, rng: np.random.RandomState) -> List[np.ndarray]:
        idx = np.arange(n)
        if self.shuffle:
            rng.shuffle(idx)
        batches = []
        for start in range(0, n, self.batch_size):
            batches.append(idx[start : start + self.batch_size])
        return batches

    def _l2_penalty(self) -> float:
        assert self.params_ is not None
        if self.l2 == 0:
            return 0.0
        s = 0.0
        for W in self.params_.W:
            s += np.sum(W * W)
        return 0.5 * self.l2 * s

    def _l2_grad(self, W: np.ndarray) -> np.ndarray:
        if self.l2 == 0:
            return 0.0
        return self.l2 * W

    # ---- must be implemented by subclasses ----
    def _compute_loss_and_dout(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
        raise NotImplementedError

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_BaseMLP":
        X, y = _check_X_y(X, y)
        n_samples, n_features = X.shape

        layer_sizes = [n_features, *list(self.hidden_layer_sizes), self._output_dim_from_y(y)]
        self.params_ = self._init_params(layer_sizes)

        rng = self._rng()
        best_loss = np.inf
        no_improve = 0
        self.loss_curve_ = []

        act_fn, act_grad = _ACTIVATIONS[self.activation]

        for it in range(self.max_iter):
            batch_ids = self._batch_indices(n_samples, rng)
            epoch_losses = []

            for bidx in batch_ids:
                Xb = X[bidx]
                yb = y[bidx]

                # forward
                Zs, As = self._forward(Xb)

                # output activation
                y_pred = self._apply_output_activation(As[-1], yb)
                As[-1] = y_pred

                # loss + gradient at output
                loss, dA = self._compute_loss_and_dout(self._prepare_y(yb), y_pred)
                loss = loss + self._l2_penalty()
                epoch_losses.append(loss)

                # backprop
                grads_W: List[np.ndarray] = [None] * len(self.params_.W)  # type: ignore
                grads_b: List[np.ndarray] = [None] * len(self.params_.b)  # type: ignore

                # output layer gradients
                # dZ_out is dA for our chosen losses (softmax+CE and sigmoid+BCE already gives dZ)
                dZ = dA
                A_prev = As[-2]
                grads_W[-1] = (A_prev.T @ dZ) / Xb.shape[0] + self._l2_grad(self.params_.W[-1])
                grads_b[-1] = np.mean(dZ, axis=0, keepdims=True)

                # hidden layers
                for li in range(len(self.params_.W) - 2, -1, -1):
                    dA_prev = dZ @ self.params_.W[li + 1].T
                    A_curr = As[li + 1]  # activation output of this hidden layer
                    dZ = dA_prev * act_grad(A_curr)

                    A_prev = As[li]
                    grads_W[li] = (A_prev.T @ dZ) / Xb.shape[0] + self._l2_grad(self.params_.W[li])
                    grads_b[li] = np.mean(dZ, axis=0, keepdims=True)

                # update
                for li in range(len(self.params_.W)):
                    self.params_.W[li] -= self.learning_rate * grads_W[li]
                    self.params_.b[li] -= self.learning_rate * grads_b[li]

            epoch_loss = float(np.mean(epoch_losses))
            self.loss_curve_.append(epoch_loss)

            if self.verbose and (it % 100 == 0 or it == self.max_iter - 1):
                print(f"[MLP] iter={it:4d} loss={epoch_loss:.6f}")

            # early stopping (training loss)
            if best_loss - epoch_loss > self.tol:
                best_loss = epoch_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.n_iter_no_change:
                    if self.verbose:
                        print(f"[MLP] early stopping at iter={it}, best_loss={best_loss:.6f}")
                    break

        return self

    def _output_dim_from_y(self, y: np.ndarray) -> int:
        raise NotImplementedError

    def _prepare_y(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _apply_output_activation(self, z_out: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MLPClassifier(_BaseMLP):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: Optional[int] = None

    def _output_dim_from_y(self, y: np.ndarray) -> int:
        y1 = np.asarray(y).reshape(-1)
        classes = np.unique(y1)
        self.classes_ = classes
        self.n_classes_ = len(classes)
        # for binary: output 1 logit (sigmoid)
        return 1 if self.n_classes_ == 2 else self.n_classes_

    def _prepare_y(self, y: np.ndarray) -> np.ndarray:
        y1 = np.asarray(y).reshape(-1)
        assert self.classes_ is not None
        # map to 0..C-1
        mapping = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.vectorize(mapping.get)(y1).astype(int)
        if self.n_classes_ == 2:
            return y_idx.reshape(-1, 1).astype(float)
        return _one_hot(y_idx, self.n_classes_)

    def _apply_output_activation(self, z_out: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        # binary => sigmoid; multiclass => softmax
        if self.n_classes_ == 2:
            return _sigmoid(z_out)
        return _stable_softmax(z_out)

    def _compute_loss_and_dout(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
        eps = 1e-12
        if self.n_classes_ == 2:
            # Binary cross-entropy, y_pred = sigmoid(z)
            y_pred = np.clip(y_pred, eps, 1.0 - eps)
            loss = -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
            # dZ for sigmoid + BCE is (y_pred - y_true)
            dZ = (y_pred - y_true)
            return float(loss), dZ
        else:
            # Categorical cross-entropy, y_pred = softmax(z)
            y_pred = np.clip(y_pred, eps, 1.0)
            loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
            # dZ for softmax + CE is (y_pred - y_true)
            dZ = (y_pred - y_true)
            return float(loss), dZ

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.params_ is None or self.n_classes_ is None:
            raise ValueError("Model is not fitted yet.")
        X = np.asarray(X, dtype=float)
        _, As = self._forward(X)
        z_out = As[-1]
        if self.n_classes_ == 2:
            p1 = _sigmoid(z_out).reshape(-1, 1)
            p0 = 1.0 - p1
            return np.hstack([p0, p1])
        return _stable_softmax(z_out)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        assert self.classes_ is not None
        return self.classes_[idx]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        y = np.asarray(y).reshape(-1)
        return float(np.mean(y_pred == y))


class MLPRegressor(_BaseMLP):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_outputs_: Optional[int] = None

    def _output_dim_from_y(self, y: np.ndarray) -> int:
        y = np.asarray(y)
        if y.ndim == 1:
            self.n_outputs_ = 1
        else:
            self.n_outputs_ = y.shape[1]
        return int(self.n_outputs_)

    def _prepare_y(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return y

    def _apply_output_activation(self, z_out: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        # linear output
        return z_out

    def _compute_loss_and_dout(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
        # MSE
        diff = (y_pred - y_true)
        loss = np.mean(diff * diff)
        # dZ for linear + MSE: 2*(y_pred - y_true)/n is handled by averaging later; keep as (y_pred - y_true)
        dZ = 2.0 * diff
        return float(loss), dZ

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.params_ is None:
            raise ValueError("Model is not fitted yet.")
        X = np.asarray(X, dtype=float)
        _, As = self._forward(X)
        y_pred = As[-1]
        if y_pred.shape[1] == 1:
            return y_pred.ravel()
        return y_pred

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        # R^2
        y = np.asarray(y, dtype=float)
        y_pred = np.asarray(self.predict(X), dtype=float)
        if y.ndim == 1:
            y = y.ravel()
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1.0 - ss_res / (ss_tot + 1e-12))
