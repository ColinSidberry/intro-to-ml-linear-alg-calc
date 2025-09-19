"""
Gradient computation utilities for educational purposes.

Provides both analytical and numerical gradient computation
for verification and understanding.
"""

import numpy as np
from typing import Callable, Tuple, List
import warnings

def numerical_gradient(f: Callable, x: np.ndarray, h: float = 1e-7) -> np.ndarray:
    """
    Compute numerical gradient using finite differences.

    This is the "brute force" method that students can compare
    against their analytical derivatives.

    Args:
        f: Function that takes array x and returns scalar
        x: Point at which to compute gradient
        h: Step size for finite differences

    Returns:
        Gradient vector at point x
    """
    grad = np.zeros_like(x)

    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += h

        x_minus = x.copy()
        x_minus[i] -= h

        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)

    return grad

def analytical_gradient_mse(weights: np.ndarray, features: np.ndarray,
                           target: float) -> np.ndarray:
    """
    Analytical gradient for Mean Squared Error loss.

    For our "Go Dolphins!" example:
    L(w) = (w^T x - y)^2
    ∇L = 2(w^T x - y) * x

    Args:
        weights: Current weight vector
        features: Feature vector for one example
        target: True label

    Returns:
        Gradient vector
    """
    prediction = np.dot(weights, features)
    error = prediction - target
    gradient = 2 * error * features

    return gradient

def analytical_gradient_mse_batch(weights: np.ndarray, X: np.ndarray,
                                 y: np.ndarray) -> np.ndarray:
    """
    Analytical gradient for MSE loss over batch of examples.

    Args:
        weights: Current weight vector (n_features,)
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)

    Returns:
        Gradient vector (n_features,)
    """
    predictions = X @ weights
    errors = predictions - y
    gradient = (2 / len(y)) * X.T @ errors

    return gradient

def verify_gradient(analytical_grad: np.ndarray, numerical_grad: np.ndarray,
                   tolerance: float = 1e-5) -> bool:
    """
    Verify analytical gradient against numerical approximation.

    Args:
        analytical_grad: Gradient computed analytically
        numerical_grad: Gradient computed numerically
        tolerance: Maximum allowed difference

    Returns:
        True if gradients match within tolerance
    """
    diff = np.abs(analytical_grad - numerical_grad)
    max_diff = np.max(diff)
    relative_diff = max_diff / (np.max(np.abs(analytical_grad)) + 1e-10)

    if max_diff < tolerance:
        print(f"✅ Gradients match! Max difference: {max_diff:.2e}")
        return True
    else:
        print(f"❌ Gradients don't match. Max difference: {max_diff:.2e}")
        print(f"Analytical: {analytical_grad}")
        print(f"Numerical:  {numerical_grad}")
        print(f"Difference: {diff}")
        return False

def gradient_descent_step(weights: np.ndarray, gradient: np.ndarray,
                         learning_rate: float) -> np.ndarray:
    """
    Perform one step of gradient descent.

    Args:
        weights: Current weights
        gradient: Gradient at current point
        learning_rate: Step size

    Returns:
        Updated weights
    """
    return weights - learning_rate * gradient

def gradient_descent_with_history(loss_function: Callable,
                                 gradient_function: Callable,
                                 initial_weights: np.ndarray,
                                 learning_rate: float,
                                 num_iterations: int,
                                 tolerance: float = 1e-8) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
    """
    Run gradient descent and track the optimization history.

    Args:
        loss_function: Function that takes weights and returns loss
        gradient_function: Function that takes weights and returns gradient
        initial_weights: Starting point
        learning_rate: Step size
        num_iterations: Maximum number of steps
        tolerance: Stop if gradient norm falls below this

    Returns:
        final_weights: Optimized weights
        loss_history: Loss at each iteration
        weight_history: Weights at each iteration
    """
    weights = initial_weights.copy()
    loss_history = []
    weight_history = [weights.copy()]

    for i in range(num_iterations):
        # Compute loss and gradient
        loss = loss_function(weights)
        gradient = gradient_function(weights)

        loss_history.append(loss)

        # Check for convergence
        if np.linalg.norm(gradient) < tolerance:
            print(f"Converged after {i} iterations (gradient norm: {np.linalg.norm(gradient):.2e})")
            break

        # Update weights
        weights = gradient_descent_step(weights, gradient, learning_rate)
        weight_history.append(weights.copy())

        # Warn about potential issues
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            warnings.warn(f"Weights became invalid at iteration {i}. Try smaller learning rate.")
            break

        if i > 0 and loss_history[-1] > loss_history[-2]:
            warnings.warn(f"Loss increased at iteration {i}. Learning rate might be too large.")

    return weights, loss_history, weight_history

def create_go_dolphins_loss_function(features: np.ndarray, target: float):
    """
    Create loss function for the specific "Go Dolphins!" example.

    Args:
        features: Feature vector [word_count, has_team, has_exclamation]
        target: True sentiment (1.0 for positive)

    Returns:
        Function that takes weights and returns MSE loss
    """
    def loss_fn(weights: np.ndarray) -> float:
        prediction = np.dot(weights, features)
        return (prediction - target) ** 2

    return loss_fn

def create_go_dolphins_gradient_function(features: np.ndarray, target: float):
    """
    Create gradient function for the specific "Go Dolphins!" example.

    Args:
        features: Feature vector [word_count, has_team, has_exclamation]
        target: True sentiment (1.0 for positive)

    Returns:
        Function that takes weights and returns gradient
    """
    def gradient_fn(weights: np.ndarray) -> np.ndarray:
        prediction = np.dot(weights, features)
        error = prediction - target
        return 2 * error * features

    return gradient_fn

def compare_learning_rates(loss_function: Callable, gradient_function: Callable,
                          initial_weights: np.ndarray,
                          learning_rates: List[float],
                          num_iterations: int = 100) -> Dict[float, Tuple[List[float], List[np.ndarray]]]:
    """
    Compare gradient descent with different learning rates.

    Args:
        loss_function: Function that takes weights and returns loss
        gradient_function: Function that takes weights and returns gradient
        initial_weights: Starting point
        learning_rates: List of learning rates to try
        num_iterations: Number of iterations per rate

    Returns:
        Dictionary mapping learning_rate -> (loss_history, weight_history)
    """
    results = {}

    for lr in learning_rates:
        print(f"Testing learning rate: {lr}")
        weights, loss_hist, weight_hist = gradient_descent_with_history(
            loss_function, gradient_function, initial_weights.copy(),
            lr, num_iterations
        )
        results[lr] = (loss_hist, weight_hist)

    return results