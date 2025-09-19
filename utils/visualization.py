"""
Visualization utilities for ML fundamentals course.

Provides consistent plotting functions for:
- Feature vectors and spaces
- Loss landscapes and gradients
- Vector fields and optimization paths
- 2D and 3D mathematical visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from typing import List, Tuple, Optional, Callable

# Set consistent style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_feature_space_2d(features: np.ndarray, labels: np.ndarray,
                         feature_names: List[str], title: str = "Feature Space"):
    """
    Plot 2D feature space with colored points by class.

    Args:
        features: (n_samples, 2) feature matrix
        labels: (n_samples,) binary labels (0/1)
        feature_names: List of 2 feature names for axes
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Separate positive and negative examples
    pos_mask = labels == 1
    neg_mask = labels == 0

    ax.scatter(features[pos_mask, 0], features[pos_mask, 1],
              c='green', label='Positive', alpha=0.7, s=100)
    ax.scatter(features[neg_mask, 0], features[neg_mask, 1],
              c='red', label='Negative', alpha=0.7, s=100)

    ax.set_xlabel(feature_names[0], fontsize=12)
    ax.set_ylabel(feature_names[1], fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig

def plot_decision_boundary(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                          bias: float, feature_names: List[str]):
    """
    Plot 2D decision boundary for linear classifier.

    Args:
        X: (n_samples, 2) feature matrix
        y: (n_samples,) binary labels
        weights: (2,) weight vector
        bias: scalar bias term
        feature_names: List of feature names
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot data points
    pos_mask = y == 1
    neg_mask = y == 0

    ax.scatter(X[pos_mask, 0], X[pos_mask, 1],
              c='green', label='Positive', alpha=0.7, s=100)
    ax.scatter(X[neg_mask, 0], X[neg_mask, 1],
              c='red', label='Negative', alpha=0.7, s=100)

    # Plot decision boundary: w1*x1 + w2*x2 + b = 0
    x_min, x_max = ax.get_xlim()
    x_boundary = np.linspace(x_min, x_max, 100)

    if abs(weights[1]) > 1e-10:  # Avoid division by zero
        y_boundary = -(weights[0] * x_boundary + bias) / weights[1]
        ax.plot(x_boundary, y_boundary, 'b-', linewidth=2, label='Decision Boundary')

    ax.set_xlabel(feature_names[0], fontsize=12)
    ax.set_ylabel(feature_names[1], fontsize=12)
    ax.set_title('Decision Boundary Visualization', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig

def plot_loss_landscape_3d(loss_function: Callable, weight_range: Tuple[float, float],
                          resolution: int = 50, title: str = "Loss Landscape"):
    """
    Create interactive 3D plot of loss landscape for 2-parameter function.

    Args:
        loss_function: Function that takes (w1, w2) and returns loss
        weight_range: (min_weight, max_weight) for both dimensions
        resolution: Number of points per dimension
        title: Plot title
    """
    w1_vals = np.linspace(weight_range[0], weight_range[1], resolution)
    w2_vals = np.linspace(weight_range[0], weight_range[1], resolution)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)

    # Calculate loss at each point
    Z = np.zeros_like(W1)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = loss_function(W1[i, j], W2[i, j])

    fig = go.Figure(data=[go.Surface(x=W1, y=W2, z=Z, colorscale='Viridis')])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Weight 1',
            yaxis_title='Weight 2',
            zaxis_title='Loss',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        font=dict(size=12)
    )

    return fig

def plot_gradient_descent_path(loss_function: Callable, gradient_function: Callable,
                              start_weights: Tuple[float, float], learning_rate: float,
                              num_steps: int, weight_range: Tuple[float, float]):
    """
    Visualize gradient descent path on 2D contour plot.

    Args:
        loss_function: Function that takes (w1, w2) and returns loss
        gradient_function: Function that takes (w1, w2) and returns (grad_w1, grad_w2)
        start_weights: Starting point (w1, w2)
        learning_rate: Step size
        num_steps: Number of optimization steps
        weight_range: (min_weight, max_weight) for plotting
    """
    # Create contour plot
    resolution = 100
    w1_vals = np.linspace(weight_range[0], weight_range[1], resolution)
    w2_vals = np.linspace(weight_range[0], weight_range[1], resolution)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)

    Z = np.zeros_like(W1)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = loss_function(W1[i, j], W2[i, j])

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot contours
    contour = ax.contour(W1, W2, Z, levels=20, alpha=0.6, colors='gray', linewidths=0.5)
    ax.contourf(W1, W2, Z, levels=20, alpha=0.3, cmap='viridis')
    ax.clabel(contour, inline=True, fontsize=8)

    # Run gradient descent and track path
    weights = list(start_weights)
    path_w1 = [weights[0]]
    path_w2 = [weights[1]]
    losses = [loss_function(weights[0], weights[1])]

    for step in range(num_steps):
        grad_w1, grad_w2 = gradient_function(weights[0], weights[1])
        weights[0] -= learning_rate * grad_w1
        weights[1] -= learning_rate * grad_w2

        path_w1.append(weights[0])
        path_w2.append(weights[1])
        losses.append(loss_function(weights[0], weights[1]))

    # Plot optimization path
    ax.plot(path_w1, path_w2, 'ro-', linewidth=2, markersize=8,
           label=f'Gradient Descent (α={learning_rate})')
    ax.plot(path_w1[0], path_w2[0], 'go', markersize=12, label='Start')
    ax.plot(path_w1[-1], path_w2[-1], 'bo', markersize=12, label='End')

    ax.set_xlabel('Weight 1', fontsize=12)
    ax.set_ylabel('Weight 2', fontsize=12)
    ax.set_title('Gradient Descent Optimization Path', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, (path_w1, path_w2, losses)

def plot_vector_field_2d(gradient_function: Callable, weight_range: Tuple[float, float],
                        resolution: int = 15, title: str = "Gradient Vector Field"):
    """
    Plot 2D vector field showing gradient directions.

    Args:
        gradient_function: Function that takes (w1, w2) and returns (grad_w1, grad_w2)
        weight_range: (min_weight, max_weight) for both dimensions
        resolution: Number of arrows per dimension
        title: Plot title
    """
    w1_vals = np.linspace(weight_range[0], weight_range[1], resolution)
    w2_vals = np.linspace(weight_range[0], weight_range[1], resolution)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)

    # Calculate gradients
    U = np.zeros_like(W1)  # grad_w1
    V = np.zeros_like(W2)  # grad_w2

    for i in range(resolution):
        for j in range(resolution):
            grad_w1, grad_w2 = gradient_function(W1[i, j], W2[i, j])
            U[i, j] = -grad_w1  # Negative for descent direction
            V[i, j] = -grad_w2

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot vector field
    ax.quiver(W1, W2, U, V, alpha=0.7, width=0.003, scale=50)

    ax.set_xlabel('Weight 1', fontsize=12)
    ax.set_ylabel('Weight 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    return fig

def create_interactive_parameter_explorer(feature_function: Callable,
                                        weight_ranges: List[Tuple[float, float]],
                                        weight_names: List[str]):
    """
    Create interactive widget for exploring how parameters affect predictions.
    This is a placeholder - actual implementation would use ipywidgets.

    Args:
        feature_function: Function that takes weights and returns prediction
        weight_ranges: List of (min, max) for each weight
        weight_names: Names for each weight parameter
    """
    print("Interactive parameter explorer would be implemented here using ipywidgets")
    print("Students could adjust sliders to see real-time prediction changes")

    # This would be implemented with ipywidgets.interact in the actual notebook
    pass