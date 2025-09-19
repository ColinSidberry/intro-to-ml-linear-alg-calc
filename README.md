# ML Fundamentals Practice Problems

Hi All 👋🏾
I created to understand machine learning concepts for my NLP class. All problems use the same "Go Dolphins!" example to build intuition from the ground up.

## My Learning Approach

Rather than jumping between abstract examples, I wanted to follow one concrete case ("Go Dolphins!" sentiment classification) from basic feature engineering all the way through advanced mathematical analysis. This helps me see how all the pieces connect.

## Structure

### Part 1: Building Your First Classifier
- **Problem 1**: Feature Engineering - Design the model's "sensors"
- **Problem 2**: Dot Products - The heartbeat of machine learning
- **Problem 3**: Loss Functions - How models measure their mistakes
- **Problem 4**: Gradient Descent - Rolling the ball downhill
- **Problem 5**: Matrix Operations - Scaling to reality

### Part 2: Advanced Mathematical Analysis
- **Problem 1**: Vector Calculus - Complete loss landscape analysis
- **Problem 2**: Multi-Layer Chain Rule - How deep networks learn
- **Problem 3**: Jacobian Analysis - Understanding system sensitivity
- **Problem 4**: Vector Fields - Visualizing optimization dynamics
- **Problem 5**: Optimization Landscapes - Advanced convergence analysis

## Setup Instructions

```bash
# Navigate to the repo
cd 2-Intro_to_ML,Linear_Alg,Calc

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start working through problems
jupyter lab

# Begin with Part 1, Problem 1
open Part1-Problems/problems/01_feature_engineering.ipynb
```

## My Study Method

**Visual + Analytical + Practical**: For each concept I try to:
- Build intuition with interactive visualizations
- Understand the mathematical foundations
- Implement working code from scratch
- Verify my understanding with multiple approaches

## Notes and Resources

- **Math references** in `docs/mathematical_reference.md`
- **Verification tests** in `tests/` to check my implementations
- **Extension challenges** when I want to go deeper

## Background Reading

These problems were inspired by and designed to accompany:
1. "What Actually Happens When You Ask ChatGPT How It Works"
2. "The Mathematical Deep Dive: Vector Calculus Behind 'Go Dolphins!'"

The articles give conceptual understanding; these problems let me practice the actual implementation and math.

---

*My goal: truly understand how AI works from first principles, one problem at a time.*