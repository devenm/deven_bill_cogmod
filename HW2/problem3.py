import numpy as np
import matplotlib.pyplot as plt

def generate_drift_rates(num_alternatives):
    return np.linspace(0.1, -0.1, num_alternatives)

def euler_maruyama_simulation(num_alternatives, num_trials, num_time_steps, alpha, sigma, threshold, non_decision_time, starting_points):
    dt = 0.01  # Time step
    response_times = []
    v = generate_drift_rates(num_alternatives)  # Generate drift rates for the current number of alternative
    for _ in range(num_trials):
        X = np.array(starting_points[num_alternatives])
        
        for _ in range(num_time_steps):
            dW = np.random.normal(0, np.sqrt(dt), num_alternatives)
            I = (alpha / num_alternatives - 1) * (np.sum(v) - v)  
            dX = (v - I) * dt + sigma * dW
            X += dX
            X = np.maximum(X, 0)  # Ensure activations are non-negative
            if np.any(X >= threshold):
                response_time = non_decision_time + dt * np.argmax(X >= threshold)
                response_times.append(response_time)
                break
    return response_times

def visualize_response_times(response_times):
    plt.hist(response_times, bins=30, color='skyblue', edgecolor='black', density=True)
    plt.xlabel('Response Time')
    plt.ylabel('Probability Density')
    plt.title('Response Time Distribution')
    plt.show()

def explore_feedforward_inhibition(num_alternatives, num_trials, num_time_steps, alphas, sigma, threshold, non_decision_time, starting_points):
    mean_response_times = []
    for alpha in alphas:
        response_times = euler_maruyama_simulation(num_alternatives, num_trials, num_time_steps, alpha, sigma, threshold, non_decision_time, starting_points)
        mean_response_time = np.mean(response_times)
        mean_response_times.append(mean_response_time)
    return mean_response_times

def run_all_alternatives(num_alternatives_values, num_trials, num_time_steps, alpha, sigma, threshold, non_decision_time):
    for num_alternatives in num_alternatives_values:
        starting_points_per_alternative = {2: [0.0, 0.0], 3: [0.0, 0.0, 0.0], 4: [0.0, 0.0, 0.0, 0.0]}
        response_times = euler_maruyama_simulation(num_alternatives, num_trials, num_time_steps, alpha, sigma, threshold, non_decision_time, starting_points_per_alternative)
        visualize_response_times(response_times)

def run_all_alpha_values(num_alternatives, num_trials, num_time_steps, alphas, sigma, threshold, non_decision_time):
    starting_points_per_alternative = {2: [0.0, 0.0], 3: [0.0, 0.0, 0.0], 4: [0.0, 0.0, 0.0, 0.0]}
    mean_response_times = explore_feedforward_inhibition(num_alternatives, num_trials, num_time_steps, alphas, sigma, threshold, non_decision_time, starting_points_per_alternative)
    plt.plot(alphas, mean_response_times, marker='o')
    plt.xlabel('Alpha')
    plt.ylabel('Mean Response Time')
    plt.title('Effect of Feedforward Inhibition (Alpha) on Mean Response Time')
    plt.show()

if __name__ == "__main__":
    num_trials = 1000
    num_time_steps = 1000
  
    alpha = 0.1
    sigma = 1
    threshold = 1
    non_decision_time = 0.2
    
    num_alternatives_values = [2, 3, 4]
    alphas = np.linspace(0.1, 1, 10)

  
    run_all_alternatives(num_alternatives_values, num_trials, num_time_steps, alpha, sigma, threshold, non_decision_time)

    num_alternatives = 3
    run_all_alpha_values(num_alternatives, num_trials, num_time_steps, alphas, sigma, threshold, non_decision_time)