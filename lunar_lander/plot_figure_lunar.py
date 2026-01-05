import pandas as pd
import matplotlib.pyplot as plt


def plot_comparison(dqn_file, c51_file):
    try:
        # Load the data
        dqn_data = pd.read_csv(dqn_file)
        c51_data = pd.read_csv(c51_file)

        # Calculate moving averages (window of 20 episodes) to smooth the curves
        dqn_data['Smooth_Reward'] = dqn_data['Reward'].rolling(window=20).mean()
        c51_data['Smooth_Reward'] = c51_data['Reward'].rolling(window=20).mean()

        plt.figure(figsize=(12, 6))

        # Plot Raw Data (faded)
        plt.plot(dqn_data['Episode'], dqn_data['Reward'], color='blue', alpha=0.15)
        plt.plot(c51_data['Episode'], c51_data['Reward'], color='orange', alpha=0.15)

        # Plot Smoothed Data (bold)
        plt.plot(dqn_data['Episode'], dqn_data['Smooth_Reward'], color='blue', label='Standard DQN', linewidth=2)
        plt.plot(c51_data['Episode'], c51_data['Smooth_Reward'], color='orange', label='C51 (Categorical DQN)', linewidth=2)

        # Formatting
        # plt.axhline(y=200, color='red', linestyle='--', label='Solved Threshold (200)')
        plt.title('DQN vs. C51 Performance on LunarLander-v3')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()

        plt.savefig('dqn_vs_c51_comparison.png')
        plt.show()

    except FileNotFoundError:
        print("Error: Ensure both CSV files exist in the current directory.")


if __name__ == "__main__":
    plot_comparison('lunar_lander/lunar_lander_dqn_results.csv', 'lunar_lander/lunar_lander_c51_results.csv')
