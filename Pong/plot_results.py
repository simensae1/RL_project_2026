import pandas as pd
import matplotlib.pyplot as plt


def plot_comparison(dqn_file, c51_file_2, c51_file_51):
    try:
        # Load the data
        dqn_data = pd.read_csv(dqn_file)
        c51_data_2 = pd.read_csv(c51_file_2)
        c51_data_51 = pd.read_csv(c51_file_51)

        # Calculate moving averages (window of 20 episodes) to smooth the curves
        dqn_data['Smooth_Reward'] = dqn_data['score'].rolling(window=20).mean()
        c51_data_2['Smooth_Reward'] = c51_data_2['score'].rolling(window=20).mean()
        c51_data_51['Smooth_Reward'] = c51_data_51['score'].rolling(window=20).mean()
        plt.figure(figsize=(12, 6))

        # Plot Raw Data (faded)
        plt.plot(dqn_data['frame'], dqn_data['score'], color='blue', alpha=0.15)
        plt.plot(c51_data_2['frame'], c51_data_2['score'], color='orange', alpha=0.15)
        plt.plot(c51_data_51['frame'], c51_data_51['score'], color='green', alpha=0.15)

        # Plot Smoothed Data (bold)
        plt.plot(dqn_data['frame'], dqn_data['Smooth_Reward'], color='blue', label='Standard DQN', linewidth=2)
        plt.plot(c51_data_2['frame'], c51_data_2['Smooth_Reward'], color='orange', label='C51 (Categorical DQN 2 atom)', linewidth=2)
        plt.plot(c51_data_51['frame'], c51_data_51['Smooth_Reward'], color='green', label='C51 (Categorical DQN 51 atoms)', linewidth=2)

        # Formatting
        plt.title('DQN vs. C51 Performance on Pong')
        plt.xlabel('frame')
        plt.ylabel('Total Score')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()

        plt.savefig('dqn_vs_c51_comparison.png')
        plt.show()

    except FileNotFoundError:
        print("Error: Ensure both CSV files exist in the current directory.")


if __name__ == "__main__":
    plot_comparison('Pong/results_Pong-v5_standardDQN_seed42.csv', 'Pong/results_Pong-v5_atoms2_seed42_10M.csv','Pong/results_Pong-v5_atoms51_seed42_10M.csv')
