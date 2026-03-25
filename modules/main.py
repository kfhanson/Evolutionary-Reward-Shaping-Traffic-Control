import os
import sumo_data
import nsga2_optimization

def main():
    csv_file = "sumo_moea_offline_data.csv"

    print("======================================================")
    print(" Pareto-Based Evolutionary Reward Shaping Framework")
    print("======================================================")

    if not os.path.exists(csv_file):
        print("\nGenerating offline traffic data using Rule-based Policy")
        sumo_data.run_sumo_and_log_data(num_simulation_steps=30000, data_collection_policy_name="rule_based")
    else:
        print(f"\nFound existing data '{csv_file}'. Skipping collection.")

    print("\nEvolving LSTM-DQN Reward Weights with NSGA-II...")
    nsga2_optimization.run_nsga2(offline_data_path=csv_file, pop_size=30, n_gen=20)

    print("\nCOMPLETED")

if __name__ == "__main__":
    main()