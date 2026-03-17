import os

# Import your custom modules
import sumo_data
import nsga2_optimization

def main():
    csv_file = "sumo_moea_offline_data.csv"

    print("======================================================")
    print(" Pareto-Based Evolutionary Reward Shaping Framework")
    print("======================================================")

    # 1. Run Data Collection if the CSV doesn't exist
    if not os.path.exists(csv_file):
        print("\n[PHASE 1] Generating offline traffic data using Rule-based Policy...")
        # 30,000 steps roughly simulates a few hours of traffic
        sumo_data.run_sumo_and_log_data(num_simulation_steps=30000, data_collection_policy_name="rule_based")
    else:
        print(f"\n[PHASE 1] Found existing data '{csv_file}'. Skipping collection.")

    # 2. Run the NSGA-II Optimization
    print("\n[PHASE 2 & 3] Evolving LSTM-DQN Reward Weights with NSGA-II...")
    # NOTE: pop_size=10 and n_gen=5 is for quick testing (takes ~5-10 minutes). 
    # For actual research results, change to pop_size=30, n_gen=20.
    nsga2_optimization.run_nsga2(offline_data_path=csv_file, pop_size=30, n_gen=20)

    print("\n[COMPLETE] Framework execution finished successfully.")

if __name__ == "__main__":
    main()