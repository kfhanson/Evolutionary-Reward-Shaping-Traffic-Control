import os
import sys
import traci
import csv
import numpy as np
import random
import traceback

try:
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
        from sumolib import checkBinary, net
    else:
        sys.exit()
except ImportError:
    sys.exit()

SUMO_BINARY = checkBinary('sumo') 
CONFIG_FILE = "aeon-froggy-middle-heady-traffic.sumocfg" 
NET_FILE = "osm.net.xml" 
TRAFFIC_LIGHT_ID = 'cluster_3639980474_3640024452_3640024453_699593332'
OUTPUT_CSV_FILE = "sumo_moea_offline_data.csv" 

CONGESTION_THRESHOLD_POLICY = 5 # vehicles
MIN_GREEN_TIME_IOT_POLICY = 10  # seconds
YELLOW_TIME_IOT_POLICY = 6      # seconds

# --- Action Space ---
# 0: South Green (Phase 0)
# 1: East Green (Phase 2)
# 2: North Green (Phase 4)
# 3: West Green (Phase 6)
AGENT_ACTION_TO_SUMO_GREEN_PHASE = {0: 0, 1: 2, 2: 4, 3: 6}
SUMO_GREEN_PHASE_TO_AGENT_ACTION = {v: k for k, v in AGENT_ACTION_TO_SUMO_GREEN_PHASE.items()}

SUMO_GREEN_TO_YELLOW_PHASE = {0: 1, 2: 3, 4: 5, 6: 7}

APPROACH_EDGES_FOR_STATE_LOGGING = {
    "north": "749313693#0",
    "south": "1053267667#1",
    "east":  "749662140#0",
    "west":  "885403818#1",
}

def discretize_value_log(value, bins):
    for i, threshold in enumerate(bins):
        if value < threshold: return i
    return len(bins)

def get_sumo_state_for_log():
    try:
        n = traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE_LOGGING["north"])
        s = traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE_LOGGING["south"])
        e = traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE_LOGGING["east"])
        w = traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE_LOGGING["west"])
        return (n, s, e, w)
    except Exception as e_state:
        print(f"Error in get_sumo_state: {e_state}")
        return None

def calculate_multi_objective_rewards():
    # Calculates r1 (Throughput), r2 (Fairness), r3 (Smoothness) for the MOEA
    try:
        halt_counts = []
        r1 = traci.simulation.getArrivedNumber()
        
        for edge_id in APPROACH_EDGES_FOR_STATE_LOGGING.values():
            halting = traci.edge.getLastStepHaltingNumber(edge_id)
            halt_counts.append(halting)

        r2 = -float(np.std(halt_counts)) # Fairness (negative variance)
        r3 = -float(np.sum(halt_counts)) # Smoothness (proxy for stops)
        
        return r1, r2, r3
    except Exception as e_reward:
        print(f"Error in rewards: {e_reward}")
        return 0.0, 0.0, 0.0

def get_rule_based_policy_action(current_tl_sumo_phase_idx):
    # Prioritize the most congested approach
    try:
        congestion_by_action = {
            2: traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE_LOGGING["north"]),
            0: traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE_LOGGING["south"]),
            1: traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE_LOGGING["east"]),
            3: traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE_LOGGING["west"])
        }
        
        sorted_actions = sorted(congestion_by_action.items(), key=lambda item: item[1], reverse=True)
        most_congested_action, max_congestion = sorted_actions[0]
        if max_congestion >= CONGESTION_THRESHOLD_POLICY:
            return most_congested_action
        current_agent_action = SUMO_GREEN_PHASE_TO_AGENT_ACTION.get(current_tl_sumo_phase_idx)
        return current_agent_action if current_agent_action is not None else 0

    except Exception as e:
        print(f"Error in rule based policy: {e}")
        return random.choice(list(AGENT_ACTION_TO_SUMO_GREEN_PHASE.keys()))

def run_sumo_and_log_data(num_simulation_steps=30000, data_collection_policy_name="rule_based"):
    sumo_cmd_list = [SUMO_BINARY, "-c", CONFIG_FILE,
                     "--waiting-time-memory", "1000", "--time-to-teleport", "-1",
                     "--no-step-log", "true", "--seed", str(random.randint(0, 100000))]

    traci.start(sumo_cmd_list)
    csv_header = [
        "state_N", "state_S", "state_E", "state_W", "action", 
        "r1_throughput", "r2_fairness", "r3_smoothness",
        "next_state_N", "next_state_S", "next_state_E", "next_state_W", "done"
    ]
    logged_transitions = 0
    try:
        with open(OUTPUT_CSV_FILE, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)

            current_step = 0
            phase_decision_timer = 0.0
            
            for _ in range(5): traci.simulationStep(); current_step+=1 
            
            state_s_for_log = get_sumo_state_for_log()
            action_applied_by_policy_for_log = -1

            while traci.simulation.getMinExpectedNumber() > 0 and current_step < num_simulation_steps:
                current_tl_sumo_phase_idx = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
                is_currently_green = current_tl_sumo_phase_idx in AGENT_ACTION_TO_SUMO_GREEN_PHASE.values()
                
                if state_s_for_log is None:
                    state_s_for_log = get_sumo_state_for_log()
                    if state_s_for_log is None:
                        traci.simulationStep(); current_step +=1; phase_decision_timer += traci.simulation.getDeltaT(); continue

                if not is_currently_green or phase_decision_timer >= MIN_GREEN_TIME_IOT_POLICY:
                    if data_collection_policy_name == "rule_based":
                        decided_agent_action = get_rule_based_policy_action(current_tl_sumo_phase_idx)
                    else:
                        decided_agent_action = random.choice(list(AGENT_ACTION_TO_SUMO_GREEN_PHASE.keys()))

                    if decided_agent_action is not None: 
                        action_applied_by_policy_for_log = decided_agent_action
                        target_sumo_green_phase = AGENT_ACTION_TO_SUMO_GREEN_PHASE[decided_agent_action]

                        if current_tl_sumo_phase_idx != target_sumo_green_phase:
                            if current_tl_sumo_phase_idx in SUMO_GREEN_TO_YELLOW_PHASE:
                                yellow_phase_idx = SUMO_GREEN_TO_YELLOW_PHASE[current_tl_sumo_phase_idx]
                                traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, yellow_phase_idx)
                                for _ in range(int(YELLOW_TIME_IOT_POLICY / traci.simulation.getDeltaT())):
                                    if traci.simulation.getMinExpectedNumber() == 0: break
                                    traci.simulationStep(); current_step += 1
                                if traci.simulation.getMinExpectedNumber() == 0: break
                            
                            traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, target_sumo_green_phase)
                        phase_decision_timer = 0.0
                    elif is_currently_green:
                         action_applied_by_policy_for_log = SUMO_GREEN_PHASE_TO_AGENT_ACTION.get(current_tl_sumo_phase_idx, -1)

                traci.simulationStep()
                current_step += 1
                phase_decision_timer += traci.simulation.getDeltaT()
                
                state_s_prime_for_log = get_sumo_state_for_log()
                r1, r2, r3 = calculate_multi_objective_rewards()
                done_for_log = traci.simulation.getMinExpectedNumber() == 0 or current_step >= num_simulation_steps

                if state_s_for_log is not None and action_applied_by_policy_for_log != -1 and state_s_prime_for_log is not None:
                    row_to_write = list(state_s_for_log) + \
                                   [action_applied_by_policy_for_log, r1, r2, r3] + \
                                   list(state_s_prime_for_log) + \
                                   [done_for_log]
                    writer.writerow(row_to_write)
                    logged_transitions += 1
                
                state_s_for_log = state_s_prime_for_log
                action_applied_by_policy_for_log = -1 

                if done_for_log: break
            
            print(f"Finished. Logged {logged_transitions} multi-objective transitions to {OUTPUT_CSV_FILE}.")

    except traci.exceptions.TraCIException as e:
        print(f"\nSUMO ERROR: {e}")
    except Exception as e:
        print(f"Unexpected error")
        traceback.print_exc()
    finally:
        try:
            traci.close()
        except Exception:
            pass

if __name__ == "__main__":
    run_sumo_and_log_data(num_simulation_steps=36000, data_collection_policy_name="rule_based")