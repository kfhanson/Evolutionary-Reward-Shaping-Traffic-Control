import os
import sys
import traci
import numpy as np
import pandas as pd
from lstm_dqn_agent import LSTMDQNAgent

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    from sumolib import checkBinary
else:
    sys.exit()

SUMO_BINARY = checkBinary('sumo')
CONFIG_FILE = "aeon-froggy-middle-heady-traffic.sumocfg" 
TRAFFIC_LIGHT_ID = 'cluster_3639980474_3640024452_3640024453_699593332'
APPROACH_EDGES = {"north": "749313693#0", "south": "1053267667#1", "east": "749662140#0", "west": "885403818#1"}
AGENT_ACTION_TO_SUMO_PHASE = {0: 0, 1: 2, 2: 4, 3: 6}
SUMO_GREEN_TO_YELLOW_PHASE = {0: 1, 2: 3, 4: 5, 6: 7}

def evaluate_baseline(mode="default", agent=None, sim_steps=1000):
    sumo_cmd = [SUMO_BINARY, "-c", CONFIG_FILE, "--no-step-log", "true", "--waiting-time-memory", "1000", "--time-to-teleport", "-1"]
    
    total_throughput = 0
    all_queues = []
    total_stops = 0
    hidden_state = None
    
    MIN_GREEN_TIME = 10
    MAX_GREEN_TIME = 60
    phase_timer = 0
    current_step = 0
    
    try:
        traci.start(sumo_cmd)
        while current_step < sim_steps:
            if traci.simulation.getMinExpectedNumber() == 0: break
            
            total_throughput += traci.simulation.getArrivedNumber()
            n = traci.edge.getLastStepHaltingNumber(APPROACH_EDGES["north"])
            s = traci.edge.getLastStepHaltingNumber(APPROACH_EDGES["south"])
            e = traci.edge.getLastStepHaltingNumber(APPROACH_EDGES["east"])
            w = traci.edge.getLastStepHaltingNumber(APPROACH_EDGES["west"])
            
            all_queues.append([n, s, e, w])
            total_stops += (n + s + e + w)
            
            current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
            is_green = current_phase in AGENT_ACTION_TO_SUMO_PHASE.values()
            target_phase = current_phase
            
            # Determine Action
            if not is_green or phase_timer >= MIN_GREEN_TIME:
                if mode == "rl" and agent is not None:
                    action, hidden_state = agent.select_action(np.array([n,s,e,w]), epsilon=0.0, hidden=hidden_state)
                    target_phase = AGENT_ACTION_TO_SUMO_PHASE[action]
                elif mode == "rule_based":
                    queues = {2: n, 0: s, 1: e, 3: w}
                    best_action = max(queues, key=queues.get)
                    if queues[best_action] >= 5: target_phase = AGENT_ACTION_TO_SUMO_PHASE[best_action]
                if is_green and phase_timer >= MAX_GREEN_TIME and target_phase == current_phase:
                    target_phase = (current_phase + 2) % 8
            
            # Execute Action
            if current_phase != target_phase and is_green:
                yellow_phase = SUMO_GREEN_TO_YELLOW_PHASE.get(current_phase)
                if yellow_phase is not None:
                    traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, yellow_phase)
                    for _ in range(6): 
                        traci.simulationStep()
                        current_step += 1
                        total_throughput += traci.simulation.getArrivedNumber()
                        n_y, s_y = traci.edge.getLastStepHaltingNumber(APPROACH_EDGES["north"]), traci.edge.getLastStepHaltingNumber(APPROACH_EDGES["south"])
                        e_y, w_y = traci.edge.getLastStepHaltingNumber(APPROACH_EDGES["east"]), traci.edge.getLastStepHaltingNumber(APPROACH_EDGES["west"])
                        all_queues.append([n_y, s_y, e_y, w_y])
                        total_stops += (n_y + s_y + e_y + w_y)
                traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, target_phase)
                phase_timer = 0
            
            traci.simulationStep()
            current_step += 1
            phase_timer += 1
            
    finally:
        traci.close()
        
    variance = np.var(all_queues) if len(all_queues) > 0 else 0
    return total_throughput, variance, total_stops

def train_single_objective_agent(csv_file, weights):
    agent = LSTMDQNAgent()
    agent.memory.load_from_csv(csv_file)
    for _ in range(3000): agent.train_step(weights=weights)
    agent.update_target_network()
    return agent

if __name__ == "__main__":
    csv_file = "sumo_moea_offline_data.csv"
    SIM_STEPS = 1000 
    results = []

    print("\n--- BASELINE 1: Default Fixed-Time ---")
    t1, v1, s1 = evaluate_baseline(mode="default", sim_steps=SIM_STEPS)
    results.append({"Policy": "Default (Fixed-Time)", "Throughput": t1, "Variance": v1, "Stops": s1})

    print("\n--- BASELINE 2: Rule-Based ---")
    t2, v2, s2 = evaluate_baseline(mode="rule_based", sim_steps=SIM_STEPS)
    results.append({"Policy": "Rule-Based Actuated", "Throughput": t2, "Variance": v2, "Stops": s2})

    print("\n--- BASELINE 3: RL (Max Throughput) ---")
    agent_t = train_single_objective_agent(csv_file, weights=[1.0, 0.0, 0.0])
    t3, v3, s3 = evaluate_baseline(mode="rl", agent=agent_t, sim_steps=SIM_STEPS)
    results.append({"Policy": "Single-Obj RL (Max Throughput)", "Throughput": t3, "Variance": v3, "Stops": s3})

    print("\n--- BASELINE 4: RL (Max Fairness) ---")
    agent_f = train_single_objective_agent(csv_file, weights=[0.0, 1.0, 0.0])
    t4, v4, s4 = evaluate_baseline(mode="rl", agent=agent_f, sim_steps=SIM_STEPS)
    results.append({"Policy": "Single-Obj RL (Max Fairness)", "Throughput": t4, "Variance": v4, "Stops": s4})

    df_results = pd.DataFrame(results)
    df_results.to_csv("baseline_results.csv", index=False)
    print("\nSaved new fixed baseline results!")