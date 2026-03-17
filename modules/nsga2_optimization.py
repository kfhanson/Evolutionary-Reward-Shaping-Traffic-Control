import os
import sys
import csv
import traci
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# Import the agent from Phase 2
from lstm_dqn_agent import LSTMDQNAgent

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    from sumolib import checkBinary
else:
    sys.exit("Please set SUMO_HOME.")

SUMO_BINARY = checkBinary('sumo') # Use 'sumo' (no GUI) for fast training
CONFIG_FILE = "aeon-froggy-middle-heady-traffic.sumocfg" 
TRAFFIC_LIGHT_ID = 'cluster_3639980474_3640024452_3640024453_699593332'
APPROACH_EDGES = {"north": "749313693#0", "south": "1053267667#1", "east": "749662140#0", "west": "885403818#1"}
AGENT_ACTION_TO_SUMO_PHASE = {0: 0, 1: 2, 2: 4, 3: 6}

def evaluate_policy_in_sumo(agent, sim_steps=3600):
    """Runs a short SUMO simulation to test the agent's fitness."""
    sumo_cmd = [SUMO_BINARY, "-c", CONFIG_FILE, "--no-step-log", "true", "--waiting-time-memory", "1000", "--time-to-teleport", "-1"]
    
    try:
        traci.start(sumo_cmd)
        all_queues = []
        total_stops = 0
        hidden_state = None
        
        for _ in range(sim_steps):
            if traci.simulation.getMinExpectedNumber() == 0: break
                
            n = traci.edge.getLastStepHaltingNumber(APPROACH_EDGES["north"])
            s = traci.edge.getLastStepHaltingNumber(APPROACH_EDGES["south"])
            e = traci.edge.getLastStepHaltingNumber(APPROACH_EDGES["east"])
            w = traci.edge.getLastStepHaltingNumber(APPROACH_EDGES["west"])
            
            all_queues.append([n, s, e, w])
            total_stops += (n + s + e + w)
            
            current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
            if current_phase in AGENT_ACTION_TO_SUMO_PHASE.values():
                action, hidden_state = agent.select_action(np.array([n,s,e,w]), epsilon=0.0, hidden=hidden_state)
                target_phase = AGENT_ACTION_TO_SUMO_PHASE[action]
                if current_phase != target_phase:
                    traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, target_phase)
            
            traci.simulationStep()
            
        throughput = traci.simulation.getArrivedNumber()
    finally:
        traci.close()
        
    variance = np.var(all_queues) if len(all_queues) > 0 else 0
    return throughput, variance, total_stops

class TrafficRewardShapingProblem(Problem):
    def __init__(self, offline_data_path):
        super().__init__(n_var=3, n_obj=3, xl=np.array([0.0,0.0,0.0]), xu=np.array([1.0,1.0,1.0]))
        self.offline_data_path = offline_data_path
        
    def _evaluate(self, X, out, *args, **kwargs):
        f1, f2, f3 = np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X))
        
        for i in range(len(X)):
            w = X[i]
            w_sum = np.sum(w)
            w = w / w_sum if w_sum > 0 else np.array([0.33, 0.33, 0.33]) # Normalize weights
            
            agent = LSTMDQNAgent()
            agent.memory.load_from_csv(self.offline_data_path)
            
            # Pre-train offline on these specific weights
            for _ in range(200): agent.train_step(weights=w.tolist())
            agent.update_target_network()
            
            # Evaluate in SUMO
            throughput, variance, stops = evaluate_policy_in_sumo(agent)
            
            # pymoo MINIMIZES. We want to maximize throughput, minimize variance, minimize stops
            f1[i] = -throughput 
            f2[i] = variance
            f3[i] = stops

        out["F"] = np.column_stack([f1, f2, f3])

def run_nsga2(offline_data_path, pop_size=10, n_gen=5):
    print(f"--- Starting NSGA-II Optimization (Pop: {pop_size}, Gen: {n_gen}) ---")
    problem = TrafficRewardShapingProblem(offline_data_path)
    
    algorithm = NSGA2(
        pop_size=pop_size,
        n_offsprings=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    res = minimize(problem, algorithm, get_termination("n_gen", n_gen), seed=1, verbose=True)
    
    print("\n=== PARETO FRONT EXTRACTED ===")
    # for idx, (weights, fitness) in enumerate(zip(res.X, res.F)):
    #     w = weights / np.sum(weights)
    #     print(f"Policy {idx+1} | Weights (T, F, S): [{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}] | "
    #           f"Metrics: Throughput={-fitness[0]}, Var={fitness[1]:.2f}, Stops={fitness[2]}")

    with open("pareto_front_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["w_throughput", "w_fairness", "w_smoothness", "throughput", "variance", "stops"])
        
        for idx, (weights, fitness) in enumerate(zip(res.X, res.F)):
            w = weights / np.sum(weights) # Normalize
            throughput = -fitness[0]
            variance = fitness[1]
            stops = fitness[2]
            
            writer.writerow([w[0], w[1], w[2], throughput, variance, stops])
            
            print(f"Policy {idx+1} | Weights: [{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}] | "
                  f"Throughput={throughput}, Var={variance:.2f}, Stops={stops}")
            
    print("Saved Pareto results to 'pareto_front_results.csv'")
    return res