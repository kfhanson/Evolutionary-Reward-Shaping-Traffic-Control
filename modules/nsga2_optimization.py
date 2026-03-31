import os
import sys
import traci
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import csv

from lstm_dqn_agent import LSTMDQNAgent

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    from sumolib import checkBinary
else:
    sys.exit("Please set SUMO_HOME.")

SUMO_BINARY = checkBinary('sumo') 
CONFIG_FILE = "aeon-froggy-middle-heady-traffic.sumocfg" 
TRAFFIC_LIGHT_ID = 'cluster_3639980474_3640024452_3640024453_699593332'
APPROACH_EDGES = {"north": "749313693#0", "south": "1053267667#1", "east": "749662140#0", "west": "885403818#1"}
AGENT_ACTION_TO_SUMO_PHASE = {0: 0, 1: 2, 2: 4, 3: 6}
SUMO_GREEN_TO_YELLOW_PHASE = {0: 1, 2: 3, 4: 5, 6: 7}

def evaluate_policy_in_sumo(agent, sim_steps=1000):
    sumo_cmd = [SUMO_BINARY, "-c", CONFIG_FILE, "--no-step-log", "true", "--waiting-time-memory", "1000", "--time-to-teleport", "-1"]
    
    total_throughput = 0
    all_queues = []
    total_stops = 0
    hidden_state = None
    
    MIN_GREEN_TIME = 10
    phase_timer = 0
    current_step = 0
    
    try:
        traci.start(sumo_cmd)
        while current_step < sim_steps:
            if traci.simulation.getMinExpectedNumber() == 0: break
            
            # PROPERLY ACCUMULATE THROUGHPUT
            total_throughput += traci.simulation.getArrivedNumber()
            
            n = traci.edge.getLastStepHaltingNumber(APPROACH_EDGES["north"])
            s = traci.edge.getLastStepHaltingNumber(APPROACH_EDGES["south"])
            e = traci.edge.getLastStepHaltingNumber(APPROACH_EDGES["east"])
            w = traci.edge.getLastStepHaltingNumber(APPROACH_EDGES["west"])
            
            all_queues.append([n, s, e, w])
            total_stops += (n + s + e + w)
            
            current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
            is_green = current_phase in AGENT_ACTION_TO_SUMO_PHASE.values()
            
            # Only change phase if it's been green for at least MIN_GREEN_TIME
            if not is_green or phase_timer >= MIN_GREEN_TIME:
                action, hidden_state = agent.select_action(np.array([n,s,e,w]), epsilon=0.0, hidden=hidden_state)
                target_phase = AGENT_ACTION_TO_SUMO_PHASE[action]
                
                if current_phase != target_phase and is_green:
                    yellow_phase = SUMO_GREEN_TO_YELLOW_PHASE.get(current_phase)
                    if yellow_phase is not None:
                        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, yellow_phase)
                        # 6 seconds of yellow light
                        for _ in range(6): 
                            traci.simulationStep()
                            current_step += 1
                            total_throughput += traci.simulation.getArrivedNumber()
                            # Record stats during yellow light
                            n_y, s_y = traci.edge.getLastStepHaltingNumber(APPROACH_EDGES["north"]), traci.edge.getLastStepHaltingNumber(APPROACH_EDGES["south"])
                            e_y, w_y = traci.edge.getLastStepHaltingNumber(APPROACH_EDGES["east"]), traci.edge.getLastStepHaltingNumber(APPROACH_EDGES["west"])
                            all_queues.append([n_y, s_y, e_y, w_y])
                            total_stops += (n_y + s_y + e_y + w_y)
                            
                    traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, target_phase)
                    phase_timer = 0
                elif not is_green and current_phase != target_phase:
                    traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, target_phase)
                    phase_timer = 0
            
            traci.simulationStep()
            current_step += 1
            phase_timer += 1
            
    finally:
        traci.close()
        
    variance = np.var(all_queues) if len(all_queues) > 0 else 0
    return total_throughput, variance, total_stops

class TrafficRewardShapingProblem(Problem):
    def __init__(self, offline_data_path):
        super().__init__(n_var=3, n_obj=3, xl=np.array([0.0,0.0,0.0]), xu=np.array([1.0,1.0,1.0]))
        self.offline_data_path = offline_data_path
        
    def _evaluate(self, X, out, *args, **kwargs):
        f1, f2, f3 = np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X))
        for i in range(len(X)):
            w = X[i]
            w_sum = np.sum(w)
            w = w / w_sum if w_sum > 0 else np.array([0.33, 0.33, 0.33]) 
            
            agent = LSTMDQNAgent()
            agent.memory.load_from_csv(self.offline_data_path)
            for _ in range(1500): 
                agent.train_step(weights=w.tolist())
            agent.update_target_network()
            
            throughput, variance, stops = evaluate_policy_in_sumo(agent, sim_steps=1000)
            f1[i], f2[i], f3[i] = -throughput, variance, stops

        out["F"] = np.column_stack([f1, f2, f3])

def run_nsga2(offline_data_path, pop_size=10, n_gen=5):
    print(f"--- Starting NSGA-II Optimization ---")
    problem = TrafficRewardShapingProblem(offline_data_path)
    algorithm = NSGA2(pop_size=pop_size, n_offsprings=pop_size, sampling=FloatRandomSampling(), crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20), eliminate_duplicates=True)
    res = minimize(problem, algorithm, get_termination("n_gen", n_gen), seed=1, verbose=True)
    
    print("\n=== PARETO FRONT EXTRACTED ===")
    with open("pareto_front_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["w_throughput", "w_fairness", "w_smoothness", "throughput", "variance", "stops"])
        for idx, (weights, fitness) in enumerate(zip(res.X, res.F)):
            w = weights / np.sum(weights)
            writer.writerow([w[0], w[1], w[2], -fitness[0], fitness[1], fitness[2]])
    print("Saved Pareto results to 'pareto_front_results.csv'")
    return res