pip import os
import sys
import traci
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
import traceback

try:
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
        from sumolib import checkBinary
    else:
        sys.exit("SUMO_HOME environment variable is not set.")
except ImportError:
    sys.exit("Please set the SUMO_HOME environment variable or ensure SUMO tools are in your Python path.")


SUMO_BINARY = checkBinary('sumo-gui')
CONFIG_FILE = "aeon-froggy-middle-heady-traffic.sumocfg"  # ← updated

# ── NEW MAP: single TL cluster ──────────────────────────────────────────────
TRAFFIC_LIGHT_ID = 'cluster_3639980474_3640024452_3640024453_699593332'  # ← updated

MODEL_LOAD_PATH = "grid_search_models/model_lr0.0005_gamma0.95_ep100_bs128_trial7.weights.h5"

STATE_FEATURES  = 4   # N, S, E, W discretized congestion counts
NUM_ACTIONS          = 3   # ← updated: new map has 3 green phases (0, 2, 4)
NUM_ACTIONS_IN_MODEL = 4   # old model was trained with 4 actions — keep for weight loading
SEQUENCE_LENGTH = 1
EVAL_EPSILON    = 0.05

MIN_GREEN_TIME = 10   # seconds
YELLOW_TIME    = 6    # seconds

# ── NEW MAP: phase mappings ─────────────────────────────────────────────────
# Phase 0 (GREEN): N+S through movements
# Phase 2 (GREEN): E+W through movements
# Phase 4 (GREEN): turning movements
# Agent action → SUMO green phase index
ACTION_TO_GREEN_PHASE = {0: 0, 1: 2, 2: 4}               # ← updated

# SUMO green phase → its yellow phase
GREEN_TO_YELLOW_PHASE = {0: 1, 2: 3, 4: 5}               # ← updated

# ── NEW MAP: approach edges ─────────────────────────────────────────────────
APPROACH_EDGES_FOR_STATE = {
    "north": "749313693#0",   # ← updated (was 754598165#2)
    "south": "1053267667#1",  # ← updated (was 1053267667#3)
    "east":  "749662140#0",   # unchanged
    "west":  "885403818#1",   # ← updated (was 885403818#2)
}

VEHICLE_BINS_FOR_STATE = [5, 15, 30]


def discretize_value(value, bins):
    for i, threshold in enumerate(bins):
        if value < threshold:
            return i
    return len(bins)


def get_sumo_state_eval():
    """Retrieves current state from SUMO for visualization."""
    try:
        n = discretize_value(traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE["north"]), VEHICLE_BINS_FOR_STATE)
        s = discretize_value(traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE["south"]), VEHICLE_BINS_FOR_STATE)
        e = discretize_value(traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE["east"]),  VEHICLE_BINS_FOR_STATE)
        w = discretize_value(traci.edge.getLastStepHaltingNumber(APPROACH_EDGES_FOR_STATE["west"]),  VEHICLE_BINS_FOR_STATE)
        state_vector = np.array([n, s, e, w], dtype=np.float32)
        return state_vector.reshape((1, SEQUENCE_LENGTH, STATE_FEATURES))
    except Exception as e_state:
        print(f"Error in get_sumo_state_eval: {e_state}")
        return None


class EvaluationAgent:
    def __init__(self, state_dims, action_size, sequence_length, model_path):
        self.state_feature_size = state_dims
        self.action_size        = action_size
        self.sequence_length    = sequence_length
        self.epsilon            = EVAL_EPSILON

        self.q_network = self._build_lstm_model()
        self.load_model(model_path)
        print("Visualization Agent Initialized and Model Loaded.")
        self.q_network.summary()

    def _build_lstm_model(self):
        # Build with NUM_ACTIONS_IN_MODEL (4) so saved weights load correctly.
        # During action selection we only use the first NUM_ACTIONS (3) outputs.
        model = Sequential([
            Input(shape=(self.sequence_length, self.state_feature_size)),
            LSTM(32, activation='relu', return_sequences=False),
            Dense(NUM_ACTIONS_IN_MODEL, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def load_model(self, filepath):
        try:
            self.q_network.load_weights(filepath)
            print(f"Visualization model weights loaded from {filepath}")
        except Exception as e:
            print(f"Error loading visualization model weights: {e}")
            raise

    def select_action(self, current_state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        if current_state is None:
            print("Warning: current_state is None in select_action, choosing random action.")
            return random.randrange(self.action_size)
        q_values = self.q_network.predict(current_state, verbose=0)
        # Mask out the 4th action (index 3) — it has no valid phase on the new map
        valid_q_values = q_values[0][:self.action_size]
        return int(np.argmax(valid_q_values))


def run_visualization_episode(agent, episode_num_vis, max_steps_vis=3600, sim_seed_offset=0):
    """Runs a single SUMO episode with the trained agent for visualization."""
    sumo_cmd_list = [SUMO_BINARY, "-c", CONFIG_FILE]
    sumo_cmd_list.extend([
        "--waiting-time-memory", "1000",
        "--time-to-teleport",    "-1",
        "--no-step-log",         "true",
        "--seed",                str(random.randint(10000, 20000) + sim_seed_offset)
    ])

    traci.start(sumo_cmd_list)
    print(f"  SUMO visualization episode {episode_num_vis} started (Epsilon: {agent.epsilon}).")

    current_step         = 0
    phase_decision_timer = 0.0

    current_sumo_state_for_eval = get_sumo_state_eval()
    if current_sumo_state_for_eval is None:
        print("  Error: Failed to get initial SUMO state for visualization. Ending episode.")
        traci.close()
        return

    try:
        while traci.simulation.getMinExpectedNumber() > 0 and current_step < max_steps_vis:
            current_tl_sumo_phase_idx = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
            is_currently_green        = current_tl_sumo_phase_idx in ACTION_TO_GREEN_PHASE.values()

            if not is_currently_green or phase_decision_timer >= MIN_GREEN_TIME:
                agent_action_choice    = agent.select_action(current_sumo_state_for_eval)
                target_sumo_green_phase = ACTION_TO_GREEN_PHASE[agent_action_choice]

                if current_tl_sumo_phase_idx != target_sumo_green_phase:
                    if current_tl_sumo_phase_idx in GREEN_TO_YELLOW_PHASE:
                        yellow_phase_idx = GREEN_TO_YELLOW_PHASE[current_tl_sumo_phase_idx]
                        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, yellow_phase_idx)
                        for _ in range(int(YELLOW_TIME / traci.simulation.getDeltaT())):
                            if traci.simulation.getMinExpectedNumber() == 0:
                                break
                            traci.simulationStep()
                            current_step += 1
                        if traci.simulation.getMinExpectedNumber() == 0:
                            break

                    traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, target_sumo_green_phase)
                    phase_decision_timer = 0.0

            traci.simulationStep()
            current_step         += 1
            phase_decision_timer += traci.simulation.getDeltaT()

            current_sumo_state_for_eval = get_sumo_state_eval()
            if current_sumo_state_for_eval is None and traci.simulation.getMinExpectedNumber() > 0:
                print("  Error: Lost SUMO state mid-visualization. Ending episode.")
                break

            if traci.simulation.getMinExpectedNumber() == 0:
                break

        print(f"  Visualization episode {episode_num_vis} finished. Steps: {current_step}.")

    except traci.exceptions.FatalTraCIError as e:
        print(f"Fatal TraCI Error during visualization: {e}")
    except KeyboardInterrupt:
        print("Visualization interrupted by user.")
    except Exception:
        print("Unexpected Python error during visualization episode:")
        traceback.print_exc()
    finally:
        try:
            if 'traci' in sys.modules:
                traci.close()
                print(f"  TraCI connection closed for episode {episode_num_vis}.")
        except traci.exceptions.TraCIException as e:
            print(f"  TraCI warning on close for episode {episode_num_vis}: {e}")
        except Exception as e_close:
            print(f"  Unexpected error during TraCI close for episode {episode_num_vis}: {e_close}")


if __name__ == "__main__":
    evaluation_agent = EvaluationAgent(STATE_FEATURES, NUM_ACTIONS, SEQUENCE_LENGTH, MODEL_LOAD_PATH)
    num_vis_episodes = 1
    for i in range(num_vis_episodes):
        run_visualization_episode(evaluation_agent, i + 1, sim_seed_offset=i * 100)
