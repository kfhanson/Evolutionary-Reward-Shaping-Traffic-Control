"""
simulation_recorder.py
───────────────────────────────────────────────────────────────────────────────
Runs a SUMO simulation with the trained RL agent (TraCI) and records per-step
metrics to a CSV file for later analysis.

Output CSV columns (one row per simulation step):
  step, sim_time_s,
  tl_phase,
  halt_north, halt_south, halt_east, halt_west,
  wait_north, wait_south, wait_east, wait_west,
  speed_north, speed_south, speed_east, speed_west,
  vehicles_in_network,
  agent_action, agent_action_label,
  phase_decision_timer
───────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import csv
import traci
import numpy as np
import random
import traceback
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam

# ── SUMO setup ──────────────────────────────────────────────────────────────
try:
    if 'SUMO_HOME' in os.environ:
        sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
        from sumolib import checkBinary
    else:
        sys.exit("Please set the SUMO_HOME environment variable.")
except ImportError:
    sys.exit("Could not import sumolib. Check your SUMO installation.")

# ── Config ───────────────────────────────────────────────────────────────────
SUMO_BINARY      = checkBinary('sumo')          # headless — faster recording
CONFIG_FILE      = "aeon-froggy-middle-heady-traffic.sumocfg"
TRAFFIC_LIGHT_ID = 'cluster_3639980474_3640024452_3640024453_699593332'
MODEL_LOAD_PATH  = "model_lr0.0005_gamma0.95_ep100_bs128_trial7.weights.h5"

# Output CSV — timestamped so multiple runs don't overwrite each other
OUTPUT_DIR = "simulation_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Agent / model params ─────────────────────────────────────────────────────
STATE_FEATURES       = 4
NUM_ACTIONS          = 3   # valid phases on new map: 0, 2, 4
NUM_ACTIONS_IN_MODEL = 4   # old saved model has 4 output neurons
SEQUENCE_LENGTH      = 1
EVAL_EPSILON         = 0.05

MIN_GREEN_TIME = 10   # seconds
YELLOW_TIME    = 6    # seconds

ACTION_TO_GREEN_PHASE = {0: 0, 1: 2, 2: 4}
GREEN_TO_YELLOW_PHASE = {0: 1, 2: 3, 4: 5}
ACTION_LABELS         = {0: "NS_green", 1: "EW_green", 2: "turn_green"}

APPROACH_EDGES = {
    "north": "749313693#0",
    "south": "1053267667#1",
    "east":  "749662140#0",
    "west":  "885403818#1",
}
VEHICLE_BINS = [5, 15, 30]

# ── CSV columns ──────────────────────────────────────────────────────────────
CSV_COLUMNS = [
    "step", "sim_time_s",
    "tl_phase",
    "halt_north", "halt_south", "halt_east", "halt_west",
    "wait_north", "wait_south", "wait_east", "wait_west",
    "speed_north", "speed_south", "speed_east", "speed_west",
    "vehicles_in_network",
    "agent_action", "agent_action_label",
    "phase_decision_timer",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def discretize(value, bins):
    for i, t in enumerate(bins):
        if value < t:
            return i
    return len(bins)


def get_state():
    """Return (state_array_for_model, raw_counts_dict) from SUMO."""
    counts = {}
    for direction, edge in APPROACH_EDGES.items():
        counts[direction] = traci.edge.getLastStepHaltingNumber(edge)
    vec = np.array([
        discretize(counts["north"], VEHICLE_BINS),
        discretize(counts["south"], VEHICLE_BINS),
        discretize(counts["east"],  VEHICLE_BINS),
        discretize(counts["west"],  VEHICLE_BINS),
    ], dtype=np.float32).reshape((1, SEQUENCE_LENGTH, STATE_FEATURES))
    return vec, counts


def record_step(writer, step, sim_time, tl_phase,
                halt, wait, speed,
                vehicles, agent_action, pdt):
    """Write one row to the CSV."""
    writer.writerow({
        "step":               step,
        "sim_time_s":         round(sim_time, 1),
        "tl_phase":           tl_phase,
        "halt_north":         halt["north"],
        "halt_south":         halt["south"],
        "halt_east":          halt["east"],
        "halt_west":          halt["west"],
        "wait_north":         round(wait["north"], 2),
        "wait_south":         round(wait["south"], 2),
        "wait_east":          round(wait["east"],  2),
        "wait_west":          round(wait["west"],  2),
        "speed_north":        round(speed["north"], 3),
        "speed_south":        round(speed["south"], 3),
        "speed_east":         round(speed["east"],  3),
        "speed_west":         round(speed["west"],  3),
        "vehicles_in_network": vehicles,
        "agent_action":       agent_action if agent_action is not None else "",
        "agent_action_label": ACTION_LABELS.get(agent_action, "") if agent_action is not None else "",
        "phase_decision_timer": round(pdt, 1),
    })


def collect_metrics():
    """Pull halting count, avg wait time, and avg speed for each approach edge."""
    halt, wait, speed = {}, {}, {}
    for direction, edge in APPROACH_EDGES.items():
        halt[direction]  = traci.edge.getLastStepHaltingNumber(edge)
        wait[direction]  = traci.edge.getWaitingTime(edge)
        speed[direction] = traci.edge.getLastStepMeanSpeed(edge)
    return halt, wait, speed


# ── Agent ────────────────────────────────────────────────────────────────────

class EvaluationAgent:
    def __init__(self, model_path):
        self.action_size = NUM_ACTIONS
        self.epsilon     = EVAL_EPSILON
        self.q_network   = self._build_model()
        self._load(model_path)
        print("Agent ready.")
        self.q_network.summary()

    def _build_model(self):
        model = Sequential([
            Input(shape=(SEQUENCE_LENGTH, STATE_FEATURES)),
            LSTM(32, activation='relu', return_sequences=False),
            Dense(NUM_ACTIONS_IN_MODEL, activation='linear'),
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def _load(self, path):
        self.q_network.load_weights(path)
        print(f"Weights loaded from: {path}")

    def select_action(self, state):
        if state is None or np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q = self.q_network.predict(state, verbose=0)[0]
        return int(np.argmax(q[:self.action_size]))   # mask out 4th action


# ── Main simulation loop ─────────────────────────────────────────────────────

def run_and_record(agent, episode=1, max_steps=3600, seed_offset=0):
    """
    Run one simulation episode, recording every step to CSV.
    Returns the path to the saved CSV file.
    """
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path   = os.path.join(OUTPUT_DIR, f"episode_{episode}_{timestamp}.csv")

    seed = random.randint(10000, 20000) + seed_offset
    sumo_cmd = [
        SUMO_BINARY, "-c", CONFIG_FILE,
        "--waiting-time-memory", "1000",
        "--time-to-teleport",    "-1",
        "--no-step-log",         "true",
        "--seed",                str(seed),
    ]

    traci.start(sumo_cmd)
    print(f"\n[Episode {episode}] SUMO started (seed={seed})")
    print(f"[Episode {episode}] Recording to: {csv_path}")

    step             = 0
    phase_timer      = 0.0
    last_action      = None

    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()

            state, _ = get_state()

            while traci.simulation.getMinExpectedNumber() > 0 and step < max_steps:
                dt              = traci.simulation.getDeltaT()
                sim_time        = traci.simulation.getTime()
                tl_phase        = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
                is_green        = tl_phase in ACTION_TO_GREEN_PHASE.values()

                # ── Agent decision ────────────────────────────────────────
                if not is_green or phase_timer >= MIN_GREEN_TIME:
                    action              = agent.select_action(state)
                    last_action         = action
                    target_green_phase  = ACTION_TO_GREEN_PHASE[action]

                    if tl_phase != target_green_phase:
                        # Yellow transition
                        if tl_phase in GREEN_TO_YELLOW_PHASE:
                            traci.trafficlight.setPhase(
                                TRAFFIC_LIGHT_ID,
                                GREEN_TO_YELLOW_PHASE[tl_phase]
                            )
                            for _ in range(int(YELLOW_TIME / dt)):
                                if traci.simulation.getMinExpectedNumber() == 0:
                                    break
                                halt, wait, speed = collect_metrics()
                                record_step(
                                    writer, step,
                                    traci.simulation.getTime(),
                                    traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID),
                                    halt, wait, speed,
                                    traci.simulation.getMinExpectedNumber(),
                                    last_action, phase_timer,
                                )
                                traci.simulationStep()
                                step += 1
                            if traci.simulation.getMinExpectedNumber() == 0:
                                break

                        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, target_green_phase)
                        phase_timer = 0.0

                # ── Collect metrics BEFORE stepping ──────────────────────
                halt, wait, speed = collect_metrics()
                vehicles          = traci.simulation.getMinExpectedNumber()

                record_step(
                    writer, step, sim_time, tl_phase,
                    halt, wait, speed,
                    vehicles, last_action, phase_timer,
                )

                # ── Advance simulation ────────────────────────────────────
                traci.simulationStep()
                step        += 1
                phase_timer += dt

                # ── Update state ──────────────────────────────────────────
                try:
                    state, _ = get_state()
                except Exception:
                    state = None

                if traci.simulation.getMinExpectedNumber() == 0:
                    break

        print(f"[Episode {episode}] Finished. Steps recorded: {step}")
        print(f"[Episode {episode}] CSV saved → {csv_path}")
        return csv_path

    except traci.exceptions.FatalTraCIError as e:
        print(f"Fatal TraCI error: {e}")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception:
        print("Unexpected error:")
        traceback.print_exc()
    finally:
        try:
            traci.close()
            print(f"[Episode {episode}] TraCI closed.")
        except Exception as e:
            print(f"[Episode {episode}] TraCI close warning: {e}")

    return csv_path


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = EvaluationAgent(MODEL_LOAD_PATH)

    NUM_EPISODES = 1   # ← change this to run multiple episodes
    saved_files  = []

    for i in range(NUM_EPISODES):
        path = run_and_record(agent, episode=i + 1, seed_offset=i * 100)
        if path:
            saved_files.append(path)

    print("\n── All done ──")
    for p in saved_files:
        print(f"  {p}")
