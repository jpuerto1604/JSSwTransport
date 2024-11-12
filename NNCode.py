import pandas as pd
from azure.storage.blob import BlobServiceClient
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import logging
import json
import os
from io import BytesIO

# Data Loading and Transport/Processing Times

# Define transport times between locations for different layouts
df1 = pd.DataFrame(np.array([
    [0, 6, 8, 10, 12],
    [12, 0, 6, 8, 10],
    [10, 6, 0, 6, 8],
    [8, 8, 6, 0, 6],
    [6, 10, 8, 6, 0]]),
    columns=["LU", "M1", "M2", "M3", "M4"], index=["LU", "M1", "M2", "M3", "M4"])

df2 = pd.DataFrame(np.array([
    [0, 4, 6, 8, 6],
    [6, 0, 2, 4, 2],
    [8, 12, 0, 2, 4],
    [6, 10, 12, 0, 2],
    [4, 8, 10, 12, 0]]),
    columns=["LU", "M1", "M2", "M3", "M4"], index=["LU", "M1", "M2", "M3", "M4"])

df3 = pd.DataFrame(np.array([
    [0, 2, 4, 10, 12],
    [12, 0, 2, 8, 10],
    [10, 12, 0, 6, 8],
    [4, 6, 8, 0, 2],
    [2, 4, 6, 12, 0]]),
    columns=["LU", "M1", "M2", "M3", "M4"], index=["LU", "M1", "M2", "M3", "M4"])

df4 = pd.DataFrame(np.array([
    [0, 4, 8, 10, 14],
    [18, 0, 4, 6, 10],
    [20, 14, 0, 8, 6],
    [12, 8, 6, 0, 6],
    [14, 14, 12, 6, 0]]),
    columns=["LU", "M1", "M2", "M3", "M4"], index=["LU", "M1", "M2", "M3", "M4"])

# Load processing times and machine assignments from Excel file in Azure Blob Storage

connection_string = "DefaultEndpointsProtocol=https;AccountName=aml13407327522;AccountKey=chc0fH0gVEM0ErWPJvcrpoW9+4FUae1Jlhf9SFETfShUkJMTATkz0GI/dVUi1VQAREQaoKR0ugAt+AStOriOMQ==;EndpointSuffix=core.windows.net"
container_name = "azureml"
blob_name = "Data.xlsx"
# Create a BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
# Get the blob client
blob_client = blob_service_client.get_blob_client(container_name, blob_name)
# Download the blob content to a BytesIO object
blob_data = blob_client.download_blob().readall()
excel_data = BytesIO(blob_data)

try:
    xls = pd.read_excel(excel_data, sheet_name='Macrodata', usecols='F:H, J:P, R:X')
except FileNotFoundError:
    logging.error("File not found")
    exit(1)  # Exit gracefully
except Exception as e:
    logging.error(f"Error loading data: {e}")
    exit(1)
data = pd.DataFrame(xls)
data.loc[:, 'nj'] = data.loc[:, 'nj'] + 1  # Adjust job numbers
data = data.fillna('')

# Extract processing times and machine assignments
p_times = pd.DataFrame(data.iloc[:, :10].to_numpy(), columns=["Set", "Job", "nj", "P1", "P2", "P3", "P4", "P5", "P6", "P7"])
m_data = pd.DataFrame(data.iloc[:, [0, 1, 2] + list(range(10, data.shape[1]))].to_numpy(),columns=["Set", "Job", "nj"] + [f"M{i}" for i in range(1, data.shape[1] - 9)])


def t_times(layout, start, end):
    '''Parameters:
    - layout: The layout number for transport times (1 to 4)
    - start: Starting location
    - end: Destination

    Returns:
    -The transport time between the start and end locations for the given layout.
    '''
    if layout == 1:
        return df1.loc[start, end]
    elif layout == 2:
        return df2.loc[start, end]
    elif layout == 3:
        return df3.loc[start, end]
    elif layout == 4:
        return df4.loc[start, end]
    else:
        raise ValueError("Invalid layout number.")
def jobs(nset):
    '''
    Parameters:
    -nset: the specific job set being used

    Returns:
    -The job data for the given set, including the job number and machine assignments.
    '''
    return m_data[m_data['Set'] == nset].iloc[:, 1:].reset_index(drop=True)

def processing(nset):
    '''
    Parameters:
    -nset: the specific job set being used

    Returns:
    -The processing times for the given set, including the job number and processing times for each machine.
    '''
    return p_times[p_times['Set'] == nset].iloc[:, 1:].reset_index(drop=True)

# Job Shop Scheduling Environment with Time Tracking

class JobShopEnv:
    def __init__(self, layout, nset, jobs_data, processing_data, num_agvs):
        """
        Job Shop Scheduling environment with 4 machines (M1 to M4) and LU (Loading-Unloading area).
        Supports multiple AGVs (from 1 to 5) for transport.
        Tracks start and end times for each job on each machine dynamically.

        Parameters:
        - layout: The layout number for transport times (1 to 4).
        - nset: The specific job set being used.
        - jobs_data: DataFrame containing job sequences (including LU).
        - processing_data: DataFrame containing processing times for each job and machine.
        - num_agvs: Number of AGVs available for transport (from 1 to 5).
        """
        self.layout = layout
        self.nset = nset
        self.jobs_data = jobs_data
        self.processing_data = processing_data
        self.num_agvs = num_agvs  # Number of AGVs available
        self.current_time = 0
        self.done = False
        self.machine_status = np.zeros(4)  # Status of machines M1 to M4; LU is always available
        self.agv_status = np.zeros(num_agvs)  # Status of each AGV (0 for free, 1 for busy)
        self.agv_locations = ["LU"] * num_agvs  # Current location of each AGV

        # Track remaining times for each job at each machine in its sequence
        self.job_times = {job_id: np.zeros(len(self.jobs_data.iloc[job_id, 2:2+self.jobs_data.iloc[job_id,1]].dropna())) for job_id in range(len(self.jobs_data))}

        # Dictionary to track which machine each job needs to go to next in its sequence
        self.job_next_machine = {job_id: 0 for job_id in range(len(self.jobs_data))}

        # Data Structure for Time Tracking
        # Dictionary to store start and end times for each job on each machine
        # Structure: {job_id: {machine: {'start': start_time, 'end': end_time}}}
        self.job_machine_times = {job_id: {} for job_id in range(len(self.jobs_data))}

        # Current location of each job (starting at LU)
        self.job_locations = {job_id: "LU" for job_id in range(len(self.jobs_data))}
        self.machine_available_at = np.zeros(4)  # Time when each machine becomes available
        self.agv_available_at = np.zeros(self.num_agvs)  # Time when each AGV becomes available


    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
        - state: The initial state of the environment.
        """
        self.current_time = 0
        self.done = False
        self.machine_status.fill(0)  # All machines (M1 to M4) start free
        self.agv_status.fill(0)      # All AGVs start free
        self.agv_locations = ["LU"] * self.num_agvs  # All AGVs at LU

        # Reset job-machine assignment tracking and job times
        self.job_next_machine = {job_id: 0 for job_id in range(len(self.jobs_data))}
        self.job_times = {job_id: np.zeros(len(self.jobs_data.iloc[job_id, 2:2+self.jobs_data.iloc[job_id,1]].dropna())) for job_id in range(len(self.jobs_data))}
        self.job_locations = {job_id: "LU" for job_id in range(len(self.jobs_data))}

        # Reset the job-machine times tracking
        self.job_machine_times = {job_id: {} for job_id in range(len(self.jobs_data))}
        self.machine_available_at.fill(0)
        self.agv_available_at.fill(0)

        logging.info("Environment reset:")
        logging.info(f"Current Time: {self.current_time}")
        logging.info(f"Machine Status: {self.machine_status}")
        logging.info(f"AGV Status: {self.agv_status}")
        logging.info(f"AGV Locations: {self.agv_locations}")

        return self._get_state()
    
    def step(self, action):
        """
        Execute the given action in the environment.

        Parameters:
        - action: A tuple (job, machine, agv), assigning a job to a machine with a specific AGV.

        Returns:
        - next_state: The updated state after the action.
        - reward: The reward obtained from taking the action.
        - done: A boolean indicating if all jobs are completed.
        """
        if action is None:
            # Advance time if no valid action is taken
            next_event_time = self.get_next_event_time()
            if next_event_time == float('inf'):
                # Ensure time advances by some small step if still `inf`
                next_event_time = self.current_time + 1
                logging.info(f"No valid action available, advancing time to {next_event_time}")
            self.current_time = next_event_time
            self.update_resource_states()  # Update machine and AGV states
            return self._get_state(), 0, self._check_done()

        #Proceed if action is valid
        job, machine, agv = action
        logging.info(f"Executing action: Job {job}, Machine {machine}, AGV {agv}")

        # Validate AGV availability
        if self.agv_status[agv] == 1:
             # AGV is busy, advance time and wait for the next available event
            logging.warning(f"AGV {agv} is currently busy. Advancing time.")
            self.current_time = self.get_next_event_time()
            self.update_resource_states()  # Update machine and AGV statesAA
            return self._get_state(), 0, self._check_done()

        # Get the job's machine sequence and the correct machine it needs to go to
        num_operations = self.jobs_data.iloc[job,1]
        job_sequence = self.jobs_data.iloc[job, 2:2+num_operations].dropna().tolist()
        next_machine_index = self.job_next_machine[job]
        if next_machine_index >= len(job_sequence):
            # Job has already completed its sequence
            return self._get_state(), 0, self._check_done()
        correct_machine = job_sequence[next_machine_index]

        # Ensure the job is assigned to the correct machine (or LU)
        if correct_machine != machine:
            logging.warning(f"Job {job} cannot be assigned to machine {machine}. It should go to machine {correct_machine}.")
            self.current_time = self.get_next_event_time()
            self.update_resource_states()
            return self._get_state(), 0, self._check_done()

        # Calculate transport time
        agv_location = self.agv_locations[agv]
        job_location = self.job_locations[job]
        empty_move_time = t_times(self.layout, agv_location, job_location) if agv_location != job_location else 0
        loaded_move_time = t_times(self.layout, job_location, machine)
        transport_time = empty_move_time + loaded_move_time

        # Update job and AGV locations
        self.agv_locations[agv] = machine
        self.job_locations[job] = machine

        if machine != "LU":
            # Machines M1 to M4

            # Check if the machine is available
            machine_index = int(machine[1]) - 1  # Convert 'M1' to index 0
            if self.machine_status[machine_index] == 1:
                # Machine is busy, advance time and wait for the next available event
                logging.warning(f"Machine {machine} is currently busy. Advancing time.")
                self.current_time = self.get_next_event_time()
                self.update_resource_states()  # Update machine and AGV states
                return self._get_state(), 0, self._check_done()

            # Calculate start and end times
            start_time = self.get_next_event_time()
            process_time = self.processing_data.iloc[job, 2 + next_machine_index]  # Get processing time for the job
            end_time = start_time + process_time

            # Update machine and AGV statuses and availability times
            self.machine_status[machine_index] = 1
            self.machine_available_at[machine_index] = end_time

            self.agv_status[agv] = 1
            self.agv_available_at[agv] = start_time + transport_time

            # Record the job's processing times on the machine
            self.job_machine_times[job][machine] = {'start': start_time, 'end': end_time}

            # Negative reward to minimize makespan
            reward = -(transport_time + process_time)

            # Mark the job's machine assignment as completed
            self.job_next_machine[job] += 1

        else:
            # If the machine is LU (Loading-Unloading area)

            # Calculate start and end times
            start_time = self.get_next_event_time()
            end_time = start_time  # No processing at LU

            # Update AGV status and availability time
            self.agv_status[agv] = 1
            self.agv_available_at[agv] = start_time

            # Record the job's time at LU
            self.job_machine_times[job][machine] = {'start': start_time, 'end': end_time}

            # Negative reward to minimize makespan
            reward = -transport_time

            # Mark the job's machine assignment as completed
            self.job_next_machine[job] += 1

        # Check if all jobs are finished
        self.done = self._check_done()

        # Get the next state of the system
        next_state = self._get_state()

        self.current_time = max(self.current_time, self.get_next_event_time())

        logging.info(f"Updated Current Time: {self.current_time}")
        logging.info(f"Machine Status: {self.machine_status}")
        logging.info(f"AGV Status: {self.agv_status}")

        return next_state, reward, self.done


    def _find_available_agv(self):
        """
        Find an available AGV.

        Returns:
        - Index of an available AGV, or None if all are busy.
        """
        for idx, status in enumerate(self.agv_status):
            if status == 0:  # AGV is free
                return idx
        return None  # No AGV is available

    def _get_state(self):
        """
        Return the current state of the environment.

        State includes:
        - Machine statuses (M1 to M4).
        - AGV statuses.
        - Job-machine times for each job.

        Returns:
        - state: A numpy array representing the current state.
        """
        # Combine machine statuses and AGV statuses
        state = np.concatenate([self.machine_status, self.agv_status])

        # Add the job-machine times for all jobs
        for job_id in range(len(self.jobs_data)):
            state = np.concatenate([state, self.job_times[job_id]])

        # Add AGV locations
        agv_location_indices = [machine_to_index(loc) for loc in self.agv_locations]
        state = np.concatenate([state, agv_location_indices])

        # Add job locations
        job_location_indices = [machine_to_index(self.job_locations[job_id]) for job_id in range(len(self.jobs_data))]
        state = np.concatenate([state, job_location_indices])

        return state

    def _check_done(self):
        """
        Check if all jobs have completed their sequences.
        
        Returns:
        - done: True if all jobs are completed, False otherwise.
        """
        return all(self.job_next_machine[job] == len(self.jobs_data.iloc[job, 2:2+self.jobs_data.iloc[job,1]].dropna()) for job in range(len(self.jobs_data)))
    
    def update_resource_states(self):
        # Update machines
        for i in range(len(self.machine_status)):
            if self.machine_status[i] == 1 and self.current_time >= self.machine_available_at[i]:
                self.machine_status[i] = 0  # Machine becomes free
                self.machine_available_at[i] = float('inf')  # Set to `inf` to indicate currently unused
                logging.info(f"Machine {i + 1} is now available at time {self.current_time}")

        # Update AGVs
        for i in range(len(self.agv_status)):
            if self.agv_status[i] == 1 and self.current_time >= self.agv_available_at[i]:
                self.agv_status[i] = 0  # AGV becomes free
                self.agv_available_at[i] = float('inf')  # Set to `inf` to indicate currently unused
                logging.info(f"AGV {i} is now available at time {self.current_time}")
        
    def get_next_event_time(self):
        # Calculate next available times for machines and AGVs
        next_machine_available = np.min(self.machine_available_at[self.machine_status == 1]) if np.any(self.machine_status == 1) else float('inf')
        next_agv_available = np.min(self.agv_available_at[self.agv_status == 1]) if np.any(self.agv_status == 1) else float('inf')

        # Find the minimum of the available times, avoiding infinite results
        next_event_time = min(next_machine_available, next_agv_available)

        # Ensure time always advances by at least a small step to avoid stagnation
        if next_event_time == float('inf') or next_event_time <= self.current_time:
            next_event_time = self.current_time + 1  # or a small epsilon increment

        logging.info(f"Next Event Time calculated: {next_event_time}")
        return next_event_time
    


# Deep Q-Network (DQN) Model

class DQNScheduler(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Deep Q-Network model to estimate Q-values for each action.

        Parameters:
        - input_dim: The number of features in the state representation.
        - output_dim: The number of possible actions (job-machine pairings).
        """
        super(DQNScheduler, self).__init__()

        # Neural network layers
        self.fc1 = nn.Linear(input_dim, 128)  # First hidden layer with 128 neurons
        self.fc2 = nn.Linear(128, 64)         # Second hidden layer with 64 neurons
        self.fc3 = nn.Linear(64, output_dim)  # Output layer to predict Q-values for actions

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the DQN.

        Parameters:
        - x: Input state tensor.

        Returns:
        - Output tensor with Q-values for each action.
        """
        x = self.relu(self.fc1(x))  # First hidden layer
        x = self.relu(self.fc2(x))  # Second hidden layer
        return self.fc3(x)          # Output layer

# Replay Buffer for Experience Storage

class ReplayBuffer:
    def __init__(self, capacity):
        """
        Replay buffer to store experiences during training.

        Parameters:
        - capacity: Maximum number of experiences to store.
        """
        self.buffer = deque(maxlen=capacity)

    def store(self, experience):
        """
        Store a new experience in the buffer.

        Parameters:
        - experience: A tuple (state, action, reward, next_state, done).
        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from the buffer.

        Parameters:
        - batch_size: Number of experiences to sample.

        Returns:
        - Tuple of arrays: (states, actions, rewards, next_states, dones).
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def size(self):
        """
        Return the current size of the buffer.

        Returns:
        - Integer representing the number of experiences stored.
        """
        return len(self.buffer)

# Training the DQN with Batch Updates

def train_dqn_batch(dqn, replay_buffer, batch_size, gamma, optimizer, num_jobs, num_machines, num_agvs):
    """
    Train the DQN model using batch updates from the replay buffer.

    Parameters:
    - dqn: The Deep Q-Network model.
    - replay_buffer: The buffer storing past experiences.
    - batch_size: Number of experiences to sample from the buffer.
    - gamma: Discount factor for future rewards.
    - optimizer: Optimizer for updating the DQN's weights.
    """
    if replay_buffer.size() < batch_size:
        return  # Skip training if not enough experiences in the buffer

    # Sample a batch of experiences from the replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Convert the experiences to PyTorch tensors and move them to the device (MPS or CPU)
    states = torch.tensor(states, dtype=torch.float32).to(device)
    # Convert complex actions (job, machine) to indices if necessary
    action_indices = torch.tensor(
        [encode_action(action[0], action[1], action[2], num_machines, num_agvs) for action in actions],
        dtype=torch.long
    ).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    # Compute Q-values for the current states
    q_values = dqn(states).gather(1, action_indices.unsqueeze(1)).squeeze(1)

    # Compute the target Q-values for the next states
    with torch.no_grad():
        next_q_values = dqn(next_states).max(1)[0]
        target_q_values = rewards + gamma * next_q_values * (1 - dones)

    # Compute the loss (MSE between current Q-values and target Q-values)
    loss = nn.MSELoss()(q_values, target_q_values)

    # Backpropagation to update the model's weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Helper function to convert machine names to indices
def machine_to_index(machine):
    machine_list = ['LU', 'M1', 'M2', 'M3', 'M4']
    return machine_list.index(machine)

# Modify the action selection to consider the availability of AGVs and machines
def select_action(state, env, dqn, epsilon):
    num_jobs = len(env.jobs_data)
    num_machines = 5  # LU, M1, M2, M3, M4
    num_agvs = env.num_agvs

    if np.random.rand() <= epsilon:
        # Exploration: randomly choose a valid action
        valid_actions = []
        for job in range(num_jobs):
            next_machine_index = env.job_next_machine[job]
            num_operations = env.jobs_data.iloc[job,1]
            job_sequence = env.jobs_data.iloc[job, 2:2+num_operations].dropna().tolist()
            #print(job_sequence)
            if next_machine_index >= len(job_sequence):
                continue  # Skip if job is already completed
            machine = job_sequence[next_machine_index]
            machine_idx = machine_to_index(machine)
            for agv in range(num_agvs):
                if env.agv_status[agv] == 0:
                    # AGV is available
                    valid_actions.append((job, machine, agv))
        if valid_actions:
            logging.info(f"Exploration: Selected random action: {valid_actions}")
            return random.choice(valid_actions)
        else:
            next_event_time = env.get_next_event_time()
            env.current_time = max(env.current_time, next_event_time)
            env.update_resource_states()
            #logging.warning("Exploration: No valid actions available")
            return None
    else:
        # Exploitation: choose the best action based on the current policy
        state_dim = (
                4 +  # Machine statuses
                num_agvs +  # AGV statuses
                sum(len(jobs_data.iloc[job, 2:2+jobs_data.iloc[job,1]].dropna()) for job in range(num_jobs)) +  # Job-machine times
                num_agvs +  # AGV locations (indices)
                num_jobs  # Job locations (indices)
            )

        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)

        q_values = dqn(state_tensor)
        sorted_indices = torch.argsort(q_values, descending=True)
        for action_idx in sorted_indices.cpu().numpy():
            job, machine_idx, agv = decode_action(action_idx, num_machines, num_agvs)
            machine_list = ['LU', 'M1', 'M2', 'M3', 'M4']
            machine = machine_list[machine_idx]
            # Check if the action is valid
            next_machine_index = env.job_next_machine.get(job, None)
            if next_machine_index is None:
                continue
            num_operations = env.jobs_data.iloc[job,1]
            #print(num_operations)
            job_sequence = env.jobs_data.iloc[job, 2:2+num_operations].dropna().tolist()
            if next_machine_index >= len(job_sequence):
                continue
            correct_machine = job_sequence[next_machine_index]
            if machine != correct_machine:
                continue
            # Check availability of AGV and machine
            if env.agv_status[agv] == 1:
                continue  # AGV is busy
            if machine != "LU":
                machine_index = int(machine[1]) - 1
                if env.machine_status[machine_index] == 1:
                    continue  # Machine is busy
            return (job, machine, agv)

        next_event_time = env.get_next_event_time()
        env.current_time = max(env.current_time, next_event_time)
        env.update_resource_states()
        return None  # No valid actions
     

def encode_action(job, machine, agv, num_machines, num_agvs):
    """
    Encodes the action of assigning a job to a machine and an AGV into a single integer.

    Parameters:
    job (int): The job index.
    machine (str): The machine identifier.
    agv (int): The AGV (Automated Guided Vehicle) index.
    num_machines (int): The total number of machines.
    num_agvs (int): The total number of AGVs.

    Returns:
    int: The encoded action as a single integer.

    Raises:
    ValueError: If job, machine_idx, or agv are not integers after conversion.
    """
    machine_idx = machine_to_index(machine) 
    job = int(job)
    agv = int(agv)
    # Ensure that job, machine_idx, and agv are all integers
    if not isinstance(job, int):
        raise ValueError("Job index must be an integer after conversion.")
    if not isinstance(machine_idx, int):
        raise ValueError("Machine index must be an integer after conversion.")
    if not isinstance(agv, int):
        raise ValueError("AGV index must be an integer after conversion.")
    e_action = job * num_machines * num_agvs + machine_idx * num_agvs + agv
    return e_action

def decode_action(action_idx, num_machines, num_agvs):
    """
    Decodes an action index into its corresponding job, machine, and AGV (Automated Guided Vehicle) indices.

    Parameters:
    action_idx (int): The action index to decode.
    num_machines (int): The number of machines available.
    num_agvs (int): The number of AGVs available.

    Returns:
    tuple: A tuple containing the job index, machine index, and AGV index.
    """
    job = action_idx // (num_machines * num_agvs)
    remainder = action_idx % (num_machines * num_agvs)
    machine_idx = remainder // num_agvs
    agv = remainder % num_agvs
    return job, machine_idx, agv

# Simulation and Training Loop with Epsilon-Greedy Policy

# Device configuration (use MPS if available, else CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Tensorboard summary writer
writer = SummaryWriter('/tmp/runs/job_shop_dqn')

#Global list to store all actions taken during training and all the performance metrics without overwriting
all_actions=[]
all_performance_metrics = []


# Number of AGVs to test (from 1 to 5)
for num_agvs in range(1, 6):
    print(f"Training with {num_agvs} AGVs")
    # Preload job and processing data for all sets (avoiding repeated loading)
    all_jobs_data = {nset: jobs(nset) for nset in range(1, 11)}        # 10 sets
    all_processing_data = {nset: processing(nset) for nset in range(1, 11)}  # 10 sets

    num_episodes = 1000  # Number of episodes per layout/set combination
    early_stopping_threshold = -100  # Threshold for early stopping based on reward

    # Initialize performance metrics storage
    performance_metrics = []

    # Loop through all layouts and sets
    for layout in range(1, 5):  # 4 layouts
        for nset in range(1, 11):  # 10 sets
            print(f"Training for Layout {layout}, Set {nset}, with {num_agvs} AGVs")

            # Retrieve jobs and processing data
            jobs_data = all_jobs_data[nset]
            processing_data = all_processing_data[nset]
            num_jobs = len(jobs_data)
            steps_per_job = len(jobs_data.iloc[0, 2:].dropna())

            # Calculate state and action dimensions
            state_dim = (
                4 +  # Machine statuses
                num_agvs +  # AGV statuses
                sum(len(jobs_data.iloc[job, 2:2+jobs_data.iloc[job,1]].dropna()) for job in range(num_jobs)) +  # Job-machine times
                num_agvs +  # AGV locations (indices)
                num_jobs  # Job locations (indices)
            )

            num_machines = 5  # LU, M1, M2, M3, M4
            action_dim = num_jobs * num_machines * num_agvs  # New action dimension

            # Initialize DQN model and optimizer
            dqn = DQNScheduler(input_dim=state_dim,output_dim=action_dim).to(device)
            optimizer = optim.Adam(dqn.parameters(), lr=0.001)  

            # Initialize replay buffer
            replay_buffer = ReplayBuffer(2000)  

            # Initialize environment
            env = JobShopEnv(layout, nset, jobs_data, processing_data, num_agvs)

            # Hyperparameters
            batch_size = 64
            gamma = 0.99
            epsilon = 1.0
            epsilon_min = 0.1
            epsilon_decay_rate = 0.995

            total_rewards = []

            # Training loop
            for episode in range(num_episodes):
                state = env.reset()
                done = False
                episode_reward = 0
                steps = 0

                # Initialize a list to store actions taken during the episode
                episode_actions = []

                while not done:
                    action = select_action(state, env, dqn, epsilon)
                    if action is None:
                        next_event_time = env.get_next_event_time()
                        env.current_time = max(env.current_time, next_event_time)
                        env.update_resource_states()
                        state = env._get_state()  # Update the state after resource states change
                        continue

                    # Store experience and train
                    next_state, reward, done = env.step(action)
                    replay_buffer.store((state, action, reward, next_state, done))
                    train_dqn_batch(dqn, replay_buffer, batch_size, gamma, optimizer, num_jobs, num_machines, num_agvs)

                    # Save the action taken in this step
                    episode_actions.append(action)

                    state = next_state
                    episode_reward += reward
                    steps += 1

                # Log episode reward
                writer.add_scalar(f'Reward/Layout_{layout}_Set_{nset}_AGVs_{num_agvs}', episode_reward, episode)

                # Epsilon decay after each episode
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay_rate

                if episode % 500 == 0:  # Save checkpoint every 500 episodes
                    torch.save(dqn.state_dict(), f'checkpoint_layout{layout}_set{nset}_agvs{num_agvs}_ep{episode}.pth')

                total_rewards.append(episode_reward)

                # Early stopping if reward threshold is met
                if episode_reward > early_stopping_threshold:
                    print(f"Early stopping at episode {episode} for Layout {layout}, Set {nset}, AGVs {num_agvs}")
                    break

                # Append the episode actions to the global list
                all_actions.append({
                    "layout": layout,
                    "set": nset,
                    "agvs": num_agvs,
                    "episode": episode,
                    "actions": episode_actions
                })

            writer.flush()

            # Store performance metrics after each set-layout combination
            performance_metrics.append({
                "layout": layout,
                "set": nset,
                "agvs": num_agvs,
                "average_reward": np.mean(total_rewards),
                "total_time": env.current_time
            })

            metrics_df = pd.DataFrame(performance_metrics)
            metrics_df.to_csv("makespan_results.csv", mode='a', index=False, header=not os.path.exists("makespan_results.csv"))

            all_performance_metrics.append({
                "layout": layout,
                "set": nset,
                "agvs": num_agvs,
                "average_reward": np.mean(total_rewards),
                "total_time": env.current_time
            })


            # Save the model after training each set-layout combination
            torch.save(dqn.state_dict(), f'model_layout{layout}_set{nset}_agvs{num_agvs}.pth')

all_metrics_df = pd.DataFrame(all_performance_metrics)
all_metrics_df.to_csv("makespan_results.csv", index=False)

# Save all actions to a JSON file after the training is complete
with open("actions_log.json", "w") as actions_file:
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    json.dump(all_actions, actions_file, indent=4, cls=NumpyEncoder)

print("All actions have been saved to actions_log.json.")