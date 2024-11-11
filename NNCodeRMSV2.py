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
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv


# Data Loading and Transport/Processing Times

load_dotenv()
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
if not connection_string:
    raise ValueError("Azure Storage connection string not found.")

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

container_name = "azureml"
blob_name = "Data.xlsx"

def load_data_from_blob(container_name, blob_name, connection_string):
    """
    Load data from an Excel file stored in Azure Blob Storage.

    Parameters:
    - container_name: The name of the Azure Blob Storage container.
    - blob_name: The name of the blob (Excel file) to load.
    - connection_string: The connection string for Azure Blob Storage.

    Returns:
    - A DataFrame containing the loaded data.
    """
    # Create a BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    # Get the blob client
    blob_client = blob_service_client.get_blob_client(container_name, blob_name)
    # Download the blob content to a BytesIO object
    blob_data = blob_client.download_blob().readall()
    excel_data = BytesIO(blob_data)
    return pd.read_excel(excel_data, sheet_name='Macrodata', usecols='F:H, J:P, R:X')

# Load the data using the new function
xls = load_data_from_blob(container_name, blob_name, connection_string)
data = pd.DataFrame(xls)
data.loc[:, 'nj'] = data.loc[:, 'nj'] + 1  # Adjust job numbers
data = data.fillna('')

# Extract processing times and machine assignments
p_times = pd.DataFrame(data.iloc[:, :10].to_numpy(), columns=["Set", "Job", "nj", "P1", "P2", "P3", "P4", "P5", "P6", "P7"])
m_data = pd.DataFrame(data.iloc[:, [0, 1, 2] + list(range(10, data.shape[1]))].to_numpy(),columns=["Set", "Job", "nj"] + [f"M{i}" for i in range(1, data.shape[1] - 9)])



def t_times(layout, start, end):
    layout_dfs = {1: df1, 2: df2, 3: df3, 4: df4}
    return layout_dfs[layout].loc[start, end]

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
            self.current_time = next_event_time
            self.update_resource_states()  # Update machine and AGV states
            return self._get_state(), 0, self._check_done()

        #Proceed if action is valid
        job, machine, agv = action
        job_location = self.job_locations[job]

        # Validate AGV availability
        if self.agv_status[agv] == 1:
             # AGV is busy, advance time and wait for the next available event
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
            self.current_time = self.get_next_event_time()
            self.update_resource_states()
            return self._get_state(), 0, self._check_done()

        # Calculate transport time
        agv_location = self.agv_locations[agv]
        empty_move_time = t_times(self.layout, agv_location, job_location) if agv_location != job_location else 0
        loaded_move_time = t_times(self.layout, job_location, machine)
        transport_time = empty_move_time + loaded_move_time

        if machine != "LU":
            machine_index = int(machine[1]) - 1  # Convert 'M1' to index 0
            if self.machine_status[machine_index] == 1:
                # Machine is busy, advance time and wait for the next available event
                self.current_time = self.get_next_event_time()
                self.update_resource_states()  # Update machine and AGV states
                return self._get_state(), 0, self._check_done()
        
        # Update AGV and job locations after transport
        self.agv_locations[agv] = machine
        self.job_locations[job] = machine
        self.agv_status[agv] = 1
        self.agv_available_at[agv] = self.current_time + transport_time

        if machine != "LU":
            # Machines M1 to M4
            # Check if the machine is available
            # Get processing time for the job on the machine
            process_time = self.processing_data.iloc[job, 2 + next_machine_index]  
            start_time = self.get_next_event_time()
            end_time = start_time + process_time

            # Update machine status and availability times
            self.machine_status[machine_index] = 1
            self.machine_available_at[machine_index] = end_time
            # Record the job's processing times on the machine
            self.job_machine_times[job][machine] = {'start': start_time, 'end': end_time}

            # Negative reward to minimize makespan
            reward = -(transport_time + process_time)

            # Mark the job's machine assignment as completed
            self.job_next_machine[job] += 1

        else:
            # If the machine is LU (Loading-Unloading area)
            start_time = self.get_next_event_time()
            end_time = start_time  # No processing at LU

            # Update AGV status and availability time
            self.agv_available_at[agv] = start_time
            self.job_machine_times[job][machine] = {'start': start_time, 'end': end_time}

            # Negative reward to minimize makespan
            reward = -transport_time

            # Mark the job's machine assignment as completed
            self.job_next_machine[job] += 1

        # Check if all jobs are finished
        self.done = self._check_done()
        if self.done:
            # When all jobs are done, set the makespan (total_time)
            self.total_time = self.current_time

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
        """
        Updates the states of machines and AGVs (Automated Guided Vehicles) based on the current time.
        
        For machines:
        - If a machine is currently in use (status is 1) and the current time is greater than or equal to the time it becomes available,
          the machine's status is set to free (status is 0) and its available time is set to infinity.

        For AGVs:
        - If an AGV is currently in use (status is 1) and the current time is greater than or equal to the time it becomes available,
          the AGV's status is set to free (status is 0) and its available time is set to infinity.

        Logging:
        - Logs an info message when a machine or AGV becomes available.
        """
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
        """
        Calculate the next event time based on the availability of machines and AGVs.

        This method determines the next available times for machines and AGVs and 
        returns the minimum of these times. If no machines or AGVs are available, 
        it ensures that the time advances by at least a small step to avoid stagnation.

        Returns:
            float: The next event time.
        """
        # Calculate next available times for machines and AGVs
        next_machine_available = np.min(self.machine_available_at[self.machine_status == 1]) if np.any(self.machine_status == 1) else float('inf')
        next_agv_available = np.min(self.agv_available_at[self.agv_status == 1]) if np.any(self.agv_status == 1) else float('inf')

        # Find the minimum of the available times, avoiding infinite results
        next_event_time = min(next_machine_available, next_agv_available)

        # Ensure time always advances by at least a small step to avoid stagnation
        if next_event_time == float('inf') or next_event_time <= self.current_time:
            next_event_time = self.current_time + 1  # or a small epsilon increment

        #logging.info(f"Next Event Time calculated: {next_event_time}")
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

        # Updated model architecture with increased complexity
        self.fc1 = nn.Linear(input_dim, 256)   # First hidden layer with 256 neurons
        self.fc2 = nn.Linear(256, 128)         # Second hidden layer with 128 neurons
        self.fc3 = nn.Linear(128, 64)          # Third hidden layer with 64 neurons
        self.fc4 = nn.Linear(64, output_dim)   # Output layer to predict Q-values for actions

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
        x = self.relu(self.fc3(x))  # Second hidden layer
        return self.fc4(x)          # Output layer

# Replay Buffer for Experience Storage

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha

    def store(self, experience, error):
        """
        Stores an experience in the buffer with a priority based on the given error.

        Args:
            experience (object): The experience to be stored.
            error (float): The error associated with the experience, used to calculate its priority.
        """
        priority = (error + 1e-5) ** self.alpha
        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of experiences from the replay buffer with prioritized sampling.

        Args:
            batch_size (int): The number of experiences to sample.
            beta (float, optional): The importance-sampling exponent. Default is 0.4.

        Returns:
            tuple: A tuple containing:
                - states (np.array): Array of states.
                - actions (np.array): Array of actions.
                - rewards (np.array): Array of rewards.
                - next_states (np.array): Array of next states.
                - dones (np.array): Array of done flags.
                - weights (np.array): Array of importance-sampling weights.
                - indices (np.array): Array of sampled indices.
        """
        priorities = np.array(self.priorities)
        priorities = np.nan_to_num(priorities, nan=0.0)
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        states, actions, rewards, next_states, dones = zip(*experiences)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones), weights, indices

    def update_priorities(self, batch_indices, batch_errors):
        """
        Update the priorities of the samples in the replay buffer.

        This method updates the priorities of the samples in the replay buffer
        based on the provided batch indices and batch errors. The priority of
        each sample is calculated as (|error| + 1e-5) ** alpha, where alpha is
        a parameter that controls the degree of prioritization.

        Args:
            batch_indices (list of int): The indices of the samples in the replay buffer.
            batch_errors (list of float): The errors corresponding to the samples.
        """

        for idx, error in zip(batch_indices, batch_errors):
            error = abs(error)  # Ensure error is non-negative
            self.priorities[idx] = (error + 1e-5) ** self.alpha
    
    def size(self):
        """
        Returns the size of the buffer.
        Returns:
            int: The number of elements in the buffer.
        """

        return len(self.buffer)

# Training the DQN with Batch Updates

# Training function with Double DQN
def train_double_dqn(dqn, target_dqn, replay_buffer, batch_size, gamma, optimizer, beta):
    """
    Trains a Double DQN (Deep Q-Network) using a replay buffer.

    Args:
        dqn (torch.nn.Module): The main DQN network.
        target_dqn (torch.nn.Module): The target DQN network.
        replay_buffer (ReplayBuffer): The replay buffer containing experience tuples.
        batch_size (int): The number of samples to draw from the replay buffer for each training step.
        gamma (float): The discount factor for future rewards.
        optimizer (torch.optim.Optimizer): The optimizer used for training the DQN.
        beta (float): The importance sampling exponent for prioritized experience replay.

    Returns:
        None
    """
    if replay_buffer.size() < batch_size:
        return
    states, actions, rewards, next_states, dones, weights, indices = replay_buffer.sample(batch_size, beta=beta)
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor([encode_action(*action, num_machines, num_agvs) for action in actions], dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        max_next_actions = dqn(next_states).argmax(1)
        next_q_values = target_dqn(next_states).gather(1, max_next_actions.unsqueeze(1)).squeeze(1)
        target_q_values = rewards + gamma * next_q_values * (1 - dones)

    loss = (weights * (q_values - target_q_values) ** 2).mean()
    errors = (q_values - target_q_values).detach().cpu().numpy()
    replay_buffer.update_priorities(indices, errors)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Helper function to convert machine names to indices
def machine_to_index(machine):
    """
    Convert a machine identifier to its corresponding index in the machine list.

    Args:
        machine (str): The machine identifier, which should be one of ['LU', 'M1', 'M2', 'M3', 'M4'].

    Returns:
        int: The index of the machine identifier in the machine list.
    """
    machine_list = ['LU', 'M1', 'M2', 'M3', 'M4']
    return machine_list.index(machine)

# Modify the action selection to consider the availability of AGVs and machines
def select_action(state, env, dqn, epsilon):
    """
    Selects an action based on the current state, environment, DQN model, and epsilon value.

    Parameters:
    state (array-like): The current state of the environment.
    env (object): The environment object containing job and resource information.
    dqn (object): The deep Q-network model used for action selection.
    epsilon (float): The exploration-exploitation trade-off parameter.

    Returns:
    tuple: A tuple (job, machine, agv) representing the selected action, or None if no valid action is available.
    """
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
    agv (int): The AGV index.
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
    action_idx (int): The index representing a specific action.
    num_jobs (int): The total number of jobs.
    num_machines (int): The total number of machines.
    num_agvs (int): The total number of AGVs.

    Returns:
    tuple: A tuple containing:
        - job (int): The job index.
        - machine_idx (int): The machine index.
        - agv (int): The AGV index.
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
writer = SummaryWriter('runs/job_shop_dqn')

#Global list to store all actions taken during training and all the performance metrics without overwriting
all_actions=[]
all_performance_metrics = []

'''
Training Loop:
- Iterat