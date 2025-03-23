import connection
import random

class QLearning:
    def __init__(self, alpha=0.1, gamma=0.97, epsilon=0.9, 
                epsilon_decay=0.995, episodes=500, table_file = "resultado.txt"):
        """
        Parameters:
        - alpha: Learning rate (0 < alpha ≤ 1)
        - gamma: Discount factor for future rewards (0 ≤ gamma < 1)
        - epsilon: Initial exploration probability
        - epsilon_decay: Decay rate for epsilon after each episode
        - episodes: Number of training episodes
        - table_file: Filename to save the Q-table
        """
        # Learning parameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        self.table_file = table_file
                
        # Environment configuration
        self.actions = ["left", "right", "jump"]
        self.num_states = 96
        self.num_actions = len(self.actions)
        
        # Initialize Q-table with zeros
        self.q_table = self.initialize_q_table()
        self.memory_size = 5  
    
    def initialize_q_table(self):
        return [[0 for _ in range(self.num_actions)] for _ in range(self.num_states)]


    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            weights = [0.2, 0.2, 0.6]  # Higher chance of jumping
            return random.choices([0, 1, 2], weights=weights)[0]
        else:
            return self.q_table[state].index(max(self.q_table[state]))  


    def update_q_table(self, state, action, reward, next_state):
        """
        Updates the Q-table using the Bellman equation for Q-Learning:        
        """
        
        max_q_next_state = max(self.q_table[next_state])
        
        # Apply Bellman equation to update Q(s,a)
        target = reward + self.gamma * max_q_next_state
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])

        

    def update_epsilon(self):
        """
        Updates epsilon value after each episode.
        Gradually decreases epsilon to the minimum value to reduce exploration.
        """
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01)

        return self.epsilon
        
        
    def save_q_table(self, filename=None):
        """Saves the Q-table to a text file."""
        if filename is None:
            filename = self.table_file
            
        with open(filename, "w") as file:
            for row in self.q_table:
                file.write(" ".join(map(str, row)) + "\n")
        print(f"Q-table successfully saved to {filename}")


    def train_agent(self, socket):
        """
        1. Starting an episode in a random state
        2. Choosing actions using the ε-greedy policy
        3. Observing rewards and next states
        4. Updating the Q-table
        5. Repeating until termination criteria is met
        6. Reducing ε for the next episode (less exploration)
        """
        
        for episode in range(self.episodes):
            print(f"Starting episode {episode + 1}/{self.episodes}...")
            
            current_state = 0
            steps = 0
            done = False
            
            while not done:
                action_index = self.choose_action(current_state)
                action = self.actions[action_index]
                
                next_state, reward = connection.get_state_reward(socket, action)
                next_state = int(next_state, 2)
                reward = float(reward)
                
                self.update_q_table(current_state, action_index, reward, next_state)
                current_state = next_state
                steps += 1
                
                done = self.sucess_criterion(reward, steps)

                
        self.epsilon = self.update_epsilon()
        if episode % 10 == 0:
            print(f"Episode {episode + 1}/{self.episodes} completed.")
        
        # Save the learned Q-table
        self.save_q_table()
        print("Training completed.")


    def sucess_criterion(self, reward, step_count):
        if reward == -1:
            print("Success!")
            return True
        elif reward == -100:
            print("Failure!")
            return True
        elif step_count >= 150:
            print("Timeout.")
            return True
        
        return False

    def test_policy(self, socket, num_tests=20):
        """
        Tests the policy learned by the agent.
        Returns the success rate in reaching the final platform.
        """
        num_successes = 0
        
        for test in range(num_tests):
            print(f"Starting test {test + 1}/{num_tests}...")
            
            current_state = 0
            step_count = 0
            done = False
            
            while not done:
                action_index = self.q_table[current_state].index(max(self.q_table[current_state]))
                action = self.actions[action_index]

                # Execute action and observe result
                next_state, reward = connection.get_state_reward(socket, action)
                next_state = int(next_state, 2)
                reward = float(reward)
                
                done = self.sucess_criterion(reward, step_count)

                current_state = next_state
                step_count += 1
                
        success_rate = (num_successes / num_tests) * 100
        return success_rate


# Connect to the server
socket = connection.connect(2037)

# Create and train the agent
agent = QLearning()
agent.train_agent(socket)

# Test the learned policy
success_rate = agent.test_policy(socket)
print(f"Success rate: {success_rate:.2f}%")