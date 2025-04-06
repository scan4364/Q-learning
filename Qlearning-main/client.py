import connection
import random
import os

class QLearning:
    def __init__(self, alpha=0.1, gamma=0.97, epsilon=0.9, 
                epsilon_decay=0.995, episodes=1000, loop_threshold = 5,  table_file="resultado.txt"):
        """
        Parâmetros:
        - alpha: Taxa de aprendizagem (0 < alpha ≤ 1)
        - gamma: Fator de desconto para recompensas futuras (0 ≤ gamma < 1)
        - epsilon: Probabilidade inicial de exploração
        - epsilon_decay: Taxa de decaimento do epsilon após cada episódio
        - episodes: Número de episódios de treinamento
        - table_file: Nome do arquivo para salvar/carregar tabela Q
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        self.table_file = table_file
        self.loop_threshold = loop_threshold
        self.actions = ["left", "right", "jump"]
        self.num_states = 96
        self.num_actions = len(self.actions)
        self.q_table = self.load_q_table()


    def load_q_table(self):
        """Carrega a tabela Q de um arquivo"""
        if os.path.exists(self.table_file):
            with open(self.table_file, 'r') as f:
                lines = f.readlines()
            
            q_table = []
            for line in lines:
                values = list(map(float, line.strip().split()))
                if len(values) == self.num_actions:
                    q_table.append(values)
            
            # Verifica se está toda zerada
            all_zeros = all(all(val == 0.0 for val in row) for row in q_table)
            
            if all_zeros:
                print("Substituindo por valores otimistas (5.0).")
                q_table = [[5.0 for _ in range(self.num_actions)] for _ in range(self.num_states)]
            else:
                print(f"Tabela Q carregada de {self.table_file}")
            
            return q_table


            
                
    def save_q_table(self):
        """Salva a tabela Q em um arquivo"""
        try:
            with open(self.table_file, 'w') as f:
                for state_values in self.q_table:
                    f.write(' '.join(map(str, state_values)) + '\n')
            print(f"Tabela Q salva em {self.table_file}")
        except Exception as e:
            print(f"Erro ao salvar tabela Q: {e}")


    def choose_action(self, state):
        """
        Retorna um índice de ação baseado na política ε-greedy.
            - Com prob. epsilon, escolhe ação aleatória (com pesos).
            - Caso contrário, escolhe a ação com valor Q máximo.
        """
        if random.random() < self.epsilon:
            return random.choices([0, 1, 2], weights=[0.3, 0.3, 0.4])[0]
        else:
            return self.q_table[state].index(max(self.q_table[state]))
    
    
    def update_q_table(self, state, action, reward, next_state, done):
        """
        Atualiza a tabela Q usando a equação de Bellman para Q-Learning        
        """
        max_q_next = 0 if done else max(self.q_table[next_state])
        target = reward + self.gamma * max_q_next
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])

    def apply_epsilon_decay(self):
        """Atualiza o valor de epsilon, aplicando o decaimento"""
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01)
        return self.epsilon
        
    def is_episode_done(self, reward, step_count):
        """
        Verifica se o episódio terminou:
          - reward == 300: sucesso
          - reward == -100: falha
          - step_count > 500: timeout
        """
        if reward == 300:
            print("Sucesso!")
            return True
        if reward == -100:
            print("Falha!")
            return True
        if step_count > 500:
            print("Timeout - muitos passos!")
            return True
        return False
    
    
    def detect_loop(self, visited_count):
        """
        Verifica se há loop no episódio: se um estado for visitado
        mais vezes que o loop_threshold, considera loop.
        """
        return any(count > self.loop_threshold for count in visited_count.values())

    def train_agent(self, socket):
        """
        Executa o processo de treinamento do agente ao longo de 'episodes':
            1. Inicializa estado aleatório.
            2. Escolhe ação (ε-greedy).
            3. Observa recompensa e próximo estado.
            4. Atualiza Q-table.
            5. Encerra se critérios de término forem satisfeitos.
            6. Decai epsilon a cada episódio.
            7. Salva a Q-table periodicamente.
        """
        total_success = 0

        for episode in range(self.episodes):
            print(f"--- Episódio {episode + 1}/{self.episodes} ---")
            current_state = 0
            step_count = 0
            done = False
            visited_count = {}

            while not done:
                visited_count[current_state] = visited_count.get(current_state, 0) + 1

                if self.detect_loop(visited_count):
                    print("Ciclo detectado! Forçando salto.")
                    action_index = 2  # 'jump'
                else:
                    action_index = self.choose_action(current_state)

                action = self.actions[action_index]
                next_state_str, reward_str = connection.get_state_reward(socket, action)
                next_state = int(next_state_str, 2)
                reward = float(reward_str)

                done = self.is_episode_done(reward, step_count)
                self.update_q_table(current_state, action_index, reward, next_state, done)

                current_state = next_state
                step_count += 1

                if reward == 300:
                    total_success += 1

            self.apply_epsilon_decay()


        self.save_q_table()
        print("Treinamento concluído.")
        print(f"Taxa de sucesso final: {(total_success / self.episodes) * 100:.2f}%")


    def test_policy(self, socket, num_tests=20):
        """
        Testa a política aprendida pelo agente.
        Retorna a taxa de sucesso em alcançar a plataforma final.
        """
        num_successes = 0
        
        for test in range(num_tests):
            print(f"--- Teste {test + 1}/{num_tests} ---")
            current_state = 0
            step_count = 0
            done = False

            while not done and step_count < 300:
                # Seleciona ação com maior valor Q
                action_index = self.q_table[current_state].index(max(self.q_table[current_state]))
                action = self.actions[action_index]

                next_state_str, reward_str = connection.get_state_reward(socket, action)
                next_state = int(next_state_str, 2)
                reward = float(reward_str)

                done = self.is_episode_done(reward, step_count)

                if reward == 300:
                    num_successes += 1

                current_state = next_state
                step_count += 1

        return (num_successes / num_tests) * 100


if __name__ == "__main__":
    socket = connection.connect(2037)
    
    # Define o caminho do arquivo de resultados
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(current_dir, "resultado.txt")

    agent = QLearning(table_file=result_path)
    
    # Descomente para treinar novamente:
    # agent.train_agent(socket)

    # Testa a política aprendida
    success_rate = agent.test_policy(socket)
    print(f"Taxa de sucesso: {success_rate:.2f}%")