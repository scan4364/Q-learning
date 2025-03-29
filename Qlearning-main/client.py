import connection
import random

class QLearning:
    def __init__(self, alpha=0.1, gamma=0.97, epsilon=0.9, 
                epsilon_decay=0.995, episodes=1000, table_file = "resultado.txt"):
        """
        Parâmetros:
        - alpha: Taxa de aprendizagem (0 < alpha ≤ 1)
        - gamma: Fator de desconto para recompensas futuras (0 ≤ gamma < 1)
        - epsilon: Probabilidade inicial de exploração
        - epsilon_decay: Taxa de decaimento do epsilon após cada episódio
        - episodes: Número de episódios de treinamento
        - table_file: Nome do arquivo para salvar a tabela Q
        """
        # Parâmetros de aprendizagem
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        self.table_file = table_file
                
        # Configuração do ambiente
        self.actions = ["left", "right", "jump"]
        self.num_states = 96
        self.num_actions = len(self.actions)
        
        # Inicializa a tabela Q com zeros
        self.q_table = self.initialize_q_table()
        self.memory_size = 5  # Tamanho da memória para detecção de loops
    
    def initialize_q_table(self):
        """Inicializa a tabela Q com valores 1.0"""
        return [[1.0 for _ in range(self.num_actions)] for _ in range(self.num_states)]


    def choose_action(self, state):
        """
        Escolhe uma ação usando a política ε-greedy:
        - Com probabilidade ε: escolhe ação aleatória
        - Caso contrário: escolhe a ação com maior valor Q
        """
        if random.uniform(0, 1) < self.epsilon:
            weights = [0.3, 0.3, 0.4]  # Pesos para left, right, jump
            return random.choices([0, 1, 2], weights=weights)[0]
        else:
            return self.q_table[state].index(max(self.q_table[state]))  


    def update_q_table(self, state, action, reward, next_state, done):
        """
        Atualiza a tabela Q usando a equação de Bellman para Q-Learning:        
        """
        
        # Normaliza a tabela Q
        max_q_next = 0 if done else max(self.q_table[next_state])
        
        # Aplica a equação de Bellman para atualizar Q(s,a)
        target = reward + self.gamma * max_q_next
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])

        

    def update_epsilon(self):
        """Atualiza o valor de epsilon, aplicando o decaimento"""
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01)

        return self.epsilon
        
        
    def save_q_table(self, filename=None):
        """Salva a tabela Q em um arquivo de texto."""
        if filename is None:
            filename = self.table_file
            
        with open(filename, "w") as file:
            for row in self.q_table:
                file.write(" ".join(map(str, row)) + "\n")
        print(f"Q-table salva com sucesso em {filename}")


    def sucess_criterion(self, reward, step_count):
        """Determina se o episódio terminou (sucesso ou falha)"""
        if reward == 300:
            print("Sucesso!")
            return True
        elif reward == -100:
            print("Falha!")
            return True

        return False

    def check_loops(self, visited_count):
        """Verifica se o agente está em um loop (visitando os mesmos estados repetidamente)"""
        for state, count in visited_count.items():
            if count > self.memory_size:
                return True
        return False

    def train_agent(self, socket):
        """
        Treina o agente:
        1. Inicia um episódio em um estado aleatório
        2. Escolhe ações usando a política ε-greedy
        3. Observa recompensas e próximos estados
        4. Atualiza a tabela Q
        5. Repete até que os critérios de término sejam atendidos
        6. Reduz ε para o próximo episódio (menos exploração)
        """
        
        for episode in range(self.episodes):
            print(f"Iniciando episódio {episode + 1}/{self.episodes}...")
            
            current_state = 0
            step_count = 0
            done = False
            visited_count = {}  # Para detectar ciclos

            
            while not done:
                # Contagem de visitas para detectar ciclos
                visited_count[current_state] = visited_count.get(current_state, 0) + 1
                
                if self.check_loops(visited_count):
                    print("Ciclo detectado!")
                    action_index = 2  # Força um pulo para tentar sair do ciclo
                else:
                    action_index = self.choose_action(current_state)
                
                action = self.actions[action_index]
                
                next_state, reward = connection.get_state_reward(socket, action)
                next_state = int(next_state, 2)  # Converte binário para inteiro
                reward = float(reward)
                
                
                done = self.sucess_criterion(reward, step_count)
                self.update_q_table(current_state, action_index, reward, next_state, done)
                current_state = next_state
                step_count += 1
                            
            self.epsilon = self.update_epsilon()
        
        if episode % 10 == 0:
            print(f"Episódio {episode + 1}/{self.episodes} completado.")

        
        # Salva a tabela Q aprendida
        self.save_q_table()
        print("Treinamento concluído.")


    def test_policy(self, socket, num_tests=20):
        """
        Testa a política aprendida pelo agente.
        Retorna a taxa de sucesso em alcançar a plataforma final.
        """
        num_successes = 0
        
        for test in range(num_tests):
            print(f"Iniciando teste {test + 1}/{num_tests}...")
            
            current_state = 0
            step_count = 0
            done = False
            
            while not done:
                action_index = self.q_table[current_state].index(max(self.q_table[current_state]))
                action = self.actions[action_index]

                # Executa a ação e observa o resultado
                next_state, reward = connection.get_state_reward(socket, action)
                next_state = int(next_state, 2)
                reward = float(reward)
                
                done = self.sucess_criterion(reward, step_count)
                
                if reward == 300:
                    num_successes += 1
                current_state = next_state
                step_count += 1
                
        success_rate = (num_successes / num_tests) * 100
        return success_rate


# Conecta ao servidor
socket = connection.connect(2037)

# Cria e treina o agente
agent = QLearning()
agent.train_agent(socket)

# Testa a política aprendida
success_rate = agent.test_policy(socket)
print(f"Taxa de sucesso: {success_rate:.2f}%")