import connection
import random
import os

class QLearning:
    def __init__(self, alpha=0.1, gamma=0.97, epsilon=0.9, 
                epsilon_decay=0.995, episodes=1000, table_file="resultado.txt"):
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
        
        # Tenta carregar a tabela Q existente ou inicializa uma nova
        self.q_table = self.load_q_table()
        self.memory_size = 5 

    def load_q_table(self):
        """Carrega a tabela Q de um arquivo"""
        try:
            if os.path.exists(self.table_file):
                with open(self.table_file, 'r') as f:
                    lines = f.readlines()
                
                q_table = []
                for line in lines:
                    values = list(map(float, line.strip().split()))
                    if len(values) == self.num_actions:
                        q_table.append(values)
                
                if len(q_table) == self.num_states:
                    # Verifica se está toda zerada
                    all_zeros = all(all(val == 0.0 for val in row) for row in q_table)
                    
                    if all_zeros:
                        print("Q-table está toda zerada. Substituindo por valores otimistas (5.0).")
                        q_table = [[5.0 for _ in range(self.num_actions)] for _ in range(self.num_states)]
                    else:
                        print(f"Tabela Q carregada de {self.table_file}")
                    
                    return q_table
        except Exception as e:
            print(f"Erro ao carregar tabela Q: {e}")
        
        print("Inicializando nova tabela Q com valores otimistas.")
        return self.initialize_q_table()
    
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
        Atualiza a tabela Q usando a equação de Bellman para Q-Learning        
        """
        max_q_next = 0 if done else max(self.q_table[next_state])
        
        target = reward + self.gamma * max_q_next
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])

    def update_epsilon(self):
        """Atualiza o valor de epsilon, aplicando o decaimento"""
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01)
        return self.epsilon
        
    def sucess_criterion(self, reward, step_count):
        """Determina se o episódio terminou (sucesso ou falha)"""
        if reward == 300:
            print("Sucesso!")
            return True
        elif reward == -100:
            print("Falha!")
            return True
        elif step_count > 500:
            print("Timeout - muitos passos!")
            return True
        return False

    def check_loops(self, visited_count):
        """Verifica se o agente está em um loop (visitando os mesmos estados repetidamente)"""
        for state, count in visited_count.items():
            if count > self.memory_size:
                return True
        return False

    def train_agent(self, socket, save_interval=10):
        """
        Treina o agente:
        1. Inicia um episódio em um estado aleatório
        2. Escolhe ações usando a política ε-greedy
        3. Observa recompensas e próximos estados
        4. Atualiza a tabela Q
        5. Repete até que os critérios de término sejam atendidos
        6. Reduz ε para o próximo episódio (menos exploração)
        """
        total_success = 0
        
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
                next_state = int(next_state, 2) 
                reward = float(reward)
                
                done = self.sucess_criterion(reward, step_count)
                
                self.update_q_table(current_state, action_index, reward, next_state, done)
                current_state = next_state
                step_count += 1
                
                if reward == 300:
                    total_success += 1
                            
            self.epsilon = self.update_epsilon()
            
            # Salva a tabela Q periodicamente
            if (episode + 1) % save_interval == 0:
                self.save_q_table()
                print(f"Episódio {episode + 1}/{self.episodes} completado.")
                print(f"Taxa de sucesso até agora: {(total_success/(episode+1))*100:.2f}%")
        
        # Salva a tabela Q ao final do treinamento
        self.save_q_table()
        print("Treinamento concluído.")
        print(f"Taxa de sucesso final: {(total_success/self.episodes)*100:.2f}%")

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
            
            while not done and step_count < 300:  # Limite de passos para testes
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


if __name__ == "__main__":
    socket = connection.connect(2037)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(current_dir, "resultado.txt")

    # Cria e treina o agente
    agent = QLearning(table_file=result_path)
    agent.train_agent(socket)

    # Testa a política aprendida
    success_rate = agent.test_policy(socket)
    print(f"Taxa de sucesso: {success_rate:.2f}%")