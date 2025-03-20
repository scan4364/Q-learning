import connection
import random

# parâmetros do q-Learning variaveis
ALPHA = 0.1         # taxa de aprendizado -- parâmetro que pode ser variado
GAMMA = 0.95        # fator de desconto -- parâmetro que pode ser variado
EPSILON = 1.0       # taxa inicial de exploração -- parâmetro que pode ser variado
EPSILON_MIN = 0.01  # taxa de exploração minima -- parâmetro que pode ser variado
EPSILON_DECAY = 0.995  # Decaimento da exploração -- parâmetro que pode ser variado
EPISODIOS = 2000     # Número de episódios -- parâmetro que pode ser variado
# parâmetros do q-learning fixos
ACOES = ["left", "right", "jump"]
NUM_ESTADOS = 96     # 24 plataformas × 4 direções
NUM_ACOES = len(ACOES)

# inicializar a tabela q
TABELA_Q = [[0 for _ in range(NUM_ACOES)] for _ in range(NUM_ESTADOS)]

def escolher_acao(estado, epsilon):
    """Escolhe uma ação usando a política ε-greedy."""
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, NUM_ACOES - 1)  # Exploração: ação aleatória
    else:
        return TABELA_Q[estado].index(max(TABELA_Q[estado]))  # Exploração: melhor ação

def atualiza_tabela_q(estado, acao, recompensa, prox_estado):
    recompensa_normalizada = recompensa / 14

    # seleciona a melhor ação no próximo estado
    melhor_prox_acao = TABELA_Q[prox_estado].index(max(TABELA_Q[prox_estado]))
    TABELA_Q[estado][acao] += ALPHA * (
        recompensa_normalizada +
        GAMMA * TABELA_Q[prox_estado][melhor_prox_acao] -
        TABELA_Q[estado][acao]
    )
    
def salva_tabela_q(tabela_q, nome_arquivo):
    # salva a tabela Q em um arquivo resultado.txt.
    with open(nome_arquivo, "w") as file:
        for row in tabela_q:
            file.write(" ".join(map(str, row)) + "\n")
    print(f"Tabela Q salva com sucesso em {nome_arquivo}")

        
def treinamedo_agente():
    # treina o agente usando q-Learning com penalizações opcionais.
    socket = connection.connect(2037)
    epsilon = EPSILON

    for episodeo in range(EPISODIOS):
        estado_atual = random.randint(0, NUM_ESTADOS - 1)
        done = False
        historico_acoes = []  # Histórico de ações para detectar giros consecutivos

        while not done:
            
            indece_acao = escolher_acao(estado_atual, epsilon)
            acao = ACOES[indece_acao]

            # Envia ação e recebe próximo estado e recompensa
            prox_estado, recompensa = connection.get_state_reward(socket, acao)
            if isinstance(prox_estado, str):
                prox_estado = int(prox_estado, 2)
                
            if prox_estado == estado_atual:
                recompensa = 0

            atualiza_tabela_q(estado_atual, indece_acao, recompensa, prox_estado)
            estado_atual = prox_estado

            # critério de término do episódio -- parâmetro que pode ser variado  -- estou com problemas nele
            # pois não estou com muota certeza se essa é a abordagem correta
            if recompensa <= -14:
                done = True

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        if (episodeo + 1) % 2 == 0:
            print(f"Episódio {episodeo + 1}/{EPISODIOS} concluído.")

    salva_tabela_q(TABELA_Q, "resultado.txt")
    print("Treinamento concluído.")

if __name__ == "__main__":
    treinamedo_agente()
