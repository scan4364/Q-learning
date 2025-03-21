# 🚀 Projeto Q-Learning: Navegação do Amongois

## 📌 Descrição do Projeto
Este projeto implementa o algoritmo de **Q-Learning** para treinar o personagem **Amongois** a navegar por plataformas em um ambiente de simulação. O objetivo é que o agente aprenda a alcançar uma **plataforma final (bloco preto)**, evitando quedas e otimizando seu caminho.

---

## 🎮 O Jogo
Neste jogo, controlamos o personagem **Amongois**, que precisa passar por diversas plataformas para chegar ao seu objetivo final (**bloco preto**). Para isso, ele pode realizar **três movimentos**:

➡️ **Girar para a esquerda** (`left`)
➡️ **Girar para a direita** (`right`)
⬆️ **Pular para a frente** (`jump`)

---

## 🤖 Implementação do Q-Learning

### 🔍 Visão Geral do Algoritmo
O **Q-Learning** é um algoritmo de **aprendizagem por reforço** que permite ao agente aprender uma política ótima através da interação com o ambiente. O agente aprende a **mapear estados para ações** de forma a **maximizar uma recompensa acumulada**.

### 🏗️ Estrutura do Projeto
📂 **Classe QLearning**: Implementa o algoritmo com métodos para inicialização, escolha de ações, atualização da Q-table e treinamento.
📂 **Conexão com o Jogo**: Utiliza o módulo `connection.py` para comunicação com o ambiente de simulação.
📂 **Processo de Treinamento e Teste**: Métodos para treinar o agente e avaliar seu desempenho.

---

## 📊 Representação de Estados e Ações

### 🏠 **Estados**
- **96 estados possíveis** (**24 plataformas × 4 direções**)
- Cada estado é representado como um vetor binário que concatena a informação da plataforma (5 bits) e da direção (2 bits)
- **Direções:**
  - `00` = Norte
  - `01` = Leste
  - `10` = Sul
  - `11` = Oeste

### 🎯 **Ações**
- `⬅️` **Girar para Esquerda**
- `➡️` **Girar para Direita**
- `⬆️` **Pular para Frente**

### 💰 **Recompensas**
- **Objetivo alcançado**: `-1`
- **Queda da plataforma**: `-14`

---

## ▶️ Como Executar
1. **Certifique-se de que o executável do jogo esteja em execução** 🕹️
2. **Execute o script principal**:
   ```bash
   python client.py
   ```
3. O agente começará seu **treinamento** e, após concluído, **testará a política aprendida**.


---

## 🎮 Jogando Manualmente

🕹️ **Comandos:**
- `⬅️` / `➡️` **Setas de direção**: Girar para esquerda/direita.
- `⏹️` **Barra de espaço**: Pular para frente.

⚙️ **Teclas de atalho:**
- `1` 🔼 Aumenta a velocidade do Amongois
- `2` 🔽 Diminui a velocidade do Amongois
- `3-7` 🔲 Ajustam o tamanho da tela progressivamente

---

## 📊 Resultados e Métricas
📌 **Taxa de Sucesso**: Percentual de testes em que o agente atinge o objetivo 🏆.
📌 **Q-table Final**: Armazenada em arquivo texto para **análise** ou **uso futuro**.

---

