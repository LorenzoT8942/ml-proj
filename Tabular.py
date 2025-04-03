import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm
from SlipperyCliffWalkingEnv import SlipperyCliffWalkingEnv

custom_env = True

# Creazione dell'ambiente CliffWalking nella versione slippery
env = SlipperyCliffWalkingEnv(slip_chance=0.2) if custom_env else gym.make('CliffWalking-v0', is_slippery =True)

print(f"Ambiente custom: {custom_env}")

# Parametri per il Q-learning
alpha = 0.1       # Tasso di apprendimento
gamma = 0.99      # Fattore di sconto
epsilon = 1.0     # Parametro di esplorazione (epsilon-greedy)
epsilon_decay = 0.995  # Decadimento di epsilon
epsilon_min = 0.01     # Valore minimo di epsilon
episodes = 10000  # Numero di episodi
evaluation_frequency = 100  # Frequenza di valutazione

# Inizializzazione della Q-table
num_states = env.observation_space.n
num_actions = env.action_space.n
q_table = np.zeros((num_states, num_actions))

# Liste per memorizzare le metriche di performance
rewards_per_episode = []
success_rates = []
evaluation_episodes = []
avg_rewards = []
avg_steps = []

# Funzione di valutazione dell'agente addestrato
def evaluate_agent(q_table, num_eval_episodes=100):
    # Ambiente di valutazione (senza rendering)
    eval_env = SlipperyCliffWalkingEnv(slip_chance=0.2) if custom_env else gym.make('CliffWalking-v0', is_slippery =True)
    
    success_count = 0
    total_rewards = 0
    total_steps = 0
    
    for _ in range(num_eval_episodes):
        state, _ = eval_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        
        while not (done or truncated):
            # Politica greedy per la valutazione (sempre la migliore azione)
            action = np.argmax(q_table[state, :])
            next_state, reward, done, truncated, _ = eval_env.step(action)
            state = next_state
            episode_reward += reward
            steps += 1
        
        # Se l'episodio è terminato con successo (raggiungimento dell'obiettivo)
        if episode_reward > -100:  # Soglia di successo
            success_count += 1
        
        total_rewards += episode_reward
        total_steps += steps
    
    # Calcolo delle metriche
    success_rate = success_count / num_eval_episodes
    avg_reward = total_rewards / num_eval_episodes
    avg_step = total_steps / num_eval_episodes
    
    eval_env.close()
    return success_rate, avg_reward, avg_step

# Training dell'agente con monitoraggio delle performance
for episode in tqdm(range(episodes)):
    state, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    
    while not (done or truncated):
        # Scelta dell'azione con politica epsilon-greedy
        if np.random.random() < epsilon:
            # Esplorazione: scegliamo un'azione casuale
            action = env.action_space.sample()
        else:
            # Sfruttamento: scegliamo l'azione con il valore Q più alto
            action = np.argmax(q_table[state, :])
        
        # Eseguiamo l'azione e osserviamo il nuovo stato e la ricompensa
        next_state, reward, done, truncated, _ = env.step(action)
        
        # Aggiorniamo la Q-table usando la formula del Q-learning
        best_next_action = np.argmax(q_table[next_state, :])
        td_target = reward + gamma * q_table[next_state, best_next_action] * (not done)
        td_error = td_target - q_table[state, action]
        q_table[state, action] += alpha * td_error
        
        # Passiamo al nuovo stato
        state = next_state
        total_reward += reward
    
    # Memorizziamo la ricompensa totale dell'episodio
    rewards_per_episode.append(total_reward)
    
    # Valutiamo l'agente periodicamente per monitorare le performance
    if (episode + 1) % evaluation_frequency == 0:
        success_rate, avg_reward, avg_step = evaluate_agent(q_table)
        success_rates.append(success_rate)
        avg_rewards.append(avg_reward)
        avg_steps.append(avg_step)
        evaluation_episodes.append(episode + 1)
        
        # Stampiamo le metriche di performance
        print(f"Episodio {episode + 1}/{episodes}")
        print(f"Success Rate: {success_rate:.4f}")
        print(f"Ricompensa Media: {avg_reward:.2f}")
        print(f"Passi Medi: {avg_step:.2f}")
        print(f"Epsilon attuale: {epsilon:.4f}")
        print("-" * 30)
    
    # Riduciamo epsilon (riduciamo l'esplorazione nel tempo)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Calcolo della media mobile delle ricompense per visualizzare il trend
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Visualizzazione delle curve di apprendimento (multiple metriche)
window_size = 100
moving_avg = moving_average(rewards_per_episode, window_size)

plt.figure(figsize=(15, 12))

# Figura 1: Ricompensa media durante il training
plt.subplot(4, 1, 1)
plt.plot(moving_avg)
plt.title(f'Media mobile delle ricompense (finestra di {window_size} episodi)')
plt.xlabel('Episodi')
plt.ylabel('Ricompensa media')
plt.grid(True)

# Figura 2: Success Rate durante la valutazione
plt.subplot(4, 1, 2)
plt.plot(evaluation_episodes, success_rates, 'g-o')
plt.title('Success Rate durante la valutazione')
plt.xlabel('Episodi')
plt.ylabel('Success Rate')
plt.ylim([0, 1.05])
plt.grid(True)

# Figura 3: Ricompensa media durante la valutazione
plt.subplot(4, 1, 3)
plt.plot(evaluation_episodes, avg_rewards, 'r-o')
plt.title('Ricompensa media durante la valutazione')
plt.xlabel('Episodi')
plt.ylabel('Ricompensa media')
plt.grid(True)

# Figura 4: Numero medio di passi per episodio
plt.subplot(4, 1, 4)
plt.plot(evaluation_episodes, avg_steps, 'm-o')
plt.title('Numero medio di passi per episodio')
plt.xlabel('Episodi')
plt.ylabel('Passi medi')
plt.grid(True)

plt.tight_layout()
plt.savefig('learning_metrics.png')
plt.show()

# Visualizzazione della Q-table finale
plt.figure(figsize=(12, 10))
sns.heatmap(q_table, cmap='viridis')
plt.title('Q-table finale')
plt.xlabel('Azioni')
plt.ylabel('Stati')
plt.savefig('q_table.png')
plt.show()

# Funzione per visualizzare la politica ottimale come una griglia con frecce
def visualize_policy(q_table, env_name='CliffWalking-v0'):
    # Ottieni le dimensioni della griglia
    env_temp = gym.make(env_name)
    if env_name == 'CliffWalking-v0':
        nrow, ncol = 4, 12
    else:
        # Per altri ambienti, potremmo dover estrarre le dimensioni in modo diverso
        raise ValueError("Ambiente non supportato per la visualizzazione")
    env_temp.close()
    
    # Mappa delle azioni ai vettori di direzione (UP, RIGHT, DOWN, LEFT)
    # 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
    action_vectors = {
        0: (-1, 0),  # UP
        1: (0, 1),   # RIGHT
        2: (1, 0),   # DOWN
        3: (0, -1)   # LEFT
    }
    
    # Mappa delle azioni ai simboli delle frecce
    action_arrows = {
        0: '↑',  # UP
        1: '→',  # RIGHT
        2: '↓',  # DOWN
        3: '←'   # LEFT
    }
    
    # Crea una figura
    plt.figure(figsize=(12, 5))
    
    # Posizioni di cliff (solo per CliffWalking)
    cliff_positions = [(3, j) for j in range(1, 11)]
    
    # Disegnamo la griglia
    for i in range(nrow+1):
        plt.axhline(i, color='black', linestyle='-')
    for j in range(ncol+1):
        plt.axvline(j, color='black', linestyle='-')
    
    # Coloriamo le posizioni speciali
    start_position = (3, 0)  # Bottom-left
    goal_position = (3, 11)  # Bottom-right
    
    # Riempiamo le celle con colori e frecce
    for i in range(nrow):
        for j in range(ncol):
            # Converti le coordinate (i,j) in un indice di stato
            state = i * ncol + j
            
            # Determina il colore della cella
            if (i, j) == start_position:
                cell_color = 'lightgreen'
                plt.text(j + 0.5, i + 0.5, 'S', fontsize=15, ha='center', va='center')
            elif (i, j) == goal_position:
                cell_color = 'gold'
                plt.text(j + 0.5, i + 0.5, 'G', fontsize=15, ha='center', va='center')
            elif (i, j) in cliff_positions:
                cell_color = 'tomato'
                plt.text(j + 0.5, i + 0.5, 'C', fontsize=15, ha='center', va='center')
            else:
                # Determiniamo l'azione migliore per questo stato
                best_action = np.argmax(q_table[state])
                cell_color = 'white'
                
                # Disegniamo una freccia che indica la direzione migliore
                di, dj = action_vectors[best_action]
                arrow_text = action_arrows[best_action]
                
                # Aggiungiamo un valore Q per mostrare la fiducia nell'azione
                q_value = q_table[state, best_action]
                
                plt.text(j + 0.5, i + 0.5, arrow_text, fontsize=15, ha='center', va='center')
                # Aggiungiamo un valore Q piccolo
                plt.text(j + 0.5, i + 0.75, f"{q_value:.1f}", fontsize=8, ha='center', va='center', alpha=0.7)
            
            # Colora la cella
            plt.fill_between([j, j+1], [i, i], [i+1, i+1], color=cell_color, alpha=0.3)
    
    # Aggiungiamo una legenda
    plt.plot([], [], 's', color='lightgreen', alpha=0.3, label='Start')
    plt.plot([], [], 's', color='gold', alpha=0.3, label='Goal')
    plt.plot([], [], 's', color='tomato', alpha=0.3, label='Cliff')
    plt.plot([], [], 's', color='white', alpha=0.3, label='Safe Path')
    
    # Imposta i limiti e le etichette degli assi
    plt.xlim(0, ncol)
    plt.ylim(nrow, 0)  # Invertiamo l'asse y per avere (0,0) in alto a sinistra
    plt.xticks(np.arange(0.5, ncol, 1), np.arange(ncol))
    plt.yticks(np.arange(0.5, nrow, 1), np.arange(nrow))
    plt.xlabel('Colonna')
    plt.ylabel('Riga')
    plt.title('Politica Ottimale per CliffWalking (frecce indicano la migliore azione)')
    plt.grid(False)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    
    plt.tight_layout()
    plt.savefig('optimal_policy.png')
    plt.show()

# Funzione per visualizzare la politica ottimale con frecce direzionali
def visualize_policy_quiver(q_table, env_name='CliffWalking-v0'):
    # Ottieni le dimensioni della griglia
    env_temp = gym.make(env_name)
    if env_name == 'CliffWalking-v0':
        nrow, ncol = 4, 12
    else:
        # Per altri ambienti, potremmo dover estrarre le dimensioni in modo diverso
        raise ValueError("Ambiente non supportato per la visualizzazione")
    env_temp.close()
    
    # Mappa delle azioni ai vettori di direzione (UP, RIGHT, DOWN, LEFT)
    # 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
    action_vectors = {
        0: (0, 1),    # UP (in matplotlib, y aumenta verso l'alto)
        1: (1, 0),    # RIGHT
        2: (0, -1),   # DOWN
        3: (-1, 0)    # LEFT
    }
    
    # Prepara gli array per le frecce
    X, Y = np.meshgrid(np.arange(ncol), np.arange(nrow))
    U = np.zeros((nrow, ncol))
    V = np.zeros((nrow, ncol))
    
    # Calcola i vettori di direzione per ogni stato
    for i in range(nrow):
        for j in range(ncol):
            state = i * ncol + j
            best_action = np.argmax(q_table[state])
            dx, dy = action_vectors[best_action]
            U[i, j] = dx
            V[i, j] = dy
    
    # Crea la figura
    plt.figure(figsize=(14, 6))
    
    # Posizioni speciali
    start_position = (3, 0)  # Bottom-left
    goal_position = (3, 11)  # Bottom-right
    cliff_positions = [(3, j) for j in range(1, 11)]
    
    # Disegna la griglia base
    for i in range(nrow):
        for j in range(ncol):
            if (i, j) == start_position:
                plt.fill_between([j-0.5, j+0.5], [i-0.5, i-0.5], [i+0.5, i+0.5], color='lightgreen', alpha=0.5)
                plt.text(j, i, 'S', fontsize=15, ha='center', va='center')
            elif (i, j) == goal_position:
                plt.fill_between([j-0.5, j+0.5], [i-0.5, i-0.5], [i+0.5, i+0.5], color='gold', alpha=0.5)
                plt.text(j, i, 'G', fontsize=15, ha='center', va='center')
            elif (i, j) in cliff_positions:
                plt.fill_between([j-0.5, j+0.5], [i-0.5, i-0.5], [i+0.5, i+0.5], color='tomato', alpha=0.5)
                plt.text(j, i, 'C', fontsize=10, ha='center', va='center')
    
    # Disegna le frecce direzionali
    plt.quiver(X, Y, U, V, scale=1, scale_units='xy', angles='xy', color='blue', 
               width=0.007, headwidth=4, headlength=4, headaxislength=3.5)
    
    # Personalizza il grafico
    plt.xlim(-0.5, ncol - 0.5)
    plt.ylim(-0.5, nrow - 0.5)
    plt.xticks(np.arange(ncol))
    plt.yticks(np.arange(nrow))
    plt.title('Politica Ottimale per CliffWalking (Le frecce indicano la direzione ottimale da seguire)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Inverti l'asse y per avere (0,0) in alto a sinistra (come è tipico in una griglia)
    plt.gca().invert_yaxis()
    
    # Aggiungi la legenda
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color='lightgreen', alpha=0.5, label='Start'),
        plt.Rectangle((0, 0), 1, 1, color='gold', alpha=0.5, label='Goal'),
        plt.Rectangle((0, 0), 1, 1, color='tomato', alpha=0.5, label='Cliff')
    ]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    plt.tight_layout()
    plt.savefig('optimal_policy_quiver.png')
    plt.show()

# Funzione per eseguire un test statistico completo dell'agente
def test_trained_agent(q_table, num_test_episodes=1000):
    print("\n" + "="*50)
    print("STATISTICHE DI TEST DELL'AGENTE ADDESTRATO")
    print("="*50)
    
    # Ambiente di test
    test_env = SlipperyCliffWalkingEnv(slip_chance=0.2)
    
    # Metriche
    rewards = []
    steps_list = []
    success_count = 0
    paths = []  # Memorizza i percorsi per visualizzazione
    
    # Esecuzione dei test
    print(f"Esecuzione di {num_test_episodes} episodi di test...")
    
    for episode in tqdm(range(num_test_episodes)):
        state, _ = test_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        
        # Memorizza il percorso di questo episodio
        path = []
        
        while not (done or truncated):
            # Aggiungi lo stato attuale al percorso
            path.append(state)
            
            # Politica greedy per il test
            action = np.argmax(q_table[state])
            next_state, reward, done, truncated, _ = test_env.step(action)
            state = next_state
            episode_reward += reward
            steps += 1
        
        # Aggiungi l'ultimo stato
        path.append(state)
        paths.append(path)
        
        rewards.append(episode_reward)
        steps_list.append(steps)
        
        if episode_reward > -100:  # Soglia di successo
            success_count += 1
    
    # Calcolo delle statistiche
    success_rate = success_count / num_test_episodes
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    
    avg_steps = np.mean(steps_list)
    std_steps = np.std(steps_list)
    min_steps = np.min(steps_list)
    max_steps = np.max(steps_list)
    
    # Stampa dei risultati
    print("\nRISULTATI DEL TEST:")
    print(f"Numero di episodi: {num_test_episodes}")
    print(f"Success Rate: {success_rate:.4f} ({success_count}/{num_test_episodes})")
    
    print("\nSTATISTICHE REWARD:")
    print(f"Reward Medio: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Reward Minimo: {min_reward:.2f}")
    print(f"Reward Massimo: {max_reward:.2f}")
    
    print("\nSTATISTICHE PASSI:")
    print(f"Passi Medi: {avg_steps:.2f} ± {std_steps:.2f}")
    print(f"Passi Minimi: {min_steps}")
    print(f"Passi Massimi: {max_steps}")
    
    # Calcolo della distribuzione delle performance
    reward_bins = np.linspace(min_reward, max_reward, 10)
    reward_hist, _ = np.histogram(rewards, bins=reward_bins)
    reward_dist = reward_hist / num_test_episodes
    
    print("\nDISTRIBUZIONE DEI REWARD:")
    for i in range(len(reward_bins)-1):
        print(f"  {reward_bins[i]:.1f} a {reward_bins[i+1]:.1f}: {reward_dist[i]*100:.2f}%")
    
    # Salvataggio delle statistiche in un file
    with open('test_results.txt', 'w') as f:
        f.write("RISULTATI DEL TEST DELL'AGENTE Q-LEARNING\n")
        f.write("="*50 + "\n")
        f.write(f"Numero di episodi: {num_test_episodes}\n")
        f.write(f"Success Rate: {success_rate:.4f} ({success_count}/{num_test_episodes})\n\n")
        
        f.write("STATISTICHE REWARD:\n")
        f.write(f"Reward Medio: {avg_reward:.2f} ± {std_reward:.2f}\n")
        f.write(f"Reward Minimo: {min_reward:.2f}\n")
        f.write(f"Reward Massimo: {max_reward:.2f}\n\n")
        
        f.write("STATISTICHE PASSI:\n")
        f.write(f"Passi Medi: {avg_steps:.2f} ± {std_steps:.2f}\n")
        f.write(f"Passi Minimi: {min_steps}\n")
        f.write(f"Passi Massimi: {max_steps}\n\n")
        
        f.write("DISTRIBUZIONE DEI REWARD:\n")
        for i in range(len(reward_bins)-1):
            f.write(f"  {reward_bins[i]:.1f} a {reward_bins[i+1]:.1f}: {reward_dist[i]*100:.2f}%\n")
    
    print("\nStatistiche complete salvate in 'test_results.txt'")
    
    # Visualizzazione della distribuzione dei reward
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(avg_reward, color='r', linestyle='--', linewidth=1.5, label=f'Media: {avg_reward:.2f}')
    plt.title('Distribuzione dei reward')
    plt.xlabel('Reward')
    plt.ylabel('Frequenza')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('reward_distribution.png')
    plt.show()
    
    # Visualizzazione della distribuzione dei passi
    plt.figure(figsize=(10, 6))
    plt.hist(steps_list, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
    plt.axvline(avg_steps, color='r', linestyle='--', linewidth=1.5, label=f'Media: {avg_steps:.2f}')
    plt.title('Distribuzione del numero di passi')
    plt.xlabel('Passi')
    plt.ylabel('Frequenza')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('steps_distribution.png')
    plt.show()
    
    test_env.close()
    return success_rate, avg_reward, avg_steps, paths

# Visualizza la politica ottimale come una griglia con frecce
print("\nVisualizzazione della politica ottimale...")
visualize_policy(q_table)
visualize_policy_quiver(q_table)

# Esecuzione del test statistico finale
print("\nEsecuzione del test statistico finale dell'agente addestrato...")
success_rate, avg_reward, avg_steps, paths = test_trained_agent(q_table, num_test_episodes=1000)

# Chiudiamo l'ambiente
env.close()

# Salvataggio della Q-table per uso futuro
np.save('q_table_cliffwalking_slippery.npy', q_table)
print("Q-table salvata con successo in 'q_table_cliffwalking_slippery.npy'")