import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.envs.toy_text.cliffwalking import CliffWalkingEnv
import random
from collections import deque
import time
import itertools
import os

# Verifica se è disponibile la GPU
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(f"Utilizzo device: {device}")

# Creazione dell'ambiente Cliff Walking Slippery
class SlipperyCliffWalkingEnv(CliffWalkingEnv):
    def __init__(self, slip_chance=0.2):
        super(SlipperyCliffWalkingEnv, self).__init__()
        self.slip_chance = slip_chance
        
    def step(self, action):
        # Con una certa probabilità, esegui un'azione casuale invece di quella scelta
        if random.random() < self.slip_chance:
            action = random.randint(0, 3)
        
        return super().step(action)

# Definizione della Policy Network con architettura configurabile
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers_config):
        """
        Inizializza la policy network con architettura configurabile
        
        Args:
            state_size: dimensione dello spazio degli stati
            action_size: dimensione dello spazio delle azioni
            hidden_layers_config: lista di interi, ogni intero rappresenta il numero di nodi in un layer nascosto
        """
        super(PolicyNetwork, self).__init__()
        
        # Costruzione dinamica dei layer
        self.layers = nn.ModuleList()
        
        # Input layer -> primo hidden layer
        self.layers.append(nn.Linear(state_size, hidden_layers_config[0]))
        
        # Hidden layers
        for i in range(len(hidden_layers_config) - 1):
            self.layers.append(nn.Linear(hidden_layers_config[i], hidden_layers_config[i+1]))
        
        # Ultimo hidden layer -> output layer
        self.layers.append(nn.Linear(hidden_layers_config[-1], action_size))
        
    def forward(self, x):
        # Passaggio attraverso tutti i layer tranne l'ultimo con attivazione ReLU
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        
        # Layer di output senza attivazione (per valori Q)
        return self.layers[-1](x)

# Classe per l'agente di apprendimento
class CliffWalkingAgent:
    def __init__(self, state_size, action_size, hidden_layers_config, learning_rate=0.001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, buffer_size=2000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Fattore di sconto
        self.hidden_layers_config = hidden_layers_config
        
        # Parametri per l'esplorazione epsilon-greedy
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Experience replay
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
        # Policy network
        self.policy_net = PolicyNetwork(state_size, action_size, hidden_layers_config).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
    
    def remember(self, state, action, reward, next_state, done):
        # Memorizza l'esperienza nel buffer
        self.memory.append((state, action, reward, next_state, done))
    
    
    def act(self, state, training=True):
        """
        Se training è True, esegui un'azione in modo epsilon-greedy.
        Altrimenti, esegui l'azione con il valore Q massimo stimato dalla rete.
        """
        # Epsilon-greedy per l'esplorazione
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        # Conversione dello stato in un tensore
        state_tensor = torch.FloatTensor(self.state_to_one_hot(state)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values).item()
    
    def state_to_one_hot(self, state):
        # Converte lo stato (intero) in una rappresentazione one-hot
        one_hot = np.zeros(self.state_size)
        one_hot[state] = 1.0
        return one_hot
    
    def replay(self):
        # Verifica se ci sono abbastanza esperienze nel buffer
        if len(self.memory) < self.batch_size:
            return 0
        
        # Campiona un batch di esperienze
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = []
        targets = []
        
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(self.state_to_one_hot(state)).to(device)
            next_state_tensor = torch.FloatTensor(self.state_to_one_hot(next_state)).to(device)
            
            # Calcolo del target Q
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                next_q_values = self.policy_net(next_state_tensor)
                target_q = q_values.clone()
                
                if done:
                    target_q[action] = reward
                else:
                    # Aggiornamento basato sulla formula di Bellman
                    target_q[action] = reward + self.gamma * torch.max(next_q_values)
            
            states.append(state_tensor)
            targets.append(target_q)
        
        # Trasformazione in tensori batch
        states_batch = torch.stack(states)
        targets_batch = torch.stack(targets)
        
        # Calcolo della loss e ottimizzazione
        predictions = self.policy_net(states_batch)
        loss = F.mse_loss(predictions, targets_batch)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Aggiornamento del parametro epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()
    
    def save_model(self, filename):
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hidden_layers_config': self.hidden_layers_config,
            'epsilon': self.epsilon
        }, filename)
    
    def load_model(self, filename):
        checkpoint = torch.load(filename)
        # Ricreare la rete con la configurazione salvata
        self.hidden_layers_config = checkpoint['hidden_layers_config']
        self.policy_net = PolicyNetwork(self.state_size, self.action_size, self.hidden_layers_config).to(device)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

# Funzione principale di training
def train_agent(env, agent, episodes, max_steps=200, render_interval=100, early_stop_threshold=None):
    rewards = []
    losses = []
    epsilon_history = []
    best_avg_score = float('-inf')
    no_improvement_count = 0
    
    for episode in range(1, episodes+1):
        state, _ = env.reset()
        episode_reward = 0
        episode_losses = []
        
        for step in range(max_steps):
            # Selezione dell'azione
            action = agent.act(state)
            
            # Esecuzione dell'azione
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            # Memorizzazione dell'esperienza
            agent.remember(state, action, reward, next_state, done)
            
            # Aggiornamento dello stato
            state = next_state
            episode_reward += reward
            
            # Training
            loss = agent.replay()
            if loss > 0:
                episode_losses.append(loss)
            
            if done:
                break
        
        # Salvataggio delle metriche
        rewards.append(episode_reward)
        if episode_losses:
            losses.append(np.mean(episode_losses))
        epsilon_history.append(agent.epsilon)
        
        # Calcolo della media del punteggio
        avg_score = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        
        # Controllo per l'early stopping
        if early_stop_threshold is not None and episode > 100:
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            if no_improvement_count >= early_stop_threshold:
                print(f"Early stopping a episodio {episode} - Nessun miglioramento per {early_stop_threshold} episodi")
                break
        
        # Stampa delle statistiche
        if episode % render_interval == 0 or episode == 1:
            avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses) if losses else 0
            print(f"Episodio {episode}/{episodes}, Reward: {episode_reward}, Media Rewards (100 ep): {avg_score:.2f}, "
                  f"Loss media: {avg_loss:.4f}, Epsilon: {agent.epsilon:.4f}")
    
    return rewards, losses, epsilon_history

# Funzione per visualizzare l'agente
def evaluate_agent(env, agent, episodes=100, render=False):
    total_rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        step = 0
        successes = 0
        
        while not (done or truncated) and step < 200:
            action = agent.act(state, training=False)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            total_reward += reward
            state = next_state
            step += 1
            
            if render:
                env.render()
                time.sleep(0.1)
        
        if not truncated or (total_reward > -100):
            successes += 1
        
        total_rewards.append(total_reward)
        print(f"Valutazione episodio {episode+1}: Reward totale = {total_reward}")
    
    avg_reward = np.mean(total_rewards)
    print(f"Reward media su {episodes} episodi: {avg_reward:.2f}")
    print(f"Successi: {successes}/{episodes} episodi -- {successes/episodes*100:.2f}% success rate")
    return avg_reward

# Visualizzazione delle performance
def plot_results(config_results, title="Confronto Configurazioni"):
    plt.figure(figsize=(12, 8))
    
    for config, scores in config_results.items():
        # Calcolo della media mobile per smussare la curva
        window_size = min(100, len(scores) // 10) if len(scores) > 10 else 1
        moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        plt.plot(moving_avg, label=f"Config: {config}")
    
    plt.title(title)
    plt.xlabel('Episodio')
    plt.ylabel('Punteggio (media mobile)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.show()

# Funzione per testare diverse configurazioni
def test_configurations(hidden_layer_configs, episodes_per_config=500, training_params=None):
    if training_params is None:
        training_params = {
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'buffer_size': 2000,
            'batch_size': 64
        }
    
    results = {}
    evaluation_scores = {}
    
    # Creazione della directory per i modelli se non esiste
    models_dir = "cliff_models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    for config in hidden_layer_configs:
        config_name = "_".join(map(str, config))
        print(f"\n\n{'='*50}")
        print(f"Training con configurazione: {config}")
        print(f"{'='*50}")
        
        # Creazione dell'ambiente
        env = SlipperyCliffWalkingEnv(slip_chance=0.2)
        
        # Inizializzazione dell'agente
        state_size = env.observation_space.n
        action_size = env.action_space.n
        agent = CliffWalkingAgent(
            state_size, action_size, 
            hidden_layers_config=config,
            **training_params
        )
        
        # Training
        scores, _, _ = train_agent(
            env, agent, 
            episodes=episodes_per_config, 
            max_steps=200, 
            render_interval=100,
            early_stop_threshold=50  # Ferma se non c'è miglioramento per 50 episodi
        )
        
        # Salvataggio del modello
        model_path = os.path.join(models_dir, f"cliff_walking_model_{config_name}.pth")
        agent.save_model(model_path)
        print(f"Modello salvato in: {model_path}")
        
        # Valutazione
        print(f"\nValutazione della configurazione {config}:")
        avg_reward = evaluate_agent(env, agent, episodes=100, render=False)
        evaluation_scores[str(config)] = avg_reward
        
        # Salvataggio dei risultati
        results[str(config)] = scores
    
    return results, evaluation_scores

# Visualizzazione comparativa delle configurazioni
def compare_configurations(evaluation_scores):
    configs = list(evaluation_scores.keys())
    scores = list(evaluation_scores.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(configs, scores, color='skyblue')
    
    # Aggiungi i valori sopra le barre
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.title('Confronto delle Performance delle Configurazioni')
    plt.xlabel('Configurazione Hidden Layers')
    plt.ylabel('Reward Media')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig('comparison_hidden_layers.png')
    plt.tight_layout()
    plt.show()

# Esecuzione del test di configurazioni multiple
if __name__ == "__main__":
    # Definiamo diverse configurazioni da testare
    hidden_layer_configs = [
               # Due layer con stessa dimensione
        [128, 128],            # Due layer 128
                 # Tre layer 32
        [64, 64, 64]     # Tre layer 64
    ]
    
    print("Inizio test delle configurazioni degli hidden layers...")
    
    # Parametri di training
    training_params = {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_size': 2000,
        'batch_size': 64
    }
    
    # Test di tutte le configurazioni
    results, evaluation_scores = test_configurations(
        hidden_layer_configs, 
        episodes_per_config=500,  # Ridotto per testare più configurazioni in tempo ragionevole
        training_params=training_params
    )
    
    # Visualizzazione comparativa delle curve di apprendimento
    plot_results(results, title="Confronto Curve di Apprendimento")
    
    # Visualizzazione delle performance finali
    compare_configurations(evaluation_scores)
    
    # Identifica la migliore configurazione
    best_config = max(evaluation_scores, key=evaluation_scores.get)
    best_score = evaluation_scores[best_config]
    print(f"\nLa configurazione migliore è: {best_config} con reward media: {best_score:.2f}")
    
    # Training più lungo sulla configurazione migliore
    print(f"\n\n{'='*50}")
    print(f"Training esteso sulla configurazione migliore: {best_config}")
    print(f"{'='*50}")
    
    # Conversione della stringa di configurazione in lista di interi
    best_config_list = [int(x) for x in best_config.strip('[]').split(', ')]
    
    # Creazione dell'ambiente
    env = SlipperyCliffWalkingEnv(slip_chance=0.2)
    
    # Inizializzazione dell'agente con la migliore configurazione
    state_size = env.observation_space.n
    action_size = env.action_space.n
    best_agent = CliffWalkingAgent(
        state_size, action_size, 
        hidden_layers_config=best_config_list,
        **training_params
    )
    
    # Training esteso
    best_scores, best_losses, best_epsilons = train_agent(
        env, best_agent, 
        episodes=1000,  # Training più lungo per la configurazione migliore
        max_steps=200, 
        render_interval=100
    )
    
    # Salvataggio del modello finale
    best_model_path = f"cliff_walking_best_model.pth"
    best_agent.save_model(best_model_path)
    print(f"Modello migliore salvato in: {best_model_path}")
    
    # Visualizzazione delle performance finali
    plt.figure(figsize=(12, 8))
    
    # Calcolo della media mobile per smussare la curva
    window_size = min(100, len(best_scores) // 10)
    moving_avg = np.convolve(best_scores, np.ones(window_size)/window_size, mode='valid')
    
    plt.plot(moving_avg)
    plt.title(f'Performance della Configurazione Migliore: {best_config}')
    plt.xlabel('Episodio')
    plt.ylabel('Punteggio (media mobile)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('best_configuration_performance.png')
    plt.show()
    
    # Valutazione finale
    print(f"\nValutazione finale della configurazione migliore {best_config}:")
    evaluate_agent(env, best_agent, episodes=100, render=False)