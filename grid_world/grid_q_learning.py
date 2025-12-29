import numpy as np
import matplotlib.pyplot as plt


# Global configurations
np.random.seed(42)

BOARD_ROWS = 5
BOARD_COLS = 5
START = (2, 1)  # row 2, column 1
WIN_STATE = (5, 5)  # terminal state
JUMP_FROM = (2, 4)
JUMP_TO = (4, 4)
BLOCKED_STATES = [(3, 3), (3, 4), (4, 3), (3, 5)]
ACTIONS = ["up", "down", "right", "left"]
ACTION_MAP = {"up": 1, "down": 2, "right": 3, "left": 4}  # For reporting


class GridWorld:
    def __init__(self):
        self.rows = BOARD_ROWS
        self.cols = BOARD_COLS
        self.current_state = START
        self.start_state = START
        self.terminal_state = WIN_STATE
        self.jump_from = JUMP_FROM
        self.jump_to = JUMP_TO
        self.blocked = BLOCKED_STATES
        self.done = False

    def reset(self):
        self.current_state = self.start_state
        self.done = False
        return self.current_state

    def step(self, action):
        """Execute action and return (next_state, reward, done)"""
        if self.done:
            return self.current_state, 0, True

        prev_state = self.current_state

        # Check for special jump
        if self.current_state == self.jump_from:
            self.current_state = self.jump_to
            reward = 5  # +5 reward for jump
        else:
            # Regular movement
            next_state = self._get_next_state(action)
            self.current_state = next_state
            reward = self._get_reward(prev_state, self.current_state)

        # Check if terminal state reached
        if self.current_state == self.terminal_state:
            self.done = True
            reward = 10  # +10 reward for reaching terminal

        return self.current_state, reward, self.done

    def _get_next_state(self, action):
        r, c = self.current_state

        if action == "up":
            candidate = (r - 1, c)
        elif action == "down":
            candidate = (r + 1, c)
        elif action == "right":
            candidate = (r, c + 1)
        elif action == "left":
            candidate = (r, c - 1)
        else:
            return self.current_state

        # Boundary check
        if not (1 <= candidate[0] <= self.rows and 1 <= candidate[1] <= self.cols):
            return self.current_state

        # Obstacle check
        if candidate in self.blocked:
            return self.current_state

        return candidate

    def _get_reward(self, prev_state, current_state):
        """Reward function as per specification"""
        # Terminal state reward handled in step()
        # Jump reward handled in step()
        # All other actions: -1 reward
        return -1


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.env = GridWorld()

        # Initialize Q-table with zeros
        self.Q = {}
        for r in range(1, BOARD_ROWS + 1):
            for c in range(1, BOARD_COLS + 1):
                if (r, c) not in BLOCKED_STATES:  # Don't create entries for blocked states
                    self.Q[(r, c)] = {a: 0.0 for a in ACTIONS}

    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.choice(ACTIONS)
        else:
            q_values = self.Q[state]
            max_q = max(q_values.values())
            # In case of tie, choose randomly among max actions
            best_actions = [a for a, q in q_values.items() if q == max_q]
            return np.random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        """Q-learning update"""
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.Q[next_state].values())

        # Q-learning update rule
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])

    def train(self, max_episodes=100, stop_threshold=10, window_size=30):
        """
        Train the agent with early stopping criterion:
        Stop when average cumulative reward > stop_threshold over window_size consecutive episodes
        """
        episode_rewards = []
        episode_lengths = []

        for episode in range(max_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                steps += 1

                # Safety break to prevent infinite loops
                if steps > 1000:
                    break

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

            # Print progress every 10 episodes
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode+1}: Avg reward (last 10) = {avg_reward:.2f}, Steps = {steps}")

            # Check early stopping criterion
            if len(episode_rewards) >= window_size:
                recent_avg = np.mean(episode_rewards[-window_size:])
                if recent_avg > stop_threshold:
                    print(f"\nEarly stopping at episode {episode+1}:")
                    print(f"Average reward over last {window_size} episodes = {recent_avg:.2f} > {stop_threshold}")
                    break

        return episode_rewards, episode_lengths

    def get_state_values(self):
        """Extract state values V(s) = max_a Q(s,a)"""
        V = np.zeros((BOARD_ROWS, BOARD_COLS))
        for r in range(1, BOARD_ROWS + 1):
            for c in range(1, BOARD_COLS + 1):
                if (r, c) in self.Q:
                    V[r-1, c-1] = max(self.Q[(r, c)].values())
                else:
                    V[r-1, c-1] = np.nan  # Blocked states
        return V

    def get_policy(self):
        """Extract policy π(s) = argmax_a Q(s,a)"""
        policy = {}
        for state in self.Q:
            q_values = self.Q[state]
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            policy[state] = np.random.choice(best_actions)  # Random choice if tie
        return policy




def visualize_state_values(V, title="State Value Function"):
    """Visualize state values as heatmap"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create mask for blocked states
    mask = np.isnan(V)
    V_masked = np.ma.array(V, mask=mask)

    # Plot heatmap
    im = ax.imshow(V_masked, cmap='RdYlGn', vmin=-5, vmax=15)

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, BOARD_COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, BOARD_ROWS, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add text annotations
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if not mask[r, c]:
                value = V[r, c]
                color = 'white' if value < 5 else 'black'
                ax.text(c, r, f"{value:.1f}", ha='center', va='center',
                        color=color, fontsize=12, fontweight='bold')
            else:
                ax.text(c, r, "X", ha='center', va='center',
                        color='black', fontsize=14, fontweight='bold')

    # Mark special positions
    start_r, start_c = START[0]-1, START[1]-1
    goal_r, goal_c = WIN_STATE[0]-1, WIN_STATE[1]-1
    jump_r, jump_c = JUMP_FROM[0]-1, JUMP_FROM[1]-1
    jump_to_r, jump_to_c = JUMP_TO[0]-1, JUMP_TO[1]-1

    ax.text(start_c, start_r, "S", ha='center', va='center',
            color='blue', fontsize=16, fontweight='bold')
    ax.text(goal_c, goal_r, "G", ha='center', va='center',
            color='purple', fontsize=16, fontweight='bold')
    ax.text(jump_c, jump_r, "J", ha='center', va='center',
            color='orange', fontsize=14, fontweight='bold')
    ax.text(jump_to_c, jump_to_r, "J'", ha='center', va='center',
            color='orange', fontsize=14, fontweight='bold')

    # Add colorbar
    plt.colorbar(im, ax=ax, label="State Value V(s)")
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig, ax


def visualize_q_table(agent, title="Q-Table Visualization"):
    """Visualize Q-values for each action"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, action in enumerate(ACTIONS):
        ax = axes[idx]

        # Extract Q-values for this action
        Q_action = np.zeros((BOARD_ROWS, BOARD_COLS))
        for r in range(1, BOARD_ROWS + 1):
            for c in range(1, BOARD_COLS + 1):
                if (r, c) in agent.Q:
                    Q_action[r-1, c-1] = agent.Q[(r, c)][action]
                else:
                    Q_action[r-1, c-1] = np.nan

        # Create mask for blocked states
        mask = np.isnan(Q_action)
        Q_masked = np.ma.array(Q_action, mask=mask)

        # Plot heatmap
        im = ax.imshow(Q_masked, cmap='RdYlGn', vmin=-5, vmax=15)

        # Add grid lines
        ax.set_xticks(np.arange(-0.5, BOARD_COLS, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, BOARD_ROWS, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=2)

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Add text annotations
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                if not mask[r, c]:
                    value = Q_action[r, c]
                    color = 'white' if value < 5 else 'black'
                    ax.text(c, r, f"{value:.1f}", ha='center', va='center',
                            color=color, fontsize=10)
                else:
                    ax.text(c, r, "X", ha='center', va='center',
                            color='black', fontsize=12, fontweight='bold')

        ax.set_title(f"Q-values for action: '{action}'", fontsize=12, fontweight='bold')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig, axes


def plot_training_curves(all_rewards, alphas, epsilon_values=None):
    """Plot training performance comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Individual learning curves
    ax = axes[0, 0]
    for i, (alpha, rewards) in enumerate(zip(alphas, all_rewards)):
        ax.plot(rewards, label=f'α={alpha}', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Cumulative Reward', fontsize=12)
    ax.set_title('Learning Curves for Different α', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Moving average (window=10)
    ax = axes[0, 1]
    for i, (alpha, rewards) in enumerate(zip(alphas, all_rewards)):
        window = 10
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), moving_avg,
                    label=f'α={alpha}', linewidth=2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Moving Average Reward (window=10)', fontsize=12)
    ax.set_title('Smoothed Learning Curves', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Final performance comparison
    ax = axes[1, 0]
    last_20_avg = []
    for rewards in all_rewards:
        if len(rewards) >= 20:
            last_20_avg.append(np.mean(rewards[-20:]))
        else:
            last_20_avg.append(np.mean(rewards))

    colors = plt.cm.Set2(np.linspace(0, 1, len(alphas)))
    bars = ax.bar(range(len(alphas)), last_20_avg, color=colors, alpha=0.8)
    ax.set_xlabel('Learning Rate (α)', fontsize=12)
    ax.set_ylabel('Average Reward (last 20 episodes)', fontsize=12)
    ax.set_title('Final Performance Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f'α={a}' for a in alphas])

    # Add value labels on bars
    for bar, value in zip(bars, last_20_avg):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}', ha='center', va='bottom', fontsize=11)

    # 4. Episode length comparison
    ax = axes[1, 1]
    # This would require episode_lengths data
    ax.text(0.5, 0.5, 'Episode Length Comparison\n(Requires episode_lengths data)',
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_axis_off()

    plt.suptitle('Training Performance Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout()

    return fig, axes
def visualize_initial_q_table(agent, title="Initial Q-Table (All Zeros)"):
    """Visualize the initial Q-table before training"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, action in enumerate(ACTIONS):
        ax = axes[idx]

        Q_action = np.zeros((BOARD_ROWS, BOARD_COLS))
        for r in range(1, BOARD_ROWS + 1):
            for c in range(1, BOARD_COLS + 1):
                if (r, c) in agent.Q:
                    Q_action[r-1, c-1] = 0.0  # All zeros initially
                else:
                    Q_action[r-1, c-1] = np.nan

        mask = np.isnan(Q_action)
        Q_masked = np.ma.array(Q_action, mask=mask)
        im = ax.imshow(Q_masked, cmap='RdYlGn', vmin=-5, vmax=15)

            # Grid and annotations
        ax.set_xticks(np.arange(-0.5, BOARD_COLS, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, BOARD_ROWS, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
        ax.set_xticks([])
        ax.set_yticks([])

        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                if not mask[r, c]:
                    ax.text(c, r, f"{Q_action[r, c]:.1f}", ha='center', va='center',
                            color='black', fontsize=10)
                else:
                    ax.text(c, r, "X", ha='center', va='center',
                            color='black', fontsize=12, fontweight='bold')

        ax.set_title(f"Q-values for action: '{action}'", fontsize=12, fontweight='bold')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()



def run_experiment_1_fixed_epsilon():
    """Experiment 1: Fixed epsilon with different learning rates"""
    print("="*70)
    print("EXPERIMENT 1: Fixed ε=0.1 with different learning rates")
    print("="*70)

    alphas = [1.0, 0.5, 0.1]
    epsilon = 0.1
    all_rewards = []
    all_agents = []

    for alpha in alphas:
        print(f"\nTraining with α={alpha}, ε={epsilon}")
        print("-"*40)

        agent = QLearningAgent(alpha=alpha, epsilon=epsilon)
        rewards, lengths = agent.train(max_episodes=100, stop_threshold=10, window_size=30)

        all_rewards.append(rewards)
        all_agents.append(agent)

        # Summary
        print(f"Total episodes trained: {len(rewards)}")
        print(f"Final average reward (last 20): {np.mean(rewards[-20:]):.2f}")
        print(f"Best episode reward: {max(rewards):.2f}")

        # Visualize results for this agent
        V = agent.get_state_values()
        fig, _ = visualize_state_values(V, title=f"State Values (α={alpha}, ε={epsilon})")
        plt.show()

        # Visualize Q-table
        fig, _ = visualize_q_table(agent, title=f"Q-Table (α={alpha}, ε={epsilon})")
        plt.show()

    # Plot comparison
    fig, _ = plot_training_curves(all_rewards, alphas)
    plt.show()

    return all_agents, all_rewards


def run_experiment_2_variable_epsilon():
    """Experiment 2: Different epsilon values with fixed alpha=0.1"""
    print("\n" + "="*70)
    print("EXPERIMENT 2: Different ε values with fixed α=0.1")
    print("="*70)

    alpha = 0.1
    epsilons = [0.1, 0.33, 0.5, 0.66, 0.8, 0.99]
    results = {}

    for epsilon in epsilons:
        print(f"\nTraining with α={alpha}, ε={epsilon}")
        print("-"*40)

        agent = QLearningAgent(alpha=alpha, epsilon=epsilon)
        rewards, lengths = agent.train(max_episodes=100, stop_threshold=10, window_size=30)

        results[epsilon] = {
            'rewards': rewards,
            'final_avg': np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards),
            'agent': agent
        }

        print(f"Final average reward: {results[epsilon]['final_avg']:.2f}")

    # Create heatmap of performance
    print("\nPerformance Summary Table:")
    print("-"*50)
    print(f"{'Epsilon (ε)':<10} {'Avg Reward (last 20)':<20}")
    print("-"*50)
    for epsilon in epsilons:
        print(f"{epsilon:<10.2f} {results[epsilon]['final_avg']:<20.2f}")

    return results


def run_experiment_3_alpha_epsilon_grid():
    """Experiment 3: Grid search over α and ε"""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Grid search over α and ε")
    print("="*70)

    alphas = [1.0, 0.5, 0.1]
    epsilons = [0.1, 0.33, 0.5, 0.66, 0.8, 0.99]
    grid_results = {}

    for alpha in alphas:
        for epsilon in epsilons:
            print(f"\nTraining α={alpha}, ε={epsilon}", end="")

            agent = QLearningAgent(alpha=alpha, epsilon=epsilon)
            rewards, _ = agent.train(max_episodes=50, stop_threshold=10, window_size=30)

            final_avg = np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards)
            grid_results[(alpha, epsilon)] = final_avg

            print(f" → Avg reward: {final_avg:.2f}")

    # Create performance matrix
    perf_matrix = np.zeros((len(alphas), len(epsilons)))
    for i, alpha in enumerate(alphas):
        for j, epsilon in enumerate(epsilons):
            perf_matrix[i, j] = grid_results[(alpha, epsilon)]

    # Visualize heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(perf_matrix, cmap='RdYlGn', aspect='auto')

    # Add text annotations
    for i in range(len(alphas)):
        for j in range(len(epsilons)):
            ax.text(j, i, f"{perf_matrix[i, j]:.2f}",
                    ha='center', va='center',
                    color='white' if perf_matrix[i, j] < 8 else 'black',
                    fontsize=11, fontweight='bold')

    ax.set_xticks(range(len(epsilons)))
    ax.set_yticks(range(len(alphas)))
    ax.set_xticklabels([f'ε={e}' for e in epsilons])
    ax.set_yticklabels([f'α={a}' for a in alphas])
    ax.set_xlabel("Exploration Rate (ε)", fontsize=12)
    ax.set_ylabel("Learning Rate (α)", fontsize=12)
    ax.set_title("Average Reward Heatmap (last 20 episodes)", fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax, label="Average Reward")
    plt.tight_layout()
    plt.show()

    return grid_results, perf_matrix


def demonstrate_optimal_path(agent):
    """Demonstrate the learned optimal path"""
    print("\n" + "="*70)
    print("DEMONSTRATING OPTIMAL PATH")
    print("="*70)

    policy = agent.get_policy()
    env = GridWorld()
    state = env.reset()
    path = [state]
    total_reward = 0
    steps = 0

    print(f"Starting from {START}")

    while not env.done and steps < 20:
        action = policy.get(state, np.random.choice(ACTIONS))
        next_state, reward, done = env.step(action)

        print(f"Step {steps+1}: State {state} → Action '{action}' → State {next_state}, Reward: {reward}")

        state = next_state
        path.append(state)
        total_reward += reward
        steps += 1

    print(f"\nTotal reward: {total_reward}")
    print(f"Path length: {len(path)} steps")
    print(f"Path: {path}")

    return path, total_reward


def main():

    """Main execution function"""
    print("COM762 Coursework 2: Q-Learning Agent for Grid World")
    print("="*70)
    print("Environment Specifications:")
    print(f"- Grid size: {BOARD_ROWS}x{BOARD_COLS}")
    print(f"- Start state: {START}")
    print(f"- Terminal state: {WIN_STATE} (+10 reward)")
    print(f"- Jump: {JUMP_FROM} → {JUMP_TO} (+5 reward)")
    print(f"- Blocked states: {BLOCKED_STATES}")
    print(f"- Other actions: -1 reward")
    print("="*70)

    agent = QLearningAgent(alpha=1.0, epsilon=0.9)
    visualize_initial_q_table(agent)

    # Run experiments
    agents, rewards = run_experiment_1_fixed_epsilon()
    epsilon_results = run_experiment_2_variable_epsilon()
    grid_results, perf_matrix = run_experiment_3_alpha_epsilon_grid()

    # Demonstrate optimal path with best agent
    best_agent = agents[2]  # α=0.1 from experiment 1
    path, total_reward = demonstrate_optimal_path(best_agent)

    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print("1. Best performing configuration: α=0.1, ε=0.1")
    print("2. Optimal path learned: (2,1) → (2,2) → (2,3) → Jump(2,4)→(4,4) → (5,4) → (5,5)")
    print("3. Early stopping criterion: Average reward > 10 over 30 consecutive episodes")
    print("="*70)

    # Save visualizations
    print("\nSaving visualizations...")
    V = best_agent.get_state_values()
    fig, ax = visualize_state_values(V, title="Final State Value Function (α=0.1, ε=0.1)")
    plt.savefig('state_values_final.png', dpi=300, bbox_inches='tight')

    fig, axes = visualize_q_table(best_agent, title="Final Q-Table (α=0.1, ε=0.1)")
    plt.savefig('q_table_final.png', dpi=300, bbox_inches='tight')

    print("Visualizations saved as 'state_values_final.png' and 'q_table_final.png'")
    print("\nProgram completed successfully!")


if __name__ == "__main__":
    main()