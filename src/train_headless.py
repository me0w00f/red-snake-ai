"""
Headless DQN Snake training - no display needed.
Reimplements the game logic without Pygame rendering.
"""
import torch
import numpy as np
import random
import math
import os
import logging
import json
from collections import deque
from dqn_agent_fast import DQNAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class HeadlessSnakeGame:
    """Snake game without Pygame - pure logic only."""
    
    def __init__(self, width=640, height=480, scale=20):
        self.width = width
        self.height = height
        self.scale = scale
        self.SCORE_VALUE = 1
        self.reset()
    
    def reset(self):
        self.snake_pos = [[self.width // 2, self.height // 2]]
        self.snake_direction = [self.scale, 0]
        self.food_pos = self._generate_food()
        self.score = 0
        self.game_over = False
        return self._get_state()
    
    def _generate_food(self):
        while True:
            x = random.randint(0, (self.width - self.scale) // self.scale) * self.scale
            y = random.randint(0, (self.height - self.scale) // self.scale) * self.scale
            if [x, y] not in self.snake_pos:
                return [x, y]
    
    def _get_state(self):
        head = self.snake_pos[0]
        state = np.zeros(14)
        
        # Danger detection
        state[0] = self._is_collision([head[0] - self.scale, head[1]])
        state[1] = self._is_collision([head[0] + self.scale, head[1]])
        state[2] = self._is_collision([head[0], head[1] - self.scale])
        state[3] = self._is_collision([head[0], head[1] + self.scale])
        
        # Direction
        state[4] = self.snake_direction[0] == -self.scale
        state[5] = self.snake_direction[0] == self.scale
        state[6] = self.snake_direction[1] == -self.scale
        state[7] = self.snake_direction[1] == self.scale
        
        # Food relative position
        state[8] = (self.food_pos[0] < head[0])
        state[9] = (self.food_pos[0] > head[0])
        state[10] = (self.food_pos[1] < head[1])
        state[11] = (self.food_pos[1] > head[1])
        
        # Normalized distance
        state[12] = abs(self.food_pos[0] - head[0]) / self.width
        state[13] = abs(self.food_pos[1] - head[1]) / self.height
        
        return state
    
    def _is_collision(self, pos):
        return (pos[0] >= self.width or pos[0] < 0 or
                pos[1] >= self.height or pos[1] < 0 or
                pos in self.snake_pos[1:])
    
    def step(self, action):
        prev_distance = math.sqrt(
            (self.snake_pos[0][0] - self.food_pos[0]) ** 2 +
            (self.snake_pos[0][1] - self.food_pos[1]) ** 2
        )
        
        if action == 0:
            self.snake_direction = [-self.scale, 0]
        elif action == 1:
            self.snake_direction = [self.scale, 0]
        elif action == 2:
            self.snake_direction = [0, -self.scale]
        elif action == 3:
            self.snake_direction = [0, self.scale]
        
        head = [
            self.snake_pos[0][0] + self.snake_direction[0],
            self.snake_pos[0][1] + self.snake_direction[1]
        ]
        
        new_distance = math.sqrt(
            (head[0] - self.food_pos[0]) ** 2 +
            (head[1] - self.food_pos[1]) ** 2
        )
        
        reward = 0
        self.game_over = self._is_collision(head)
        
        if self.game_over:
            reward = -10
            return self._get_state(), reward, True
        
        self.snake_pos.insert(0, head)
        
        if head == self.food_pos:
            reward = 20
            self.score += self.SCORE_VALUE
            self.food_pos = self._generate_food()
        else:
            self.snake_pos.pop()
            if new_distance < prev_distance:
                reward = 0.1
            else:
                reward = -0.1
            reward += 0.01
        
        return self._get_state(), reward, False
    
    def close(self):
        pass


def train(episodes=500, save_interval=100):
    logging.info("=" * 50)
    logging.info("Starting HEADLESS DQN Snake Training")
    logging.info(f"Episodes: {episodes}")
    logging.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    logging.info("=" * 50)
    
    env = HeadlessSnakeGame(width=640, height=480, scale=20)
    state_size = 14  # Fixed: matches actual state size
    action_size = 4
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    
    # Override epsilon decay to be per-episode, not per-step
    agent.epsilon = 1.0
    agent.epsilon_min = 0.01
    agent.epsilon_decay = 0.995  # Will apply once per episode instead
    
    # Stats tracking
    scores = []
    best_score = 0
    recent_scores = deque(maxlen=50)
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 500  # Tighter limit to keep episodes fast
        losses = []
        
        while steps < max_steps:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            
            # Train every 4 steps (faster than every step)
            if steps % 4 == 0 and len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                if loss is not None:
                    losses.append(loss)
            
            if done:
                agent.update_target_model()
                break
        
        # Decay epsilon once per episode
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
        scores.append(env.score)
        recent_scores.append(env.score)
        
        if env.score > best_score:
            best_score = env.score
        
        avg_recent = np.mean(recent_scores) if recent_scores else 0
        avg_loss = np.mean(losses) if losses else 0
        
        if (e + 1) % 10 == 0:
            logging.info(
                f"Ep {e+1:4d}/{episodes} | "
                f"Score: {env.score:3d} | "
                f"Best: {best_score:3d} | "
                f"Avg(50): {avg_recent:6.2f} | "
                f"ε: {agent.epsilon:.4f} | "
                f"Steps: {steps:4d} | "
                f"Loss: {avg_loss:.4f}"
            )
        
        if (e + 1) % save_interval == 0:
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"model_checkpoint_{e+1}.pth")
            agent.save(model_path)
            logging.info(f"💾 Saved checkpoint: {model_path}")
    
    # Save final model
    os.makedirs("models", exist_ok=True)
    agent.save("models/model_final.pth")
    
    # Save training stats
    stats = {
        "episodes": episodes,
        "best_score": best_score,
        "final_avg_50": float(np.mean(list(recent_scores)[-50:])),
        "final_epsilon": agent.epsilon,
        "scores": scores
    }
    with open("training_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    logging.info("=" * 50)
    logging.info(f"Training complete!")
    logging.info(f"Best score: {best_score}")
    logging.info(f"Final avg (50 ep): {np.mean(list(recent_scores)[-50:]):.2f}")
    logging.info("=" * 50)
    
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes')
    parser.add_argument('--save-interval', type=int, default=100, help='Save model every N episodes')
    args = parser.parse_args()
    train(episodes=args.episodes, save_interval=args.save_interval)
