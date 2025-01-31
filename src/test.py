from game import SnakeGame
from dqn_agent import DQNAgent
import time
import logging
import numpy as np
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test(model_path, num_episodes=5, render=True):
    env = SnakeGame()
    state_size = 11
    action_size = 4
    agent = DQNAgent(state_size, action_size)
    
    logging.info(f"Loading model from {model_path}")
    agent.load(model_path)
    agent.epsilon = 0
    
    scores = []
    steps_list = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        logging.info(f"Starting episode {episode + 1}/{num_episodes}")
        
        while True:
            if render:
                env.render()
                
            action = agent.act(state)
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                scores.append(env.score)
                steps_list.append(steps)
                logging.info(f"Episode {episode + 1} - Score: {env.score} - Steps: {steps}")
                logging.info(f"Total Reward: {total_reward:.2f}")
                time.sleep(1)
                break
    
    # 显示统计信息
    avg_score = np.mean(scores)
    avg_steps = np.mean(steps_list)
    logging.info(f"\nTesting Results:")
    logging.info(f"Average Score: {avg_score:.2f}")
    logging.info(f"Average Steps: {avg_steps:.2f}")
    logging.info(f"Max Score: {max(scores)}")
    logging.info(f"Min Score: {min(scores)}")
                
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="model_checkpoint_1000.pth", 
                      help='Path to the model file')
    parser.add_argument('--episodes', type=int, default=5, 
                      help='Number of episodes to test')
    parser.add_argument('--no-render', action='store_true', 
                      help='Disable game rendering')
    
    args = parser.parse_args()
    test(args.model, args.episodes, not args.no_render)
