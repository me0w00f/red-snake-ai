from game import SnakeGame
from dqn_agent import DQNAgent
import wandb
import numpy as np
import logging
import argparse

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def train(render=False):
    wandb.init(project="snake-dqn", name="training_run")
    logging.info("Starting training process...")
    
    env = SnakeGame()
    state_size = 11
    action_size = 4
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    episodes = 1000

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        logging.info(f"Episode {e+1}/{episodes} - Epsilon: {agent.epsilon:.4f}")
        
        while True:
            if render:
                env.render()
            
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                wandb.log({
                    "loss": loss,
                    "epsilon": agent.epsilon,
                    "step": steps
                })

            if done:
                agent.update_target_model()
                logging.info(f"Episode: {e+1} - Score: {env.score} - Steps: {steps}")
                break
                
        wandb.log({
            "episode": e,
            "score": env.score,
            "total_reward": total_reward,
            "total_steps": steps
        })
        
        if (e + 1) % 100 == 0:
            model_path = f"model_checkpoint_{e+1}.pth"
            agent.save(model_path)
            logging.info(f"Saved model checkpoint: {model_path}")
            
    env.close()
    wandb.finish()
    logging.info("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render the game during training')
    args = parser.parse_args()
    train(render=args.render)
