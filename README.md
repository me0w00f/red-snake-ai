<!-- Demo Video -->
<video width="100%" controls>
    <source src="video/demo.mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>

# 🐍 Red Snake AI

A DQN-based snake game with customizable skins. Train an AI to master the classic snake game or play it yourself!

## Features

- Deep Q-Learning AI agent
- Multiple snake skins (Classic Red/Gold/Neon)
- Human playable mode
- Real-time training visualization
- Performance tracking via wandb

## Quick Start


1. Clone the repository:
    ```sh
    git clone https://github.com/me0w00f/red-snake-ai.git
    ```
2. Navigate to the project directory:
    ```sh
    cd red-snake-ai
    ```
3. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Game Details

### Controls
- Use **Arrow Keys** to change direction
- Press **P** to pause the game
- Press **ESC** to quit

### Gameplay
- The snake starts with a length of 3 segments
- Each food item eaten adds 1 segment and 10 points
- The snake speed increases every 50 points
- Game ends if the snake hits the walls or itself

### Display
- Score is shown in the top-right corner
- High score is saved between sessions
- Game grid is 20x20 cells
- Food appears randomly on the grid

## AI training
This project serves as an excellent dataset for AI training purposes. The game's deterministic nature and clear rule set make it ideal for:

- Reinforcement Learning algorithms
- Deep Q-Learning implementations
- Path-finding optimization
- Pattern recognition training

The game state can be easily converted into numerical data, allowing AI models to:
- Learn optimal movement patterns
- Predict collision scenarios
- Develop food-seeking strategies
- Analyze performance metrics

## Contributing

Contributions are welcome! If you have any ideas or improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.