# Custom PPO: Proximal Policy Optimization from Scratch 

This repository offers a custom implementation of the Proximal Policy Optimization (PPO) algorithm in Python, emphasizing clarity and educational value. It includes both a foundational version and an optimized variant, facilitating a deeper understanding of PPO's mechanics and potential enhancements.

## Features
- **Core PPO Implementation**: A straightforward PPO algorithm built from the ground up.
- **Optimized PPO Variant**: An enhanced version incorporating performance improvements.
- **Training Visualizations**: Graphs illustrating rewards per episode for both implementations.
- **Modular Code Structure**: Separation of concerns across different modules for clarity and reusability.

## Repository Structure
- `main.py`: Entry point for training the PPO agent.
- `ppo.py`: Contains the standard implementation of the PPO algorithm.
- `ppo_with_optimizations.py`: Houses the optimized PPO variant with performance enhancements.
- `network.py`: Defines the neural network architecture used by the PPO agent.
- `requirements.txt`: Lists all Python dependencies required to run the project.
- `rewards_per_episode.png`: Graph depicting training rewards for the standard PPO.
- `rewards_per_episode_with_optimizations.png`: Graph showing training rewards for the optimized PPO.

## Getting Started
### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/pritmanvar/customPPO.git
   cd customPPO
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the PPO Agent
- **Standard PPO**:
```bash
  python main.py
  ```

- **Optimized PPO**:
  To utilize the optimized version, modify `main.py` to import from `ppo_with_optimizations.py` instead of `ppo.py`.

## Visualizations
Training progress is visualized using reward graphs:
- `rewards_per_episode.png`: Rewards over episodes for the standard PPO.
- `rewards_per_episode_with_optimizations.png`: Rewards over episodes for the optimized PPO.

## Contributing
Contributions are welcome! Feel free to fork the repository, submit issues, or propose pull requests to enhance the project.

## License
This project is licensed under the MIT License.
