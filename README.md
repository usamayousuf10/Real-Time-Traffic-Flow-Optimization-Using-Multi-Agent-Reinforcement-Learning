# CityFlow Traffic Simulation Project

This project uses CityFlow to simulate traffic scenarios.

## Installation

To run this project, you need to have Python and Node.js installed.

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   ```

2. **Install Python dependencies:**
   This project requires `cityflow`, `torch`, and `numpy`. As we discovered, `cityflow` is not available on PyPI. Please follow the official installation instructions for your operating system. For Windows, it is recommended to use WSL or Docker.
   ```bash
   pip install torch numpy
   ```

3. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

## Usage

You can use the following scripts to run the different parts of the project:

- **`npm start` or `npm run dev`:**
  Runs the main CityFlow simulation setup.
  ```bash
  npm start
  ```

- **`npm run analyze`:**
  Runs the analysis and visualization script.
  ```bash
  npm run analyze
  ```

- **`npm run presslight`:**
  Runs the PressLight script.
  ```bash
  npm run presslight
  ```
