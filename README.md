# Medical Transcribe

Welcome to **Medical Transcribe**, a comprehensive solution for transcribing medical audio recordings using a Python backend and an Angular frontend. This repository provides instructions to set up the project on **Linux**, **macOS**, and **Windows** operating systems.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Python Backend](#python-backend)
    - [Linux](#linux)
    - [macOS](#macos)
    - [Windows](#windows)
  - [Angular Frontend](#angular-frontend)
    - [Linux](#linux-1)
    - [macOS](#macos-1)
    - [Windows](#windows-1)
- [Usage](#usage)
- [Contributing](#contributing)


## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.6+**
- **Node.js and npm**
- **Git**

## Installation

### Python Backend

#### Linux

1. **Install Python 3 and pip:**

    ```bash
    sudo apt update
    sudo apt install python3 python3-pip python3-venv libsndfile1 ffmpeg build-essential -y
    ```

2. **Set `pip` alias to `pip3`:**

    ```bash
    echo "alias pip=pip3" >> ~/.bashrc
    source ~/.bashrc
    ```

3. **Clone the repository and navigate to the backend directory:**

    ```bash
    git clone https://github.com/yourusername/Medical-transcribe.git
    cd Medical-transcribe
    ```

4. **Create and activate a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

5. **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

6. **Run the backend server:**

    ```bash
    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4 --reload
    ```

#### macOS

1. **Install Homebrew** (if not already installed):

    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

2. **Install Python 3 and pip:**

    ```bash
    brew update
    brew install python3
    ```

3. **Set `pip` alias to `pip3`:**

    ```bash
    echo "alias pip=pip3" >> ~/.bash_profile
    source ~/.bash_profile
    ```

4. **Clone the repository and navigate to the backend directory:**

    ```bash
    git clone https://github.com/yourusername/Medical-transcribe.git
    cd Medical-transcribe
    ```

5. **Create and activate a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

6. **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

7. **Install additional dependencies:**

    ```bash
    brew install libsndfile ffmpeg
    ```

8. **Run the backend server:**

    ```bash
    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4 --reload
    ```

#### Windows

1. **Install Python 3:**

    - Download and install Python from the [official website](https://www.python.org/downloads/windows/).
    - During installation, ensure you check the box to **Add Python to PATH**.

2. **Set `pip` alias to `pip3`:**

    - Open Command Prompt and run:

      ```cmd
      doskey pip=pip3
      ```

    - To make this permanent, add the alias to your PowerShell profile or use a tool like `alias` in Git Bash.

3. **Clone the repository and navigate to the backend directory:**

    ```cmd
    git clone https://github.com/yourusername/Medical-transcribe.git
    cd Medical-transcribe
    ```

4. **Create and activate a virtual environment:**

    ```cmd
    python -m venv venv
    venv\Scripts\activate
    ```

5. **Install the required Python packages:**

    ```cmd
    pip install -r requirements.txt
    ```

6. **Install additional dependencies:**

    - **libsndfile:** Download and install from [libsndfile](http://www.mega-nerd.com/libsndfile/).
    - **FFmpeg:** Download and install from [FFmpeg](https://ffmpeg.org/download.html), and add it to your PATH.

7. **Run the backend server:**

    ```cmd
    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4 --reload
    ```

### Angular Frontend

#### Linux

1. **Navigate to the frontend directory:**

    ```bash
    cd ~/audio-transcriber
    ```

2. **Install Angular CLI globally:**

    ```bash
    sudo npm install -g @angular/cli
    ```

3. **Install frontend dependencies:**

    ```bash
    npm install
    ```

4. **Run the Angular development server:**

    ```bash
    ng serve --host 0.0.0.0 --port 4200
    ```

#### macOS

1. **Navigate to the frontend directory:**

    ```bash
    cd ~/audio-transcriber
    ```

2. **Install Angular CLI globally:**

    ```bash
    sudo npm install -g @angular/cli
    ```

3. **Install frontend dependencies:**

    ```bash
    npm install
    ```

4. **Run the Angular development server:**

    ```bash
    ng serve --host 0.0.0.0 --port 4200
    ```

#### Windows

1. **Navigate to the frontend directory:**

    ```cmd
    cd \path\to\audio-transcriber
    ```

2. **Install Angular CLI globally:**

    ```cmd
    npm install -g @angular/cli
    ```

3. **Install frontend dependencies:**

    ```cmd
    npm install
    ```

4. **Run the Angular development server:**

    ```cmd
    ng serve --host 0.0.0.0 --port 4200
    ```

## Usage

1. **Start the Python backend server** by following the [Python Backend](#python-backend) instructions.
2. **Start the Angular frontend server** by following the [Angular Frontend](#angular-frontend) instructions.
3. Open your browser and navigate to `http://localhost:4200` to access the application.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

