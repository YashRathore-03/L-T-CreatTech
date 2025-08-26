# Prestressed Concrete Design Assistant (Flask Version)

## Overview
This is a Flask-based web application for optimizing beam structures. Users can select a beam type, enter parameters, and visualize optimization results.

## Features
- Select beam type from a dropdown menu
- Input parameters for optimization
- Display computed results dynamically
- Generate and display an optimization curve using Matplotlib
- Responsive UI using Bootstrap

## Installation
### Prerequisites
Ensure you have Python 3 installed.

### Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/YashRathore-03/L-T-CreatTech.git
   cd tendon_profiling_project
   ```

2. Create a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Running the Application
Start the Flask server with:
```sh flask app.py
```
The application will be available at `http://127.0.0.1:5000/`.

## Project Structure
```
beamoptimizer-visualizer/
├── static/
│   ├── css/
│   │   ├── styles.css
├── templates/
│   ├── index.html
│   ├── results.html
├── models/
├── app.py
├── requirements.txt
├── README.md
```

## License
This project is licensed under the MIT License.
