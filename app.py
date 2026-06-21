from flask import Flask, request, jsonify, send_from_directory
import os
import glob
from game.state import State
from agents import DQNAgent, PPOAgent, AlphaZeroAgent

app = Flask(__name__, static_folder='static', static_url_path='')

# Set strict CSP headers as required by security guidelines
@app.after_request
def add_security_headers(response):
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self';"
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    return response

models_cache = {}

def get_agent(model_name):
    if model_name not in models_cache:
        path = os.path.join('./trained', model_name)
        if not os.path.exists(path):
            raise ValueError(f"Model {model_name} not found")
            
        if model_name.startswith('ppo_'):
            agent = PPOAgent(15)
        elif model_name.startswith('az_'):
            agent = AlphaZeroAgent(15)
        else:
            agent = DQNAgent(15)
            
        agent.load(path)
        if hasattr(agent, 'epsilon'):
            agent.epsilon = 0.0
        models_cache[model_name] = agent
    return models_cache[model_name]

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/models', methods=['GET'])
def get_models():
    # Find all .h5 files in ./trained
    if not os.path.exists('./trained'):
        return jsonify([])
    files = glob.glob('./trained/*.h5')
    models = [os.path.basename(f) for f in files]
    
    # Sort them nicely (if they have numbers, try to sort numerically)
    def extract_ep(name):
        import re
        m = re.search(r'ep(\d+)', name)
        return int(m.group(1)) if m else -1
        
    models.sort(key=extract_ep)
    return jsonify(models)

@app.route('/play', methods=['POST'])
def play():
    data = request.get_json()
    if not data or 'board' not in data or 'model' not in data or 'turn' not in data:
        return jsonify({"error": "Invalid payload"}), 400

    board = data['board']
    model_name = data['model']
    turn = data['turn']

    # Input validation (must be 15x15)
    if not isinstance(board, list) or len(board) != 15 or any(not isinstance(row, list) or len(row) != 15 for row in board):
        return jsonify({"error": "Invalid board size"}), 400
        
    # Validate integers
    for r in range(15):
        for c in range(15):
            if board[r][c] not in (0, 1, 2):
                return jsonify({"error": "Invalid cell value"}), 400

    try:
        agent = get_agent(model_name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

    state = State(15, board, turn)
    
    # We don't want agent to explore randomly when playing against human
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    _, best_move = agent.find_best_move(state)
    agent.epsilon = original_epsilon
    
    r = best_move // 15
    c = best_move % 15

    return jsonify({"move": {"row": r, "col": c}})

if __name__ == '__main__':
    # Listen on localhost to comply with security guidelines
    app.run(host='127.0.0.1', port=5000, debug=True)
