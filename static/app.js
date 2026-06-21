const SIZE = 15;
let board = [];
let currentTurn = 1; // 1 = Black (Player), 2 = White (AI)
let gameOver = false;

const boardElement = document.getElementById('board');
const statusText = document.getElementById('statusText');
const resetBtn = document.getElementById('resetBtn');
const modelSelect = document.getElementById('modelSelect');

// Initialize board
function initBoard() {
    boardElement.replaceChildren();
    board = Array(SIZE).fill(null).map(() => Array(SIZE).fill(0));
    currentTurn = 1;
    gameOver = false;
    statusText.textContent = "Your turn (Black)";

    for (let r = 0; r < SIZE; r++) {
        for (let c = 0; c < SIZE; c++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.dataset.r = r;
            cell.dataset.c = c;
            cell.addEventListener('click', onCellClick);
            boardElement.appendChild(cell);
        }
    }
}

// Fetch available models
async function fetchModels() {
    try {
        const response = await fetch('/models');
        const models = await response.json();
        
        modelSelect.replaceChildren();
        if (models.length === 0) {
            const option = document.createElement('option');
            option.value = "";
            option.textContent = "No models available";
            modelSelect.appendChild(option);
            return;
        }

        const optgroups = {
            'ppo': document.createElement('optgroup'),
            'az': document.createElement('optgroup'),
            'dqn': document.createElement('optgroup'),
            'other': document.createElement('optgroup')
        };
        optgroups['ppo'].label = "PPO Models";
        optgroups['az'].label = "AlphaZero Models";
        optgroups['dqn'].label = "DQN Models";
        optgroups['other'].label = "Other Models";

        for (const model of models) {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            
            if (model.includes('final') || model === models[models.length - 1]) {
                option.selected = true;
            }
            
            if (model.startsWith('ppo_')) optgroups['ppo'].appendChild(option);
            else if (model.startsWith('az_')) optgroups['az'].appendChild(option);
            else if (model.startsWith('agent_') || model.startsWith('dqn_')) optgroups['dqn'].appendChild(option);
            else optgroups['other'].appendChild(option);
        }

        for (const group in optgroups) {
            if (optgroups[group].children.length > 0) {
                modelSelect.appendChild(optgroups[group]);
            }
        }
    } catch (err) {
        console.error("Failed to load models:", err);
    }
}

function checkWin(r, c, turn) {
    const directions = [[0, 1], [1, 0], [1, 1], [1, -1]];
    for (const [dr, dc] of directions) {
        let count = 1;
        
        // Forward
        for (let step = 1; step < 5; step++) {
            const nr = r + step * dr;
            const nc = c + step * dc;
            if (nr >= 0 && nr < SIZE && nc >= 0 && nc < SIZE && board[nr][nc] === turn) count++;
            else break;
        }
        
        // Backward
        for (let step = 1; step < 5; step++) {
            const nr = r - step * dr;
            const nc = c - step * dc;
            if (nr >= 0 && nr < SIZE && nc >= 0 && nc < SIZE && board[nr][nc] === turn) count++;
            else break;
        }

        if (count >= 5) return true;
    }
    return false;
}

async function onCellClick(e) {
    if (gameOver || currentTurn !== 1) return;

    const r = parseInt(e.target.dataset.r);
    const c = parseInt(e.target.dataset.c);

    if (board[r][c] !== 0) return;

    // Player move
    placePiece(r, c, 1);
    
    if (checkWin(r, c, 1)) {
        endGame("You win!");
        return;
    }

    currentTurn = 2;
    statusText.textContent = "AI is thinking...";
    
    await makeAIMove();
}

function placePiece(r, c, turn) {
    board[r][c] = turn;
    const cellIndex = r * SIZE + c;
    const cell = boardElement.children[cellIndex];
    
    const piece = document.createElement('div');
    piece.className = 'piece ' + (turn === 1 ? 'black' : 'white');
    cell.appendChild(piece);
}

async function makeAIMove() {
    const model = modelSelect.value;
    if (!model) {
        alert("Please select a model first");
        statusText.textContent = "Your turn (Black)";
        currentTurn = 1;
        return;
    }

    try {
        const response = await fetch('/play', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ board, model, turn: 2 })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || "Unknown error");
        }

        const data = await response.json();
        const r = data.move.row;
        const c = data.move.col;

        placePiece(r, c, 2);

        if (checkWin(r, c, 2)) {
            endGame("AI wins!");
            return;
        }

        currentTurn = 1;
        statusText.textContent = "Your turn (Black)";

    } catch (err) {
        console.error(err);
        statusText.textContent = "Error: " + err.message;
        currentTurn = 1;
    }
}

function endGame(message) {
    gameOver = true;
    statusText.textContent = message;
}

resetBtn.addEventListener('click', initBoard);

// Startup
initBoard();
fetchModels();
