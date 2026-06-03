import { api } from './api.js';
import { BoardView } from './board.js';
import { KifuViewer } from './kifu_viewer.js';

const BLACK = 1, WHITE = 2;
const colorToInt = (s) => s === 'black' ? BLACK : s === 'white' ? WHITE : null;

const els = {
  board:        document.getElementById('board'),
  turnDot:      document.querySelector('#status-turn .turn-dot'),
  turnText:     document.querySelector('#status-turn .turn-text'),
  capBlack:     document.getElementById('cap-black'),
  capWhite:     document.getElementById('cap-white'),
  metaSize:     document.getElementById('meta-size'),
  metaKomi:     document.getElementById('meta-komi'),
  btnNew:       document.getElementById('btn-new'),
  btnPass:      document.getElementById('btn-pass'),
  btnUndo:      document.getElementById('btn-undo'),
  btnResign:    document.getElementById('btn-resign'),
  aiLastMove:   document.getElementById('ai-lastmove'),
  aiSims:       document.getElementById('ai-sims'),
  aiWinrate:    document.getElementById('ai-winrate'),
  aiMode:       document.getElementById('ai-mode'),
  aiPulse:      document.getElementById('ai-pulse'),
  kifu:         document.getElementById('kifu'),
  toast:        document.getElementById('toast'),
  sizeSelect:   document.getElementById('size-select'),
  btnEstimate:  document.getElementById('btn-estimate'),
  btnKifus:     document.getElementById('btn-kifus'),
  panelScore:   document.getElementById('panel-score'),
  scoreFillBlack: document.getElementById('score-fill-black'),
  scoreFillWhite: document.getElementById('score-fill-white'),
  scoreBlack:   document.getElementById('score-black'),
  scoreWhite:   document.getElementById('score-white'),
  scoreVerdict: document.getElementById('score-verdict'),
  scoreNote:    document.getElementById('score-note'),
};

/* ----- state ----- */
let state = null;
let selectedSize = 9;
let selectedAi = 'mcts';
let territoryShown = false;

/* ----- board view ----- */
const boardView = new BoardView(els.board, {
  onClick: handleBoardClick,
  onHover: handleBoardHover,
  onLeave: () => {},
});

/* ----- helpers ----- */
function toast(msg, ms = 2000) {
  els.toast.textContent = msg;
  els.toast.classList.add('show');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => els.toast.classList.remove('show'), ms);
}

function isHumanTurn() {
  if (!state) return false;
  if (state.is_over) return false;
  if (state.ai_color === null) return true;  // human vs human
  return state.turn !== state.ai_color;
}

/* ----- rendering ----- */
function render() {
  if (!state) return;

  boardView.render(state);

  // status bar
  const turnInt = colorToInt(state.turn);
  els.turnDot.classList.toggle('turn-black', turnInt === BLACK);
  els.turnDot.classList.toggle('turn-white', turnInt === WHITE);

  if (state.is_over) {
    if (state.resigned_by) {
      const winner = state.resigned_by === 'black' ? 'White' : 'Black';
      els.turnText.textContent = `${winner} wins by resignation`;
    } else {
      els.turnText.textContent = 'Game over — both passed';
    }
  } else {
    const who = state.turn === state.human_color ? 'you' :
                state.turn === state.ai_color    ? 'AI'  : '';
    const colorName = state.turn[0].toUpperCase() + state.turn.slice(1);
    els.turnText.textContent = who ? `${colorName} to play (${who})` : `${colorName} to play`;
  }

  // captures (Note: 'captures.black' is stones black has captured, i.e. white stones removed)
  els.capBlack.textContent = state.captures.black;
  els.capWhite.textContent = state.captures.white;

  // meta
  els.metaSize.textContent = `${state.size} × ${state.size}`;
  els.metaKomi.textContent = state.komi.toFixed(1);

  // kifu (move history)
  renderKifu();

  // buttons
  els.btnPass.disabled   = !isHumanTurn();
  els.btnResign.disabled = state.is_over;
  els.btnUndo.disabled   = state.history.length === 0;
}

function renderKifu() {
  els.kifu.replaceChildren();
  if (!state.history.length) {
    const empty = document.createElement('li');
    empty.className = 'kifu-empty';
    empty.textContent = 'no moves yet';
    els.kifu.appendChild(empty);
    return;
  }
  state.history.forEach((m, i) => {
    const li = document.createElement('li');
    li.className = 'kifu-item';

    const num = document.createElement('span');
    num.className = 'kifu-num';
    num.textContent = String(i + 1);
    li.appendChild(num);

    const stone = document.createElement('span');
    stone.className = `kifu-stone ${m.color}`;
    li.appendChild(stone);

    const label = document.createElement('span');
    label.className = 'kifu-label';
    if (m.kind === 'play') label.textContent = m.label;
    else if (m.kind === 'pass') label.textContent = 'pass';
    else label.textContent = 'resign';
    li.appendChild(label);

    const meta = document.createElement('span');
    meta.className = 'kifu-meta';
    if (m.captured > 0) meta.textContent = `+${m.captured}`;
    li.appendChild(meta);

    els.kifu.appendChild(li);
  });
  // auto-scroll to latest
  els.kifu.scrollTop = els.kifu.scrollHeight;
}

function renderAIInfo(info) {
  if (!info || !info.kind) {
    els.aiLastMove.textContent = '—';
    els.aiSims.textContent = '—';
    els.aiWinrate.textContent = '—';
    return;
  }
  const lastPlay = [...state.history].reverse().find(m => m.kind === 'play' && m.color === state.ai_color);
  if (info.kind === 'play' && lastPlay) {
    els.aiLastMove.textContent = lastPlay.label;
  } else if (info.kind === 'pass') {
    els.aiLastMove.textContent = 'pass';
  } else if (info.kind === 'resign') {
    els.aiLastMove.textContent = 'resign';
  }
  els.aiSims.textContent    = info.simulations > 0 ? info.simulations : '—';
  els.aiWinrate.textContent = info.winrate != null ? `${(info.winrate * 100).toFixed(1)}%` : '—';
}

/* ----- event handlers ----- */
async function handleBoardClick(r, c) {
  if (!state || !isHumanTurn()) return;
  try {
    state = await api.playMove(state.game_id, r, c);
    render();
    // Trigger AI response if it's now AI's turn
    if (state.ai_color !== null && state.turn === state.ai_color && !state.is_over) {
      await triggerAI();
    }
  } catch (e) {
    toast(e.message || 'illegal move');
  }
}

function handleBoardHover(r, c) {
  if (!state || !isHumanTurn()) return;
  if (state.board[r][c] !== 0) return;
  const colorInt = colorToInt(state.turn);
  boardView.drawHover(r, c, colorInt);
}

async function triggerAI() {
  els.aiPulse.classList.add('active');
  try {
    // MCTS genuinely takes a second or two — the pulse shows real thinking.
    state = await api.aiMove(state.game_id);
    render();
    renderAIInfo(state.ai_info);
    if (state.is_over && !state.resigned_by && !territoryShown) {
      await toggleEstimate();
    }
  } catch (e) {
    toast(`AI: ${e.message}`);
  } finally {
    els.aiPulse.classList.remove('active');
  }
}

async function newGame() {
  try {
    hideScore();
    state = await api.newGame({ size: selectedSize, komi: null, aiColor: 'white', aiType: selectedAi });
    render();
    renderAIInfo(null);
    // Reflect current opponent in the AI info panel.
    const modeEl = document.getElementById('ai-mode');
    if (modeEl) modeEl.textContent = selectedAi === 'neural' ? 'Trained' : 'MCTS';
  } catch (e) {
    toast(e.message);
  }
}

async function doPass() {
  if (!isHumanTurn()) return;
  try {
    state = await api.playPass(state.game_id);
    render();
    if (!state.is_over && state.ai_color !== null && state.turn === state.ai_color) {
      await triggerAI();
    }
    // If two passes ended the game, show the final count automatically.
    if (state.is_over && !state.resigned_by && !territoryShown) {
      await toggleEstimate();
    }
  } catch (e) { toast(e.message); }
}

async function doUndo() {
  try {
    hideScore();
    state = await api.undo(state.game_id);
    render();
    renderAIInfo(null);
  } catch (e) { toast(e.message); }
}

async function doResign() {
  if (state.is_over) return;
  try {
    state = await api.resign(state.game_id);
    render();
    toast('You resigned');
  } catch (e) { toast(e.message); }
}

/* ----- scoring / position estimate ----- */
function hideScore() {
  territoryShown = false;
  els.panelScore.style.display = 'none';
  boardView.clearTerritory();
  els.btnEstimate.textContent = 'Estimate score';
}

async function toggleEstimate() {
  if (!state) return;
  // Toggle off if already showing.
  if (territoryShown) {
    hideScore();
    return;
  }
  try {
    const score = await api.score(state.game_id);
    boardView.showTerritory(score.ownership, state.board);
    territoryShown = true;
    els.btnEstimate.textContent = 'Hide estimate';
    renderScorePanel(score);
  } catch (e) {
    toast(e.message);
  }
}

function renderScorePanel(score) {
  els.panelScore.style.display = 'block';

  const b = score.black, w = score.white;
  const total = b + w || 1;
  els.scoreFillBlack.style.width = `${(b / total) * 100}%`;
  els.scoreFillWhite.style.width = `${(w / total) * 100}%`;

  els.scoreBlack.textContent = b.toFixed(1);
  els.scoreWhite.textContent = w.toFixed(1);

  const winnerName = score.winner === 'black' ? 'Black' : 'White';
  const verb = score.is_estimate ? 'leads' : 'wins';
  els.scoreVerdict.textContent = `${winnerName} ${verb} by ${score.margin.toFixed(1)}`;

  // Honest note about what this number means.
  const parts = [];
  parts.push(`B: ${score.black_stones} stones + ${score.black_territory} territory`);
  parts.push(`W: ${score.white_stones} + ${score.white_territory} + ${score.komi} komi`);
  if (score.dame > 0) {
    parts.push(`${score.dame} neutral/unsettled points — boundaries not yet closed.`);
  }
  if (score.is_estimate) {
    parts.push('Estimate only: no dead-stone detection yet.');
  }
  els.scoreNote.textContent = parts.join('  ·  ');
}

function selectSize(size) {
  if (size === selectedSize && state && state.size === size) return;
  selectedSize = size;
  els.sizeSelect.querySelectorAll('.size-opt').forEach(btn => {
    btn.classList.toggle('is-active', Number(btn.dataset.size) === size);
  });
  // Switching board size starts a fresh game on that board immediately.
  newGame();
}

/* ----- wire up ----- */
els.btnNew.addEventListener('click', newGame);
els.btnPass.addEventListener('click', doPass);
els.btnUndo.addEventListener('click', doUndo);
els.btnResign.addEventListener('click', doResign);
els.btnEstimate.addEventListener('click', toggleEstimate);

// Kifu library — lazy-init the viewer on first open so it doesn't add DOM
// nodes (and a global key listener) until the user actually wants it.
let kifuViewer = null;
els.btnKifus.addEventListener('click', () => {
  if (!kifuViewer) kifuViewer = new KifuViewer();
  kifuViewer.open();
});

els.sizeSelect.addEventListener('click', (e) => {
  const btn = e.target.closest('.size-opt');
  if (!btn) return;
  selectSize(Number(btn.dataset.size));
});

// AI opponent selector — switching starts a fresh game using the new opponent.
const aiSelect = document.getElementById('ai-select');
if (aiSelect) {
  aiSelect.addEventListener('click', (e) => {
    const btn = e.target.closest('.ai-opt');
    if (!btn) return;
    const newAi = btn.dataset.ai;
    if (newAi === selectedAi) return;
    selectedAi = newAi;
    aiSelect.querySelectorAll('.ai-opt').forEach(b => {
      b.classList.toggle('is-active', b.dataset.ai === newAi);
    });
    newGame();
  });
}

// start
newGame();
