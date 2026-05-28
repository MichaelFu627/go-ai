/**
 * Thin wrapper around the FastAPI backend.
 * All endpoints return the full game state object (or throw on error).
 */

const API = '/api';

async function request(path, opts = {}) {
  const res = await fetch(API + path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!res.ok) {
    let detail = `${res.status}`;
    try { const j = await res.json(); detail = j.detail || detail; } catch {}
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  newGame: ({ size = 9, komi = 7.5, aiColor = 'white', seed = null,
              aiType = 'mcts', aiStrength = null } = {}) =>
    request('/game/new', {
      method: 'POST',
      body: JSON.stringify({ size, komi, ai_color: aiColor, seed,
                             ai_type: aiType, ai_strength: aiStrength }),
    }),

  getState:  (id) => request(`/game/${id}/state`),
  playMove:  (id, row, col) => request(`/game/${id}/move`, { method: 'POST', body: JSON.stringify({ row, col }) }),
  playPass:  (id) => request(`/game/${id}/pass`,   { method: 'POST' }),
  resign:    (id) => request(`/game/${id}/resign`, { method: 'POST' }),
  undo:      (id) => request(`/game/${id}/undo`,   { method: 'POST' }),
  aiMove:    (id) => request(`/game/${id}/ai-move`, { method: 'POST' }),
  score:     (id) => request(`/game/${id}/score`),
};
