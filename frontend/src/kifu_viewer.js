/**
 * Kifu library — a drawer that slides in from the right showing all saved
 * self-play games. Click a game to load it into a mini replay viewer with
 * step controls.
 *
 * Reuses BoardView for the actual board rendering (DRY: same renderer in
 * both the main game and the replay).
 */

import { api } from './api.js';
import { BoardView } from './board.js';

const BLACK = 1, WHITE = 2;

export class KifuViewer {
  constructor() {
    this.kifus = [];
    this.currentKifu = null;        // full loaded kifu
    this.moveIndex = 0;             // how many moves have been played in replay
    this.replayBoard = null;        // BoardView instance for the drawer
    this.playTimer = null;
    this.playSpeed = 500;           // ms per move when auto-playing

    this._buildDOM();
  }

  _buildDOM() {
    // Drawer scaffold appended to body
    this.root = document.createElement('div');
    this.root.className = 'kifu-drawer';
    this.root.innerHTML = `
      <div class="kifu-overlay"></div>
      <div class="kifu-panel">
        <div class="kifu-header">
          <div class="kifu-title">Kifu library <span class="panel-title-sub">棋譜庫</span></div>
          <button class="kifu-close" aria-label="Close">×</button>
        </div>
        <div class="kifu-body">
          <aside class="kifu-list-pane">
            <div class="kifu-list-header">
              <span class="kifu-count" id="kifu-count">—</span>
              <button class="kifu-refresh" id="kifu-refresh" title="Refresh">↻</button>
            </div>
            <ol class="kifu-list" id="kifu-list"></ol>
          </aside>
          <section class="kifu-replay-pane">
            <div class="kifu-empty-state" id="kifu-empty">
              Select a game from the list.
            </div>
            <div class="kifu-replay" id="kifu-replay" style="display:none;">
              <div class="kifu-meta" id="kifu-meta"></div>
              <div class="kifu-board-wrap">
                <svg class="kifu-mini-board" viewBox="0 0 600 600" xmlns="http://www.w3.org/2000/svg">
                  <g id="board-bg"></g>
                  <g id="board-grid"></g>
                  <g id="board-stars"></g>
                  <g id="board-labels"></g>
                  <g id="board-stones"></g>
                  <g id="board-territory"></g>
                  <g id="board-marks"></g>
                  <g id="board-hover"></g>
                </svg>
              </div>
              <div class="kifu-controls">
                <button class="kifu-ctrl" id="kifu-first" title="First (Home)">⏮</button>
                <button class="kifu-ctrl" id="kifu-prev"  title="Previous (←)">◀</button>
                <button class="kifu-ctrl kifu-play" id="kifu-play" title="Play / Pause (Space)">▶</button>
                <button class="kifu-ctrl" id="kifu-next"  title="Next (→)">▶</button>
                <button class="kifu-ctrl" id="kifu-last"  title="Last (End)">⏭</button>
                <input type="range" class="kifu-slider" id="kifu-slider" min="0" max="0" value="0">
                <span class="kifu-move-counter mono" id="kifu-counter">0 / 0</span>
              </div>
            </div>
          </section>
        </div>
      </div>
    `;
    document.body.appendChild(this.root);

    // Wire close
    this.root.querySelector('.kifu-overlay').addEventListener('click', () => this.close());
    this.root.querySelector('.kifu-close').addEventListener('click', () => this.close());
    this.root.querySelector('#kifu-refresh').addEventListener('click', () => this.loadList());

    // Replay controls
    const $ = (id) => this.root.querySelector('#' + id);
    $('kifu-first').addEventListener('click', () => this.goTo(0));
    $('kifu-prev').addEventListener('click',  () => this.goTo(this.moveIndex - 1));
    $('kifu-next').addEventListener('click',  () => this.goTo(this.moveIndex + 1));
    $('kifu-last').addEventListener('click',  () => this.goTo(this._totalMoves()));
    $('kifu-play').addEventListener('click',  () => this.togglePlay());
    $('kifu-slider').addEventListener('input', (e) => this.goTo(parseInt(e.target.value, 10)));

    // Keyboard while drawer is open
    this._keyHandler = (e) => {
      if (!this.isOpen()) return;
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === 'ArrowRight') { e.preventDefault(); this.goTo(this.moveIndex + 1); }
      else if (e.key === 'ArrowLeft')  { e.preventDefault(); this.goTo(this.moveIndex - 1); }
      else if (e.key === 'Home')       { e.preventDefault(); this.goTo(0); }
      else if (e.key === 'End')        { e.preventDefault(); this.goTo(this._totalMoves()); }
      else if (e.key === ' ')          { e.preventDefault(); this.togglePlay(); }
      else if (e.key === 'Escape')     { this.close(); }
    };
    document.addEventListener('keydown', this._keyHandler);

    // BoardView uses svg.querySelector which is SCOPED TO THIS SVG, so it's
    // safe to reuse the same ids inside two SVGs in the same page.
    const svg = this.root.querySelector('.kifu-mini-board');
    this.replayBoard = new BoardView(svg, { onClick: () => {}, onHover: () => {} });
  }

  // ---- public API ----
  async open() {
    this.root.classList.add('is-open');
    document.body.style.overflow = 'hidden';
    await this.loadList();
  }

  close() {
    this._stopPlay();
    this.root.classList.remove('is-open');
    document.body.style.overflow = '';
  }

  isOpen() {
    return this.root.classList.contains('is-open');
  }

  destroy() {
    document.removeEventListener('keydown', this._keyHandler);
    this.root.remove();
  }

  // ---- list ----
  async loadList() {
    try {
      const res = await api.listKifus();
      this.kifus = res.kifus || [];
      this._renderList();
    } catch (e) {
      this.root.querySelector('#kifu-list').innerHTML =
        `<li class="kifu-list-error">Failed to load: ${e.message}</li>`;
    }
  }

  _renderList() {
    const list = this.root.querySelector('#kifu-list');
    const count = this.root.querySelector('#kifu-count');
    count.textContent = `${this.kifus.length} game${this.kifus.length === 1 ? '' : 's'}`;
    if (this.kifus.length === 0) {
      list.innerHTML = `<li class="kifu-list-empty">No saved games yet. Run <code>python train.py</code> to generate some.</li>`;
      return;
    }
    list.innerHTML = '';
    this.kifus.forEach((k) => {
      const li = document.createElement('li');
      li.className = 'kifu-list-item';
      const meta = k.meta || {};
      const iter = meta.iteration != null ? `iter ${meta.iteration}` : '';
      const gameIdx = meta.game_idx != null ? `· g${meta.game_idx}` : '';
      const winner = meta.winner || '?';
      const moves = meta.move_count != null ? `${meta.move_count} moves` : '';
      const dotClass = winner === 'black' ? 'black' : winner === 'white' ? 'white' : '';
      li.innerHTML = `
        <div class="kifu-row-top">
          <span class="kifu-row-tag">${iter} ${gameIdx}</span>
          <span class="kifu-row-winner"><span class="kifu-dot ${dotClass}"></span>${winner} won</span>
        </div>
        <div class="kifu-row-bottom mono">${moves}</div>
      `;
      li.addEventListener('click', () => this._selectKifu(k.id, li));
      list.appendChild(li);
    });
  }

  // ---- replay ----
  async _selectKifu(kifuId, liEl) {
    this._stopPlay();
    // highlight
    this.root.querySelectorAll('.kifu-list-item.is-active').forEach(el => el.classList.remove('is-active'));
    liEl.classList.add('is-active');

    try {
      this.currentKifu = await api.getKifu(kifuId);
    } catch (e) {
      alert('Failed to load kifu: ' + e.message);
      return;
    }
    this.root.querySelector('#kifu-empty').style.display = 'none';
    this.root.querySelector('#kifu-replay').style.display = 'flex';

    // configure slider
    const total = this._totalMoves();
    const slider = this.root.querySelector('#kifu-slider');
    slider.max = total;
    slider.value = total;            // show finished position by default

    this._renderMeta();
    this.goTo(total);
  }

  _renderMeta() {
    const k = this.currentKifu;
    const meta = k.meta || {};
    const settings = k.settings || {};
    const html = `
      <span class="kifu-meta-item"><b>${settings.size || '?'}×${settings.size || '?'}</b></span>
      <span class="kifu-meta-item">komi ${settings.komi ?? '?'}</span>
      <span class="kifu-meta-item">${meta.move_count} moves</span>
      <span class="kifu-meta-item">winner: <span class="kifu-dot ${meta.winner || ''}"></span>${meta.winner || '?'}</span>
      ${meta.score ? `<span class="kifu-meta-item mono">B ${meta.score.black?.toFixed?.(1) ?? '?'} · W ${meta.score.white?.toFixed?.(1) ?? '?'}</span>` : ''}
      ${meta.iteration != null ? `<span class="kifu-meta-item">iter ${meta.iteration}</span>` : ''}
    `;
    this.root.querySelector('#kifu-meta').innerHTML = html;
  }

  _totalMoves() {
    return this.currentKifu ? this.currentKifu.moves.length : 0;
  }

  goTo(index) {
    if (!this.currentKifu) return;
    const total = this._totalMoves();
    index = Math.max(0, Math.min(total, index));
    this.moveIndex = index;

    // Reconstruct the position after `index` moves by replaying.
    const size = this.currentKifu.settings.size;
    const board = Array.from({ length: size }, () => Array(size).fill(0));
    const captures = { black: 0, white: 0 };
    const moves = this.currentKifu.moves;
    let lastPlay = null;
    for (let i = 0; i < index; i++) {
      const m = moves[i];
      if (m.kind !== 'play') continue;
      const color = m.color === 'black' ? BLACK : WHITE;
      const opp = color === BLACK ? WHITE : BLACK;
      board[m.row][m.col] = color;
      // Capture: remove opponent groups with 0 liberties adjacent to this stone.
      this._captureAfter(board, m.row, m.col, opp, size, captures, m.color);
      lastPlay = m;
    }

    // Build a synthetic "state" object the BoardView understands.
    const state = {
      size,
      board,
      history: lastPlay ? [{ kind: 'play', row: lastPlay.row, col: lastPlay.col }] : [],
    };
    this.replayBoard.render(state);

    // Update slider + counter
    this.root.querySelector('#kifu-slider').value = index;
    this.root.querySelector('#kifu-counter').textContent = `${index} / ${total}`;
  }

  _captureAfter(board, r, c, opp, size, captures, mover) {
    // For each orthogonal neighbor of (r,c) that is `opp`, do a BFS of its
    // connected group and count its liberties. If zero, remove the group.
    const neighbors = (rr, cc) => {
      const out = [];
      if (rr > 0)        out.push([rr-1, cc]);
      if (rr < size - 1) out.push([rr+1, cc]);
      if (cc > 0)        out.push([rr, cc-1]);
      if (cc < size - 1) out.push([rr, cc+1]);
      return out;
    };
    for (const [nr, nc] of neighbors(r, c)) {
      if (board[nr][nc] !== opp) continue;
      // BFS group
      const seen = new Set([nr * size + nc]);
      const stack = [[nr, nc]];
      const group = [];
      let libs = 0;
      const libSeen = new Set();
      while (stack.length) {
        const [cr, cc2] = stack.pop();
        group.push([cr, cc2]);
        for (const [mr, mc] of neighbors(cr, cc2)) {
          const key = mr * size + mc;
          if (board[mr][mc] === opp && !seen.has(key)) {
            seen.add(key); stack.push([mr, mc]);
          } else if (board[mr][mc] === 0 && !libSeen.has(key)) {
            libSeen.add(key); libs++;
          }
        }
      }
      if (libs === 0) {
        for (const [gr, gc] of group) board[gr][gc] = 0;
        captures[mover] += group.length;
      }
    }
  }

  // ---- play/pause ----
  togglePlay() {
    if (!this.currentKifu) return;
    if (this.playTimer) this._stopPlay();
    else this._startPlay();
  }

  _startPlay() {
    if (this.moveIndex >= this._totalMoves()) this.goTo(0);  // wrap to start
    this.root.querySelector('#kifu-play').textContent = '⏸';
    this.playTimer = setInterval(() => {
      if (this.moveIndex >= this._totalMoves()) { this._stopPlay(); return; }
      this.goTo(this.moveIndex + 1);
    }, this.playSpeed);
  }

  _stopPlay() {
    if (this.playTimer) { clearInterval(this.playTimer); this.playTimer = null; }
    const btn = this.root.querySelector('#kifu-play');
    if (btn) btn.textContent = '▶';
  }
}
