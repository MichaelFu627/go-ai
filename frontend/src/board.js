/**
 * SVG board rendering. The board is laid out in a 600x600 viewBox; everything
 * scales with the SVG. We compute padding so coordinate labels fit comfortably.
 */

const VB = 600;        // viewBox size
const PAD = 38;        // padding from edge to first line

const BLACK = 1;
const WHITE = 2;

const COLS = 'ABCDEFGHJKLMNOPQRSTUVWXYZ';  // skip 'I' (Go convention)

export class BoardView {
  constructor(svg, { onClick, onHover, onLeave } = {}) {
    this.svg = svg;
    this.size = 9;
    this.step = (VB - 2 * PAD) / (this.size - 1);
    this.onClick = onClick;
    this.onHover = onHover;
    this.onLeave = onLeave;

    this.layers = {
      bg:     svg.querySelector('#board-bg'),
      grid:   svg.querySelector('#board-grid'),
      stars:  svg.querySelector('#board-stars'),
      labels: svg.querySelector('#board-labels'),
      stones: svg.querySelector('#board-stones'),
      territory: svg.querySelector('#board-territory'),
      marks:  svg.querySelector('#board-marks'),
      hover:  svg.querySelector('#board-hover'),
    };

    svg.addEventListener('click',     (e) => this._handlePointer(e, 'click'));
    svg.addEventListener('mousemove', (e) => this._handlePointer(e, 'hover'));
    svg.addEventListener('mouseleave', () => {
      this.layers.hover.replaceChildren();
      if (this.onLeave) this.onLeave();
    });

    // Draw the grid/stars/labels right away. Previously this only ran via
    // setSize(), which render() skips when the server's size matches the
    // default — so the static layers never appeared.
    this._drawStaticLayers();
  }

  setSize(size) {
    if (size === this.size) {
      // Size unchanged but ensure static layers exist (e.g. first render).
      this._drawStaticLayers();
      return;
    }
    this.size = size;
    this.step = (VB - 2 * PAD) / (size - 1);
    this._drawStaticLayers();
  }

  _drawStaticLayers() {
    const { bg, grid, stars, labels } = this.layers;
    const ns = 'http://www.w3.org/2000/svg';

    // background — solid warm wood. SVG attributes don't support CSS vars,
    // so we set explicit colors (matching --wood in style.css).
    bg.replaceChildren();
    const rect = document.createElementNS(ns, 'rect');
    rect.setAttribute('width', VB);
    rect.setAttribute('height', VB);
    rect.setAttribute('fill', '#d9b88a');
    rect.setAttribute('rx', '3');
    bg.appendChild(rect);

    // grid lines — explicit stroke so they don't depend on CSS class loading
    grid.replaceChildren();
    for (let i = 0; i < this.size; i++) {
      const p = PAD + i * this.step;
      const lineA = document.createElementNS(ns, 'line');
      lineA.setAttribute('x1', PAD); lineA.setAttribute('y1', p);
      lineA.setAttribute('x2', VB - PAD); lineA.setAttribute('y2', p);
      lineA.setAttribute('stroke', '#3a2a14');
      lineA.setAttribute('stroke-width', '1');
      lineA.setAttribute('stroke-linecap', 'square');
      grid.appendChild(lineA);
      const lineB = document.createElementNS(ns, 'line');
      lineB.setAttribute('x1', p); lineB.setAttribute('y1', PAD);
      lineB.setAttribute('x2', p); lineB.setAttribute('y2', VB - PAD);
      lineB.setAttribute('stroke', '#3a2a14');
      lineB.setAttribute('stroke-width', '1');
      lineB.setAttribute('stroke-linecap', 'square');
      grid.appendChild(lineB);
    }
    // make outer border slightly bolder for visual weight
    const border = document.createElementNS(ns, 'rect');
    border.setAttribute('x', PAD); border.setAttribute('y', PAD);
    border.setAttribute('width', VB - 2 * PAD);
    border.setAttribute('height', VB - 2 * PAD);
    border.setAttribute('fill', 'none');
    border.setAttribute('stroke', '#3a2a14');
    border.setAttribute('stroke-width', '1.6');
    grid.appendChild(border);

    // star points (hoshi) — for 9x9: 2-2, 2-6, 6-2, 6-6, and tengen 4-4
    stars.replaceChildren();
    const hoshi = this._hoshiPoints(this.size);
    for (const [r, c] of hoshi) {
      const [x, y] = this._pos(r, c);
      const star = document.createElementNS(ns, 'circle');
      star.setAttribute('cx', x); star.setAttribute('cy', y);
      star.setAttribute('r', 3.5);
      star.setAttribute('fill', '#3a2a14');
      stars.appendChild(star);
    }

    // coordinate labels
    labels.replaceChildren();
    for (let i = 0; i < this.size; i++) {
      const p = PAD + i * this.step;
      // top column labels
      const topLabel = document.createElementNS(ns, 'text');
      topLabel.setAttribute('x', p); topLabel.setAttribute('y', PAD - 14);
      topLabel.setAttribute('text-anchor', 'middle');
      topLabel.setAttribute('font-family', "'JetBrains Mono', monospace");
      topLabel.setAttribute('font-size', '11');
      topLabel.setAttribute('fill', '#3a2a14');
      topLabel.setAttribute('fill-opacity', '0.55');
      topLabel.textContent = COLS[i];
      labels.appendChild(topLabel);
      // left row labels (Go counts from the bottom, so row 0 displays as size)
      const leftLabel = document.createElementNS(ns, 'text');
      leftLabel.setAttribute('x', PAD - 16); leftLabel.setAttribute('y', p + 4);
      leftLabel.setAttribute('text-anchor', 'middle');
      leftLabel.setAttribute('font-family', "'JetBrains Mono', monospace");
      leftLabel.setAttribute('font-size', '11');
      leftLabel.setAttribute('fill', '#3a2a14');
      leftLabel.setAttribute('fill-opacity', '0.55');
      leftLabel.textContent = String(this.size - i);
      labels.appendChild(leftLabel);
    }
  }

  _hoshiPoints(size) {
    if (size === 9)  return [[2,2],[2,6],[6,2],[6,6],[4,4]];
    if (size === 13) return [[3,3],[3,9],[9,3],[9,9],[6,6]];
    if (size === 19) return [[3,3],[3,9],[3,15],[9,3],[9,9],[9,15],[15,3],[15,9],[15,15]];
    return [];
  }

  _pos(r, c) {
    return [PAD + c * this.step, PAD + r * this.step];
  }

  _eventToPoint(evt) {
    const rect = this.svg.getBoundingClientRect();
    const x = ((evt.clientX - rect.left) / rect.width) * VB;
    const y = ((evt.clientY - rect.top) / rect.height) * VB;
    const c = Math.round((x - PAD) / this.step);
    const r = Math.round((y - PAD) / this.step);
    if (r < 0 || r >= this.size || c < 0 || c >= this.size) return null;
    // reject clicks too far from a real intersection
    const [px, py] = this._pos(r, c);
    const dx = x - px, dy = y - py;
    if (Math.hypot(dx, dy) > this.step * 0.45) return null;
    return [r, c];
  }

  _handlePointer(evt, kind) {
    const p = this._eventToPoint(evt);
    if (kind === 'click') {
      if (p && this.onClick) this.onClick(p[0], p[1]);
    } else if (kind === 'hover') {
      this.layers.hover.replaceChildren();
      if (p && this.onHover) this.onHover(p[0], p[1]);
    }
  }

  /** Draw a hover preview stone at (r, c) of the given color (1 or 2). */
  drawHover(r, c, color) {
    if (color !== BLACK && color !== WHITE) return;
    const ns = 'http://www.w3.org/2000/svg';
    const [x, y] = this._pos(r, c);
    const stone = document.createElementNS(ns, 'circle');
    stone.setAttribute('cx', x); stone.setAttribute('cy', y);
    stone.setAttribute('r', this.step * 0.46);
    stone.setAttribute('fill', color === BLACK ? 'url(#stone-black)' : 'url(#stone-white)');
    stone.setAttribute('class', 'hover-stone');
    this.layers.hover.replaceChildren(stone);
  }

  /** Render the full board state from the server. */
  render(state) {
    if (state.size !== this.size) this.setSize(state.size);

    const ns = 'http://www.w3.org/2000/svg';
    this.layers.stones.replaceChildren();
    this.layers.marks.replaceChildren();
    this.layers.territory.replaceChildren();  // stale once the board changes

    for (let r = 0; r < this.size; r++) {
      for (let c = 0; c < this.size; c++) {
        const v = state.board[r][c];
        if (v === 0) continue;
        const [x, y] = this._pos(r, c);
        const stone = document.createElementNS(ns, 'circle');
        stone.setAttribute('cx', x); stone.setAttribute('cy', y);
        stone.setAttribute('r', this.step * 0.46);
        stone.setAttribute('fill', v === BLACK ? 'url(#stone-black)' : 'url(#stone-white)');
        stone.setAttribute('filter', 'url(#stone-shadow)');
        stone.setAttribute('class', 'stone');
        this.layers.stones.appendChild(stone);
      }
    }

    // last-move marker (vermilion dot)
    const lastPlay = [...state.history].reverse().find(m => m.kind === 'play');
    if (lastPlay) {
      const [x, y] = this._pos(lastPlay.row, lastPlay.col);
      const dot = document.createElementNS(ns, 'circle');
      dot.setAttribute('cx', x); dot.setAttribute('cy', y);
      dot.setAttribute('r', this.step * 0.13);
      dot.setAttribute('class', 'last-move-dot');
      this.layers.marks.appendChild(dot);
    }
  }

  /** Overlay territory/ownership markers from a score result.
   *  ownership is a size×size grid of 'black' | 'white' | 'neutral'.
   *  We draw small squares: filled for territory (empty points), and a
   *  smaller hollow marker on stones so dead-vs-alive reads clearly. */
  showTerritory(ownership, board) {
    const ns = 'http://www.w3.org/2000/svg';
    this.layers.territory.replaceChildren();
    if (!ownership) return;

    const sq = this.step * 0.30;  // territory marker size
    for (let r = 0; r < this.size; r++) {
      for (let c = 0; c < this.size; c++) {
        const owner = ownership[r][c];
        if (owner === 'neutral') continue;
        const [x, y] = this._pos(r, c);
        const onStone = board[r][c] !== 0;

        const rect = document.createElementNS(ns, 'rect');
        rect.setAttribute('x', x - sq / 2);
        rect.setAttribute('y', y - sq / 2);
        rect.setAttribute('width', sq);
        rect.setAttribute('height', sq);
        rect.setAttribute('rx', sq * 0.18);
        const fill = owner === 'black' ? '#1a1614' : '#f8f3e4';
        rect.setAttribute('fill', fill);
        // On stones, show only a faint outline so we don't hide the stone;
        // on empty territory, show a solid (semi-transparent) marker.
        rect.setAttribute('fill-opacity', onStone ? '0' : '0.78');
        rect.setAttribute('stroke', fill);
        rect.setAttribute('stroke-width', onStone ? '1.4' : '0.5');
        rect.setAttribute('stroke-opacity', onStone ? '0.5' : '0.3');
        rect.setAttribute('class', 'territory-mark');
        this.layers.territory.appendChild(rect);
      }
    }
  }

  clearTerritory() {
    this.layers.territory.replaceChildren();
  }
}
