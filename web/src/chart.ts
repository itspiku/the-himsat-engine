/**
 * Minimal dependency-free SVG time-series chart: one series, optional ±1σ band, optional event
 * marker, a recessive grid, and a crosshair tooltip. Single series, so the chart title names it
 * (no legend). Text uses ink tokens, never the series colour.
 */
export interface Point {
  t: number; // epoch ms
  v: number;
  se?: number;
  label?: string;
}

export interface ChartOpts {
  title: string;
  unit?: string;
  height?: number;
  eventTime?: number | null;
  eventLabel?: string;
  zeroLine?: boolean;
  fmt?: (v: number) => string;
  fmtT?: (t: number) => string;
  step?: boolean;
  tableLabel?: string;
}

const NS = "http://www.w3.org/2000/svg";

function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>): SVGElementTagNameMap[K] {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
}

function niceTicks(lo: number, hi: number, n = 4): number[] {
  const span = hi - lo || Math.abs(hi) || 1;
  const raw = span / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

export function lineChart(container: HTMLElement, pts: Point[], o: ChartOpts): void {
  container.innerHTML = "";
  const fig = document.createElement("figure");
  fig.className = "chart";
  const cap = document.createElement("figcaption");
  cap.textContent = o.title;
  fig.appendChild(cap);
  container.appendChild(fig);
  if (!pts.length) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = "—";
    fig.appendChild(p);
    return;
  }
  const W = 340;
  const H = o.height ?? 150;
  const m = { l: 42, r: 10, t: 8, b: 22 };
  const fmt = o.fmt ?? ((v: number) => v.toFixed(2));
  const fmtT = o.fmtT ?? ((t: number) => new Date(t).toISOString().slice(0, 10));
  pts = [...pts].sort((a, b) => a.t - b.t);
  let t0 = pts[0].t;
  let t1 = pts[pts.length - 1].t;
  if (o.eventTime) {
    t0 = Math.min(t0, o.eventTime);
    t1 = Math.max(t1, o.eventTime);
  }
  if (t1 === t0) t1 = t0 + 86400000;
  let lo = Math.min(...pts.map((p) => p.v - (p.se ?? 0)));
  let hi = Math.max(...pts.map((p) => p.v + (p.se ?? 0)));
  if (o.zeroLine) {
    lo = Math.min(lo, 0);
    hi = Math.max(hi, 0);
  }
  if (hi === lo) hi = lo + 1;
  const pad = (hi - lo) * 0.08;
  lo -= pad;
  hi += pad;
  const x = (t: number) => m.l + ((t - t0) / (t1 - t0)) * (W - m.l - m.r);
  const y = (v: number) => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.title, class: "chart-svg" });
  for (const v of niceTicks(lo, hi)) {
    svg.appendChild(el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: v === 0 ? "axis" : "grid" }));
    const tx = el("text", { x: m.l - 6, y: y(v) + 3, class: "tick", "text-anchor": "end" });
    tx.textContent = fmt(v);
    svg.appendChild(tx);
  }
  // time ticks: first / middle / last
  for (const t of [t0, (t0 + t1) / 2, t1]) {
    const tx = el("text", { x: x(t), y: H - 6, class: "tick", "text-anchor": t === t0 ? "start" : t === t1 ? "end" : "middle" });
    tx.textContent = fmtT(t);
    svg.appendChild(tx);
  }
  if (pts.some((p) => p.se)) {
    const up = pts.map((p) => `${x(p.t)},${y(p.v + (p.se ?? 0))}`);
    const dn = pts.map((p) => `${x(p.t)},${y(p.v - (p.se ?? 0))}`).reverse();
    svg.appendChild(el("polygon", { points: [...up, ...dn].join(" "), class: "band" }));
  }
  if (o.eventTime) {
    const ex = x(o.eventTime);
    svg.appendChild(el("line", { x1: ex, x2: ex, y1: m.t, y2: H - m.b, class: "event" }));
    const tx = el("text", { x: ex - 3, y: m.t + 9, class: "event-label", "text-anchor": "end" });
    tx.textContent = o.eventLabel ?? "";
    svg.appendChild(tx);
  }
  let d = "";
  pts.forEach((p, i) => {
    if (o.step && i > 0) d += `H${x(p.t)}V${y(p.v)}`;
    else d += `${i ? "L" : "M"}${x(p.t)},${y(p.v)}`;
  });
  svg.appendChild(el("path", { d, class: "series" }));
  if (pts.length <= 60) for (const p of pts) svg.appendChild(el("circle", { cx: x(p.t), cy: y(p.v), r: 2.5, class: "dot" }));

  // crosshair + tooltip
  const cross = el("line", { y1: m.t, y2: H - m.b, class: "crosshair", visibility: "hidden" });
  const focus = el("circle", { r: 4.5, class: "focus", visibility: "hidden" });
  svg.append(cross, focus);
  const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" });
  svg.appendChild(hit);
  const tip = document.createElement("div");
  tip.className = "tooltip";
  tip.hidden = true;
  // accessible table view of the same data (screen readers, print, colour-independent reading)
  const tbl = document.createElement("details");
  tbl.className = "chart-data";
  const rows = pts.map((p) => `<tr><td>${fmtT(p.t)}</td><td>${fmt(p.v)}${p.se ? ` ±${fmt(p.se)}` : ""}</td>${
    p.label ? `<td>${p.label}</td>` : ""}</tr>`).join("");
  tbl.innerHTML = `<summary>${o.tableLabel ?? "Data"}</summary><table class="tbl"><tbody>${rows}</tbody></table>`;
  fig.append(svg, tip, tbl);
  const show = (clientX: number) => {
    const r = svg.getBoundingClientRect();
    const sx = ((clientX - r.left) / r.width) * W;
    let best = pts[0];
    for (const p of pts) if (Math.abs(x(p.t) - sx) < Math.abs(x(best.t) - sx)) best = p;
    cross.setAttribute("x1", String(x(best.t)));
    cross.setAttribute("x2", String(x(best.t)));
    focus.setAttribute("cx", String(x(best.t)));
    focus.setAttribute("cy", String(y(best.v)));
    cross.setAttribute("visibility", "visible");
    focus.setAttribute("visibility", "visible");
    tip.hidden = false;
    tip.innerHTML = `<b>${fmt(best.v)}${o.unit ? " " + o.unit : ""}</b>${best.se ? ` ±${fmt(best.se)}` : ""}<br><span>${fmtT(best.t)}</span>${best.label ? `<br><span>${best.label}</span>` : ""}`;
    const px = (x(best.t) / W) * r.width;
    tip.style.left = `${Math.min(Math.max(px, 60), r.width - 60)}px`;
  };
  hit.addEventListener("pointermove", (e) => show(e.clientX));
  hit.addEventListener("pointerdown", (e) => show(e.clientX));
  hit.addEventListener("pointerleave", () => {
    cross.setAttribute("visibility", "hidden");
    focus.setAttribute("visibility", "hidden");
    tip.hidden = true;
  });
}
