#!/usr/bin/env python3
"""Generate publication-grade DeepSeek-V4.1-Flash architecture diagram (SVG + high-res PNG/JPG).

Style modeled after CalvinXKY/InfraTech DeepSeek-V4 architecture diagrams:
- Clean white background, vibrant semantic color-coded modules, trapezoid projections,
  red KV cache blocks with pink token-slot bars, dashed zoom-out connectors, and exact
  tensor shapes & hyperparameters verified from DeepSeek-V4.1-Flash config.json,
  model.safetensors.index.json, and SGLang DeepseekV41ForCausalLM source code.
"""

from __future__ import annotations

import html
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

W, H = 3600, 2980

FONT_SANS = '"PingFang SC", "SF Pro Display", "Helvetica Neue", Arial, sans-serif'
FONT_MONO = '"SF Mono", Menlo, Consolas, "PingFang SC", monospace'

# Semantic color palette
C_INK = "#1E293B"
C_MUTED = "#475569"
C_SUBTLE = "#64748B"

C_BLUE = "#5B7CFA"
C_BLUE_DARK = "#3B5BDB"
C_BLUE_LIGHT = "#EEF2FF"

C_ORANGE = "#F5931E"
C_ORANGE_DARK = "#D97706"
C_ORANGE_LIGHT = "#FFF8EB"

C_PERI = "#7492FA"
C_PERI_DARK = "#4C6EF5"
C_PERI_LIGHT = "#F0F4FF"

C_GREEN = "#7CB342"
C_GREEN_DARK = "#558B2F"
C_GREEN_LIGHT = "#F4FBE9"

C_TEAL = "#26B5AC"
C_TEAL_DARK = "#0F8B83"
C_TEAL_LIGHT = "#ECFEFC"

C_RED = "#E51C23"
C_RED_DARK = "#B91C1C"
C_PINK_BAR = "#FBB6B6"

C_INDIGO = "#5C6BC0"
C_INDIGO_DARK = "#3F51B5"
C_INDIGO_LIGHT = "#EEF0FF"

C_ROSE = "#D83B78"
C_ROSE_DARK = "#AD1F57"
C_ROSE_LIGHT = "#FFF1F6"

C_SKY = "#0EA5E9"
C_SKY_DARK = "#0369A1"
C_SKY_LIGHT = "#F0F9FF"

C_VIOLET = "#8B5CF6"
C_VIOLET_DARK = "#6D28D9"
C_VIOLET_LIGHT = "#F5F3FF"

C_AMBER = "#F59E0B"
C_EMERALD = "#10B981"
C_PURPLE_DARK = "#6D3598"

MARKER_COLORS = {
    "blk": "#1E293B",
    "gray": "#64748B",
    "red": C_RED,
    "green": C_GREEN_DARK,
    "orange": C_ORANGE_DARK,
    "blue": C_BLUE_DARK,
    "peri": C_PERI_DARK,
    "teal": C_TEAL_DARK,
    "indigo": C_INDIGO_DARK,
    "rose": C_ROSE_DARK,
    "sky": C_SKY_DARK,
    "violet": C_VIOLET_DARK,
    "emerald": "#059669",
    "amber": "#B45309",
    "purple": C_PURPLE_DARK,
}


def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


class SVGBuilder:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.defs: List[str] = []
        self.layers: dict[str, List[str]] = {
            "bg": [],
            "connectors": [],
            "panels": [],
            "shapes": [],
            "arrows": [],
            "labels": [],
        }
        self._init_defs()

    def _init_defs(self) -> None:
        for name, color in MARKER_COLORS.items():
            self.defs.append(
                f'<marker id="arr-{name}" viewBox="0 0 10 10" refX="8.5" refY="5" '
                f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M 0 1.2 L 9.5 5 L 0 8.8 z" fill="{color}"/></marker>'
            )

    def add(self, elem: str, layer: str = "shapes") -> None:
        self.layers[layer].append(elem)

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str = "#FFFFFF",
        stroke: str = "none",
        stroke_width: float = 1.5,
        rx: float = 8,
        dash: Optional[str] = None,
        opacity: float = 1.0,
        layer: str = "shapes",
    ) -> None:
        d_attr = f' stroke-dasharray="{dash}"' if dash else ""
        op_attr = f' opacity="{opacity}"' if opacity < 1.0 else ""
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx:.1f}" ry="{rx:.1f}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width:.1f}"{d_attr}{op_attr}/>',
            layer=layer,
        )

    def text(
        self,
        x: float,
        y: float,
        content: str,
        *,
        size: float = 14,
        weight: str = "500",
        fill: str = C_INK,
        anchor: str = "middle",
        mono: bool = False,
        layer: str = "labels",
        baseline: str = "central",
    ) -> None:
        font = FONT_MONO if mono else FONT_SANS
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'dominant-baseline="{baseline}" font-family="{esc(font)}" '
            f'font-size="{size:.1f}" font-weight="{weight}" fill="{fill}">'
            f"{esc(content)}</text>",
            layer=layer,
        )

    def text_lines(
        self,
        x: float,
        y: float,
        lines: Sequence[str],
        *,
        size: float = 13.5,
        line_h: float = 19.5,
        weight: str = "500",
        fill: str = C_INK,
        anchor: str = "start",
        mono: bool = False,
        layer: str = "labels",
    ) -> None:
        for i, line in enumerate(lines):
            self.text(
                x,
                y + i * line_h,
                line,
                size=size,
                weight=weight,
                fill=fill,
                anchor=anchor,
                mono=mono,
                layer=layer,
                baseline="hanging",
            )

    def panel(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        *,
        color: str,
        fill: str = "#FFFFFF",
        subtitle: Optional[str] = None,
        dash: str = "10,6",
        title_size: float = 20.5,
    ) -> None:
        self.rect(
            x,
            y,
            w,
            h,
            fill=fill,
            stroke=color,
            stroke_width=2.2,
            rx=14,
            dash=dash,
            layer="panels",
        )
        self.text(
            x + 18,
            y + 24,
            title,
            size=title_size,
            weight="700",
            fill=color,
            anchor="start",
            layer="labels",
        )
        if subtitle:
            self.text(
                x + w - 18,
                y + 24,
                subtitle,
                size=13.2,
                weight="600",
                fill=C_MUTED,
                anchor="end",
                layer="labels",
            )

    def box(
        self,
        cx: float,
        cy: float,
        w: float,
        h: float,
        label: str,
        *,
        fill: str,
        text_color: str = "#FFFFFF",
        stroke: str = "none",
        stroke_width: float = 1.6,
        rx: float = 9,
        size: float = 14.5,
        weight: str = "700",
        sublabel: Optional[str] = None,
        sub_size: float = 11.5,
        sub_color: Optional[str] = None,
        dash: Optional[str] = None,
        layer: str = "shapes",
    ) -> Tuple[float, float, float, float]:
        x, y = cx - w / 2, cy - h / 2
        self.rect(
            x,
            y,
            w,
            h,
            fill=fill,
            stroke=stroke,
            stroke_width=stroke_width,
            rx=rx,
            dash=dash,
            layer=layer,
        )
        if sublabel:
            self.text(
                cx,
                cy - h * 0.19,
                label,
                size=size,
                weight=weight,
                fill=text_color,
                layer="labels",
            )
            self.text(
                cx,
                cy + h * 0.21,
                sublabel,
                size=sub_size,
                weight="500",
                fill=sub_color or text_color,
                layer="labels",
            )
        else:
            self.text(
                cx,
                cy,
                label,
                size=size,
                weight=weight,
                fill=text_color,
                layer="labels",
            )
        return (x, y, x + w, y + h)

    def trap(
        self,
        cx: float,
        cy: float,
        w_bot: float,
        w_top: float,
        h: float,
        label: str,
        *,
        fill: str,
        text_color: str = "#FFFFFF",
        stroke: str = "none",
        stroke_width: float = 1.5,
        size: float = 14,
        weight: str = "700",
        sublabel: Optional[str] = None,
        sub_size: float = 11.2,
    ) -> Tuple[float, float, float, float]:
        y_top = cy - h / 2
        y_bot = cy + h / 2
        pts = [
            (cx - w_top / 2, y_top),
            (cx + w_top / 2, y_top),
            (cx + w_bot / 2, y_bot),
            (cx - w_bot / 2, y_bot),
        ]
        p_str = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        self.add(
            f'<polygon points="{p_str}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width:.1f}" stroke-linejoin="round"/>',
            layer="shapes",
        )
        if sublabel:
            self.text(
                cx,
                cy - h * 0.17,
                label,
                size=size,
                weight=weight,
                fill=text_color,
                layer="labels",
            )
            self.text(
                cx,
                cy + h * 0.21,
                sublabel,
                size=sub_size,
                weight="500",
                fill=text_color,
                layer="labels",
            )
        else:
            self.text(
                cx,
                cy,
                label,
                size=size,
                weight=weight,
                fill=text_color,
                layer="labels",
            )
        return (cx - max(w_bot, w_top) / 2, y_top, cx + max(w_bot, w_top) / 2, y_bot)

    def circle_op(
        self,
        cx: float,
        cy: float,
        r: float,
        op: str,
        *,
        fill: str,
        stroke: str = "#1E293B",
        sym_color: str = "#FFFFFF",
        stroke_width: float = 1.8,
    ) -> None:
        self.add(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width:.1f}"/>',
            layer="shapes",
        )
        d = r * 0.54
        if op == "plus":
            self.add(
                f'<line x1="{cx-d:.1f}" y1="{cy:.1f}" x2="{cx+d:.1f}" y2="{cy:.1f}" '
                f'stroke="{sym_color}" stroke-width="2.4" stroke-linecap="round"/>'
                f'<line x1="{cx:.1f}" y1="{cy-d:.1f}" x2="{cx:.1f}" y2="{cy+d:.1f}" '
                f'stroke="{sym_color}" stroke-width="2.4" stroke-linecap="round"/>',
                layer="labels",
            )
        elif op == "mul":
            dm = r * 0.44
            self.add(
                f'<line x1="{cx-dm:.1f}" y1="{cy-dm:.1f}" x2="{cx+dm:.1f}" y2="{cy+dm:.1f}" '
                f'stroke="{sym_color}" stroke-width="2.4" stroke-linecap="round"/>'
                f'<line x1="{cx-dm:.1f}" y1="{cy+dm:.1f}" x2="{cx+dm:.1f}" y2="{cy-dm:.1f}" '
                f'stroke="{sym_color}" stroke-width="2.4" stroke-linecap="round"/>',
                layer="labels",
            )

    def arrow(
        self,
        pts: Sequence[Tuple[float, float]],
        *,
        color_key: str = "blk",
        width: float = 1.9,
        dash: Optional[str] = None,
        head_end: bool = True,
        head_start: bool = False,
        layer: str = "arrows",
    ) -> None:
        color = MARKER_COLORS[color_key]
        p_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        d_attr = f' stroke-dasharray="{dash}"' if dash else ""
        me = f' marker-end="url(#arr-{color_key})"' if head_end else ""
        ms = f' marker-start="url(#arr-{color_key})"' if head_start else ""
        self.add(
            f'<polyline points="{p_str}" fill="none" stroke="{color}" '
            f'stroke-width="{width:.1f}" stroke-linejoin="round"{d_attr}{me}{ms}/>',
            layer=layer,
        )

    def zoom_line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str,
        dash: str = "8,6",
        width: float = 1.6,
        opacity: float = 0.72,
    ) -> None:
        self.add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width:.1f}" stroke-dasharray="{dash}" '
            f'opacity="{opacity:.2f}"/>',
            layer="connectors",
        )

    def kv_slots(
        self,
        cx: float,
        cy: float,
        *,
        bar_w: float = 14,
        bar_h: float = 34,
        left_n: int = 2,
        right_n: int = 2,
        fill: str = C_PINK_BAR,
        stroke: str = "#7F1D1D",
        gap: float = 0,
        dots_w: float = 34,
    ) -> Tuple[float, float, float, float]:
        total_w = (
            (left_n + right_n) * bar_w
            + max(0, left_n - 1) * gap
            + max(0, right_n - 1) * gap
            + (dots_w if right_n > 0 else 0)
        )
        x0 = cx - total_w / 2
        y0 = cy - bar_h / 2
        cur = x0
        for _ in range(left_n):
            self.rect(cur, y0, bar_w, bar_h, fill=fill, stroke=stroke, stroke_width=1.2, rx=1.5)
            cur += bar_w + gap
        if right_n > 0:
            self.text(cur + dots_w / 2, cy, "• • •", size=11.5, weight="700", fill=C_INK)
            cur += dots_w
            for _ in range(right_n):
                self.rect(cur, y0, bar_w, bar_h, fill=fill, stroke=stroke, stroke_width=1.2, rx=1.5)
                cur += bar_w + gap
        return (x0, y0, x0 + total_w, y0 + bar_h)

    def badge(
        self,
        cx: float,
        cy: float,
        text: str,
        *,
        fill: str,
        text_color: str = "#FFFFFF",
        stroke: str = "none",
        w: Optional[float] = None,
        h: float = 24,
        size: float = 11.5,
        rx: float = 12,
    ) -> None:
        bw = w if w is not None else max(48, len(text) * (size * 0.62) + 18)
        self.box(
            cx,
            cy,
            bw,
            h,
            text,
            fill=fill,
            text_color=text_color,
            stroke=stroke,
            stroke_width=1.2 if stroke != "none" else 0,
            rx=rx,
            size=size,
            weight="700",
        )

    def render(self) -> str:
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">',
            "<defs>",
            *self.defs,
            "</defs>",
            f'<rect width="{self.width}" height="{self.height}" fill="#FFFFFF"/>',
        ]
        for layer in ("bg", "connectors", "panels", "shapes", "arrows", "labels"):
            parts.append(f'<g id="layer-{layer}">')
            parts.extend(self.layers[layer])
            parts.append("</g>")
        parts.append("</svg>")
        return "\n".join(parts)


# =============================================================================
# Section builders
# =============================================================================

def draw_header_and_layer_map(svg: SVGBuilder) -> None:
    # Title
    svg.text(
        W / 2,
        44,
        "DeepSeek-V4.1-Flash 架构全景图 (763B Total · 552B Backbone + 196.9B Engram + 14.2B DSpark + 0.5B Vision)",
        size=34,
        weight="800",
        fill="#0F172A",
    )

    # Spec pills bar
    pills = [
        ("Active: ~8B Prefill / ~16B Decode", "#DBEAFE", "#1D4ED8"),
        ("40L CED (20L Encoder + 20L Decoder) + 3L DSpark MTP", "#FEF3C7", "#B45309"),
        ("Context: 1M (YaRN 16×) · Vocab: 129,280", "#E0E7FF", "#3730A3"),
        ("Global KV + Idx Cache: 890 B/tok (FP4)", "#FEE2E2", "#B91C1C"),
        ("Quant: FP8 32×32 + MXFP4 Routed Experts", "#DCFCE7", "#166534"),
    ]
    pill_ws = [285, 420, 325, 315, 330]
    gap = 14
    total_w = sum(pill_ws) + gap * (len(pills) - 1)
    x = (W - total_w) / 2
    for (txt, bg, fg), pw in zip(pills, pill_ws):
        svg.box(x + pw / 2, 88, pw, 30, txt, fill=bg, text_color=fg, stroke=fg, stroke_width=1.0, rx=15, size=13.5, weight="700")
        x += pw + gap

    # 40L + 3 MTP Layer Topology Strip (y=118..282, h=164)
    sx, sy, sw, sh = 50, 118, 3500, 164
    svg.rect(sx, sy, sw, sh, fill="#F8FAFC", stroke="#CBD5E1", stroke_width=1.6, rx=12, layer="panels")
    svg.text(
        sx + 18,
        sy + 19,
        "全模型 40 层主干 + 3 层 DSpark MTP 层级映射图（每层 Attention 模式 / KV 共享 / Engram / DSpark Tap）",
        size=14,
        weight="700",
        fill=C_INK,
        anchor="start",
    )

    idx_sources = {2, 8, 14, 20, 24, 28, 32, 36}
    engram_layers = {1, 14}
    dspark_taps = {37, 38, 39}

    cw, cg = 72, 6
    ced_gap = 18  # extra visual gap between L19 and L20 for clean CED cut line
    x_start = 100
    cy_cell = sy + 82
    ch_cell = 46

    def cell_x(i: int) -> float:
        extra = ced_gap if i >= 20 else 0
        return x_start + i * (cw + cg) + extra

    def draw_bracket(bx1: float, bx2: float, label: str, color: str, y_b: float) -> None:
        svg.add(
            f'<path d="M {bx1:.1f} {y_b+7:.1f} L {bx1:.1f} {y_b:.1f} L {bx2:.1f} {y_b:.1f} L {bx2:.1f} {y_b+7:.1f}" '
            f'fill="none" stroke="{color}" stroke-width="1.6"/>',
            layer="arrows",
        )
        svg.text((bx1 + bx2) / 2, y_b - 9, label, size=12, weight="700", fill=color)

    draw_bracket(
        cell_x(0),
        cell_x(19) + cw,
        "Causal Encoder · L0–L19 (~8B Active Prefill · L0–1 SWA, L2–19 3×6 CSA2-C2 Groups)",
        C_ORANGE_DARK,
        sy + 46,
    )
    draw_bracket(
        cell_x(20),
        cell_x(39) + cw,
        "Decoder · L20–L39 (~16B Active Decode · 全 20 层共享 L20 C1 KV & Index-K; L24/28/32/36 在 L20 的 2048 候选块内 Reindex)",
        C_BLUE_DARK,
        sy + 46,
    )

    # CED cut line cleanly centered in the gap between L19 and L20
    cut_x = (cell_x(19) + cw + cell_x(20)) / 2
    svg.add(
        f'<line x1="{cut_x:.1f}" y1="{sy+36:.1f}" x2="{cut_x:.1f}" y2="{sy+122:.1f}" '
        f'stroke="{C_RED}" stroke-width="2.2" stroke-dasharray="5,4"/>',
        layer="arrows",
    )
    svg.badge(cut_x, sy + 123, "CED Cut", fill=C_RED, w=68, h=20, size=10.5)

    for i in range(40):
        cx = cell_x(i) + cw / 2
        if i < 2:
            fill, tc, stroke, mode = "#E2E8F0", "#1E293B", "#64748B", "SWA"
        elif i in {2, 8, 14}:
            fill, tc, stroke, mode = C_ORANGE, "#FFFFFF", C_ORANGE_DARK, "Full·C2"
        elif i < 20:
            fill, tc, stroke, mode = "#FDE68A", "#92400E", "#D97706", "Reuse·C2"
        elif i == 20:
            fill, tc, stroke, mode = "#DC2626", "#FFFFFF", "#991B1B", "C1+Cand"
        elif i in idx_sources:
            fill, tc, stroke, mode = C_GREEN, "#FFFFFF", C_GREEN_DARK, "Reindex"
        else:
            fill, tc, stroke, mode = "#DBEAFE", "#1E40AF", "#3B82F6", "Reuse·C1"

        svg.box(
            cx,
            cy_cell,
            cw,
            ch_cell,
            f"L{i}",
            fill=fill,
            text_color=tc,
            stroke=stroke,
            stroke_width=1.4,
            rx=6,
            size=12.5,
            sublabel=mode,
            sub_size=10.2,
            sub_color=tc,
        )
        if i in engram_layers:
            svg.badge(cx + cw / 2 - 4, cy_cell - ch_cell / 2 + 2, "E", fill=C_ROSE, w=20, h=17, size=10.5, rx=8)
        if i in dspark_taps:
            svg.badge(cx + cw / 2 - 4, cy_cell - ch_cell / 2 + 2, "T", fill=C_VIOLET, w=20, h=17, size=10.5, rx=8)

    # MTP 0..2
    mtp_x0 = cell_x(39) + cw + 30
    draw_bracket(mtp_x0, mtp_x0 + 3 * cw + 2 * cg, "DSpark MTP (3L)", C_VIOLET_DARK, sy + 46)

    for m in range(3):
        cx = mtp_x0 + m * (cw + cg) + cw / 2
        svg.box(
            cx,
            cy_cell,
            cw,
            ch_cell,
            f"MTP{m}",
            fill=C_VIOLET,
            text_color="#FFFFFF",
            stroke=C_VIOLET_DARK,
            stroke_width=1.4,
            rx=6,
            size=12,
            sublabel="SWA·128E",
            sub_size=9.8,
        )

    # Legend strip along bottom (clean vertical clearance below CED Cut pill!)
    legend_items = [
        ("#E2E8F0", "#64748B", "SWA-Only (r=0, win=128)"),
        (C_ORANGE, C_ORANGE_DARK, "CSA2-Full C2 (L2/8/14: 建 C2 KV + Idx-K)"),
        ("#FDE68A", "#D97706", "CSA2-Reuse C2 (复用本组 KV+Top512)"),
        ("#DC2626", "#991B1B", "L20 C1+Cand (Decoder 唯一 KV/Idx-K + 2048 候选块源)"),
        (C_GREEN, C_GREEN_DARK, "CSA2-Reindex (L24/28/32/36: 仅扫 2048 块重选 Top512)"),
        ("#DBEAFE", "#3B82F6", "CSA2-Reuse C1 (复用 L20 KV + 组内 Top512)"),
        (C_VIOLET, C_VIOLET_DARK, "DSpark MTP0–2 (SWA+128E top3)"),
        (C_ROSE, C_ROSE_DARK, "[E] Engram (L1/14)"),
        (C_VIOLET, C_VIOLET_DARK, "[T] DSpark Tap (L37–39)"),
    ]
    lx = sx + 22
    ly = sy + 148
    for fill, stroke, lbl in legend_items:
        svg.rect(lx, ly - 7, 15, 14, fill=fill, stroke=stroke, stroke_width=1.2, rx=3)
        svg.text(lx + 21, ly, lbl, size=11.8, weight="600", fill=C_INK, anchor="start")
        est_w = sum(12.2 if ord(ch) > 127 else 6.8 for ch in lbl)
        lx += 21 + est_w + 18


def draw_left_column(svg: SVGBuilder) -> None:
    # =========================================================================
    # L1a: Router (Gate)  (x=50..600, y=306..846)
    # =========================================================================
    svg.panel(50, 306, 550, 540, "Router (Gate)", color=C_TEAL_DARK, subtitle="sqrtsoftplus · noaux_tc")

    rx = 415
    lx = 230

    # Input arrow (shifted down cleanly so top weights label clears subtitle!)
    svg.arrow([(rx, 830), (rx, 796)], color_key="blk")
    svg.text(rx + 66, 818, "[seq, 5120]", size=13, mono=True, fill=C_INK)

    svg.trap(rx, 772, 200, 164, 42, "Linear (gate)", sublabel="5120 → 384 (FP32)", fill=C_TEAL, stroke=C_TEAL_DARK, size=13.5, sub_size=11)
    svg.arrow([(rx, 751), (rx, 722)], color_key="blk")
    svg.text(rx + 94, 736, "[seq, 384]", size=12.5, mono=True, fill=C_INK)

    svg.box(rx, 702, 164, 36, "Softplus", fill=C_TEAL, stroke=C_TEAL_DARK, size=14)
    svg.arrow([(rx, 684), (rx, 656)], color_key="blk")

    svg.box(rx, 636, 164, 36, "Sqrt", fill=C_TEAL, stroke=C_TEAL_DARK, size=14)
    svg.text(rx + 104, 636, "scores", size=12.5, weight="600", fill=C_MUTED)

    # Split from Sqrt top: straight up to Gather & left-up to +bias
    svg.arrow([(rx, 618), (rx, 522)], color_key="blk")
    svg.text(rx + 46, 574, "unbiased", size=11.5, fill=C_MUTED)
    svg.arrow([(rx, 600), (lx, 600), (lx, 588)], color_key="blk")

    # Bias selection box
    svg.box(
        122,
        640,
        122,
        52,
        "bias / bias_vl",
        sublabel="text vs image",
        fill="#CCFBF1",
        text_color=C_TEAL_DARK,
        stroke=C_TEAL_DARK,
        stroke_width=1.4,
        size=12.5,
        sub_size=10.8,
        rx=8,
    )
    svg.arrow([(122, 614), (122, 574), (lx - 15, 574)], color_key="teal")

    svg.circle_op(lx, 574, 14, "plus", fill=C_TEAL, stroke=C_TEAL_DARK)
    svg.arrow([(lx, 560), (lx, 522)], color_key="blk")

    svg.box(lx, 500, 140, 38, "TopK (k=6)", fill=C_TEAL, stroke=C_TEAL_DARK, size=14)
    svg.arrow([(lx, 481), (lx, 366)], color_key="blk")
    svg.text(lx - 48, 404, "indices", size=13, weight="700", fill=C_INK)
    svg.text(lx + 46, 404, "[seq, 6]", size=12.5, mono=True, fill=C_INK)
    svg.arrow([(lx + 70, 500), (rx - 74, 500)], color_key="blk")

    svg.box(rx, 500, 144, 38, "Gather", fill=C_TEAL, stroke=C_TEAL_DARK, size=14)
    svg.arrow([(rx, 481), (rx, 456)], color_key="blk")

    svg.box(rx, 438, 144, 34, "/= sum", fill=C_TEAL, stroke=C_TEAL_DARK, size=13.5)
    svg.arrow([(rx, 421), (rx, 400)], color_key="blk")

    svg.box(rx, 382, 144, 32, "× 1.5 (scale)", fill=C_TEAL_DARK, stroke="#115E59", size=13)
    svg.arrow([(rx, 366), (rx, 344)], color_key="blk")
    svg.text(rx + 84, 355, "weights [seq, 6]", size=12, mono=True, fill=C_INK)

    # Bottom-left info box
    svg.rect(68, 708, 224, 122, fill=C_TEAL_LIGHT, stroke=C_TEAL, stroke_width=1.2, rx=8)
    svg.text_lines(
        78,
        718,
        [
            "• 全 40 层均用学习型路由",
            "  (无 V4 Pro Hash/Tid2eid)",
            "• noaux_tc 无辅助 Loss 均衡",
            "• VL 双偏置: 文本 bias / 图像 bias_vl",
            "• DSpark MTP: 128 专家选 top-3",
        ],
        size=11.5,
        line_h=20,
        fill=C_INK,
    )

    # =========================================================================
    # L1b: MLP (Clamped SwiGLU)  (x=630..1310, y=306..846)
    # =========================================================================
    svg.panel(630, 306, 680, 540, "MLP (Clamped SwiGLU)", color=C_GREEN_DARK, subtitle="swiglu_limit = 10.0")

    ux = 810
    gx = 1120

    svg.arrow([(965, 828), (965, 784)], color_key="blk", head_end=False)
    svg.text(1048, 814, "[seq, 5120]", size=13, mono=True, fill=C_INK)
    svg.arrow([(965, 784), (ux, 784), (ux, 754)], color_key="blk")
    svg.arrow([(965, 784), (gx, 784), (gx, 754)], color_key="blk")

    svg.trap(ux, 728, 224, 184, 48, "Linear (up_proj / w3)", sublabel="5120 → 2304", fill=C_GREEN, stroke=C_GREEN_DARK, size=13.5, sub_size=11.2)
    svg.trap(gx, 728, 224, 184, 48, "Linear (gate_proj / w1)", sublabel="5120 → 2304", fill=C_GREEN, stroke=C_GREEN_DARK, size=13.5, sub_size=11.2)

    svg.arrow([(ux, 704), (ux, 670)], color_key="blk")
    svg.arrow([(gx, 704), (gx, 670)], color_key="blk")
    svg.text(ux - 102, 686, "[seq, 2304]", size=12, mono=True, fill=C_INK)
    svg.text(gx + 102, 686, "[seq, 2304]", size=12, mono=True, fill=C_INK)

    svg.box(ux, 648, 192, 38, "Clamp [-10.0, 10.0]", fill="#A3D95B", text_color="#1A2E05", stroke=C_GREEN_DARK, size=13)
    svg.box(gx, 648, 192, 38, "Clamp (max = 10.0)", fill="#A3D95B", text_color="#1A2E05", stroke=C_GREEN_DARK, size=13)

    svg.arrow([(ux, 629), (ux, 570)], color_key="blk")
    svg.arrow([(gx, 629), (gx, 574)], color_key="blk")

    svg.box(gx, 552, 196, 40, "SiLU Activation", fill=C_GREEN, stroke=C_GREEN_DARK, size=14)
    svg.arrow([(gx - 98, 552), (ux + 18, 552)], color_key="blk")

    svg.circle_op(ux, 552, 16, "mul", fill=C_GREEN, stroke=C_GREEN_DARK)
    svg.arrow([(ux, 536), (ux, 486)], color_key="blk")
    svg.text(ux + 72, 510, "[seq, 2304]", size=12.5, mono=True, fill=C_INK)

    svg.trap(ux, 458, 184, 224, 48, "Linear (down_proj / w2)", sublabel="2304 → 5120", fill=C_GREEN, stroke=C_GREEN_DARK, size=13.5, sub_size=11.2)
    svg.arrow([(ux, 434), (ux, 354)], color_key="blk")
    svg.text(ux + 72, 382, "[seq, 5120]", size=13, mono=True, fill=C_INK)

    svg.rect(975, 356, 314, 152, fill=C_GREEN_LIGHT, stroke=C_GREEN, stroke_width=1.4, rx=10)
    svg.text_lines(
        990,
        368,
        [
            "hidden_size = 5120",
            "moe_intermediate_size = 2304",
            "swiglu_limit = 10.0 (防 FP4/FP8 激活溢出)",
            "单专家参数 = 3 × 5120 × 2304 ≈ 35.4M",
            "Routed 专家: MXFP4 (W4A8 / W4A16)",
            "Shared 专家: FP8 (32×32 UE8M0)",
        ],
        size=12.2,
        line_h=22,
        fill=C_INK,
    )

    # =========================================================================
    # L2: MoE Layer  (x=50..1310, y=876..1216)
    # =========================================================================
    svg.panel(50, 876, 1260, 340, "MoE Layer (All 40L · 无稠密 FFN / 无 Hash)", color=C_BLUE_DARK, subtitle="384 Routed (Top-6) + 1 Shared")

    # Input bus at bottom y=1192
    svg.arrow([(640, 1210), (640, 1184)], color_key="blk", head_end=False)
    svg.text(712, 1202, "[seq, 5120]", size=13, mono=True, fill=C_INK)

    svg.arrow([(640, 1184), (420, 1184), (420, 1164)], color_key="blk")
    svg.arrow([(640, 1184), (890, 1184), (890, 1066)], color_key="blk")

    svg.box(420, 1138, 210, 44, "Router (Gate)", sublabel="top-6 active / token", fill=C_TEAL, stroke=C_TEAL_DARK, size=14, sub_size=11)

    # 384 Routed Experts sub-box (caption cleanly in top-left; distributor bar below, collector arrow above -> ZERO crossings!)
    svg.rect(86, 982, 688, 108, fill="#F8FAFC", stroke=C_BLUE, stroke_width=1.6, rx=10, dash="7,4")
    svg.text(102, 1000, "384 Routed Experts (MXFP4 · 每 token 仅激活 6 个)", size=12.5, weight="700", fill=C_BLUE_DARK, anchor="start")
    svg.box(176, 1044, 136, 40, "MLP (E0)", fill=C_GREEN, stroke=C_GREEN_DARK, size=13.5)
    svg.box(340, 1044, 136, 40, "MLP (E1)", fill=C_GREEN, stroke=C_GREEN_DARK, size=13.5)
    svg.text(468, 1044, "•  •  •", size=18, weight="800", fill=C_INK)
    svg.box(596, 1044, 152, 40, "MLP (E383)", fill=C_GREEN, stroke=C_GREEN_DARK, size=13.5)

    # Clean orthogonal distributor bar below the experts box from Router to E0/E1/E383
    svg.arrow([(420, 1116), (420, 1102)], color_key="blk", dash="5,4", head_end=False, width=1.6)
    svg.arrow([(176, 1102), (596, 1102)], color_key="blk", dash="5,4", head_end=False, width=1.6)
    for tx in (176, 340, 596):
        svg.arrow([(tx, 1102), (tx, 1066)], color_key="blk", dash="5,4", width=1.6)

    # 1 Shared Expert sub-box
    svg.rect(802, 982, 176, 108, fill=C_GREEN_LIGHT, stroke=C_BLUE, stroke_width=1.6, rx=10, dash="7,4")
    svg.text(890, 1000, "1 Shared (FP8)", size=12.5, weight="700", fill=C_BLUE_DARK)
    svg.box(890, 1044, 144, 40, "Shared MLP", fill=C_GREEN, stroke=C_GREEN_DARK, size=13.5)

    # Clean combine circle + at cx=640 (well right of panel title!)
    svg.circle_op(640, 938, 16, "plus", fill=C_BLUE, stroke=C_BLUE_DARK)
    svg.arrow([(430, 982), (430, 964), (640, 964), (640, 956)], color_key="blk", dash="5,4", width=1.8)
    svg.arrow([(890, 982), (890, 938), (658, 938)], color_key="blk")
    svg.arrow([(640, 922), (640, 888)], color_key="blk")
    svg.text(712, 904, "[seq, 5120]", size=13, mono=True, fill=C_INK)

    # Right summary card
    svg.rect(1002, 922, 286, 272, fill=C_BLUE_LIGHT, stroke=C_BLUE, stroke_width=1.4, rx=10)
    svg.text(1018, 942, "MoE 关键参数", size=14, weight="700", fill=C_BLUE_DARK, anchor="start")
    svg.text_lines(
        1018,
        962,
        [
            "• 每层总参数: ~13.63B",
            "• 每 token 激活: 6+1 = 7 专家",
            "  (~250M / 层)",
            "• 主干 MoE 合计: ~545B",
            "  (占 Backbone 98.8%)",
            "• 激活量化: MXFP8 / FP8",
            "  路由权重: MXFP4 (Humming)",
            "• DSpark MTP (3L):",
            "  128 专家选 3 + 1 共享",
        ],
        size=12.2,
        line_h=23,
        fill=C_INK,
    )

    # Zoom connectors from MoE up to Router & MLP
    svg.zoom_line(315, 1116, 50, 846, color=C_TEAL_DARK, dash="7,5")
    svg.zoom_line(525, 1116, 600, 846, color=C_TEAL_DARK, dash="7,5")
    svg.zoom_line(818, 1014, 630, 846, color=C_GREEN_DARK, dash="7,5")
    svg.zoom_line(962, 1014, 1310, 846, color=C_GREEN_DARK, dash="7,5")

    # =========================================================================
    # L3: Single-Pass mHC (hc_mult=4)  (x=50..1310, y=1246..1702)
    # =========================================================================
    svg.panel(50, 1246, 1260, 456, "Single-Pass mHC (Manifold-Constrained Hyper-Connections)", color=C_INDIGO_DARK, subtitle="hc_mult=4 · Sinkhorn 20 iters · 无独立 hc_head")

    # Left half: Coefficient Generator
    svg.rect(72, 1296, 636, 384, fill="#FAF9FF", stroke=C_INDIGO, stroke_width=1.4, rx=10, dash="6,4")
    svg.text(88, 1316, "动态混合系数生成 (hc_attn_* / hc_ffn_*)", size=14, weight="700", fill=C_INDIGO_DARK, anchor="start")

    svg.text(146, 1354, "x_l [T, 4, 5120]", size=12.5, mono=True, weight="700", fill=C_INK)
    for s in range(4):
        svg.box(102 + s * 28, 1394, 24, 44, f"s{s}", fill=C_INDIGO, text_color="#FFFFFF", stroke=C_INDIGO_DARK, rx=4, size=11)

    svg.arrow([(204, 1394), (242, 1394)], color_key="indigo")
    svg.box(316, 1394, 144, 46, "Flatten + RMSNorm", sublabel="[T, 20480]", fill=C_INDIGO, stroke=C_INDIGO_DARK, size=12.5, sub_size=11)
    svg.arrow([(388, 1394), (424, 1394)], color_key="indigo")
    svg.trap(542, 1394, 224, 186, 46, "Linear (hc_fn)", sublabel="20480 → 24 (FP32)", fill=C_INDIGO_DARK, stroke="#312E81", size=13, sub_size=11)

    svg.arrow([(542, 1417), (542, 1442)], color_key="indigo", head_end=False)
    svg.arrow([(542, 1442), (186, 1442), (186, 1468)], color_key="indigo")
    svg.arrow([(542, 1442), (402, 1442), (402, 1468)], color_key="indigo")
    svg.arrow([(542, 1442), (596, 1442), (596, 1468)], color_key="indigo")

    svg.box(186, 1508, 200, 72, "pre_next [T, 4]", sublabel="σ(m[0:4]·s₀ + b₀) + ε", fill="#E0E7FF", text_color="#1E1B4B", stroke=C_INDIGO_DARK, size=13, sub_size=11.5)
    svg.text(186, 1564, "传递给下一个子层 (Single-Pass)", size=11.5, weight="600", fill=C_INDIGO_DARK)

    svg.box(402, 1508, 188, 72, "post [T, 4]", sublabel="2·σ(m[4:8]·s₁ + b₁)", fill="#E0E7FF", text_color="#1E1B4B", stroke=C_INDIGO_DARK, size=13, sub_size=11.5)
    svg.text(402, 1564, "将子层输出广播回 4 流", size=11.5, weight="600", fill=C_INDIGO_DARK)

    svg.box(596, 1508, 178, 72, "H_res [T, 4×4]", sublabel="Sinkhorn-Knopp (20×)", fill=C_INDIGO_DARK, text_color="#FFFFFF", stroke="#312E81", size=13, sub_size=11.2)
    svg.text(596, 1564, "双随机矩阵 (行和=列和=1)", size=11.5, weight="600", fill=C_INDIGO_DARK)

    svg.rect(88, 1590, 604, 74, fill="#EEF2FF", stroke="#C7D2FE", stroke_width=1.2, rx=8)
    svg.text_lines(
        100,
        1600,
        [
            "• Single-Pass 优化 (hc_pre_from_prev_sublayer=True): 单次访存完成残差更新与下层 pre-mix 融合",
            "• 终态折叠: 最后一层 FFN 的 pre 直接执行 mhc_combine → [T, 5120] 送入 RMSNorm (省去 hc_head)",
            "• 参数: hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-20",
        ],
        size=11.6,
        line_h=20,
        fill=C_INK,
    )

    # Right half: 4-stream Sublayer Execution Diagram (clean clearance below title!)
    svg.rect(730, 1296, 558, 384, fill="#FFFFFF", stroke=C_INDIGO, stroke_width=1.4, rx=10)
    svg.text(748, 1316, "子层残差更新 (Mega-mHC 融合路径)", size=14, weight="700", fill=C_INDIGO_DARK, anchor="start")

    # Clean top-to-bottom spacing from y=1354 down to y=1642
    svg.box(1008, 1354, 246, 32, "x_{l+1} [T, 4, 5120] + pre_next", fill=C_INDIGO_DARK, stroke="#312E81", size=12.5)
    svg.arrow([(1008, 1396), (1008, 1372)], color_key="indigo")
    svg.circle_op(1008, 1412, 15, "plus", fill=C_INDIGO_DARK, stroke="#312E81")

    # Left branch: x_l -> H_res @ x_l -> +
    svg.box(854, 1642, 180, 38, "x_l [T, 4, 5120]", fill=C_INDIGO, stroke=C_INDIGO_DARK, size=13)
    svg.arrow([(854, 1623), (854, 1514)], color_key="indigo")
    svg.box(854, 1490, 176, 44, "H_res @ x_l", sublabel="4×4 双随机流混合", fill="#E0E7FF", text_color="#1E1B4B", stroke=C_INDIGO_DARK, size=13, sub_size=11)
    svg.arrow([(854, 1468), (854, 1412), (991, 1412)], color_key="indigo")

    # Right branch: mhc_combine -> RMSNorm+Sublayer -> *post -> +
    svg.arrow([(944, 1642), (1126, 1642), (1126, 1614)], color_key="blk")
    svg.box(1126, 1590, 226, 42, "mhc_combine (prev pre)", sublabel="∑ pre_c·x_c → [T, 5120]", fill=C_BLUE, stroke=C_BLUE_DARK, size=12.5, sub_size=11)
    svg.arrow([(1126, 1569), (1126, 1546)], color_key="blk")
    svg.box(1126, 1524, 226, 38, "RMSNorm → Attn / MoE", fill=C_ORANGE, stroke=C_ORANGE_DARK, size=13)
    svg.arrow([(1126, 1505), (1126, 1478)], color_key="blk")
    svg.circle_op(1126, 1460, 15, "mul", fill=C_INDIGO, stroke=C_INDIGO_DARK)
    svg.text(1234, 1460, "post [T, 4]", size=12, mono=True, weight="700", fill=C_INDIGO_DARK)
    svg.arrow([(1186, 1460), (1143, 1460)], color_key="indigo")
    svg.arrow([(1126, 1445), (1126, 1412), (1025, 1412)], color_key="indigo")

    # =========================================================================
    # L4: Engram Conditional Memory (L1 & L14, ~196.9B)  (x=50..1310, y=1732..2312)
    # =========================================================================
    svg.panel(50, 1732, 1260, 580, "Engram Conditional Memory (挂载于 L1 & L14)", color=C_ROSE_DARK, subtitle="196.6B Table + 0.32B Proj · O(1) 查表 · ~0 FLOPs")

    # Left half: Deterministic Multi-Head N-Gram Hash & Lookup (x=76..616)
    ex = 342
    svg.box(ex, 2264, 220, 36, "input_ids [T] (+ 历史前驱)", fill="#FFE4E6", text_color=C_ROSE_DARK, stroke=C_ROSE_DARK, size=13)
    svg.arrow([(ex, 2246), (ex, 2220)], color_key="rose")

    svg.box(ex, 2196, 386, 44, "Tokenizer 词表归一化压缩", sublabel="129,280 → 99,092 (NFKC/NFD/小写/去重音)", fill=C_ROSE, stroke=C_ROSE_DARK, size=13.5, sub_size=11)
    svg.arrow([(ex, 2174), (ex, 2146)], color_key="rose")

    svg.box(
        ex,
        2114,
        452,
        56,
        "Multi-Head Rolling XOR N-Gram Hash",
        sublabel="2/3/4-gram × 8 heads = 24 列 · (滚动乘异或) % prime_i + offset_i",
        fill=C_ROSE,
        stroke=C_ROSE_DARK,
        size=13.2,
        sub_size=10.8,
    )
    svg.arrow([(ex, 2086), (ex, 2058)], color_key="rose")
    svg.text(ex + 128, 2071, "hash_ids [T, 24]", size=12, mono=True, fill=C_INK)

    # Big Host/Device FP8 Embedding Table card
    svg.rect(76, 1922, 532, 132, fill=C_ROSE_LIGHT, stroke=C_ROSE_DARK, stroke_width=2.0, rx=10)
    svg.text(342, 1944, "超大规模 FP8 Embedding 查表 (支持 Host CPU RAM 卸载 / 预取)", size=13.2, weight="700", fill=C_ROSE_DARK)
    svg.box(208, 1990, 232, 46, "Layer 1 Table (98.3B)", sublabel="384,006,168 × 256 (FP8+scale)", fill="#FFFFFF", text_color=C_ROSE_DARK, stroke=C_ROSE, size=12.5, sub_size=10.8)
    svg.box(476, 1990, 232, 46, "Layer 14 Table (98.3B)", sublabel="384,016,682 × 256 (FP8+scale)", fill="#FFFFFF", text_color=C_ROSE_DARK, stroke=C_ROSE, size=12.5, sub_size=10.8)
    svg.text(342, 2034, "engram_vocab_size = 16,000,000 (递增素数分桶) · head_dim = 256", size=11.6, mono=True, fill=C_MUTED)

    svg.arrow([(ex, 1922), (ex, 1890)], color_key="rose")
    svg.text(ex + 132, 1906, "emb [T, 24×256 = 6144]", size=12, mono=True, fill=C_INK)

    svg.trap(ex, 1860, 248, 316, 52, "Linear (engram.wkv · FP8 32×32)", sublabel="6144 → (4 + 1) × 5120 = 25,600", fill=C_ROSE_DARK, stroke="#881337", size=13, sub_size=10.8)

    # Clean arrow from wkv output to Right Half (text strictly inside left side!)
    svg.arrow([(ex, 1834), (ex, 1804), (636, 1804)], color_key="rose")
    svg.text(468, 1788, "kv [T, 5×5120] → 4 Keys + 1 Value", size=12, mono=True, weight="600", fill=C_ROSE_DARK)

    # Right half: Context-Aware Gating & Residual Fusion (x=638..1288, w=650 — extra wide for formula!)
    svg.rect(638, 1776, 650, 516, fill="#FFFFFF", stroke=C_ROSE, stroke_width=1.5, rx=10)
    svg.text(656, 1796, "上下文感知门控融合 (Per-Stream Context Gate · c = 0..3)", size=14, weight="700", fill=C_ROSE_DARK, anchor="start")

    gx_c = 858   # Gate column center
    vx_c = 1172  # Value / Output column center

    svg.box(gx_c, 1852, 236, 46, "4 Per-Stream Keys", sublabel="K₀..K₃ [T, 4, 5120]", fill="#FCE7F3", text_color=C_ROSE_DARK, stroke=C_ROSE, size=13, sub_size=11)
    svg.box(vx_c, 1852, 198, 46, "1 Shared Value", sublabel="V [T, 5120] (4 流共享)", fill="#FCE7F3", text_color=C_ROSE_DARK, stroke=C_ROSE, size=13, sub_size=11)

    # Current mHC streams x_c input
    svg.box(gx_c, 1940, 236, 42, "当前残差流 x_c [T, 4, 5120]", fill=C_INDIGO, stroke=C_INDIGO_DARK, size=12.8)

    # Clean routing: Keys route around LEFT edge into dot-product box (no crossing x_c!); x_c goes straight down into dot-product box!
    svg.arrow([(gx_c - 118, 1852), (660, 1852), (660, 2032), (680, 2032)], color_key="rose")
    svg.arrow([(gx_c, 1961), (gx_c, 2000)], color_key="indigo")

    # Wide box (w=356) so full mathematical formula fits comfortably with generous padding!
    svg.box(
        gx_c,
        2032,
        356,
        60,
        "归一化内积打分 (q_w, k_w)",
        sublabel="d_c = ⟨RMSNorm(x_c)⊙q_w, RMSNorm(K_c)⊙k_w⟩ / √5120",
        fill=C_ROSE,
        stroke=C_ROSE_DARK,
        size=12.8,
        sub_size=10.5,
    )
    svg.arrow([(gx_c, 2062), (gx_c, 2092)], color_key="rose")

    svg.box(
        gx_c,
        2118,
        356,
        48,
        "Signed-Sqrt Sigmoid Gate",
        sublabel="g_c = σ( sign(d_c) · √max(|d_c|, 1e-6) )",
        fill=C_ROSE_DARK,
        stroke="#881337",
        size=12.8,
        sub_size=10.8,
    )

    # Multiply g_c * V and add to x_c
    svg.arrow([(gx_c + 178, 2118), (vx_c - 18, 2118)], color_key="rose")
    svg.arrow([(vx_c, 1875), (vx_c, 2100)], color_key="rose")
    svg.circle_op(vx_c, 2118, 16, "mul", fill=C_ROSE, stroke=C_ROSE_DARK)
    svg.arrow([(vx_c, 2134), (vx_c, 2168)], color_key="rose")

    svg.circle_op(vx_c, 2184, 16, "plus", fill=C_INDIGO_DARK, stroke="#312E81")
    # Residual shortcut from x_c cleanly routed at x=1084
    svg.arrow([(gx_c + 118, 1940), (1084, 1940), (1084, 2184), (vx_c - 18, 2184)], color_key="indigo", dash="6,4")
    svg.text(1032, 1926, "residual x_c", size=11, fill=C_INDIGO_DARK)

    # Output + image token bypass guard
    svg.arrow([(vx_c, 2200), (vx_c, 2224)], color_key="indigo")
    svg.box(
        964,
        2250,
        608,
        44,
        "x_c ← x_c + g_c · V  (注入回 4 条 mHC 流)",
        sublabel="视觉 Token 保护: 若 input_id == image_token_id (129264)，强制旁路保持 x_c 不变",
        fill=C_ROSE_LIGHT,
        text_color=C_ROSE_DARK,
        stroke=C_ROSE_DARK,
        size=13,
        sub_size=11.2,
    )

    # =========================================================================
    # L5: Vision Tower + Aligner (0.5B)  (x=50..1310, y=2342..2922)
    # =========================================================================
    svg.panel(50, 2342, 1260, 580, "Vision Tower + Aligner (0.49B · 原生多模态)", color=C_SKY_DARK, subtitle="32L ViT + 3×3 Spatial Downsample → 5120")

    vx1 = 212
    svg.box(vx1, 2844, 230, 50, "Image Input", sublabel="min_pixels = 295,936 · 动态分辨率", fill=C_SKY_LIGHT, text_color=C_SKY_DARK, stroke=C_SKY, size=13.5, sub_size=11)
    svg.arrow([(vx1, 2819), (vx1, 2784)], color_key="sky")
    svg.box(vx1, 2754, 230, 52, "Patchify (14×14×3)", sublabel="[N_p, 588]", fill=C_SKY, stroke=C_SKY_DARK, size=13.5, sub_size=11.5)
    svg.arrow([(vx1, 2728), (vx1, 2692)], color_key="sky")
    svg.trap(vx1, 2660, 190, 236, 56, "PatchEmbed Linear", sublabel="588 → 1024 (+ bias)", fill=C_SKY_DARK, stroke="#0C4A6E", size=13.5, sub_size=11.2)

    # Clean routing OUTSIDE the ViT dashed box (x=378) into the bottom of RMSNorm(norm1)!
    svg.arrow([(vx1, 2632), (vx1, 2580), (378, 2580), (378, 2864), (618, 2864), (618, 2849)], color_key="sky")
    svg.text(294, 2564, "[n_h·n_w, 1024]", size=12, mono=True, fill=C_INK)

    # Vision specs card
    svg.rect(78, 2400, 282, 132, fill=C_SKY_LIGHT, stroke=C_SKY, stroke_width=1.2, rx=8)
    svg.text_lines(
        92,
        2410,
        [
            "• 32 层 ViT · dim = 1024",
            "• 16 注意力头 (d = 64)",
            "• 2D RoPE (θ = 10,000, 半维配对)",
            "• SwiGLU intermediate = 2816",
            "• downsample_ratio = 3 (3×3)",
            "• max_image_tokens = 1024",
        ],
        size=11.8,
        line_h=19.8,
        fill=C_INK,
    )

    # Middle: 32x ViT Block (x=402..834, y=2398..2892)
    svg.rect(402, 2398, 430, 494, fill="#F8FDFF", stroke=C_SKY_DARK, stroke_width=1.8, rx=12, dash="8,5")
    svg.text(420, 2420, "32× ViT Block (全双向注意力)", size=15, weight="700", fill=C_SKY_DARK, anchor="start")

    vcx = 618
    svg.box(vcx, 2828, 290, 38, "RMSNorm (norm1 · 1024)", fill=C_SKY, stroke=C_SKY_DARK, size=13)
    svg.arrow([(vcx, 2809), (vcx, 2780)], color_key="sky")
    svg.box(vcx, 2748, 340, 56, "Bidirectional MHA (wqkv + wo)", sublabel="16 heads × 64 · 2D-RoPE (θ=10K) · FA3", fill=C_SKY_DARK, stroke="#0C4A6E", size=13.5, sub_size=11.2)
    svg.arrow([(vcx, 2720), (vcx, 2694)], color_key="sky")
    svg.circle_op(vcx, 2678, 14, "plus", fill=C_SKY, stroke=C_SKY_DARK)
    svg.arrow([(432, 2864), (432, 2678), (vcx - 16, 2678)], color_key="sky", dash="5,4")
    svg.arrow([(vcx, 2664), (vcx, 2634)], color_key="sky")

    svg.box(vcx, 2612, 290, 38, "RMSNorm (norm2 · 1024)", fill=C_SKY, stroke=C_SKY_DARK, size=13)
    svg.arrow([(vcx, 2593), (vcx, 2564)], color_key="sky")
    svg.box(vcx, 2534, 340, 52, "SwiGLU MLP (w1, w2)", sublabel="1024 → 2×2816 (SiLU⊙) → 1024", fill=C_GREEN, stroke=C_GREEN_DARK, size=13.5, sub_size=11.2)
    svg.arrow([(vcx, 2508), (vcx, 2482)], color_key="sky")
    svg.circle_op(vcx, 2466, 14, "plus", fill=C_SKY, stroke=C_SKY_DARK)
    svg.arrow([(432, 2650), (432, 2466), (vcx - 16, 2466)], color_key="sky", dash="5,4")

    # Exit from ViT top -> Right Aligner
    svg.arrow([(vcx, 2452), (vcx, 2434), (860, 2434), (860, 2844), (924, 2844)], color_key="sky")

    # Right: vision.norm + 3x3 Unfold + Aligner MLP + Markers
    ax = 1098
    svg.box(ax, 2844, 342, 44, "vision.norm (RMSNorm 1024)", fill=C_SKY, stroke=C_SKY_DARK, size=13)
    svg.arrow([(ax, 2822), (ax, 2792)], color_key="sky")

    svg.box(ax, 2760, 342, 56, "3×3 Spatial Unfold (r = 3)", sublabel="[n_h, n_w, 1024] → [⌈n_h/3⌉·⌈n_w/3⌉, 9216]", fill="#E0F2FE", text_color=C_SKY_DARK, stroke=C_SKY_DARK, size=13, sub_size=10.8)
    svg.arrow([(ax, 2732), (ax, 2702)], color_key="sky")

    svg.trap(ax, 2674, 326, 268, 48, "Aligner w1 (Linear + bias)", sublabel="9216 → 5120", fill=C_SKY_DARK, stroke="#0C4A6E", size=13, sub_size=11)
    svg.arrow([(ax, 2650), (ax, 2626)], color_key="sky")
    svg.box(ax, 2606, 170, 34, "GELU", fill=C_SKY, stroke=C_SKY_DARK, size=13.5)
    svg.arrow([(ax, 2589), (ax, 2564)], color_key="sky")
    svg.box(ax, 2536, 286, 48, "Aligner w2 (Linear + bias)", sublabel="5120 → 5120", fill=C_SKY_DARK, stroke="#0C4A6E", size=13, sub_size=11)
    svg.arrow([(ax, 2512), (ax, 2480)], color_key="sky")

    svg.box(
        ax,
        2442,
        362,
        66,
        "Pack Multimodal Span (≤1024 tokens)",
        sublabel="[image_start] + rows + [image_newline] + [image_end]",
        fill="#BAE6FD",
        text_color="#0C4A6E",
        stroke=C_SKY_DARK,
        size=12.8,
        sub_size=10.5,
    )


def draw_center_backbone(svg: SVGBuilder) -> None:
    cx = 1740

    # =========================================================================
    # Top: DSpark MTP summary box + Logits + LM-Head + RMSNorm + mhc_combine
    # =========================================================================
    svg.box(
        cx,
        364,
        360,
        94,
        "DSpark Speculative Draft (mtp.0–2)",
        sublabel="3L SWA+MoE(128E top-3) · block=5 · Markov+Conf",
        fill=C_VIOLET_LIGHT,
        text_color=C_VIOLET_DARK,
        stroke=C_VIOLET_DARK,
        stroke_width=2.0,
        dash="8,5",
        size=15,
        sub_size=12,
        rx=12,
    )
    svg.arrow([(cx, 454), (cx, 413)], color_key="violet")

    svg.box(cx, 476, 270, 42, "Next-Token Logits [seq, 129280]", fill="#F1F5F9", text_color=C_INK, stroke="#64748B", size=13.5, rx=10)
    svg.arrow([(cx, 532), (cx, 499)], color_key="blk")

    svg.box(cx, 558, 290, 48, "LM-Head (head.weight)", sublabel="5120 → 129,280 (untied)", fill=C_BLUE, stroke=C_BLUE_DARK, size=14.5, sub_size=11.5)
    svg.arrow([(cx, 616), (cx, 584)], color_key="blk")

    svg.box(cx, 640, 250, 44, "Final RMSNorm", sublabel="eps = 1e-20", fill=C_BLUE, stroke=C_BLUE_DARK, size=14, sub_size=11)
    svg.arrow([(cx, 694), (cx, 664)], color_key="blk")

    svg.box(
        cx,
        720,
        320,
        48,
        "mhc_combine (last pre)",
        sublabel="[seq, 4, 5120] → [seq, 5120] (无独立 hc_head)",
        fill=C_INDIGO_DARK,
        stroke="#312E81",
        size=13.8,
        sub_size=11,
    )
    svg.arrow([(cx, 782), (cx, 746)], color_key="indigo")

    # =========================================================================
    # Decoder Block (20x, L20..L39, y=782..1568)
    # =========================================================================
    svg.rect(1445, 782, 590, 786, fill="#FAFCFF", stroke=C_BLUE_DARK, stroke_width=2.2, rx=14, dash="10,6", layer="panels")
    svg.text(1465, 806, "Decoder · 20× (L20–L39)", size=17, weight="800", fill=C_BLUE_DARK, anchor="start")
    svg.text(2015, 806, "~16B Active Decode", size=13.5, weight="700", fill=C_BLUE_DARK, anchor="end")

    # DSpark Target Hidden Tap badge
    svg.box(
        cx,
        852,
        450,
        42,
        "DSpark Target Tap @ L37, L38, L39",
        sublabel="mhc_stream_mean([T, 4, 5120]) × 3 → [T, 15360]",
        fill=C_VIOLET_LIGHT,
        text_color=C_VIOLET_DARK,
        stroke=C_VIOLET,
        stroke_width=1.6,
        size=13,
        sub_size=11,
    )
    svg.arrow([(cx, 904), (cx, 875)], color_key="indigo")

    # Decoder MoE sub-block
    svg.box(cx, 924, 230, 36, "hc_post (4×4 mix + post)", fill=C_INDIGO, stroke=C_INDIGO_DARK, size=13)
    svg.arrow([(cx, 976), (cx, 944)], color_key="blk")
    svg.box(
        cx,
        1012,
        300,
        66,
        "MoE (384+1 · Top-6)",
        sublabel="Clamped SwiGLU (limit=10) · MXFP4",
        fill="#FFFFFF",
        text_color=C_BLUE_DARK,
        stroke=C_BLUE_DARK,
        stroke_width=2.0,
        dash="7,4",
        size=15,
        sub_size=11.5,
    )
    svg.arrow([(cx, 1076), (cx, 1047)], color_key="blk")
    svg.box(cx, 1096, 220, 36, "RMSNorm (ffn_norm)", fill=C_BLUE, stroke=C_BLUE_DARK, size=13)
    svg.arrow([(cx, 1142), (cx, 1116)], color_key="blk")
    svg.box(cx, 1160, 230, 34, "hc_pre (Single-Pass)", fill=C_INDIGO, stroke=C_INDIGO_DARK, size=13)
    svg.arrow([(cx + 115, 1160), (1970, 1160), (1970, 924), (cx + 118, 924)], color_key="indigo", width=1.8)

    svg.arrow([(cx, 1218), (cx, 1179)], color_key="indigo")

    # Decoder CSA2-C1 sub-block (explicitly spaced 3-line box!)
    svg.box(cx, 1236, 230, 36, "hc_post (4×4 mix + post)", fill=C_INDIGO, stroke=C_INDIGO_DARK, size=13)
    svg.arrow([(cx, 1288), (cx, 1256)], color_key="blk")

    svg.rect(cx - 210, 1290, 420, 108, fill=C_ORANGE_LIGHT, stroke=C_ORANGE_DARK, stroke_width=2.0, rx=10, dash="7,4")
    svg.text(cx, 1316, "CSA2-C1 (compress_ratio = 1)", size=15.5, weight="700", fill=C_ORANGE_DARK)
    svg.text(cx, 1346, "L20: Full KV+Idx-K+2048 Cand · L24/28/32/36: Reindex · 其余 Reuse", size=11.3, weight="600", fill=C_ORANGE_DARK)
    svg.text(cx, 1374, "所有 20 层共享 L20 C1 KV + 各层独立 SWA-128", size=11.4, weight="700", fill="#9A3412")

    svg.arrow([(cx, 1430), (cx, 1400)], color_key="blk")
    svg.box(cx, 1450, 220, 36, "RMSNorm (attn_norm)", fill=C_BLUE, stroke=C_BLUE_DARK, size=13)
    svg.arrow([(cx, 1496), (cx, 1470)], color_key="blk")
    svg.box(cx, 1516, 230, 36, "hc_pre (Single-Pass)", fill=C_INDIGO, stroke=C_INDIGO_DARK, size=13)
    svg.arrow([(cx + 115, 1516), (1970, 1516), (1970, 1236), (cx + 118, 1236)], color_key="indigo", width=1.8)

    # =========================================================================
    # CED Bounded Replay Divider (y=1584..1650)
    # =========================================================================
    svg.arrow([(cx, 1584), (cx, 1536)], color_key="red", width=2.2)
    svg.box(
        cx,
        1618,
        590,
        62,
        "CED Bounded Replay 切分点 (L19 → L20)",
        sublabel="Prefill: L20 为全量 Prompt 写 C1 KV/Idx-K 后，主体提前结束 (~8B)；仅末尾 128 Token & Decode 进 Decoder",
        fill="#FEF2F2",
        text_color=C_RED_DARK,
        stroke=C_RED,
        stroke_width=2.0,
        dash="6,4",
        size=14,
        sub_size=10.8,
        rx=10,
    )
    svg.arrow([(cx, 1678), (cx, 1651)], color_key="red", width=2.2)

    # =========================================================================
    # Causal Encoder Block (20x, L0..L19, y=1678..2634)
    # =========================================================================
    svg.rect(1445, 1678, 590, 956, fill="#FFFDF7", stroke=C_ORANGE_DARK, stroke_width=2.2, rx=14, dash="10,6", layer="panels")
    svg.text(1465, 1702, "Causal Encoder · 20× (L0–L19)", size=17, weight="800", fill=C_ORANGE_DARK, anchor="start")
    svg.text(2015, 1702, "~8B Active Prefill", size=13.5, weight="700", fill=C_ORANGE_DARK, anchor="end")

    # Upper sub-block: 18x (3x6 groups, L2..L19)  y=1726..2198
    svg.rect(1472, 1726, 536, 472, fill="#FFFFFF", stroke=C_ORANGE, stroke_width=1.6, rx=10, dash="6,4")
    svg.text(1488, 1746, "18× (3 组 × 6 层: L2–7, L8–13, L14–19)", size=13.5, weight="700", fill=C_ORANGE_DARK, anchor="start")

    svg.box(cx, 1788, 224, 34, "hc_post", fill=C_INDIGO, stroke=C_INDIGO_DARK, size=13)
    svg.arrow([(cx, 1830), (cx, 1807)], color_key="blk")
    svg.box(cx, 1858, 286, 52, "MoE (384+1 · Top-6)", sublabel="MXFP4 Routed + FP8 Shared", fill="#FFFFFF", text_color=C_BLUE_DARK, stroke=C_BLUE_DARK, stroke_width=1.8, dash="6,4", size=14, sub_size=11)
    svg.arrow([(cx, 1910), (cx, 1886)], color_key="blk")
    svg.box(cx, 1930, 224, 34, "hc_pre → RMSNorm", fill=C_BLUE, stroke=C_BLUE_DARK, size=12.8)
    svg.arrow([(cx + 112, 1930), (1956, 1930), (1956, 1788), (cx + 114, 1788)], color_key="indigo", width=1.6)

    svg.arrow([(cx, 1978), (cx, 1949)], color_key="indigo")
    svg.box(cx, 1996, 224, 34, "hc_post", fill=C_INDIGO, stroke=C_INDIGO_DARK, size=13)
    svg.arrow([(cx, 2036), (cx, 2015)], color_key="blk")
    svg.box(
        cx,
        2068,
        366,
        60,
        "CSA2-C2 (compress_ratio = 2)",
        sublabel="Full @ L2, L8, L14 · Reuse ×5 · SWA-128",
        fill=C_ORANGE_LIGHT,
        text_color=C_ORANGE_DARK,
        stroke=C_ORANGE_DARK,
        stroke_width=1.8,
        dash="6,4",
        size=14.2,
        sub_size=11.2,
    )
    svg.arrow([(cx, 2122), (cx, 2100)], color_key="blk")
    svg.box(cx, 2142, 224, 34, "hc_pre → RMSNorm", fill=C_BLUE, stroke=C_BLUE_DARK, size=12.8)
    svg.arrow([(cx + 112, 2142), (1956, 2142), (1956, 1996), (cx + 114, 1996)], color_key="indigo", width=1.6)

    # Engram L14 badge cleanly in bottom-left whitespace
    svg.badge(1576, 2178, "+ Engram @ L14 (98.3B)", fill=C_ROSE, w=176, h=25, size=11.5)

    svg.arrow([(cx, 2224), (cx, 2161)], color_key="indigo")

    # Lower sub-block: 2x SWA-only (L0..L1)  y=2224..2614
    svg.rect(1472, 2224, 536, 390, fill="#FFFFFF", stroke="#64748B", stroke_width=1.6, rx=10, dash="6,4")
    svg.text(1488, 2244, "2× (L0–L1 · 纯滑窗注意力 SWA)", size=13.5, weight="700", fill=C_MUTED, anchor="start")

    svg.box(cx, 2282, 224, 34, "hc_post", fill=C_INDIGO, stroke=C_INDIGO_DARK, size=13)
    svg.arrow([(cx, 2320), (cx, 2301)], color_key="blk")
    svg.box(cx, 2344, 286, 44, "MoE (384+1 · Top-6)", fill="#FFFFFF", text_color=C_BLUE_DARK, stroke=C_BLUE_DARK, stroke_width=1.8, dash="6,4", size=14)
    svg.arrow([(cx, 2386), (cx, 2368)], color_key="blk")
    svg.box(cx, 2404, 224, 32, "hc_pre → RMSNorm", fill=C_BLUE, stroke=C_BLUE_DARK, size=12.8)
    svg.arrow([(cx + 112, 2404), (1956, 2404), (1956, 2282), (cx + 114, 2282)], color_key="indigo", width=1.6)

    svg.arrow([(cx, 2444), (cx, 2422)], color_key="indigo")
    svg.box(cx, 2462, 224, 32, "hc_post", fill=C_INDIGO, stroke=C_INDIGO_DARK, size=13)
    svg.arrow([(cx, 2496), (cx, 2480)], color_key="blk")
    svg.box(cx, 2518, 350, 42, "SWA-Only MQA (r=0, win=128, θ=10K)", fill="#F1F5F9", text_color=C_INK, stroke="#475569", stroke_width=1.8, size=13.2)
    svg.arrow([(cx, 2562), (cx, 2541)], color_key="blk")
    svg.box(cx, 2576, 212, 32, "hc_pre → RMSNorm", fill=C_BLUE, stroke=C_BLUE_DARK, size=12.5)
    svg.arrow([(cx + 106, 2576), (1956, 2576), (1956, 2462), (cx + 114, 2462)], color_key="indigo", width=1.6)

    # Engram L1 badge cleanly to the left of hc_pre->RMSNorm (ZERO overlap!)
    svg.badge(1556, 2594, "+ Engram @ L1 (98B)", fill=C_ROSE, w=150, h=24, size=11)

    # =========================================================================
    # Bottom Inputs (y=2656..2922)
    # =========================================================================
    svg.arrow([(cx, 2668), (cx, 2594)], color_key="indigo")
    svg.box(cx, 2690, 320, 42, "hc_broadcast (×4 streams)", sublabel="[seq, 5120] → [seq, 4, 5120]", fill=C_INDIGO, stroke=C_INDIGO_DARK, size=13.5, sub_size=11)
    svg.arrow([(cx, 2738), (cx, 2713)], color_key="blk")

    svg.box(cx, 2764, 300, 48, "Embedding (embed.weight)", sublabel="vocab 129,280 × 5120 (+ 视觉 Token 拼接)", fill=C_BLUE, stroke=C_BLUE_DARK, size=14, sub_size=11)
    svg.arrow([(cx, 2818), (cx, 2790)], color_key="blk")

    svg.box(cx, 2844, 230, 44, "Tokenizer", fill=C_BLUE, stroke=C_BLUE_DARK, size=14.5)
    svg.arrow([(cx, 2892), (cx, 2868)], color_key="blk")
    svg.text(cx, 2908, "Text Input (context_length = 1M)", size=14.5, weight="700", fill=C_INK)

    # Vision -> Embedding connector cleanly routed in center gutter
    svg.arrow([(1279, 2442), (1352, 2442), (1352, 2764), (1588, 2764)], color_key="sky", width=2.2)
    svg.text(1470, 2748, "Image Spans [N_img, 5120]", size=11.5, weight="700", fill=C_SKY_DARK)

    # =========================================================================
    # Clean Dashed Zoom-out Connectors (Center Spine -> Left & Right Panels)
    # =========================================================================
    svg.zoom_line(1590, 979, 1310, 876, color=C_BLUE_DARK)
    svg.zoom_line(1590, 1045, 1310, 1216, color=C_BLUE_DARK)

    svg.zoom_line(1625, 1236, 1310, 1246, color=C_INDIGO_DARK)
    svg.zoom_line(1625, 1516, 1310, 1702, color=C_INDIGO_DARK)

    svg.zoom_line(1488, 2178, 1310, 1732, color=C_ROSE_DARK)
    svg.zoom_line(1481, 2594, 1310, 2312, color=C_ROSE_DARK)

    svg.zoom_line(1920, 317, 2170, 306, color=C_VIOLET_DARK)
    svg.zoom_line(1920, 411, 2170, 886, color=C_VIOLET_DARK)
    svg.arrow([(1965, 852), (2135, 852), (2135, 766), (2216, 766)], color_key="violet", width=2.2)

    svg.zoom_line(1950, 1344, 2170, 1506, color=C_ORANGE_DARK)
    svg.zoom_line(1923, 2068, 2170, 2266, color=C_ORANGE_DARK)


def draw_right_column(svg: SVGBuilder) -> None:
    # =========================================================================
    # R1: DSpark Speculative Decoding (mtp.0-2, 14.2B, block_size=5)  (x=2170..3550, y=306..886)
    # =========================================================================
    svg.panel(2170, 306, 1380, 580, "DSpark Speculative Decoding (mtp.0–2 · 14.2B)", color=C_VIOLET_DARK, subtitle="block_size=5 · 128E top-3 · Markov + Confidence")

    # Left sub-column (x=2196..2566): Target Context KV Injection
    svg.rect(2196, 354, 366, 510, fill=C_VIOLET_LIGHT, stroke=C_VIOLET, stroke_width=1.4, rx=10, dash="6,4")
    svg.text(2212, 374, "① 主干上下文注入 (Target KV)", size=13.5, weight="700", fill=C_VIOLET_DARK, anchor="start")

    tx = 2379
    svg.box(tx, 766, 316, 56, "Concat [h₃₇, h₃₈, h₃₉]", sublabel="每层取 4 流均值 mhc_stream_mean → [T, 15360]", fill="#EDE9FE", text_color=C_VIOLET_DARK, stroke=C_VIOLET_DARK, size=13, sub_size=10.5)
    svg.arrow([(tx, 738), (tx, 700)], color_key="violet")

    svg.trap(tx, 670, 290, 228, 52, "Linear (mtp.0.main_proj)", sublabel="15360 → 5120 (FP8)", fill=C_VIOLET, stroke=C_VIOLET_DARK, size=13, sub_size=11)
    svg.arrow([(tx, 644), (tx, 608)], color_key="violet")

    svg.box(tx, 584, 240, 42, "RMSNorm (main_norm)", fill=C_VIOLET, stroke=C_VIOLET_DARK, size=13)
    svg.arrow([(tx, 563), (tx, 526)], color_key="violet")
    svg.text(tx + 74, 544, "main_x [T, 5120]", size=11.8, mono=True, fill=C_INK)

    # Explicitly spaced 3-line Commit Context KV box (ZERO text overlap!)
    svg.rect(tx - 160, 424, 320, 96, fill="#FFFFFF", stroke=C_VIOLET_DARK, stroke_width=1.6, rx=10)
    svg.text(tx, 446, "Commit Context KV → mtp.0 / 1 / 2", size=13.2, weight="700", fill=C_VIOLET_DARK)
    svg.text(tx, 472, "经各 stage 的 wkv → kv_norm → RoPE(θ=10K)", size=11, weight="500", fill=C_INK)
    svg.text(tx, 498, "写入各自独立的 SWA-128 KV Ring", size=11.2, weight="700", fill=C_RED_DARK)

    # Clean arrow from Commit KV directly into DSpark Attention (SWA-128, r=0) at y=598!
    svg.arrow([(2539, 472), (2588, 472), (2588, 598), (2632, 598)], color_key="violet", width=2.2)

    svg.text_lines(
        2210,
        810,
        [
            "• dspark_target_layer_ids = [37, 38, 39]",
            "• dspark_noise_token_id = 128799",
        ],
        size=11.5,
        line_h=18,
        mono=True,
        fill=C_MUTED,
    )

    # Middle sub-column (x=2614..3006): 3x DSpark Draft Backbone
    dx = 2810
    svg.box(dx, 832, 370, 44, "Draft Block: [anchor, noise×4] (5 tokens)", sublabel="noise_id = 128799 · 并行草稿块", fill="#EDE9FE", text_color=C_VIOLET_DARK, stroke=C_VIOLET_DARK, size=12.8, sub_size=10.8)
    svg.arrow([(dx, 810), (dx, 784)], color_key="blk")

    svg.box(dx, 760, 340, 40, "Shared Embedding → hc_broadcast (×4)", fill=C_BLUE, stroke=C_BLUE_DARK, size=13)
    svg.arrow([(dx, 740), (dx, 714)], color_key="indigo")

    svg.rect(2614, 422, 392, 290, fill="#FFFFFF", stroke=C_VIOLET_DARK, stroke_width=2.0, rx=12, dash="8,5")
    svg.text(2632, 444, "3× DSpark Stage (mtp.0, mtp.1, mtp.2)", size=14, weight="700", fill=C_VIOLET_DARK, anchor="start")

    svg.box(dx, 666, 336, 40, "hc_pre → RMSNorm (Single-Pass mHC)", fill=C_INDIGO, stroke=C_INDIGO_DARK, size=12.5)
    svg.arrow([(dx, 646), (dx, 626)], color_key="blk")
    svg.box(dx, 598, 350, 50, "DSpark Attention (SWA-128, r=0)", sublabel="64Q : 1KV · 注意到注入的 Target KV + 块内 KV", fill=C_ORANGE, stroke=C_ORANGE_DARK, size=13, sub_size=10.5)
    svg.arrow([(dx, 573), (dx, 552)], color_key="blk")
    svg.box(dx, 524, 350, 50, "Draft MoE (128 Routed Top-3 + 1 Shared)", sublabel="MXFP4 · SwiGLU limit=10 · 每层 4.6B", fill=C_GREEN, stroke=C_GREEN_DARK, size=12.8, sub_size=10.5)
    svg.arrow([(dx, 499), (dx, 478)], color_key="blk")
    svg.box(dx, 466, 220, 24, "hc_post", fill=C_INDIGO, stroke=C_INDIGO_DARK, size=11.5)

    svg.arrow([(dx, 422), (dx, 400)], color_key="indigo")
    svg.box(dx, 376, 356, 42, "mhc_combine → RMSNorm (mtp.2.norm)", sublabel="h_draft [5, 5120]", fill=C_VIOLET_DARK, stroke="#4C1D95", size=12.8, sub_size=11)

    # Right sub-column (x=3056..3528): Shared LM-Head + Markov Head + Confidence Head
    svg.arrow([(2988, 376), (3056, 376)], color_key="violet")
    svg.arrow([(3030, 376), (3030, 752), (3066, 752)], color_key="violet")

    svg.box(3290, 376, 444, 48, "Shared LM-Head (5120 → 129,280)", sublabel="并行输出 base_logits [5, 129280]", fill=C_BLUE, stroke=C_BLUE_DARK, size=13.5, sub_size=11)

    # Markov Head box (y=454..654) — cleanly separated green vertical line from formula!
    svg.rect(3066, 454, 456, 198, fill="#ECFDF5", stroke="#059669", stroke_width=1.8, rx=10)
    svg.text(3082, 474, "② Markov Head (rank = 256 · 消除并行后缀衰减)", size=13.2, weight="700", fill="#065F46", anchor="start")

    svg.box(3164, 530, 168, 46, "prev_tok", sublabel="Embed(129280→256)", fill=C_EMERALD, stroke="#047857", size=12.5, sub_size=10.5)
    svg.arrow([(3248, 530), (3292, 530)], color_key="emerald")
    svg.box(3396, 530, 200, 46, "Linear (markov_w2)", sublabel="256 → 129,280 (bias)", fill="#059669", stroke="#065F46", size=12.5, sub_size=10.5)
    svg.circle_op(3424, 600, 15, "plus", fill="#059669", stroke="#065F46")
    svg.arrow([(3424, 553), (3424, 583)], color_key="emerald")
    svg.arrow([(3496, 400), (3496, 600), (3441, 600)], color_key="blue")

    # Route markov_emb down on the FAR LEFT (x=3100) so it NEVER crosses any formula!
    svg.arrow([(3100, 553), (3100, 686)], color_key="emerald")
    svg.text(3184, 670, "markov_emb [5, 256]", size=11.2, mono=True, fill="#047857")

    svg.text(3264, 600, "logits_t = base_t + bias(tok_{t-1})", size=11.8, mono=True, weight="600", fill="#065F46")
    svg.text(3294, 632, "→ 顺序采样草稿块 tok₁..tok₅", size=12, weight="700", fill="#047857")

    # Confidence Head box (y=686..864)
    svg.rect(3066, 686, 456, 178, fill="#FFFBEB", stroke=C_ORANGE_DARK, stroke_width=1.8, rx=10)
    svg.text(3082, 706, "③ Confidence Head (置信度动态调度验证)", size=13.2, weight="700", fill="#92400E", anchor="start")
    svg.box(3294, 752, 412, 42, "Concat [h_draft (5120), markov_emb (256)] → [5, 5376]", fill="#FEF3C7", text_color="#92400E", stroke=C_AMBER, size=12)
    svg.arrow([(3294, 773), (3294, 794)], color_key="amber")
    svg.box(3294, 822, 412, 46, "Linear (proj: 5376→1) → σ(logit / τ_sts)", sublabel="输出每 token 存活概率 p ∈ [0,1]，低置信度提前截断验证", fill=C_AMBER, text_color="#FFFFFF", stroke="#B45309", size=12.8, sub_size=10.8)

    # =========================================================================
    # R2: Hierarchical Sparse Indexer (FP4)  (x=2170..3550, y=916..1476)
    # =========================================================================
    svg.panel(2170, 916, 1380, 560, "Hierarchical Sparse Indexer (FP4 · 两级分层稀疏检索)", color=C_GREEN_DARK, subtitle="仅 KV 源层拥有 Index-K · 候选块剪枝实现 O(1) 复杂度")

    wx = 2320
    qx = 2630
    kx = 2915

    # Bottom inputs comfortably INSIDE panel bottom border (y=1476)
    svg.arrow([(wx, 1442), (wx, 1410)], color_key="green")
    svg.text(wx, 1456, "[1, 5120] (x)", size=12, mono=True, fill=C_INK)

    svg.arrow([(qx, 1442), (qx, 1410)], color_key="red")
    svg.text(qx, 1456, "[1, 1280] (q_lora)", size=12, mono=True, fill=C_RED_DARK)

    svg.trap(wx, 1384, 224, 174, 46, "Linear (weights_proj)", sublabel="5120 → 32 (BF16)", fill=C_GREEN, stroke=C_GREEN_DARK, size=13, sub_size=10.8)
    svg.trap(qx, 1384, 186, 232, 46, "Linear (indexer.wq_b)", sublabel="1280 → 32×128 (FP8)", fill=C_GREEN, stroke=C_GREEN_DARK, size=13, sub_size=10.8)

    # Index-K Cache
    svg.box(kx, 1384, 198, 54, "Index-K Cache (FP4)", sublabel="[T_c, 128] · 68 B/slot", fill=C_RED, stroke=C_RED_DARK, size=13, sub_size=11)
    svg.kv_slots(kx, 1326, bar_w=13, bar_h=30, left_n=2, right_n=2)
    svg.text(kx, 1296, "来自 L2/8/14/20 wk+k_norm", size=11, fill=C_MUTED)
    svg.arrow([(kx, 1357), (kx, 1344)], color_key="red", head_end=False)
    svg.arrow([(kx, 1284), (kx, 1188), (qx + 96, 1188)], color_key="red")

    # Q branch upward
    svg.arrow([(qx, 1361), (qx, 1338)], color_key="blk")
    svg.text(qx + 84, 1349, "[1, 32, 128]", size=11.5, mono=True, fill=C_INK)

    svg.box(qx, 1318, 186, 34, "RoPE (tail 64)", fill=C_GREEN, stroke=C_GREEN_DARK, size=13.5)
    svg.arrow([(qx, 1301), (qx, 1278)], color_key="blk")

    # Clean 2-line FP4 Quant box (zero text collision!)
    svg.box(qx, 1252, 232, 46, "FP4 Quant (E2M1/32 UE8M0)", sublabel="无 Hadamard Rotate (较 V4 Pro 简化)", fill=C_GREEN, stroke=C_GREEN_DARK, size=12.5, sub_size=10.5)
    svg.arrow([(qx, 1229), (qx, 1208)], color_key="blk")

    svg.box(qx, 1188, 188, 36, "Matmul (Q_fp4 · K_fp4ᵀ)", fill=C_GREEN_DARK, stroke="#365314", size=13)
    svg.arrow([(qx, 1170), (qx, 1142)], color_key="blk")
    svg.text(qx + 72, 1155, "[1, 32, T_c]", size=11.5, mono=True, fill=C_INK)

    svg.box(qx, 1122, 168, 36, "ReLU", fill=C_GREEN, stroke=C_GREEN_DARK, size=14)
    svg.arrow([(qx, 1104), (qx, 1074)], color_key="blk")

    svg.arrow([(wx, 1361), (wx, 1078)], color_key="green")
    svg.text(wx - 42, 1260, "[1, 32]", size=11.5, mono=True, fill=C_INK)
    svg.box(wx, 1056, 200, 40, "Scale (1/√128 · 1/√32)", fill=C_GREEN, stroke=C_GREEN_DARK, size=12.8)
    svg.arrow([(wx + 100, 1056), (qx - 18, 1056)], color_key="green")

    svg.circle_op(qx, 1056, 16, "mul", fill=C_GREEN, stroke=C_GREEN_DARK)
    svg.arrow([(qx, 1040), (qx, 1012)], color_key="blk")

    svg.box(qx, 990, 180, 38, "Sum (32 heads)", fill=C_GREEN, stroke=C_GREEN_DARK, size=13.5)
    svg.arrow([(qx + 90, 990), (3036, 990)], color_key="green", width=2.2)
    svg.text(2860, 976, "scores [1, T_c] (FP32)", size=12.2, mono=True, weight="600", fill=C_GREEN_DARK)

    # Right sub-panel: Two-Stage Hierarchical Candidate Selection
    svg.rect(3038, 958, 488, 496, fill=C_GREEN_LIGHT, stroke=C_GREEN_DARK, stroke_width=1.6, rx=10)
    svg.text(3056, 980, "两级分层 Top-K 选择机制", size=14.5, weight="700", fill=C_GREEN_DARK, anchor="start")

    svg.rect(3056, 1004, 452, 168, fill="#FFFFFF", stroke=C_ORANGE_DARK, stroke_width=1.6, rx=9)
    svg.text(3072, 1024, "Stage 1 · Full Indexer (L2, L8, L14, L20)", size=13.2, weight="700", fill=C_ORANGE_DARK, anchor="start")
    svg.box(3164, 1066, 188, 36, "Causal Mask → TopK(512)", fill=C_ORANGE, stroke=C_ORANGE_DARK, size=12.2)
    svg.box(3386, 1066, 198, 36, "index_topk [1, 512]", fill="#FEF3C7", text_color="#92400E", stroke=C_ORANGE_DARK, size=12)
    svg.arrow([(3258, 1066), (3285, 1066)], color_key="orange")

    svg.box(
        3282,
        1130,
        420,
        58,
        "L20 额外发布候选块 (candidate_source_layer_id = 20)",
        sublabel="AmaxPool(block=8) → TopK_blocks(2048) → ≤16,384 候选 Token",
        fill="#FEE2E2",
        text_color="#991B1B",
        stroke="#DC2626",
        stroke_width=1.6,
        size=12.2,
        sub_size=11,
    )

    svg.arrow([(3282, 1159), (3282, 1198)], color_key="red", width=2.0)
    svg.text(3388, 1180, "candidate_blocks [2048]", size=11.5, mono=True, weight="700", fill=C_RED_DARK)

    svg.rect(3056, 1200, 452, 122, fill="#FFFFFF", stroke=C_GREEN_DARK, stroke_width=1.6, rx=9)
    svg.text(3072, 1220, "Stage 2 · Reindex Layers (L24, L28, L32, L36)", size=13.2, weight="700", fill=C_GREEN_DARK, anchor="start")
    svg.text_lines(
        3072,
        1238,
        [
            "• 复用 L20 的 Index-K Cache（自身无 wk / k_norm）",
            "• 仅在 L20 发布的 2048 候选块 (≤16K) 内打分并选 TopK(512)",
            "• 长上下文过滤 ~98% 无关列，检索开销与总长度解耦 (O(1))",
        ],
        size=11.8,
        line_h=22,
        fill=C_INK,
    )

    svg.text_lines(
        3056,
        1336,
        [
            "index_n_heads = 32   |  index_head_dim = 128",
            "rope_head_dim = 64   |  index_topk = 512",
            "candidate_block_size = 8  |  candidate_topk_blocks = 2048",
            "kv_source = [2,8,14,20] | index_source = [2,8,14,20,24,28,32,36]",
            "TP 全复制 (Replicated): 每张卡算全 32 头，免跨卡 TopK 通信",
        ],
        size=11.6,
        line_h=21,
        mono=True,
        fill=C_INK,
    )

    # =========================================================================
    # R3: CSA2 (Compressed Sparse Attention 2 · MQALayer)  (x=2170..3550, y=1506..2266)
    # =========================================================================
    svg.panel(2170, 1506, 1380, 760, "CSA2 (Compressed Sparse Attention 2 · MQALayer)", color=C_ORANGE_DARK, subtitle="Full / Reindex / Reuse 三静态模式 · 64Q : 1KV")

    svg.rect(2192, 1556, 330, 186, fill=C_ORANGE_LIGHT, stroke=C_ORANGE, stroke_width=1.3, rx=9)
    svg.text_lines(
        2204,
        1566,
        [
            "hidden_size = 5120   | head_dim = 512",
            "heads = 64 (MQA)     | kv_heads = 1",
            "rope_head_dim = 64   | q_lora_rank = 1280",
            "o_groups = 8         | o_lora_rank = 1024",
            "index_topk = 512     | window_size = 128",
            "q_head_norm = False (相对 V4 Pro 移除)",
            "RoPE: θ=10K (SWA) / θ=160K YaRN (CSA2)",
            "softmax_scale = 1 / √512",
        ],
        size=11.5,
        line_h=21,
        mono=True,
        fill=C_INK,
    )

    ox = 2890
    svg.arrow([(ox, 1548), (ox, 1514)], color_key="blk")
    svg.text(ox + 68, 1528, "[1, 5120]", size=12.5, mono=True, fill=C_INK)

    svg.trap(ox, 1570, 260, 210, 44, "Linear (wo_b · FP8)", sublabel="8192 → 5120", fill=C_ORANGE, stroke=C_ORANGE_DARK, size=13.5, sub_size=11)
    svg.arrow([(ox, 1614), (ox, 1594)], color_key="blk")
    svg.text(ox + 108, 1604, "[1, 8 × 1024 = 8192]", size=12, mono=True, fill=C_INK)

    svg.trap(ox, 1636, 304, 244, 44, "Grouped Linear (wo_a · 8 groups)", sublabel="8 × (4096 → 1024) · einsum(tgd,grd→tgr)", fill=C_ORANGE, stroke=C_ORANGE_DARK, size=13, sub_size=10.5)
    svg.arrow([(ox, 1678), (ox, 1660)], color_key="blk")
    svg.text(ox + 126, 1670, "view [1, 8, 4096]", size=11.8, mono=True, fill=C_INK)

    svg.box(ox, 1700, 224, 38, "RoPE (inverse, tail 64)", fill=C_ORANGE, stroke=C_ORANGE_DARK, size=13.5)
    svg.arrow([(ox, 1748), (ox, 1721)], color_key="blk")
    svg.text(ox + 76, 1734, "[1, 64, 512]", size=12, mono=True, fill=C_INK)

    svg.box(2862, 1770, 1296, 44, "Attention (MQA · 64 Q Heads : 1 KV Head · head_dim = 512)", fill=C_ORANGE, stroke=C_ORANGE_DARK, size=15.5)
    svg.text(2550, 1770, "KV", size=14.5, weight="800", fill="#991B1B")
    svg.text(3196, 1770, "Q", size=14.5, weight="800", fill="#991B1B")

    # Bottom Input Bus comfortably above panel border (y=2266)
    b1x = 2356
    b2x = 2766
    b3x = 3206

    svg.arrow([(2862, 2248), (2862, 2224)], color_key="blk", head_end=False)
    svg.text(2942, 2242, "[1, 5120] (x)", size=12.5, mono=True, fill=C_INK)
    svg.arrow([(2862, 2224), (b1x, 2224), (b1x, 2202)], color_key="blk")
    svg.arrow([(2862, 2224), (b2x, 2224), (b2x, 2202)], color_key="blk")
    svg.arrow([(2862, 2224), (b3x, 2224), (b3x, 2202)], color_key="blk")
    svg.arrow([(3206, 2224), (3444, 2224), (3444, 2078)], color_key="green")

    # Branch 1 (Left): Local Window KV
    svg.trap(b1x, 2180, 228, 178, 44, "Linear (wkv · 5120→512)", fill=C_ORANGE, stroke=C_ORANGE_DARK, size=13)
    svg.arrow([(b1x, 2158), (b1x, 2136)], color_key="blk")
    svg.box(b1x, 2116, 184, 36, "RMSNorm (kv_norm)", fill=C_ORANGE, stroke=C_ORANGE_DARK, size=13)

    svg.arrow([(b1x, 2098), (b1x, 2084)], color_key="blk", head_end=False)
    svg.arrow([(b1x, 2084), (b1x - 54, 2084), (b1x - 54, 2070)], color_key="blk")
    svg.arrow([(b1x, 2084), (b1x + 54, 2084), (b1x + 54, 2070)], color_key="blk")

    svg.box(b1x - 54, 2052, 96, 32, "RoPE [-64:]", fill=C_ORANGE, stroke=C_ORANGE_DARK, size=11.5)
    svg.box(b1x + 54, 2052, 96, 32, "Quant [:-64]", fill=C_ORANGE, stroke=C_ORANGE_DARK, size=11.5)

    svg.arrow([(b1x - 54, 2036), (b1x - 54, 2022), (b1x, 2022), (b1x, 2004)], color_key="blk")
    svg.arrow([(b1x + 54, 2036), (b1x + 54, 2022), (b1x, 2022)], color_key="blk", head_end=False)

    svg.box(b1x, 1968, 208, 42, "KV Cache (Window 128)", fill=C_RED, stroke=C_RED_DARK, size=13.2)
    svg.kv_slots(b1x, 1910, bar_w=13, bar_h=34, left_n=2, right_n=2)
    svg.text(b1x + 92, 1910, "[≤128, 512]", size=11.5, mono=True, fill=C_INK)
    svg.arrow([(b1x, 1947), (b1x, 1930)], color_key="blk", head_end=False)

    # Branch 2 (Center): Shared Compressed KV
    svg.trap(
        b2x,
        2168,
        304,
        244,
        66,
        "Token-Level Compressor",
        sublabel="C2 @ L2,8,14 (r=2) · C1 @ L20 (r=1)",
        fill=C_PERI,
        stroke=C_PERI_DARK,
        size=13.5,
        sub_size=11,
    )
    svg.arrow([(b2x, 2135), (b2x, 2096)], color_key="blk")
    svg.kv_slots(b2x, 2072, bar_w=13, bar_h=34, left_n=2, right_n=2)
    svg.text(b2x + 106, 2072, "[T_c, 512]", size=11.5, mono=True, fill=C_INK)
    svg.arrow([(b2x, 2055), (b2x, 2028)], color_key="blk")

    svg.box(b2x, 2002, 224, 46, "KV Cache (Compress · FP4)", sublabel="E2M1 + E4M3/16 · 288 B/slot", fill=C_RED, stroke=C_RED_DARK, size=12.8, sub_size=10.5)
    svg.arrow([(b2x, 1979), (b2x, 1952)], color_key="blk")

    svg.box(b2x, 1930, 184, 38, "Select (top-512)", fill="#FFFFFF", text_color=C_PERI_DARK, stroke=C_PERI_DARK, stroke_width=1.8, size=13.5)
    svg.arrow([(b2x, 1911), (b2x, 1888)], color_key="blk")
    svg.kv_slots(b2x, 1868, bar_w=13, bar_h=30, left_n=2, right_n=2)
    svg.text(b2x + 106, 1868, "[512, 512]", size=11.5, mono=True, fill=C_INK)

    # Concat Window + Selected Compressed -> Attention KV
    svg.box(2550, 1822, 170, 34, "Concat (128 + 512)", fill=C_ORANGE_DARK, stroke="#9A3412", size=12.5)
    svg.arrow([(b1x, 1893), (b1x, 1822), (2463, 1822)], color_key="blk")
    svg.arrow([(b2x, 1853), (b2x, 1822), (2637, 1822)], color_key="blk")
    svg.arrow([(2550, 1805), (2550, 1794)], color_key="blk")

    # Branch 3 (Right): Query & Indexer Hook
    svg.trap(b3x, 2180, 226, 172, 44, "Linear (wq_a · 5120→1280)", fill=C_ORANGE, stroke=C_ORANGE_DARK, size=12.8)
    svg.arrow([(b3x, 2158), (b3x, 2134)], color_key="blk")
    svg.text(b3x - 78, 2146, "[1, 1280]", size=11.5, mono=True, fill=C_INK)

    svg.box(b3x, 2114, 184, 36, "RMSNorm (q_norm)", fill=C_ORANGE, stroke=C_ORANGE_DARK, size=13)
    svg.arrow([(b3x, 2096), (b3x, 2058)], color_key="blk")

    svg.arrow([(b3x, 2076), (3364, 2076), (3364, 2048), (3380, 2048)], color_key="red", width=1.8)

    svg.trap(b3x, 2032, 176, 232, 46, "Linear (wq_b)", sublabel="1280 → 64 × 512", fill=C_ORANGE, stroke=C_ORANGE_DARK, size=13.2, sub_size=11)
    svg.arrow([(b3x, 2009), (b3x, 1972)], color_key="blk")
    svg.text(b3x - 94, 1990, "[1, 64, 512]", size=11.5, mono=True, fill=C_INK)

    svg.box(b3x, 1950, 176, 38, "RoPE (tail 64)", fill=C_ORANGE, stroke=C_ORANGE_DARK, size=13.5)
    svg.badge(b3x - 150, 1950, "No Q Head Norm", fill="#FEF3C7", text_color="#92400E", stroke=C_ORANGE_DARK, w=116, h=22, size=10.2)
    svg.arrow([(b3x, 1931), (b3x, 1794)], color_key="blk")

    svg.box(3378, 1854, 134, 36, "attn_sink [64]", fill=C_ORANGE_DARK, stroke="#9A3412", size=12.5)
    svg.arrow([(3378, 1836), (3378, 1794)], color_key="blk")

    svg.box(3444, 2032, 124, 88, "Indexer", sublabel="FP4 Top-512", fill=C_GREEN_LIGHT, text_color=C_GREEN_DARK, stroke=C_GREEN_DARK, stroke_width=1.8, dash="6,4", size=14.5, sub_size=11)
    svg.arrow([(3444, 1988), (3444, 1930), (2860, 1930)], color_key="green", width=2.0)
    svg.text(3032, 1916, "index_topk [1, 512]", size=12, mono=True, weight="700", fill=C_GREEN_DARK)

    # Clean zoom lines from mini Indexer up to R2 Indexer (stay strictly on right side!)
    svg.zoom_line(3444, 1988, 3360, 1476, color=C_GREEN_DARK, opacity=0.55)
    svg.zoom_line(3506, 1988, 3550, 1476, color=C_GREEN_DARK, opacity=0.70)

    # Clean zoom lines from CSA2 Compressor down to R4 Compressor
    svg.zoom_line(2614, 2201, 2170, 2296, color=C_PERI_DARK)
    svg.zoom_line(2918, 2201, 3550, 2296, color=C_PERI_DARK)

    # =========================================================================
    # R4: Token-Level Compressor (C2 & C1) + Dual FP4 Cache  (x=2170..3550, y=2296..2922)
    # =========================================================================
    svg.panel(2170, 2296, 1380, 626, "Token-Level Compressor (C2 / C1) & Dual FP4 Cache Writer", color=C_PERI_DARK, subtitle="无 Overlap · 无 APE · 全局 890 B/tok")

    # ---- Zone 1 (Left): C2 Compressor (L2, L8, L14, ratio=2) ----
    svg.rect(2196, 2348, 466, 550, fill=C_PERI_LIGHT, stroke=C_PERI, stroke_width=1.5, rx=10, dash="6,4")
    svg.text(2212, 2368, "C2 Compressor · Encoder L2, L8, L14 (ratio = 2)", size=13.5, weight="700", fill=C_PERI_DARK, anchor="start")

    c2_l = 2324
    c2_r = 2536
    c2_m = 2430

    svg.box(c2_m, 2864, 268, 34, "相邻 2 个 Token: [2, 5120]", fill="#E0E7FF", text_color="#1E1B4B", stroke=C_PERI_DARK, size=12.5)
    svg.arrow([(c2_m, 2847), (c2_m, 2828)], color_key="blk", head_end=False)
    svg.arrow([(c2_m, 2828), (c2_l, 2828), (c2_l, 2806)], color_key="blk")
    svg.arrow([(c2_m, 2828), (c2_r, 2828), (c2_r, 2806)], color_key="blk")

    svg.trap(c2_l, 2780, 192, 152, 46, "Linear (wkv)", sublabel="5120→512 (FP32)", fill=C_PERI, stroke=C_PERI_DARK, size=12.8, sub_size=10.5)
    svg.trap(c2_r, 2780, 192, 152, 46, "Linear (wgate)", sublabel="5120→512 (FP32)", fill=C_PERI, stroke=C_PERI_DARK, size=12.8, sub_size=10.5)

    svg.arrow([(c2_r, 2757), (c2_r, 2724)], color_key="blk")
    svg.box(c2_r, 2702, 168, 38, "Softmax (dim=0)", fill=C_PERI, stroke=C_PERI_DARK, size=13)
    svg.text(c2_r + 80, 2738, "[2, 512]", size=11, mono=True, fill=C_INK)

    svg.arrow([(c2_l, 2757), (c2_l, 2640), (c2_m - 18, 2640)], color_key="blk")
    svg.text(c2_l - 44, 2698, "[2, 512]", size=11, mono=True, fill=C_INK)
    svg.arrow([(c2_r, 2683), (c2_r, 2640), (c2_m + 18, 2640)], color_key="blk")

    svg.circle_op(c2_m, 2640, 15, "mul", fill=C_GREEN, stroke=C_GREEN_DARK)
    svg.arrow([(c2_m, 2625), (c2_m, 2598)], color_key="blk")

    svg.box(c2_m, 2578, 186, 36, "Sum (pair → 1)", fill=C_PERI, stroke=C_PERI_DARK, size=13)
    svg.arrow([(c2_m, 2560), (c2_m, 2532)], color_key="blk")
    svg.text(c2_m + 68, 2546, "[1, 512]", size=11.5, mono=True, fill=C_INK)

    svg.box(c2_m, 2510, 224, 38, "RMSNorm (BF16 finish)", fill=C_PERI_DARK, stroke="#3730A3", size=13)
    svg.arrow([(c2_m, 2491), (c2_m, 2458)], color_key="peri")

    svg.box(c2_m, 2436, 286, 40, "Pre-RoPE Latent [T/2, 512]", fill="#FFFFFF", text_color=C_PERI_DARK, stroke=C_PERI_DARK, stroke_width=1.8, size=13)
    svg.badge(c2_m, 2394, "无 Overlap (coff=1) · 无 APE (较 V4 Pro 简化)", fill="#DBEAFE", text_color="#1E40AF", stroke="#3B82F6", w=346, h=22, size=11)

    # ---- Zone 2 (Middle): C1 Compressor (L20, ratio=1) ----
    svg.rect(2686, 2348, 266, 550, fill="#FEF2F2", stroke="#DC2626", stroke_width=1.5, rx=10, dash="6,4")
    svg.text(2819, 2368, "C1 Compressor · L20 (r = 1)", size=13.5, weight="700", fill="#B91C1C")
    svg.text(2819, 2392, "(全 20 层 Decoder 唯一源)", size=11.5, weight="600", fill="#991B1B")

    c1x = 2819
    svg.box(c1x, 2864, 214, 34, "单 Token: [1, 5120]", fill="#FEE2E2", text_color="#991B1B", stroke="#DC2626", size=12.5)
    svg.arrow([(c1x, 2847), (c1x, 2772)], color_key="red")

    svg.trap(c1x, 2740, 214, 164, 54, "Linear (wkv)", sublabel="5120 → 512 (无 wgate)", fill=C_PERI_DARK, stroke="#3730A3", size=13, sub_size=10.8)
    svg.arrow([(c1x, 2713), (c1x, 2640)], color_key="peri")
    svg.text(c1x + 56, 2676, "[1, 512]", size=11.5, mono=True, fill=C_INK)

    svg.box(c1x, 2614, 200, 42, "RMSNorm (BF16)", fill=C_PERI_DARK, stroke="#3730A3", size=13)
    svg.arrow([(c1x, 2593), (c1x, 2514)], color_key="peri")

    svg.box(c1x, 2488, 232, 44, "Pre-RoPE Latent [T, 512]", sublabel="L20–L39 全共享", fill="#FFFFFF", text_color="#B91C1C", stroke="#DC2626", stroke_width=1.8, size=12.8, sub_size=10.8)

    # ---- Zone 3 (Right): Dual FP4 Cache Output & 890 B/tok Breakdown ----
    svg.rect(2984, 2348, 542, 550, fill="#FFFFFF", stroke=C_RED_DARK, stroke_width=1.6, rx=10)
    svg.text(3002, 2368, "双路 FP4 Cache 写入 (复用同一 Pre-RoPE Latent)", size=13.5, weight="700", fill=C_RED_DARK, anchor="start")

    # Clean fork bus from C2 & C1 Pre-RoPE Latents into BOTH Path A and Path B!
    svg.arrow([(c2_m + 143, 2436), (2968, 2436), (2968, 2466), (3016, 2466)], color_key="peri", width=2.0)
    svg.arrow([(c1x + 116, 2488), (2968, 2488), (2968, 2466)], color_key="red", width=2.0, head_end=False)
    svg.arrow([(2968, 2488), (2968, 2624), (3016, 2624)], color_key="green", width=2.0)

    # Path A (Main Compressed KV)
    svg.rect(3006, 2396, 502, 140, fill="#FFF5F5", stroke=C_RED, stroke_width=1.3, rx=8)
    svg.text(3020, 2414, "Path A · Main Compressed KV (512d → 288 B/slot)", size=12.5, weight="700", fill=C_RED_DARK, anchor="start")
    svg.box(3110, 2466, 178, 44, "RoPE (tail 64)", sublabel="θ = 160K YaRN", fill=C_PERI, stroke=C_PERI_DARK, size=12.5, sub_size=10.5)
    svg.arrow([(3199, 2466), (3226, 2466)], color_key="blk")
    svg.box(3318, 2466, 178, 44, "FP4 E2M1 Quant", sublabel="per-16 E4M3FN scale", fill=C_PERI_DARK, stroke="#3730A3", size=12.5, sub_size=10.5)
    svg.arrow([(3407, 2466), (3430, 2466)], color_key="red")
    svg.kv_slots(3468, 2466, bar_w=10, bar_h=34, left_n=2, right_n=2, dots_w=20)
    svg.text(3258, 2516, "256 B payload (512×4b) + 32 B scale (32×E4M3) = 288 B / slot", size=11.5, mono=True, weight="600", fill=C_RED_DARK)

    # Path B (Indexer K)
    svg.rect(3006, 2554, 502, 168, fill=C_GREEN_LIGHT, stroke=C_GREEN_DARK, stroke_width=1.3, rx=8)
    svg.text(3020, 2572, "Path B · Index-K Cache (512d → 128d → 68 B/slot)", size=12.5, weight="700", fill=C_GREEN_DARK, anchor="start")
    svg.trap(3092, 2624, 146, 118, 42, "Linear(wk)", sublabel="512→128", fill=C_GREEN, stroke=C_GREEN_DARK, size=12, sub_size=10.2)
    svg.arrow([(3165, 2624), (3188, 2624)], color_key="blk")
    svg.box(3246, 2624, 112, 42, "RMSNorm", sublabel="k_norm(128)", fill=C_GREEN, stroke=C_GREEN_DARK, size=11.8, sub_size=10)
    svg.arrow([(3302, 2624), (3322, 2624)], color_key="blk")
    svg.box(3410, 2624, 174, 42, "RoPE(64)+FP4", sublabel="per-32 UE8M0", fill=C_GREEN_DARK, stroke="#365314", size=12, sub_size=10.2)
    svg.text(3258, 2672, "64 B payload (128×4b) + 4 B scale (4×UE8M0) = 68 B / slot", size=11.5, mono=True, weight="600", fill=C_GREEN_DARK)
    svg.text(3258, 2700, "仅 L2, L8, L14, L20 拥有 wk / k_norm · L24/28/32/36 复用 L20 Index-K", size=11.2, weight="600", fill=C_INK)

    # Exact 890 B/tok Cache Accounting Box
    svg.rect(3006, 2740, 502, 142, fill="#1E293B", stroke="#0F172A", stroke_width=1.5, rx=10)
    svg.text(3024, 2762, "全局持久 KV + Index-K 显存精确核算 (≈ 890 B / token)", size=13.2, weight="700", fill="#FDE68A", anchor="start")
    svg.text_lines(
        3024,
        2782,
        [
            "Encoder 3 源 (L2, L8, L14, r=2): 3 × (288 + 68) / 2 = 534 B/tok",
            "Decoder 1 源 (L20, r=1):        1 × (288 + 68) / 1 = 356 B/tok",
            "合计全局持久缓存 = 534 + 356 = 890 B/tok (较 V4 降低 ~4×)",
            "+ 局部 SWA 滑窗: 固定上限 128 tok/层 (不随 1M 上下文增长)",
        ],
        size=11.8,
        line_h=22.5,
        mono=True,
        fill="#F8FAFC",
    )


def draw_footer(svg: SVGBuilder) -> None:
    svg.text(
        W / 2,
        2954,
        "Verified against DeepSeek-V4.1-Flash config.json (DeepseekV41ForCausalLM), model.safetensors.index.json (96,085 tensors / 510.3 GB) & SGLang implementation",
        size=13,
        weight="500",
        fill=C_SUBTLE,
    )


def build_svg() -> str:
    svg = SVGBuilder(W, H)
    draw_header_and_layer_map(svg)
    draw_left_column(svg)
    draw_center_backbone(svg)
    draw_right_column(svg)
    draw_footer(svg)
    return svg.render()


def render_outputs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / "deepseek_v41_flash_architecture.svg"
    png_path = out_dir / "deepseek_v41_flash_architecture.png"
    jpg_path = out_dir / "deepseek_v41_flash_architecture.jpg"

    svg_content = build_svg()
    svg_path.write_text(svg_content, encoding="utf-8")
    print(f"Wrote SVG: {svg_path} ({svg_path.stat().st_size:,} bytes)")

    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-device-scale-factor=2",
        f"--window-size={W},{H}",
        f"--screenshot={png_path}",
        f"file://{svg_path.resolve()}",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not png_path.exists():
        raise RuntimeError(f"Chrome screenshot failed:\n{res.stderr}")
    print(f"Wrote PNG: {png_path} ({png_path.stat().st_size:,} bytes)")

    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(png_path).convert("RGB")
    im.save(jpg_path, "JPEG", quality=95, subsampling=0)
    print(f"Wrote JPG: {jpg_path} ({jpg_path.stat().st_size:,} bytes, {im.size[0]}x{im.size[1]})")


if __name__ == "__main__":
    render_outputs(Path(__file__).resolve().parent)
