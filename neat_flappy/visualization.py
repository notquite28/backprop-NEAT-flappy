"""Dependency-free SVG rendering for starting and champion genomes."""
from __future__ import annotations

from collections import deque
from html import escape
from pathlib import Path
import math
import random

from .genome import (
    Genome,
    InnovationStore,
    NodeKind,
    OUTPUT_NODE_ID,
    initial_population,
    load_genome,
    validate_genome,
)


def _levels(genome: Genome, store: InnovationStore) -> dict[int, int]:
    incoming: dict[int, list[int]] = {}
    outgoing: dict[int, list[int]] = {}
    indegree = {node_id: 0 for node_id in genome.node_ids}
    for gene in genome.connections.values():
        if gene.enabled:
            incoming.setdefault(gene.dst, []).append(gene.src)
            outgoing.setdefault(gene.src, []).append(gene.dst)
            indegree[gene.dst] += 1
    queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
    levels: dict[int, int] = {}
    while queue:
        node_id = queue.popleft()
        predecessors = incoming.get(node_id, ())
        node = store.nodes[node_id]
        minimum = 1 if node.kind is NodeKind.HIDDEN else 0
        levels[node_id] = max(
            minimum,
            max((levels[source] + 1 for source in predecessors), default=0),
        )
        for destination in sorted(outgoing.get(node_id, ())):
            indegree[destination] -= 1
            if indegree[destination] == 0:
                queue.append(destination)
    if len(levels) != len(genome.node_ids):
        raise ValueError("cannot render a cyclic genome")
    hidden_levels = [
        levels[node_id]
        for node_id in genome.node_ids
        if store.nodes[node_id].kind is NodeKind.HIDDEN
    ]
    if hidden_levels:
        levels[OUTPUT_NODE_ID] = max(levels[OUTPUT_NODE_ID], max(hidden_levels) + 1)
    return levels


NODE_COLORS = {
    "input": "#4C78A8",
    "bias": "#9E9E9E",
    "output": "#F58518",
    "sigmoid": "#54A24B",
    "tanh": "#72B7B2",
    "relu": "#E45756",
    "gaussian": "#B279A2",
    "sin": "#FF9DA6",
    "abs": "#9D755D",
    "multiply": "#BAB0AC",
    "add": "#59A14F",
    "square": "#EDC948",
}
INPUT_LABELS = {
    0: "gap Δy",
    1: "velocity",
    2: "gap size",
    3: "next Δy",
    4: "pipe Δx",
}


def _node_label(node_id: int, store: InnovationStore) -> str:
    node = store.nodes[node_id]
    if node.kind is NodeKind.INPUT:
        return INPUT_LABELS.get(node_id, "input")
    if node.kind in (NodeKind.BIAS, NodeKind.OUTPUT):
        return node.kind.value
    if node.activation is None:
        raise ValueError("hidden node requires an activation")
    return node.activation.value


def _panel(
    genome: Genome,
    store: InnovationStore,
    title: str,
    x_offset: float,
    width: float,
    height: float,
) -> list[str]:
    validate_genome(genome, store)
    levels = _levels(genome, store)
    max_level = max(levels.values(), default=0)
    grouped: dict[int, list[int]] = {}
    for node_id, level in levels.items():
        grouped.setdefault(level, []).append(node_id)
    positions: dict[int, tuple[float, float]] = {}
    left = x_offset + 85
    usable_width = width - 170
    for level, node_ids in sorted(grouped.items()):
        x = left + usable_width * level / max(max_level, 1)
        ordered = sorted(node_ids)
        for row, node_id in enumerate(ordered, start=1):
            y = 100 + row * ((height - 150) / (len(ordered) + 1))
            positions[node_id] = (x, y)

    enabled_count = sum(gene.enabled for gene in genome.connections.values())
    lines = [
        f'<text x="{x_offset + width / 2:.1f}" y="38" text-anchor="middle" class="title">{escape(title)}</text>',
        f'<text x="{x_offset + width / 2:.1f}" y="66" text-anchor="middle" class="counts">'
        f"{len(genome.node_ids)} nodes · {enabled_count} enabled / "
        f"{len(genome.connections)} total connections</text>",
    ]
    for gene in sorted(genome.connections.values(), key=lambda value: (value.enabled, value.innovation)):
        x1, y1 = positions[gene.src]
        x2, y2 = positions[gene.dst]
        distance = max(math.hypot(x2 - x1, y2 - y1), 1.0)
        end_x = x2 - 39.0 * (x2 - x1) / distance
        end_y = y2 - 39.0 * (y2 - y1) / distance
        if gene.enabled:
            sign = "positive" if gene.weight >= 0.0 else "negative"
            color = "#2E7D32" if gene.weight >= 0.0 else "#C62828"
            dash = ""
            opacity = "0.85"
        else:
            sign = "disabled"
            color = "#8A8A8A"
            dash = ' stroke-dasharray="7 5"'
            opacity = "0.65"
        thickness = min(8.0, 0.8 + 2.2 * abs(gene.weight))
        lines.extend(
            (
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{end_x:.1f}" y2="{end_y:.1f}" '
                f'stroke="{color}" stroke-width="{thickness:.2f}" stroke-opacity="{opacity}"'
                f'{dash} marker-end="url(#arrow-{sign})"/>',
                f'<text x="{(x1 + x2) / 2:.1f}" y="{(y1 + y2) / 2 - 6:.1f}" '
                f'text-anchor="middle" class="edge">i{gene.innovation} · {gene.weight:+.3f}</text>',
            )
        )
    for node_id in sorted(genome.node_ids):
        node = store.nodes[node_id]
        x, y = positions[node_id]
        color_key = node.activation.value if node.activation is not None else node.kind.value
        lines.extend(
            (
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="36" fill="{NODE_COLORS[color_key]}" '
                'stroke="#222222" stroke-width="1.5"/>',
                f'<text x="{x:.1f}" y="{y - 3:.1f}" text-anchor="middle" class="node-id">'
                f"#{node_id}</text>",
                f'<text x="{x:.1f}" y="{y + 15:.1f}" text-anchor="middle" class="node">'
                f"{escape(_node_label(node_id, store))}</text>",
            )
        )
    return lines


def render_comparison(
    genome_path: Path,
    output_path: Path,
    seed: int = 0,
    best_path: Path | None = None,
) -> Path:
    """Render the starting graph, optional global best, and a saved genome."""
    start_store = InnovationStore.base()
    starting = initial_population(start_store, random.Random(seed), size=1)[0]
    saved, saved_store, saved_generation = load_genome(genome_path)
    panels = [
        (
            starting,
            start_store,
            f"Starting genome · seed {seed} · genome 0",
        )
    ]
    if best_path is not None:
        best, best_store, best_generation = load_genome(best_path)
        panels.append(
            (
                best,
                best_store,
                f"Global best · generation {best_generation} · "
                f"fitness {best.fitness:.3f} · score {best.score:.3f}",
            )
        )
    panels.append(
        (
            saved,
            saved_store,
            f"Saved genome · generation {saved_generation} · "
            f"fitness {saved.fitness:.3f} · score {saved.score:.3f}",
        )
    )
    panel_width = 700
    width = panel_width * len(panels)
    height = 560
    content = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow-positive" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#2E7D32"/>',
        "</marker>",
        '<marker id="arrow-negative" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#C62828"/>',
        "</marker>",
        '<marker id="arrow-disabled" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#8A8A8A"/>',
        "</marker>",
        "<style>",
        ".title { fill: #111111; font: 500 19px sans-serif; }",
        ".counts { fill: #222222; font: 16px sans-serif; }",
        ".node-id { fill: #111111; font: 600 12px sans-serif; }",
        ".node { fill: #111111; font: 12px sans-serif; }",
        ".edge { fill: #333333; font: 11px monospace; paint-order: stroke; stroke: #ffffff; stroke-width: 3px; }",
        "</style>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    for index in range(1, len(panels)):
        x = panel_width * index
        content.append(
            f'<line x1="{x}" y1="20" x2="{x}" y2="{height - 20}" stroke="#dddddd"/>'
        )
    for index, (genome, store, title) in enumerate(panels):
        content.extend(
            _panel(
                genome,
                store,
                title,
                panel_width * index,
                panel_width,
                height,
            )
        )
    content.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(content) + "\n")
    return output_path
