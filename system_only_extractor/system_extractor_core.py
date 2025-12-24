from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import ceil, hypot
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import fitz  # PyMuPDF
except ModuleNotFoundError:  # pragma: no cover
    import pymupdf as fitz

import re


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def expand(self, dx: float, dy: float) -> "BBox":
        return BBox(self.x0 - dx, self.y0 - dy, self.x1 + dx, self.y1 + dy)


@dataclass
class TextElement:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float


@dataclass
class EquipmentNode:
    page_index: int
    region_id: int
    name: str
    bbox: BBox
    font_size: float
    properties: List[str]

    primary_from: str = "-"
    alternate_from: str = "-"

    @property
    def type(self) -> str:
        return self.name[:3]

    @property
    def x_center(self) -> float:
        return (self.bbox.x0 + self.bbox.x1) / 2


@dataclass
class ExtractorConfig:
    min_source_font: float = 10.5
    y_group_tolerance: float = 35.0
    primary_cost_tolerance: float = 150.0
    max_downward_search: float = 200.0
    fine_vector_snap: float = 1.0
    bridge_max_gap: float = 20.0
    bridge_align_tol: float = 2.0
    alt_types: Set[str] = None  # type: ignore[assignment]
    skip_source_types: Set[str] = None  # type: ignore[assignment]

    debug_equipment: str = ""
    debug_page: int = 0

    vision_enabled: bool = False
    vision_provider: str = "OPENAI"
    vision_min_confidence: int = 55
    vision_model: str = ""

    # Alternate source type priority (higher = preferred)
    alt_type_priority: Dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.alt_types is None:
            self.alt_types = {"MBC", "MSB", "DSG", "ATS", "BMP"}
        if self.skip_source_types is None:
            self.skip_source_types = {"UPB"}
        if self.alt_type_priority is None:
            # Higher number = higher priority for alternate selection
            self.alt_type_priority = {
                "MSB": 100,
                "UPS": 90,
                "GEN": 80,
                "TRN": 70,
                "GSB": 60,
                "MDP": 50,
                "MBC": 40,
                "DSG": 30,
                "ATS": 20,
                "BMP": 10,
            }


EXCLUDE_KEYWORDS = {
    "LINETYPE",
    "LEGEND",
    "NOTES",
    "KEYED",
    "ISSUES",
    "REVISIONS",
    "PROPRIETARY",
    "CONSULTING",
    "ENGINEERS",
    "ARCHITECTS",
    "ONE-LINE GENERAL NOTES",
    "EDGECONNEX",
    "ASSET CODE",
    "IDENTIFICATION SYSTEM",
    "BURR COMPUTER",
}


EQUIPMENT_RE = re.compile(r"\b([A-Z]{3}(?:[\s\-]*[A-Z0-9]{2}[\s\-]*\d{3}|[\s\-]*\d{5}))\b")


def extract_text_elements(page: "fitz.Page") -> List[TextElement]:
    blocks = page.get_text("dict")["blocks"]
    elements: List[TextElement] = []
    for block in blocks:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = (span.get("text") or "").strip()
                if not text:
                    continue
                x0, y0, x1, y1 = span["bbox"]
                elements.append(
                    TextElement(
                        text=text,
                        x0=float(x0),
                        y0=float(y0),
                        x1=float(x1),
                        y1=float(y1),
                        font_size=float(span.get("size") or 0.0),
                    )
                )
    elements.sort(key=lambda e: (round(e.y0 / 15) * 15, e.x0))
    return elements


def _is_in_legend(elements: Sequence[TextElement], i: int) -> bool:
    base = elements[i]
    for j in range(max(0, i - 8), min(len(elements), i + 9)):
        ctx_el = elements[j]
        ctx = ctx_el.text.upper()
        if not any(k in ctx for k in EXCLUDE_KEYWORDS):
            continue
        if abs(ctx_el.y0 - base.y0) > 80:
            continue
        if abs(ctx_el.x0 - base.x0) > 650:
            continue
        return True
    return False


def _extract_properties(elements: Sequence[TextElement], idx: int) -> List[str]:
    base = elements[idx]
    base_y = base.y0
    base_x = base.x0

    base_name_match = EQUIPMENT_RE.search(base.text)
    base_name = ""
    if base_name_match:
        base_name = re.sub(r"[^A-Z0-9]", "", (base_name_match.group(1) or "").upper())
    base_type = base_name[:3] if len(base_name) >= 3 else ""

    property_patterns = [
        r"\b\d+\s*A\b",
        r"\b\d+\s*AMP(?:S)?\b",
        r"\d+kW",
        r"\d+kVA",
        r"\b\d+\s*V\b",
        r"\d+Y/\d+V",
        r"\d+kV",
        r"\d+kAIC",
        r"\d+AF|AT",
        r"\d+%",
        r"NEMA\s+\d+",
        r"LSIG|ERMS|LSI|S\.T\.U\.",
    ]

    keywords = {
        "STATIC",
        "UPS",
        "SYSTEM",
        "GENERATOR",
        "SWITCHBOARD",
        "PANEL",
        "TRANSFORMER",
        "BREAKER",
        "DISTRIBUTION",
        "BUSWAY",
        "HUMIDIFIER",
        "AHU",
    }

    skip_tokens = {
        "RE:",
        "O.F.C.I.",
        "SPD",
        "PM1",
        "PM2",
        "METER",
        "GFPE",
        "ATS PLC",
        "MANUF",
    }

    props: List[str] = []

    # RPP labels often have their electrical properties rendered as separate text lines
    # below the name, and those text elements are not guaranteed to appear right after
    # the name in the global sorted element list. For RPP, use a coordinate-based search.
    if base_type == "RPP":
        y_min = base_y
        y_max = base_y + 320
        x_center = (base.x0 + base.x1) / 2
        x_min = x_center - 420
        x_max = x_center + 420

        amp_value: str | None = None
        yv_value: str | None = None

        # Group nearby text by approximate line (y bucket), then build line strings.
        line_buckets: Dict[int, List[TextElement]] = {}
        for el in elements:
            if el.y0 < y_min or el.y0 > y_max:
                continue
            el_center_x = (el.x0 + el.x1) / 2
            if el_center_x < x_min or el_center_x > x_max:
                continue
            # Skip the equipment label itself
            if EQUIPMENT_RE.search(el.text):
                continue

            bucket = int(round(el.y0 / 6.0) * 6)
            line_buckets.setdefault(bucket, []).append(el)

        for _, els in sorted(line_buckets.items(), key=lambda kv: kv[0]):
            els.sort(key=lambda e: e.x0)
            line = " ".join(e.text for e in els if (e.text or "").strip())
            if not line:
                continue

            if any(s in line for s in skip_tokens):
                continue
            if any(k in line.upper() for k in EXCLUDE_KEYWORDS):
                continue

            if yv_value is None:
                m_yv = re.search(r"\b(\d{2,4})\s*Y/\s*(\d{2,4})\s*V\b", line, re.IGNORECASE)
                if m_yv:
                    yv_value = f"{m_yv.group(1)}Y/{m_yv.group(2)}V"

            if amp_value is None:
                m_a = re.search(r"\b(\d{2,5})\s*A\b", line, re.IGNORECASE)
                if m_a:
                    amp_value = f"{m_a.group(1)}A"

            if amp_value is not None and yv_value is not None:
                break

        if amp_value:
            props.append(amp_value)
        if yv_value:
            props.append(yv_value)

        return props

    max_scan = 60 if base_type == "RPP" else 25
    max_y_diff = 220 if base_type == "RPP" else 120
    max_x_diff = 340 if base_type == "RPP" else 260

    for j in range(idx + 1, min(idx + max_scan, len(elements))):
        nxt = elements[j]
        y_diff = nxt.y0 - base_y
        x_diff = abs(nxt.x0 - base_x)

        if y_diff > max_y_diff:
            break
        if y_diff < 0 or x_diff > max_x_diff:
            continue

        if EQUIPMENT_RE.search(nxt.text):
            break

        t = nxt.text
        is_prop = any(re.search(p, t, re.IGNORECASE) for p in property_patterns) or any(
            k in t.upper() for k in keywords
        )

        if any(s in t for s in skip_tokens):
            is_prop = False
        if any(k in t.upper() for k in EXCLUDE_KEYWORDS):
            is_prop = False

        if is_prop and t not in props:
            props.append(t)

    return props


def find_equipment_nodes(page_index: int, elements: Sequence[TextElement]) -> List[EquipmentNode]:
    candidates: List[Tuple[int, str, TextElement]] = []

    for i, el in enumerate(elements):
        m = EQUIPMENT_RE.search(el.text)
        if not m:
            continue
        if _is_in_legend(elements, i):
            continue
        name = re.sub(r"[^A-Z0-9]", "", (m.group(1) or "").upper())
        candidates.append((i, name, el))

    best_by_name: Dict[str, Tuple[int, TextElement]] = {}
    for i, name, el in candidates:
        prev = best_by_name.get(name)
        if prev is None or el.font_size > prev[1].font_size:
            best_by_name[name] = (i, el)

    nodes: List[EquipmentNode] = []
    for name, (i, el) in best_by_name.items():
        props = _extract_properties(elements, i)
        nodes.append(
            EquipmentNode(
                page_index=page_index,
                region_id=0,
                name=name,
                bbox=BBox(el.x0, el.y0, el.x1, el.y1),
                font_size=el.font_size,
                properties=props,
            )
        )

    nodes.sort(key=lambda n: (n.page_index, round(n.bbox.y0 / 15) * 15, n.bbox.x0))
    return nodes


def split_regions_by_x(nodes: Sequence[EquipmentNode], page_width: float) -> List[Tuple[float, float]]:
    if len(nodes) < 25:
        return [(0.0, page_width)]

    xs = sorted(n.x_center for n in nodes)
    gaps = [(xs[i + 1] - xs[i], i) for i in range(len(xs) - 1)]
    gap, idx = max(gaps, key=lambda t: t[0])

    if gap < page_width * 0.12:
        return [(0.0, page_width)]

    split_x = (xs[idx] + xs[idx + 1]) / 2
    return [(0.0, split_x), (split_x, page_width)]


def assign_region_ids(nodes: List[EquipmentNode], regions: Sequence[Tuple[float, float]]) -> None:
    if len(regions) == 1:
        for n in nodes:
            n.region_id = 0
        return

    for n in nodes:
        for rid, (x0, x1) in enumerate(regions):
            if x0 <= n.x_center <= x1:
                n.region_id = rid
                break


PointKey = Tuple[int, int]


def _snap_point(x: float, y: float, snap: float) -> PointKey:
    return (int(round(x / snap)), int(round(y / snap)))


def build_vector_graph(
    page: "fitz.Page", region_x0: float, region_x1: float, snap: float = 2.0
) -> Tuple[
    Dict[PointKey, Set[PointKey]],
    Dict[PointKey, Set[PointKey]],
    Dict[PointKey, Tuple[float, float]],
    Set[PointKey],
]:
    black_graph: Dict[PointKey, Set[PointKey]] = {}
    red_graph: Dict[PointKey, Set[PointKey]] = {}
    coords: Dict[PointKey, Tuple[float, float]] = {}
    junction_markers: Set[PointKey] = set()

    def is_red_color(c: Optional[Tuple[float, float, float]]) -> bool:
        if c is None:
            return False

        r: float
        g: float
        b: float
        if isinstance(c, int):
            r = float((c >> 16) & 255) / 255.0
            g = float((c >> 8) & 255) / 255.0
            b = float(c & 255) / 255.0
        else:
            try:
                r, g, b = c  # type: ignore[misc]
            except Exception:
                return False
            if max(r, g, b) > 1.001:
                r = float(r) / 255.0
                g = float(g) / 255.0
                b = float(b) / 255.0

        return r >= 0.55 and (r - max(g, b)) >= 0.22 and g <= 0.55 and b <= 0.55

    def add_node(target: Dict[PointKey, Set[PointKey]], k: PointKey, xy: Tuple[float, float]) -> None:
        if k not in target:
            target[k] = set()
        if k not in coords:
            coords[k] = xy

    def add_edge(target: Dict[PointKey, Set[PointKey]], a_xy: Tuple[float, float], b_xy: Tuple[float, float]) -> None:
        ax, ay = a_xy
        bx, by = b_xy
        a = _snap_point(ax, ay, snap)
        b = _snap_point(bx, by, snap)
        add_node(target, a, (ax, ay))
        add_node(target, b, (bx, by))
        target[a].add(b)
        target[b].add(a)

    def add_segment_densified(a_xy: Tuple[float, float], b_xy: Tuple[float, float]) -> None:
        x0, y0 = a_xy
        x1, y1 = b_xy
        dx = x1 - x0
        dy = y1 - y0
        length = hypot(dx, dy)
        if length <= step:
            add_edge(active_graph, a_xy, b_xy)
            return

        n = int(ceil(length / step))
        prev = (float(x0), float(y0))
        for i in range(1, n + 1):
            t = i / n
            cur = (float(x0 + dx * t), float(y0 + dy * t))
            add_edge(active_graph, prev, cur)
            prev = cur

    # Smaller step increases junction/connectivity fidelity for wiring diagrams.
    step = max(1.0, snap * 1.2)

    drawings = page.get_drawings()
    for d in drawings:
        draw_color = d.get("color") or d.get("fill")
        active_graph = red_graph if is_red_color(draw_color) else black_graph
        for item in d.get("items", []):
            if not item:
                continue
            t = item[0]

            if t == "l":
                (x0, y0) = item[1]
                (x1, y1) = item[2]
                seg_min_x = min(x0, x1)
                seg_max_x = max(x0, x1)
                if seg_max_x < region_x0 - 50 or seg_min_x > region_x1 + 50:
                    continue
                add_segment_densified((float(x0), float(y0)), (float(x1), float(y1)))
                continue

            if t == "re":
                r = item[1]
                w = float(r.x1 - r.x0)
                h = float(r.y1 - r.y0)
                w = abs(w)
                h = abs(h)
                min_side = min(w, h)
                max_side = max(w, h)
                aspect = max_side / max(0.001, min_side)

                if max_side <= 4.5 and min_side <= 4.5:
                    cx = float(r.x0 + r.x1) / 2.0
                    cy = float(r.y0 + r.y1) / 2.0
                    junction_markers.add(_snap_point(cx, cy, snap))

                if not (
                    (max_side <= 25.0 and min_side <= 25.0)
                    or (aspect >= 8.0 and min_side <= 20.0 and max_side <= 450.0)
                ):
                    continue
                if float(r.x1) < region_x0 - 50 or float(r.x0) > region_x1 + 50:
                    continue
                p1 = (float(r.x0), float(r.y0))
                p2 = (float(r.x1), float(r.y0))
                p3 = (float(r.x1), float(r.y1))
                p4 = (float(r.x0), float(r.y1))
                add_segment_densified(p1, p2)
                add_segment_densified(p2, p3)
                add_segment_densified(p3, p4)
                add_segment_densified(p4, p1)
                continue

            if t == "c":
                p0 = item[1]
                p1 = item[2]
                p2 = item[3]
                p3 = item[4]
                xs = [float(p0.x), float(p1.x), float(p2.x), float(p3.x)]
                ys = [float(p0.y), float(p1.y), float(p2.y), float(p3.y)]
                if max(xs) < region_x0 - 50 or min(xs) > region_x1 + 50:
                    continue

                max_side = max(max(xs) - min(xs), max(ys) - min(ys))
                w = max(xs) - min(xs)
                h = max(ys) - min(ys)
                min_side = min(w, h)
                aspect = max_side / max(0.001, min_side)
                if max_side <= 10.0:
                    cx = (min(xs) + max(xs)) / 2.0
                    cy = (min(ys) + max(ys)) / 2.0
                    junction_markers.add(_snap_point(cx, cy, snap))
                if max_side > 20.0:
                    continue

                if max_side >= 9.0 and aspect <= 1.8:
                    cx = (min(xs) + max(xs)) / 2.0
                    cy = (min(ys) + max(ys)) / 2.0
                    junction_markers.add(_snap_point(cx, cy, snap))
                    continue

                est = (
                    hypot(float(p1.x - p0.x), float(p1.y - p0.y))
                    + hypot(float(p2.x - p1.x), float(p2.y - p1.y))
                    + hypot(float(p3.x - p2.x), float(p3.y - p2.y))
                )
                n = max(2, int(ceil(est / step)))
                prev = (float(p0.x), float(p0.y))
                for i in range(1, n + 1):
                    u = i / n
                    um = 1.0 - u
                    x = (
                        (um**3) * float(p0.x)
                        + 3 * (um**2) * u * float(p1.x)
                        + 3 * um * (u**2) * float(p2.x)
                        + (u**3) * float(p3.x)
                    )
                    y = (
                        (um**3) * float(p0.y)
                        + 3 * (um**2) * u * float(p1.y)
                        + 3 * um * (u**2) * float(p2.y)
                        + (u**3) * float(p3.y)
                    )
                    cur = (float(x), float(y))
                    add_edge(active_graph, prev, cur)
                    prev = cur
                continue

    return black_graph, red_graph, coords, junction_markers


def _bridge_collinear_gaps(
    graph: Dict[PointKey, Set[PointKey]],
    coords: Dict[PointKey, Tuple[float, float]],
    max_gap: float = 8.0,
    align_tol: float = 1.6,
) -> int:
    endpoints = [k for k, nbs in graph.items() if len(nbs) == 1]
    if not endpoints:
        return 0

    cell = max(4.0, max_gap)
    grid: Dict[Tuple[int, int], List[PointKey]] = {}
    for k in endpoints:
        x, y = coords[k]
        grid.setdefault((int(x // cell), int(y // cell)), []).append(k)

    added = 0
    used: Set[Tuple[PointKey, PointKey]] = set()

    for a in endpoints:
        if len(graph.get(a, ())) != 1:
            continue
        ax, ay = coords[a]
        na = next(iter(graph[a]))
        nx, ny = coords[na]
        vax = nx - ax
        vay = ny - ay
        ac = (int(ax // cell), int(ay // cell))

        best_b: Optional[PointKey] = None
        best_d = float("inf")
        for gx in (ac[0] - 1, ac[0], ac[0] + 1):
            for gy in (ac[1] - 1, ac[1], ac[1] + 1):
                for b in grid.get((gx, gy), []):
                    if b == a:
                        continue
                    if len(graph.get(b, ())) != 1:
                        continue
                    bx, by = coords[b]
                    dx = bx - ax
                    dy = by - ay
                    d = hypot(dx, dy)
                    if d <= 0.0 or d > max_gap:
                        continue

                    if abs(dx) >= abs(dy):
                        if abs(dy) > align_tol:
                            continue
                    else:
                        if abs(dx) > align_tol:
                            continue

                    if vax * dx + vay * dy >= 0.0:
                        continue
                    nb = next(iter(graph[b]))
                    nbx, nby = coords[nb]
                    vbx = nbx - bx
                    vby = nby - by
                    if vbx * (-dx) + vby * (-dy) >= 0.0:
                        continue

                    if d < best_d:
                        best_d = d
                        best_b = b

        if best_b is None:
            continue
        p = (a, best_b) if a < best_b else (best_b, a)
        if p in used:
            continue
        used.add(p)
        if len(graph.get(a, ())) != 1 or len(graph.get(best_b, ())) != 1:
            continue
        graph[a].add(best_b)
        graph[best_b].add(a)
        added += 1

    return added


def _compute_no_turn_nodes(
    graph: Dict[PointKey, Set[PointKey]],
    coords: Dict[PointKey, Tuple[float, float]],
    junction_markers: Set[PointKey],
    ortho_tol: float = 1.8,
) -> Set[PointKey]:
    no_turn: Set[PointKey] = set()
    for k, nbs in graph.items():
        if len(nbs) != 4:
            continue
        if k in junction_markers:
            continue
        x, y = coords[k]
        h = 0
        v = 0
        ok = True
        for nb in nbs:
            nx, ny = coords[nb]
            dx = nx - x
            dy = ny - y
            if abs(dy) <= ortho_tol and abs(dx) > abs(dy):
                h += 1
            elif abs(dx) <= ortho_tol and abs(dy) > abs(dx):
                v += 1
            else:
                ok = False
                break
        if ok and h == 2 and v == 2:
            no_turn.add(k)
    return no_turn


def build_spatial_index_for_nodes(coords: Dict[PointKey, Tuple[float, float]], cell: float = 50.0) -> Dict[Tuple[int, int], List[PointKey]]:
    idx: Dict[Tuple[int, int], List[PointKey]] = {}
    for k, (x, y) in coords.items():
        cx = int(x // cell)
        cy = int(y // cell)
        idx.setdefault((cx, cy), []).append(k)
    return idx


def _iter_cells_for_rect(x0: float, y0: float, x1: float, y1: float, cell: float) -> Iterable[Tuple[int, int]]:
    cx0 = int(x0 // cell)
    cy0 = int(y0 // cell)
    cx1 = int(x1 // cell)
    cy1 = int(y1 // cell)
    for cx in range(cx0, cx1 + 1):
        for cy in range(cy0, cy1 + 1):
            yield (cx, cy)


def build_spatial_index_for_equipment(nodes: Sequence[EquipmentNode], cell: float = 200.0, pad: float = 40.0) -> Dict[Tuple[int, int], List[EquipmentNode]]:
    idx: Dict[Tuple[int, int], List[EquipmentNode]] = {}
    for n in nodes:
        b = n.bbox.expand(pad, pad)
        for c in _iter_cells_for_rect(b.x0, b.y0, b.x1, b.y1, cell):
            idx.setdefault(c, []).append(n)
    return idx


def find_port_nodes(
    equipment: EquipmentNode,
    node_index: Dict[Tuple[int, int], List[PointKey]],
    coords: Dict[PointKey, Tuple[float, float]],
    graph: Dict[PointKey, Set[PointKey]],
    cell: float = 50.0,
) -> List[PointKey]:
    b = equipment.bbox
    pb = b.expand(110, 110)
    zones = [
        BBox(b.x0 - 50, b.y0 - 110, b.x1 + 50, b.y1 + 110),
        BBox(b.x0 - 220, b.y0 - 700, b.x1 + 220, b.y1 + 180),
    ]

    cand: List[Tuple[int, int, float, float, float, PointKey]] = []
    for zone in zones:
        for c in _iter_cells_for_rect(zone.x0, zone.y0, zone.x1, zone.y1, cell):
            for k in node_index.get(c, []):
                x, y = coords[k]
                if not zone.contains(x, y):
                    continue

                dx_pb = 0.0
                if x < pb.x0:
                    dx_pb = pb.x0 - x
                elif x > pb.x1:
                    dx_pb = x - pb.x1
                dy_pb = 0.0
                if y < pb.y0:
                    dy_pb = pb.y0 - y
                elif y > pb.y1:
                    dy_pb = y - pb.y1
                dist_pb = hypot(dx_pb, dy_pb)
                if dist_pb > 120.0:
                    continue

                dx_t = 0.0
                if x < b.x0:
                    dx_t = b.x0 - x
                elif x > b.x1:
                    dx_t = x - b.x1
                dy_t = 0.0
                if y < b.y0:
                    dy_t = b.y0 - y
                elif y > b.y1:
                    dy_t = y - b.y1
                dist = hypot(dx_t, dy_t)

                up = 0
                out = 0
                for nb in graph.get(k, ()):  # pragma: no branch
                    nx, ny = coords[nb]
                    if not b.contains(nx, ny):
                        out = 1
                    if ny < b.y0 - 2.0:
                        up = 1
                if out == 0:
                    continue
                cand.append((up, out, dist, x, y, k))

        if cand:
            break

    if not cand:
        return []

    cand.sort(key=lambda t: (-t[0], -t[1], t[2], t[3], t[4]))
    cand = cand[:140]

    selected: List[PointKey] = []
    selected_xy: List[Tuple[float, float]] = []
    min_sep = 16.0
    max_ports = 8

    for __, ___, _, x, y, k in cand:
        ok = True
        for sx, sy in selected_xy:
            if hypot(x - sx, y - sy) < min_sep:
                ok = False
                break
        if not ok:
            continue
        selected.append(k)
        selected_xy.append((x, y))
        if len(selected) >= max_ports:
            break

    return selected


def _edge_cost(a_xy: Tuple[float, float], b_xy: Tuple[float, float]) -> float:
    ax, ay = a_xy
    bx, by = b_xy
    dx = bx - ax
    dy = by - ay
    return hypot(dx, dy)


def trace_upstream_source(
    start: PointKey,
    graph: Dict[PointKey, Set[PointKey]],
    coords: Dict[PointKey, Tuple[float, float]],
    no_turn_nodes: Optional[Set[PointKey]],
    equip_index: Dict[Tuple[int, int], List[EquipmentNode]],
    current: EquipmentNode,
    cell: float = 200.0,
    max_cost: float = 8000.0,
    config: Optional[ExtractorConfig] = None,

    # Return up to N upstream hits found by shortest-path cost (total line length)
    max_sources: int = 4,
) -> List[Tuple[EquipmentNode, float]]:
    pq: List[Tuple[float, PointKey, int]] = []
    heappush(pq, (0.0, start, 0))
    best: Dict[Tuple[PointKey, int], float] = {(start, 0): 0.0}

    best_major_by_name: Dict[str, Tuple[EquipmentNode, float]] = {}
    best_minor_by_name: Dict[str, Tuple[EquipmentNode, float]] = {}

    def is_minor_source(node: EquipmentNode) -> bool:
        if config is None:
            return False
        if node.type in config.skip_source_types:
            return True
        if node.font_size and node.font_size < config.min_source_font:
            return True
        return False

    while pq:
        cost, node, cur_dir = heappop(pq)
        if cost > max_cost:
            break
        if cost != best.get((node, cur_dir)):
            continue

        x, y = coords[node]

        if config is not None:
            if y > current.bbox.y1 + config.max_downward_search:
                continue

        if y < current.bbox.y0 - 5:
            if not (no_turn_nodes is not None and node in no_turn_nodes):
                cx = int(x // cell)
                cy = int(y // cell)
                done = False
                for nx in (cx - 1, cx, cx + 1):
                    for ny in (cy - 1, cy, cy + 1):
                        for cand in equip_index.get((nx, ny), []):
                            if cand.name == current.name:
                                continue
                            if current.type in {"RPP", "RPU"} and cand.type in {"RPP", "RPU"}:
                                continue
                            if current.type == "PDU" and cand.type in {"PDU", "RPP", "RPU"}:
                                continue
                            # Labels/symbols are often offset from the wire connection.
                            if not cand.bbox.expand(75, 75).contains(x, y):
                                continue
                            if cand.bbox.y0 >= current.bbox.y0:
                                continue

                            if is_minor_source(cand):
                                prev = best_minor_by_name.get(cand.name)
                                if prev is None or cost < prev[1]:
                                    best_minor_by_name[cand.name] = (cand, cost)
                            else:
                                prev = best_major_by_name.get(cand.name)
                                if prev is None or cost < prev[1]:
                                    best_major_by_name[cand.name] = (cand, cost)
                                if len(best_major_by_name) >= max_sources:
                                    done = True
                                    break
                        if done:
                            break
                    if done:
                        break

        for nb in graph.get(node, ()):  # pragma: no branch
            ax, ay = coords[node]
            bx, by = coords[nb]
            dx = bx - ax
            dy = by - ay
            if abs(dx) >= abs(dy) * 2.0:
                edge_dir = 1
            elif abs(dy) >= abs(dx) * 2.0:
                edge_dir = 2
            else:
                edge_dir = 3

            next_dir = edge_dir if cur_dir == 0 else cur_dir
            if cur_dir != 0 and edge_dir != cur_dir:
                if no_turn_nodes is not None and node in no_turn_nodes and {cur_dir, edge_dir} <= {1, 2}:
                    continue
                next_dir = edge_dir

            ncost = cost + hypot(dx, dy)
            if ncost < best.get((nb, next_dir), float("inf")):
                best[(nb, next_dir)] = ncost
                heappush(pq, (ncost, nb, next_dir))

        if len(best_major_by_name) >= max_sources:
            break

    if best_major_by_name:
        best_major = list(best_major_by_name.values())
        best_major.sort(key=lambda t: t[1])
        return best_major[:max_sources]

    if best_minor_by_name:
        best_minor = list(best_minor_by_name.values())
        best_minor.sort(key=lambda t: t[1])
        return best_minor[:max_sources]

    return []


def proximity_sources(elements: Sequence[TextElement], equipment: EquipmentNode) -> List[str]:
    base_x = equipment.bbox.x0
    base_y = equipment.bbox.y0

    above: List[Tuple[float, float, str]] = []
    for el in elements:
        if el.y0 >= base_y or el.y0 < base_y - 220:
            continue
        if el.x0 < base_x - 360 or el.x0 > equipment.bbox.x1 + 360:
            continue
        m = EQUIPMENT_RE.search(el.text)
        if not m:
            continue
        name = re.sub(r"[^A-Z0-9]", "", (m.group(1) or "").upper())
        if name == equipment.name:
            continue
        if equipment.type in {"RPP", "RPU"} and name.startswith(("RPP", "RPU")):
            continue
        if equipment.type == "PDU" and name.startswith(("PDU", "RPP", "RPU")):
            continue
        above.append((base_y - el.y0, abs(el.x0 - base_x), name))

    above.sort(key=lambda t: (t[0], t[1]))
    out: List[str] = []
    for _, _, n in above:
        if n not in out:
            out.append(n)
        if len(out) >= 2:
            break
    return out


def _proximity_major_source_nodes(
    *,
    elements: Sequence[TextElement],
    equipment: EquipmentNode,
    name_to_node: Dict[str, EquipmentNode],
    region_x0: float,
    region_x1: float,
    config: ExtractorConfig,
    max_up: float = 1200.0,
    limit: int = 6,
) -> List[EquipmentNode]:
    base_y = equipment.bbox.y0

    scored: List[Tuple[float, float, EquipmentNode]] = []
    for el in elements:
        if el.y0 >= base_y or el.y0 < base_y - max_up:
            continue
        if el.x0 < region_x0 or el.x0 > region_x1:
            continue
        m = EQUIPMENT_RE.search(el.text)
        if not m:
            continue
        nm = re.sub(r"[^A-Z0-9]", "", (m.group(1) or "").upper())
        if nm == equipment.name:
            continue
        node = name_to_node.get(nm)
        if node is None:
            continue
        if equipment.type in {"RPP", "RPU"} and node.type in {"RPP", "RPU"}:
            continue
        if equipment.type == "PDU" and node.type in {"PDU", "RPP", "RPU"}:
            continue
        if node.type in config.skip_source_types:
            continue
        if node.font_size and node.font_size < config.min_source_font:
            continue

        dy = base_y - node.bbox.y0
        dx = abs(equipment.x_center - node.x_center)
        scored.append((dy, dx, node))

    scored.sort(key=lambda t: (t[0], t[1]))
    out: List[EquipmentNode] = []
    for _, __, n in scored:
        if n.name not in {x.name for x in out}:
            out.append(n)
        if len(out) >= limit:
            break
    return out


ALT_TYPES = {"MBC", "MSB", "DSG", "ATS", "BMP"}


def _rank_sources(eq: EquipmentNode, sources: Sequence[EquipmentNode], config: ExtractorConfig) -> List[EquipmentNode]:
    ranked: List[Tuple[float, EquipmentNode]] = []
    for s in sources:
        dy = eq.bbox.y0 - s.bbox.y0
        if dy <= 0:
            continue
        minor_penalty = 0.0
        if s.type in config.skip_source_types or s.font_size < config.min_source_font:
            minor_penalty = 10_000.0
        score = dy + minor_penalty
        ranked.append((score, s))
    ranked.sort(key=lambda t: t[0])
    out: List[EquipmentNode] = []
    for _, n in ranked:
        if n.name not in {x.name for x in out}:
            out.append(n)
    return out


def _rank_sources_with_cost(
    eq: EquipmentNode,
    sources: Sequence[Tuple[EquipmentNode, float]],
    config: ExtractorConfig,
) -> List[Tuple[float, EquipmentNode]]:
    ranked: List[Tuple[float, float, EquipmentNode]] = []
    for s, cost in sources:
        if cost == float("inf"):
            continue
        dy = eq.bbox.y0 - s.bbox.y0
        if dy <= 0:
            continue
        sort_cost = cost
        if s.region_id != eq.region_id:
            sort_cost = sort_cost + 1200.0
        if s.type in config.skip_source_types or s.font_size < config.min_source_font:
            sort_cost = cost + 10_000.0
        ranked.append((sort_cost, cost, s))

    ranked.sort(key=lambda t: t[0])
    out: List[Tuple[float, EquipmentNode]] = []
    for _, raw_cost, n in ranked:
        if n.name not in {x.name for __, x in out}:
            out.append((raw_cost, n))
    return out


def _choose_primary_alternate(
    eq: EquipmentNode,
    ranked: Sequence[Tuple[float, EquipmentNode]],
    config: ExtractorConfig,
    vector_found: bool,
) -> Tuple[str, str, int]:
    if not ranked:
        return "-", "-", 0

    best_cost, best = ranked[0]
    primary = best.name

    remaining = [(c, s) for (c, s) in ranked[1:]]
    alternate = "-"
    if eq.type in config.alt_types and remaining:
        remaining.sort(key=lambda t: (t[0], t[1].bbox.y0, -config.alt_type_priority.get(t[1].type, 0)))
        alt = remaining[0][1]
        if alt.type in config.skip_source_types or alt.font_size < config.min_source_font:
            # Don't report intermediate/minor devices as alternates.
            alternate = "-"
        else:
            alternate = alt.name

    confidence = 90 if vector_found else 35
    if best.type in config.skip_source_types or best.font_size < config.min_source_font:
        confidence = 35
    return primary, alternate, confidence


def assign_sources_for_page(page: "fitz.Page", page_index: int, config: Optional[ExtractorConfig] = None) -> List[dict]:
    if config is None:
        config = ExtractorConfig()
    elements = extract_text_elements(page)
    nodes = find_equipment_nodes(page_index, elements)
    regions = split_regions_by_x(nodes, page.rect.width)
    assign_region_ids(nodes, regions)

    black_graph, red_graph, coords, junctions = build_vector_graph(page, 0.0, float(page.rect.width), snap=2.0)
    bridged_black = _bridge_collinear_gaps(
        black_graph,
        coords,
        max_gap=float(config.bridge_max_gap),
        align_tol=float(config.bridge_align_tol),
    )
    bridged_red = _bridge_collinear_gaps(
        red_graph,
        coords,
        max_gap=float(config.bridge_max_gap),
        align_tol=float(config.bridge_align_tol),
    )
    no_turn_black = _compute_no_turn_nodes(black_graph, coords, junctions) if black_graph else set()
    no_turn_red = _compute_no_turn_nodes(red_graph, coords, junctions) if red_graph else set()
    node_index = build_spatial_index_for_nodes(coords, cell=50.0) if (black_graph or red_graph) else {}
    equip_index = build_spatial_index_for_equipment(nodes, cell=200.0, pad=45.0)
    name_to_node = {n.name: n for n in nodes}

    fine_black_graph: Optional[Dict[PointKey, Set[PointKey]]] = None
    fine_red_graph: Optional[Dict[PointKey, Set[PointKey]]] = None
    fine_coords: Optional[Dict[PointKey, Tuple[float, float]]] = None
    fine_junctions: Optional[Set[PointKey]] = None
    fine_node_index: Optional[Dict[Tuple[int, int], List[PointKey]]] = None
    fine_no_turn_black: Optional[Set[PointKey]] = None
    fine_no_turn_red: Optional[Set[PointKey]] = None

    if config.debug_page in {0, page_index + 1} and config.debug_equipment:
        b_edges = sum(len(v) for v in black_graph.values()) // 2 if black_graph else 0
        r_edges = sum(len(v) for v in red_graph.values()) // 2 if red_graph else 0
        print(
            f"DEBUG page={page_index + 1} black nodes={len(black_graph)} edges={b_edges} bridged={bridged_black} no_turn={len(no_turn_black)} red nodes={len(red_graph)} edges={r_edges} bridged={bridged_red} no_turn={len(no_turn_red)} junctions={len(junctions)} vision={config.vision_enabled} provider={config.vision_provider.upper()} model={config.vision_model}"
        )

    for eq in nodes:
        ports: List[PointKey] = []
        if black_graph:
            ports = find_port_nodes(eq, node_index, coords, black_graph, cell=50.0)

        if config.debug_equipment and eq.name == config.debug_equipment and (config.debug_page in {0, eq.page_index + 1}):
            port_desc = ", ".join([f"{p}:{len(black_graph.get(p, ())) }@({coords[p][0]:.1f},{coords[p][1]:.1f})" for p in ports[:10]])
            print(f"DEBUG {eq.name} page={eq.page_index + 1} ports={len(ports)} {port_desc}")

        rx0, rx1 = regions[eq.region_id]

        vector_found = False
        sources_with_cost: List[Tuple[EquipmentNode, float]] = []
        best_cost_by_name: Dict[str, float] = {}
        best_node_by_name: Dict[str, EquipmentNode] = {}
        if black_graph:
            for p in ports[:6]:
                hits = trace_upstream_source(p, black_graph, coords, no_turn_black, equip_index, eq, config=config)
                if config.debug_equipment and eq.name == config.debug_equipment and (config.debug_page in {0, eq.page_index + 1}):
                    if hits:
                        print(
                            f"DEBUG {eq.name} page={eq.page_index + 1} port={p} hits="
                            + ", ".join([f"{h[0].name}:{h[1]:.1f}" for h in hits])
                        )
                    else:
                        print(f"DEBUG {eq.name} page={eq.page_index + 1} port={p} hits=")

                if not hits:
                    continue
                vector_found = True

                for src, cost in hits:
                    prev = best_cost_by_name.get(src.name)
                    if prev is None or cost < prev:
                        best_cost_by_name[src.name] = cost
                        best_node_by_name[src.name] = src

        for nm, node in best_node_by_name.items():
            sources_with_cost.append((node, best_cost_by_name[nm]))

        if not sources_with_cost:
            prox = proximity_sources(elements, eq)
            for nm in prox:
                node = name_to_node.get(nm)
                if node is None:
                    continue
                if node.type == eq.type:
                    continue
                dx = abs(eq.x_center - node.x_center)
                dy = abs(eq.bbox.y0 - node.bbox.y0)
                sources_with_cost.append((node, hypot(dx, dy)))

        if not sources_with_cost:
            major_candidates = _proximity_major_source_nodes(
                elements=elements,
                equipment=eq,
                name_to_node=name_to_node,
                region_x0=rx0,
                region_x1=rx1,
                config=config,
            )
            for cand in major_candidates:
                if cand.name not in {n.name for n, _ in sources_with_cost}:
                    dx = abs(eq.x_center - cand.x_center)
                    dy = abs(eq.bbox.y0 - cand.bbox.y0)
                    sources_with_cost.append((cand, hypot(dx, dy)))

        ranked = _rank_sources_with_cost(eq, sources_with_cost, config)

        if config.debug_equipment and eq.name == config.debug_equipment and (config.debug_page in {0, eq.page_index + 1}):
            print(
                f"DEBUG {eq.name} page={eq.page_index + 1} ranked="
                + ", ".join([f"{n.name}:{c:.1f}" for c, n in ranked[:10]])
            )

            # If vector tracing only found minor/intermediate devices, try a broader
            # region-wide proximity scan for major upstream equipment.
        if ranked and (
            ranked[0][1].type in config.skip_source_types
            or (ranked[0][1].font_size and ranked[0][1].font_size < config.min_source_font)
        ):
            major_candidates = _proximity_major_source_nodes(
                elements=elements,
                equipment=eq,
                name_to_node=name_to_node,
                region_x0=rx0,
                region_x1=rx1,
                config=config,
            )
            for cand in major_candidates:
                if cand.name not in {n.name for n, _ in sources_with_cost}:
                    dx = abs(eq.x_center - cand.x_center)
                    dy = abs(eq.bbox.y0 - cand.bbox.y0)
                    sources_with_cost.append((cand, hypot(dx, dy)))
            ranked = _rank_sources_with_cost(eq, sources_with_cost, config)

        primary, alternate, confidence = _choose_primary_alternate(eq, ranked, config, vector_found)
        eq.primary_from = primary
        eq.alternate_from = alternate

        if eq.type in config.alt_types and red_graph:
            red_ports = find_port_nodes(eq, node_index, coords, red_graph, cell=50.0)
            best_red_cost: Dict[str, float] = {}
            best_red_node: Dict[str, EquipmentNode] = {}
            for p in red_ports[:6]:
                hits = trace_upstream_source(p, red_graph, coords, no_turn_red, equip_index, eq, config=config)
                if config.debug_equipment and eq.name == config.debug_equipment and (config.debug_page in {0, eq.page_index + 1}):
                    if hits:
                        print(
                            f"DEBUG {eq.name} page={eq.page_index + 1} red_port={p} hits="
                            + ", ".join([f"{h[0].name}:{h[1]:.1f}" for h in hits])
                        )
                for src, cost in hits:
                    prev = best_red_cost.get(src.name)
                    if prev is None or cost < prev:
                        best_red_cost[src.name] = cost
                        best_red_node[src.name] = src

            if best_red_node:
                red_ranked = _rank_sources_with_cost(eq, [(best_red_node[nm], best_red_cost[nm]) for nm in best_red_node], config)
                for _, cand in red_ranked:
                    if cand.name != eq.primary_from:
                        eq.alternate_from = cand.name
                        break

        if (
            black_graph
            and vector_found
            and eq.type in config.alt_types
            and eq.alternate_from == "-"
            and len(best_node_by_name) < 2
        ):
            if fine_black_graph is None or fine_coords is None or fine_node_index is None:
                fine_black_graph, fine_red_graph, fine_coords, fine_junctions = build_vector_graph(
                    page,
                    0.0,
                    float(page.rect.width),
                    snap=max(0.8, float(config.fine_vector_snap)),
                )
                if fine_black_graph:
                    _bridge_collinear_gaps(fine_black_graph, fine_coords)
                if fine_red_graph:
                    _bridge_collinear_gaps(fine_red_graph, fine_coords)
                fine_no_turn_black = _compute_no_turn_nodes(fine_black_graph, fine_coords, fine_junctions or set()) if fine_black_graph else set()
                fine_no_turn_red = _compute_no_turn_nodes(fine_red_graph, fine_coords, fine_junctions or set()) if fine_red_graph else set()
                fine_node_index = build_spatial_index_for_nodes(fine_coords, cell=50.0) if (fine_black_graph or fine_red_graph) else {}

            if fine_black_graph and fine_coords and fine_node_index:
                fine_ports = find_port_nodes(eq, fine_node_index, fine_coords, fine_black_graph, cell=50.0)
                if config.debug_equipment and eq.name == config.debug_equipment and (config.debug_page in {0, eq.page_index + 1}):
                    print(f"DEBUG {eq.name} page={eq.page_index + 1} fine_ports={len(fine_ports)}")

                for p in fine_ports[:6]:
                    hits = trace_upstream_source(p, fine_black_graph, fine_coords, fine_no_turn_black, equip_index, eq, config=config)
                    if not hits:
                        continue
                    vector_found = True
                    if config.debug_equipment and eq.name == config.debug_equipment and (config.debug_page in {0, eq.page_index + 1}):
                        print(
                            f"DEBUG {eq.name} page={eq.page_index + 1} fine_port={p} hits="
                            + ", ".join([f"{h[0].name}:{h[1]:.1f}" for h in hits])
                        )
                    for src, cost in hits:
                        prev = best_cost_by_name.get(src.name)
                        if prev is None or cost < prev:
                            best_cost_by_name[src.name] = cost
                            best_node_by_name[src.name] = src

                sources_with_cost = [(best_node_by_name[nm], best_cost_by_name[nm]) for nm in best_node_by_name]
                ranked = _rank_sources_with_cost(eq, sources_with_cost, config)
                primary, alternate, confidence = _choose_primary_alternate(eq, ranked, config, vector_found)
                eq.primary_from = primary
                eq.alternate_from = alternate

        should_try_vision = False
        if eq.type in config.alt_types:
            if confidence < config.vision_min_confidence:
                should_try_vision = True
            elif eq.alternate_from == "-":
                should_try_vision = True

        if config.vision_enabled and should_try_vision:
            try:
                from vision_clients import ask_vision_for_sources
                import json
                import re

                if config.debug_equipment and eq.name == config.debug_equipment and (config.debug_page in {0, eq.page_index + 1}):
                    print(
                        f"DEBUG {eq.name} page={eq.page_index + 1} vision provider={config.vision_provider.upper()} model={config.vision_model}"
                    )

                clip = fitz.Rect(
                    max(0, eq.bbox.x0 - 600),
                    max(0, eq.bbox.y0 - 1000),
                    min(page.rect.width, eq.bbox.x1 + 600),
                    min(page.rect.height, eq.bbox.y1 + 250),
                )
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip, alpha=False)
                png = pix.tobytes("png")

                instruction = "Alternative sources exist mainly for MBC, MSB, DSG, ATS, BMP."
                resp = ask_vision_for_sources(
                    png_bytes=png,
                    equipment_name=eq.name,
                    instruction=instruction,
                    provider=config.vision_provider.upper(),
                    model=config.vision_model,
                )

                payload = resp.strip()
                if not payload.startswith("{"):
                    m = re.search(r"\{[\s\S]*\}", payload)
                    if m:
                        payload = m.group(0)
                data = json.loads(payload)
                v_primary = str(data.get("primary", "-") or "-")
                v_alt = str(data.get("alternate", "-") or "-")
                eq.primary_from = v_primary if v_primary else "-"
                eq.alternate_from = v_alt if (eq.type in config.alt_types and v_alt) else "-"
            except Exception:
                pass

        if config.debug_equipment and eq.name == config.debug_equipment and (config.debug_page in {0, eq.page_index + 1}):
            print(
                f"DEBUG {eq.name} page={eq.page_index + 1} chosen primary={eq.primary_from} alternate={eq.alternate_from} confidence={confidence} vector={vector_found}"
            )

    out: List[dict] = []
    for n in nodes:
        out.append(
            {
                "Equipment": n.name,
                "Type": n.type,
                "Properties": ", ".join(n.properties) if n.properties else "-",
                "Primary From": n.primary_from or "-",
                "Alternate From": n.alternate_from or "-",
            }
        )

    out.sort(key=lambda r: r["Equipment"])
    return out


def extract_system_pdf(pdf_path: str) -> List[dict]:
    return extract_system_pdf_with_config(pdf_path, ExtractorConfig())


def extract_system_pdf_with_config(pdf_path: str, config: ExtractorConfig) -> List[dict]:
    doc = fitz.open(pdf_path)
    try:
        all_rows: List[dict] = []
        for page_index in range(len(doc)):
            page = doc[page_index]
            all_rows.extend(assign_sources_for_page(page, page_index, config=config))
        return all_rows
    finally:
        doc.close()
