"""Картинка связей для Телеграма: сущность в центре, вокруг — с чем она связана.

Телеграм не показывает интерактивный граф, как страница, — только картинки и
кнопки. Поэтому бот рисует окрестность одной сущности: её связи из фактов
(стрелка — направление факта, подпись — связь словами статьи) и то, что в неё
входит. Кнопки под картинкой ведут к соседям — так по графу ходят, как щелчками
на странице.

Весь граф (сотни узлов) на картинке был бы кашей, поэтому показываются самые
подтверждённые связи — по числу статей, до NEIGHBORS соседей.

Рисует Pillow (уже стоит вместе с sentence-transformers); раскладка — своя, кольцами.
Шрифт — системный с кириллицей: Segoe UI / Arial на Windows, DejaVu на Linux.
"""

from __future__ import annotations

import io
import math
import textwrap
from pathlib import Path
from typing import Any

NEIGHBORS = 12  # соседей на картинке самое большее
W, H = 1400, 1000
PAD_X, PAD_Y = 200, 150  # поля: подписи узлов не должны вылезать за край

BG = (246, 243, 238)
INK = (31, 27, 22)
MUTED = (107, 100, 92)
CENTER = (194, 86, 43)  # сама сущность
PART = (122, 92, 158)  # то, что в неё входит
NODE = (59, 110, 165)  # связанные сущности
EDGE = (201, 119, 46)
EDGE_TEXT = (140, 70, 16)
PART_OF = "входит в"

_FONTS = {
    False: [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ],
    True: [
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ],
}


def font(size: int, bold: bool = False) -> Any:
    from PIL import ImageFont

    for path in _FONTS[bold] + _FONTS[False]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def neighborhood(
    data: dict[str, Any], limit: int = NEIGHBORS
) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Факты → рёбра (от, связь, к) и соседи по порядку важности.

    Факты в данных уже отсортированы по числу статей. Факт о том, что входит в
    сущность («зона — …»), рисуется от этой части, а часть соединяется с центром
    стрелкой «входит в»."""
    center = data["name"]
    edges: list[tuple[str, str, str]] = []
    order: list[str] = []

    def node(name: str) -> bool:
        if name == center or name in order:
            return True
        if len(order) >= limit:
            return False
        order.append(name)
        return True

    for row in data.get("facts") or []:
        about, other = row["about"], row["other"]
        if about != center and not node(about):
            continue
        if not node(other):
            continue
        if about != center and (about, PART_OF, center) not in edges:
            edges.append((about, PART_OF, center))
        a, b = (about, other) if row["direction"] == "→" else (other, about)
        if (a, row["relation"], b) not in edges:
            edges.append((a, row["relation"], b))
    for part in data.get("includes") or []:  # части без своих фактов — тоже видны
        if node(part) and (part, PART_OF, center) not in edges:
            edges.append((part, PART_OF, center))
    return edges, order


def layout(center: str, names: list[str], edges: Any) -> dict[str, tuple[float, float]]:
    """Кольца: связанные с центром — на внутреннем, связанные через них — на внешнем,
    в секторе своего «родителя». Сектор тем шире, чем больше у узла детей: так
    подписи не наезжают друг на друга, как было при пружинной раскладке."""
    if not names:
        return {center: (W / 2, H / 2)}
    near = [n for n in names if any(center in (a, b) and n in (a, b) for a, _, b in edges)]
    children: dict[str, list[str]] = {n: [] for n in near}
    for n in names:
        if n in children:
            continue
        parent = next(
            (
                a if b == n else b
                for a, _, b in edges
                if n in (a, b) and (a if b == n else b) in children
            ),
            None,
        )
        if parent is None:  # связан ни с чем из ближних — в ближние
            children[n] = []
            near.append(n)
        else:
            children[parent].append(n)
    outer = any(children.values())
    r_near = 0.56 if outer else 1.0
    weights = [max(1, len(children[n])) for n in near]
    total = sum(weights)
    rx, ry = W / 2 - PAD_X, H / 2 - PAD_Y
    pos = {center: (W / 2, H / 2)}

    def at(angle: float, r: float) -> tuple[float, float]:
        return W / 2 + math.cos(angle) * r * rx, H / 2 + math.sin(angle) * r * ry

    angle = -math.pi / 2 - math.pi * weights[0] / total  # первый сектор — сверху
    for n, w in zip(near, weights, strict=True):
        sector = 2 * math.pi * w / total
        pos[n] = at(angle + sector / 2, r_near)
        kids = children[n]
        for i, kid in enumerate(kids):
            pos[kid] = at(angle + sector * (i + 0.5) / len(kids), 1.0)
        angle += sector
    return pos


def _arrow(draw: Any, a: Any, b: Any, r_to: float, width: int = 3) -> None:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    tip = (b[0] - ux * (r_to + 4), b[1] - uy * (r_to + 4))
    draw.line([a, tip], fill=EDGE, width=width)
    size = 16
    left = (tip[0] - ux * size - uy * size * 0.55, tip[1] - uy * size + ux * size * 0.55)
    right = (tip[0] - ux * size + uy * size * 0.55, tip[1] - uy * size - ux * size * 0.55)
    draw.polygon([tip, left, right], fill=EDGE)


def _label(
    draw: Any, xy: Any, text: str, fnt: Any, fill: Any, box: bool = False, width: int = 22
) -> None:
    lines = textwrap.wrap(text, width=width)[:3]
    if len(textwrap.wrap(text, width=width)) > 3:
        lines[-1] = lines[-1][: width - 1] + "…"
    body = "\n".join(lines)
    left, top, right, bottom = draw.multiline_textbbox(
        xy, body, font=fnt, anchor="ma", align="center", spacing=2
    )
    if box:
        draw.rounded_rectangle((left - 6, top - 3, right + 6, bottom + 4), radius=6, fill=BG)
    draw.multiline_text(
        xy,
        body,
        font=fnt,
        fill=fill,
        anchor="ma",
        align="center",
        spacing=2,
        stroke_width=0 if box else 3,
        stroke_fill=BG,
    )


def render(data: dict[str, Any], limit: int = NEIGHBORS) -> tuple[bytes, list[str]]:
    """Картинка PNG и соседи в том порядке, в каком их показывать кнопками."""
    from PIL import Image, ImageDraw

    center = data["name"]
    edges, names = neighborhood(data, limit)
    pos = layout(center, names, edges)
    parts = set(data.get("includes") or [])
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    radius = {n: 30 if n == center else 18 for n in [center, *names]}

    for a, _, b in edges:
        _arrow(draw, pos[a], pos[b], radius[b])
    outer = {n for n in names if all(center not in (a, b) for a, _, b in edges if n in (a, b))}
    name_font, small_name, center_font = font(22), font(19), font(28, bold=True)
    for n in [*names, center]:
        x, y = pos[n]
        r = radius[n]
        color = CENTER if n == center else PART if n in parts else NODE
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=BG, width=3)
        two_rings = bool(outer)
        fnt = center_font if n == center else small_name if n in outer or two_rings else name_font
        _label(draw, (x, y + r + 6), n, fnt, INK, width=20 if n == center or not two_rings else 18)

    # Подписи связей — последними, на подложке: читаются, даже если легли на имя узла.
    small = font(19)
    for a, relation, b in edges:
        if relation == PART_OF and b == center:
            continue  # «входит в» — цвет узла, подпись лишняя
        t = 0.45 if center in (a, b) else 0.5
        x = pos[a][0] + (pos[b][0] - pos[a][0]) * t
        y = pos[a][1] + (pos[b][1] - pos[a][1]) * t - 12
        dx, dy = pos[b][0] - pos[a][0], pos[b][1] - pos[a][1]
        if abs(dx) < abs(dy) / 2:  # почти вертикальное ребро — подпись сбоку
            x += 12 + draw.textlength(relation[:22], font=small) / 2
        _label(draw, (x, y), relation, small, EDGE_TEXT, box=True, width=22)

    title = font(30, bold=True)
    draw.text((32, 24), center, font=title, fill=INK)
    facts = len(data.get("facts") or [])
    shown = f"на картинке — {len(names)} главных" if facts > len(names) else "все"
    draw.text(
        (32, 66),
        f"фактов: {facts}, статей: {data.get('documents', 0)} · {shown}. "
        "Стрелка — направление факта, подпись — связь из статьи.",
        font=font(20),
        fill=MUTED,
    )
    legend = [(CENTER, "сущность"), (PART, "входит в неё"), (NODE, "связана с ней")]
    x = 32
    for color, text in legend:
        draw.ellipse((x, H - 44, x + 18, H - 26), fill=color)
        draw.text((x + 26, H - 48), text, font=font(20), fill=MUTED)
        x += 60 + int(draw.textlength(text, font=font(20)))
    draw.text(
        (W - 32, H - 48), "KBRAG · факты из статей базы", font=font(20), fill=MUTED, anchor="ra"
    )

    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue(), names
