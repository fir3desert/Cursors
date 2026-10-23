#!/usr/bin/env python3
"""Modern Plasma - generador de tema de cursores para Windows (.cur / .ani).
Uso: python generar_cursores.py [carpeta_salida] [tamano_animados]
Requiere: pillow, numpy, scipy
"""
import os, io, struct, math, sys, colorsys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageFont
from scipy import ndimage as ndi

OUT = sys.argv[1] if len(sys.argv) > 1 else "tema"
ANI_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 48
CUR_SIZES = [32, 48, 64, 96, 128]
SS = 4
ACCENT = (53, 132, 228)
RED = (229, 72, 77)
WOOD = (240, 208, 160)
GRAPH = (40, 40, 48)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
VARIANT = sys.argv[3] if len(sys.argv) > 3 else "oscuro"
LIGHT = VARIANT == "blanco"
RGB = VARIANT == "rgb"
HUE = 0.0
THEME = "Modern Plasma Blanco" if LIGHT else ("Modern Plasma RGB" if RGB else "Modern Plasma")
EDGE = (22, 22, 28) if LIGHT else (255, 255, 255)
G1, G2 = ((255, 255, 255), (214, 217, 226)) if LIGHT else ((62, 64, 80), (10, 10, 14))
SEP = (120, 124, 140) if LIGHT else (150, 152, 165)


# ---------------------------------------------------------------- primitivas
class Cv:
    def __init__(s, N):
        s.N = N
        s.S = N * SS

    def new(s):
        return Image.new("L", (s.S, s.S), 0)

    def _open(s, m, rr):
        if rr <= 0:
            return m
        r = rr * s.S
        a = np.array(m) > 127
        e = ndi.distance_transform_edt(a) > r
        d = ndi.distance_transform_edt(~e) <= r
        return Image.fromarray((d * 255).astype(np.uint8))

    def poly(s, pts, rr=0.0):
        m = s.new()
        ImageDraw.Draw(m).polygon([(x * s.S, y * s.S) for x, y in pts], fill=255)
        return s._open(m, rr)

    def rrect(s, x0, y0, x1, y1, r=0.0):
        m = s.new()
        ImageDraw.Draw(m).rounded_rectangle(
            [x0 * s.S, y0 * s.S, x1 * s.S, y1 * s.S], radius=r * s.S, fill=255)
        return m

    def circle(s, cx, cy, r):
        m = s.new()
        ImageDraw.Draw(m).ellipse(
            [(cx - r) * s.S, (cy - r) * s.S, (cx + r) * s.S, (cy + r) * s.S], fill=255)
        return m

    def line(s, pts, w):
        m = s.new()
        d = ImageDraw.Draw(m)
        P = [(x * s.S, y * s.S) for x, y in pts]
        d.line(P, fill=255, width=max(1, int(w * s.S)), joint="curve")
        r = w * s.S / 2
        for x, y in (P[0], P[-1]):
            d.ellipse([x - r, y - r, x + r, y + r], fill=255)
        return m

    def rot(s, m, ang, cx=0.5, cy=0.5):
        return m.rotate(ang, resample=Image.BICUBIC, center=(cx * s.S, cy * s.S))

    def text(s, ch, cx, cy, size):
        m = s.new()
        f = ImageFont.truetype(FONT, int(size * s.S))
        ImageDraw.Draw(m).text((cx * s.S, cy * s.S), ch, font=f, fill=255, anchor="mm")
        return m


def U(*ms):
    out = ms[0]
    for m in ms[1:]:
        out = ImageChops.lighter(out, m)
    return out


def L(m, f="dark", a=1.0):
    return (m, f, a)


_grad = {}


def hcol(h, s=0.85, v=1.0):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def hsv_np(h, s, v):
    h = h % 1.0
    i = np.floor(h * 6).astype(int)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i = i % 6
    return (np.choose(i, [v, q, p, p, t, v]), np.choose(i, [t, v, v, q, p, p]), np.choose(i, [p, p, t, v, v, q]))


def gradient(S):
    if RGB:
        key = (S, round(HUE, 4))
        if key not in _grad:
            if len(_grad) > 40:
                _grad.clear()
            yy, xx = np.mgrid[0:S, 0:S]
            t = np.clip((xx + yy) / (S * 1.1), 0, 1)
            r, g, b = hsv_np(HUE + 0.15 * t, 0.88, 1 - 0.28 * t)
            _grad[key] = Image.fromarray(np.dstack([(r * 255).astype(np.uint8), (g * 255).astype(np.uint8),
                                                    (b * 255).astype(np.uint8), np.full((S, S), 255, np.uint8)]), "RGBA")
        return _grad[key]
    if S not in _grad:
        yy, xx = np.mgrid[0:S, 0:S]
        t = np.clip((xx + yy) / (S * 1.1), 0, 1)[..., None]
        c1 = np.array(G1)
        c2 = np.array(G2)
        g = (c1 * (1 - t) + c2 * t).astype(np.uint8)
        _grad[S] = Image.fromarray(np.dstack([g, np.full((S, S, 1), 255, np.uint8)]), "RGBA")
    return _grad[S]


def compose(c, layers, outline=0.02):
    S, N = c.S, c.N
    union = Image.new("L", (S, S), 0)
    for m, f, a in layers:
        union = ImageChops.lighter(union, m)
    inv = ~(np.array(union) > 127)
    outl = Image.fromarray(((ndi.distance_transform_edt(inv) <= outline * S) * 255).astype(np.uint8))
    sh = outl.filter(ImageFilter.GaussianBlur(0.03 * S))
    sh = sh.transform(sh.size, Image.AFFINE, (1, 0, -0.025 * S, 0, 1, -0.04 * S)).point(lambda v: int(v * 0.5))
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    if RGB:
        glow = Image.fromarray(((ndi.distance_transform_edt(inv) <= (outline + 0.025) * S) * 255).astype(np.uint8))
        glow = glow.filter(ImageFilter.GaussianBlur(0.045 * S)).point(lambda v: int(v * 0.6))
        img.paste(hcol(HUE, 0.9, 1.0) + (255,), (0, 0), glow)
        sh = sh.point(lambda v: int(v * 0.7))
    img.paste((0, 0, 0, 255), (0, 0), sh)
    img.paste(EDGE + (255,), (0, 0), outl)
    for m, f, a in layers:
        fill = gradient(S) if f == "dark" else Image.new("RGBA", (S, S), tuple(f) + (255,))
        mm = m if a >= 1 else m.point(lambda v, a=a: int(v * a))
        img.paste(fill, (0, 0), mm)
    return img.resize((N, N), Image.LANCZOS)


# ---------------------------------------------------------------- formas
ARROW = [(0, 0), (144, 58), (77, 82), (60, 142)]


def arrow(c, ext=0.70, pad=0.06, rr=0.018):
    k = ext / 144
    return c.poly([(pad + x * k, pad + y * k) for x, y in ARROW], rr)


def spinner(c, cx, cy, R, dr, n, t):
    out = []
    head = t * 2 * math.pi
    for i in range(n):
        ang = i * 2 * math.pi / n
        d = (head - ang) % (2 * math.pi)
        w = max(0.0, 1 - d / (2 * math.pi * 0.8))
        a = 0.15 + 0.85 * w
        sc = 0.6 + 0.4 * w
        out.append(L(c.circle(cx + R * math.sin(ang), cy - R * math.cos(ang), dr * sc), "dark", a))
    return out


def badge(c, cx=0.72, cy=0.72, r=0.19, color=ACCENT):
    return c.circle(cx, cy, r)


def dbl_arrow(c, ang=0, h=0.40, hl_=0.21, hh=0.17):
    shaft = c.rrect(0.5 - (h - hl_ + 0.03), 0.5 - 0.035, 0.5 + (h - hl_ + 0.03), 0.5 + 0.035, 0.0)
    hl = c.poly([(0.5 - h, 0.5), (0.5 - h + hl_, 0.5 - hh), (0.5 - h + hl_, 0.5 + hh)], 0.0)
    hr = c.poly([(0.5 + h, 0.5), (0.5 + h - hl_, 0.5 - hh), (0.5 + h - hl_, 0.5 + hh)], 0.0)
    m = c._open(U(shaft, hl, hr), 0.012)
    return c.rot(m, ang) if ang else m


# ---------------------------------------------------------------- cursores
# cada funcion: (Cv, t) -> (layers, hotspot normalizado)
def c_normal(c, t):
    return [L(arrow(c))], (0.08, 0.08)


def c_help(c, t):
    b = badge(c)
    q = c.text("?", 0.72, 0.735, 0.30)
    return [L(arrow(c, 0.58, 0.06, 0.014)), L(b, ACCENT), L(q, (255, 255, 255))], (0.077, 0.077)


def c_working(c, t):
    return ([L(arrow(c, 0.58, 0.06, 0.014))] + spinner(c, 0.72, 0.72, 0.15, 0.033, 8, t)), (0.077, 0.077)


def c_busy(c, t):
    return spinner(c, 0.5, 0.5, 0.30, 0.058, 12, t), (0.5, 0.5)


def c_cross(c, t):
    m = U(c.rrect(0.10, 0.475, 0.42, 0.525, 0.02), c.rrect(0.58, 0.475, 0.90, 0.525, 0.02),
          c.rrect(0.475, 0.10, 0.525, 0.42, 0.02), c.rrect(0.475, 0.58, 0.525, 0.90, 0.02))
    return [L(m)], (0.5, 0.5)


def c_text(c, t):
    m = U(c.rrect(0.465, 0.12, 0.535, 0.88, 0.02), c.rrect(0.35, 0.12, 0.65, 0.19, 0.03),
          c.rrect(0.35, 0.81, 0.65, 0.88, 0.03))
    return [L(m)], (0.5, 0.5)


def c_pen(c, t):
    x0, y0 = 0.12, 0.84
    prog = min(1.0, t / 0.8)

    def path(u):
        return (x0 + 0.42 * u, y0 + 0.05 * math.sin(u * math.pi * 5) * (0.4 + 0.6 * u))
    pts = [path(prog * i / 40) for i in range(41)]
    trail = c.line(pts, 0.045) if prog > 0 else c.circle(x0, y0, 0.022)
    tx, ty = pts[-1]
    cone = c.poly([(tx, ty), (tx - 0.05, ty - 0.14), (tx + 0.05, ty - 0.14)], 0.0)
    body = c.rrect(tx - 0.05, ty - 0.54, tx + 0.05, ty - 0.12, 0.025)
    band = c.rrect(tx - 0.05, ty - 0.54, tx + 0.05, ty - 0.45, 0.025)
    lead = c.poly([(tx, ty), (tx - 0.02, ty - 0.055), (tx + 0.02, ty - 0.055)], 0.0)
    A = -30
    cone, body, band, lead = [c.rot(m, A, tx, ty) for m in (cone, body, band, lead)]
    return [L(trail), L(body), L(cone, WOOD), L(lead, GRAPH), L(band, hcol(HUE + 0.5) if RGB else ACCENT)], (x0, y0)


def c_no(c, t):
    b = badge(c)
    slash = c.line([(0.72 - 0.085, 0.72 + 0.085), (0.72 + 0.085, 0.72 - 0.085)], 0.05)
    return [L(arrow(c, 0.58, 0.06, 0.014)), L(b, RED), L(slash, (255, 255, 255))], (0.077, 0.077)


def c_vert(c, t):
    return [L(dbl_arrow(c, 90))], (0.5, 0.5)


def c_horz(c, t):
    return [L(dbl_arrow(c, 0))], (0.5, 0.5)


def c_dgn1(c, t):  # NO-SE  (\)
    return [L(dbl_arrow(c, -45))], (0.5, 0.5)


def c_dgn2(c, t):  # NE-SO  (/)
    return [L(dbl_arrow(c, 45))], (0.5, 0.5)


def c_move(c, t):
    m = U(dbl_arrow(c, 0, 0.42, 0.15, 0.13), dbl_arrow(c, 90, 0.42, 0.15, 0.13))
    return [L(m)], (0.5, 0.5)


def c_alt(c, t):
    m = c.poly([(0.5, 0.06), (0.70, 0.34), (0.57, 0.34), (0.57, 0.92), (0.43, 0.92),
                (0.43, 0.34), (0.30, 0.34)], 0.015)
    return [L(m)], (0.5, 0.085)


def c_link(c, t):
    index = c.rrect(0.30, 0.08, 0.42, 0.58, 0.06)
    palm = c.rrect(0.26, 0.42, 0.72, 0.84, 0.10)
    f2 = c.rrect(0.42, 0.38, 0.52, 0.62, 0.05)
    f3 = c.rrect(0.52, 0.40, 0.62, 0.62, 0.05)
    f4 = c.rrect(0.62, 0.44, 0.72, 0.64, 0.05)
    thumb = c.rot(c.rrect(0.12, 0.50, 0.36, 0.63, 0.06), 25, 0.34, 0.60)
    seps = U(c.line([(0.42, 0.47), (0.42, 0.60)], 0.018), c.line([(0.52, 0.46), (0.52, 0.60)], 0.018),
             c.line([(0.62, 0.48), (0.62, 0.62)], 0.018))
    return [L(U(index, palm, f2, f3, f4, thumb)), L(seps, SEP)], (0.36, 0.095)


def c_pin(c, t):
    m = U(c.circle(0.5, 0.35, 0.24), c.poly([(0.5 - 0.215, 0.44), (0.5 + 0.215, 0.44), (0.5, 0.92)], 0.0))
    return [L(m, RED), L(c.circle(0.5, 0.35, 0.085), (255, 255, 255))], (0.5, 0.9)


def c_person(c, t):
    b = badge(c)
    head = c.circle(0.72, 0.675, 0.045)
    sh = ImageChops.multiply(c.circle(0.72, 0.865, 0.105), c.circle(0.72, 0.72, 0.175))
    return [L(arrow(c, 0.58, 0.06, 0.014)), L(b, ACCENT), L(U(head, sh), (255, 255, 255))], (0.077, 0.077)


# clave, etiqueta, archivo, funcion, animacion (frames, jiffies por frame) o None
CURSORS = [
    ("Arrow", "Normal", "normal", c_normal, None),
    ("Help", "Ayuda", "ayuda", c_help, None),
    ("AppStarting", "Trabajando", "trabajando", c_working, (24, 2)),
    ("Wait", "Ocupado", "ocupado", c_busy, (24, 2)),
    ("Crosshair", "Precision", "precision", c_cross, None),
    ("IBeam", "Texto", "texto", c_text, None),
    ("NWPen", "Escritura", "escritura", c_pen, (28, 3)),
    ("No", "No disponible", "no_disponible", c_no, None),
    ("SizeNS", "Vertical", "vertical", c_vert, None),
    ("SizeWE", "Horizontal", "horizontal", c_horz, None),
    ("SizeNWSE", "Diagonal 1", "diagonal1", c_dgn1, None),
    ("SizeNESW", "Diagonal 2", "diagonal2", c_dgn2, None),
    ("SizeAll", "Mover", "mover", c_move, None),
    ("UpArrow", "Alternativa", "alternativa", c_alt, None),
    ("Hand", "Enlace", "enlace", c_link, None),
    ("Pin", "Ubicacion", "ubicacion", c_pin, None),
    ("Person", "Persona", "persona", c_person, None),
]


def render(fn, N, t=0.0, hue=0.0):
    global HUE
    HUE = hue
    c = Cv(N)
    layers, hot = fn(c, t)
    return compose(c, layers), (int(round(hot[0] * N)), int(round(hot[1] * N)))


# ---------------------------------------------------------------- formatos
def dib(im):
    N = im.width
    a = np.array(im.convert("RGBA"))
    bgra = a[..., [2, 1, 0, 3]][::-1]
    pix = bgra.tobytes()
    mask = bytes(((N + 31) // 32) * 4 * N)
    hdr = struct.pack("<IiiHHIIiiII", 40, N, 2 * N, 1, 32, 0, len(pix) + len(mask), 0, 0, 0, 0)
    return hdr + pix + mask


def cur_bytes(items):  # items: [(img, hot)]
    data = [dib(im) for im, _ in items]
    out = struct.pack("<HHH", 0, 2, len(items))
    off = 6 + 16 * len(items)
    for (im, hot), d in zip(items, data):
        n = im.width
        out += struct.pack("<BBBBHHII", n % 256, n % 256, 0, 0, hot[0], hot[1], len(d), off)
        off += len(d)
    return out + b"".join(data)


def chunk(tag, data):
    pad = b"\0" if len(data) % 2 else b""
    return tag + struct.pack("<I", len(data)) + data + pad


def ani_bytes(frames, hot, rates):
    n = frames[0].width
    jif = rates[0]
    anih = struct.pack("<IIIIIIIII", 36, len(frames), len(frames), n, n, 32, 1, jif, 1)
    rate = struct.pack("<%dI" % len(frames), *rates)
    fram = b"fram" + b"".join(chunk(b"icon", cur_bytes([(f, hot)])) for f in frames)
    body = b"ACON" + chunk(b"anih", anih) + chunk(b"rate", rate) + chunk(b"LIST", fram)
    return b"RIFF" + struct.pack("<I", len(body)) + body


# ---------------------------------------------------------------- build
def build():
    os.makedirs(OUT, exist_ok=True)
    for key, label, fname, fn, anim in CURSORS:
        if RGB:
            frames, hot, rates = [], None, []
            for i, (t, hue, rate) in enumerate(rgb_schedule(key, anim)):
                im, hot = render(fn, ANI_SIZE, t, hue)
                frames.append(im)
                rates.append(rate)
            open(os.path.join(OUT, fname + ".ani"), "wb").write(ani_bytes(frames, hot, rates))
        elif anim is None:
            items = [render(fn, n) for n in CUR_SIZES]
            open(os.path.join(OUT, fname + ".cur"), "wb").write(cur_bytes(items))
        else:
            nf, jif = anim
            frames, hot = [], None
            for i in range(nf):
                t = i / nf if key in ("AppStarting", "Wait") else i / (nf - 1)
                im, hot = render(fn, ANI_SIZE, t)
                frames.append(im)
            rates = [jif] * (nf - 1) + [jif if jif > 2 else 25]
            open(os.path.join(OUT, fname + ".ani"), "wb").write(ani_bytes(frames, hot, rates))
    write_inf()


def rgb_schedule(key, anim):
    """(t, hue, jiffies) por fotograma. Ciclo de color lento (~8 s)."""
    out = []
    if anim is None:
        n = 60
        for i in range(n):
            out.append((0.0, i / n, 8))
    elif key in ("AppStarting", "Wait"):
        n = 72                      # 3 vueltas del spinner por 1 ciclo de color (~2.4 s)
        for i in range(n):
            out.append(((i % 24) / 24, i / n, 2))
    else:                           # escritura: 3 pasadas por ciclo de color
        n = 28
        for i in range(n * 3):
            out.append(((i % n) / (n - 1), i / (n * 3), 22 if i % n == n - 1 else 3))
    return out


def fname_of(key):
    for k, _, f, _, a in CURSORS:
        if k == key:
            return f + (".ani" if (a or RGB) else ".cur")


def write_inf():
    order = ["Arrow", "Help", "AppStarting", "Wait", "Crosshair", "IBeam", "NWPen", "No", "SizeNS",
             "SizeWE", "SizeNWSE", "SizeNESW", "SizeAll", "UpArrow", "Hand", "Pin", "Person"]
    files = [fname_of(k) for k in order]
    L_ = ["[Version]", 'signature="$CHICAGO$"', "", "[DefaultInstall]",
          "CopyFiles = Scheme.Cur", "AddReg = Scheme.Reg,Wreg", "", "[DestinationDirs]",
          'Scheme.Cur = 10,"%CUR_DIR%"', "", "[Scheme.Reg]",
          'HKCU,"Control Panel\\Cursors\\Schemes","%SCHEME_NAME%",,"' +
          ",".join("%10%\\%CUR_DIR%\\" + f for f in files) + '"', "", "[Wreg]",
          'HKCU,"Control Panel\\Cursors",,0x00020000,"%SCHEME_NAME%"',
          'HKCU,"Control Panel\\Cursors","Scheme Source",0x00010001,1']
    for k, f in zip(order, files):
        L_.append('HKCU,"Control Panel\\Cursors","%s",0x00020000,"%%10%%\\%%CUR_DIR%%\\%s"' % (k, f))
    L_ += ["", "[Scheme.Cur]"] + files + ["", "[Strings]", 'CUR_DIR = "Cursors\\%s"' % THEME,
                                           'SCHEME_NAME = "%s"' % THEME]
    open(os.path.join(OUT, "install.inf"), "w", newline="").write("\r\n".join(L_) + "\r\n")


def previews():
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    cols, cw, ch = 6, 140, 120
    rows = math.ceil(len(CURSORS) / cols)
    sheet = Image.new("RGB", (cols * cw, rows * ch * 2), (255, 255, 255))
    for half, bg, fg in ((0, (236, 238, 242), (40, 40, 40)), (1, (30, 30, 36), (220, 220, 220))):
        ImageDraw.Draw(sheet).rectangle([0, half * rows * ch, cols * cw, (half + 1) * rows * ch], fill=bg)
        for i, (key, label, fname, fn, anim) in enumerate(CURSORS):
            im, _ = render(fn, 72, 0.3 if anim else 0.0, (i / len(CURSORS)) if RGB else 0.0)
            x, y = (i % cols) * cw, half * rows * ch + (i // cols) * ch
            sheet.paste(im, (x + (cw - 72) // 2, y + 10), im)
            d = ImageDraw.Draw(sheet)
            d.text((x + cw // 2, y + 98), label + (" (anim)" if anim else ""), font=f, fill=fg, anchor="mm")
    sheet.save(os.path.join(OUT, "vista_previa.png"))
    # gif animado
    cell = 130
    names = ["Arrow", "Wait", "AppStarting", "NWPen", "Hand", "Crosshair"] if RGB else ["AppStarting", "Wait", "NWPen"]
    K = len(names)
    gif = []
    for i in range(72 if RGB else 28):
        bg = Image.new("RGB", (cell * K, cell), (28, 28, 34) if RGB else (236, 238, 242))
        for j, nm in enumerate(names):
            for key, label, fname, fn, anim in CURSORS:
                if key == nm:
                    t = (i % 24) / 24 if nm != "NWPen" else (i % 28) / 27
                    im, _ = render(fn, 96, t, i / 72 if RGB else 0.0)
                    bg.paste(im, (j * cell + 17, 17), im)
        gif.append(bg)
    gif[0].save(os.path.join(OUT, "vista_previa_animada.gif"), save_all=True, append_images=gif[1:],
                duration=50, loop=0)


if __name__ == "__main__":
    build()
    previews()
    print("OK ->", OUT)
