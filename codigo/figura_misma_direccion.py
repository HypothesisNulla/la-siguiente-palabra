#!/usr/bin/env python3
"""
Capítulo 5 — ¿apuntan al mismo lado los caminos del género? (L24)

Un «camino» es la resta de dos listas de cien números (de «padre» a «madre»). Dos caminos se
pueden dibujar a la vez sin aplanar nada: dos flechas que salen del mismo punto están siempre en
un plano, y en ese plano el ángulo entre ellas es el de verdad. La figura dibuja tres parejas de
flechas: la pareja de caminos del género que más se parece, la que menos, y lo que dan, de media,
dos caminos entre palabras sacadas al azar.

Desde la Orden 150 (11 nov 2026, regla 11.9) salen DOS figuras:
  - `misma_direccion.png`, la del cuerpo del capítulo 5: sin los «parecido 0,59 / 0,28 / 0,008», y con la clave «Cómo se
    lee: flechas una encima de otra, al mismo lado; en escuadra, nada que ver.»;
  - `misma_direccion_numeros.png`, la de antes, con el «parecido» de cada pareja, para el anexo A5.

Nada se calcula aquí: parecidos y largos se leen del bloque 7 de
`datos/salidas/palabras_de_cerca.txt`.

Uso:
    python figura_misma_direccion.py --selftest
    python figura_misma_direccion.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/palabras_de_cerca.txt"
DESTINO = "../figuras/misma_direccion.png"                    # la del cuerpo, sin números
DESTINO_NUMEROS = "../figuras/misma_direccion_numeros.png"    # la del anexo, con el parecido de cada pareja
PIE_CUERPO = "Cómo se lee: flechas una encima de otra, al mismo lado; en escuadra, nada que ver."
ALTO = 2.75
LARGO_MAXIMO = 17.0          # unidades del lienzo para la flecha más larga

# ==========================================================

import argparse
import math
import re
import sys
from pathlib import Path

from matplotlib.patches import FancyArrow

from formato import leer_tablas
from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent
NUM = r"(-?\d+,\d+)"


def f(x):
    return float(x.replace(",", "."))


def leer(ruta):
    """Los largos y los parecidos, de las dos tablas de los caminos (L24, 9 de octubre: la
    salida ya no es texto sangrado sino tablas editoriales). Los nombres de los caminos se
    devuelven como antes, «rey -> reina», que es lo que rotula la figura."""
    tablas = leer_tablas(Path(ruta).read_text(encoding="utf-8"))
    for t in ("El tamaño de cada camino", "Los caminos del género, de dos en dos"):
        assert t in tablas, f"se esperaba la tabla «{t}» en {ruta}"
    camino = lambda c: re.sub(r"^de (\w+) a (\w+)$", r"\1 -> \2", c)
    largos = {camino(c): f(v) for c, v in tablas["El tamaño de cada camino"][1]}
    pares, azar = [], None
    for a, b, v in tablas["Los caminos del género, de dos en dos"][1]:
        if re.match(r"^de \w+ a \w+$", a) and re.match(r"^de \w+ a \w+$", b):
            pares.append((camino(a), camino(b), f(v)))
        elif re.match(r"^media de \d+ comparaciones$", b):
            azar = f(v)
    assert largos and len(pares) >= 2 and azar is not None, "no se pudieron leer los caminos"
    return largos, pares, azar


def paneles(largos, pares, azar):
    mas = max(pares, key=lambda t: t[2])
    menos = min(pares, key=lambda t: t[2])
    return [(mas[0], mas[1], mas[2], largos[mas[0]], largos[mas[1]], "la pareja que más se parece"),
            (menos[0], menos[1], menos[2], largos[menos[0]], largos[menos[1]], "la que menos"),
            ("un camino al azar", "otro al azar", azar, None, None, "dos al azar, de media")]


def dibujar(largos, pares, azar, paleta, ruta, con_numeros=True):
    L = Lienzo("¿Apuntan al mismo lado?",
               "Cada flecha es un camino: la resta de dos listas de cien números.\n"
               "El ángulo entre dos flechas es el de verdad, sin aplanar nada.", paleta, alto=ALTO)
    p, ax = L.p, L.ax
    maxlargo = max(largos.values())
    angulos = []
    for k, (a, b, s, la, lb, titulo) in enumerate(paneles(largos, pares, azar)):
        x0 = 6 + k * 32
        y0 = 19
        ang = math.acos(max(-1.0, min(1.0, s)))
        angulos.append(ang)
        la_ = LARGO_MAXIMO * (la / maxlargo if la else 0.75)
        lb_ = LARGO_MAXIMO * (lb / maxlargo if lb else 0.75)
        for largo, th, rot in ((la_, 0.0, a), (lb_, ang, b)):
            dx, dy = largo * math.cos(th), largo * math.sin(th)
            ax.add_patch(FancyArrow(x0, y0, dx, dy, width=0.35, head_width=1.6, head_length=1.4,
                                    length_includes_head=True, facecolor=p.tinta, edgecolor="none"))
            L.texto(x0 + dx + (0.8 if th < 0.3 else -1.0), y0 + dy + (1.3 if th > 0.3 else -1.6),
                    rot.replace("->", "→"), tam=6.8,
                    ha="left" if th < 0.3 else "center")
        L.texto(x0 + 13, 13.0, titulo, tam=7.2, ha="center", negrita=True)
        if con_numeros:
            L.texto(x0 + 13, 9.5, f"parecido {str(s).replace('.', ',')}", tam=7.2, ha="center")
    if con_numeros:
        L.pie("Cómo se lee: flechas encima una de otra, parecido 1; en ángulo recto, 0; opuestas, -1.\n"
              "Largo de cada flecha: el tamaño del camino.")
    else:
        L.pie(PIE_CUERPO)
    L.guardar(ruta)
    return angulos


def selftest():
    fallos = []
    largos, pares, azar = leer(AQUI / SALIDA)
    # 1. TEST NULO — si todos los caminos se parecieran 0, las flechas saldrían en ángulo recto.
    nulos = [(a, b, 0.0) for a, b, _ in pares]
    ang = dibujar(largos, nulos, 0.0, GRIS, "/dev/null")
    ok = all(abs(x - math.pi / 2) < 1e-9 for x in ang)
    print(f"[1] test nulo         parecido 0 -> ángulo recto en los tres: {ok}")
    if not ok:
        fallos.append("test nulo: parecido 0 no da ángulo recto")
    # 2. SEÑAL — un parecido de 1 da dos flechas encima una de otra.
    unos = [(a, b, 1.0) for a, b, _ in pares]
    ang = dibujar(largos, unos, 1.0, GRIS, "/dev/null")
    print(f"[2] señal             parecido 1 -> ángulo 0: {all(abs(x) < 1e-9 for x in ang)}")
    if not all(abs(x) < 1e-9 for x in ang):
        fallos.append("señal: parecido 1 no da flechas superpuestas")
    # 3. INVARIANTE — el panel de la izquierda es el de mayor parecido y el del medio el de menor,
    #    y a más parecido, menos ángulo; se dibuja en color y en gris.
    pan = paneles(largos, pares, azar)
    ang = dibujar(largos, pares, azar, GRIS, "/dev/null")
    dibujar(largos, pares, azar, COLOR, "/dev/null")
    # Orden 150: la figura del cuerpo no lleva ninguna cifra con decimales (11.9), y sus ángulos son los mismos
    ang_cuerpo = dibujar(largos, pares, azar, GRIS, "/dev/null", con_numeros=False)
    ok = (pan[0][2] == max(t[2] for t in pares) and pan[1][2] == min(t[2] for t in pares)
          and ang[0] <= ang[1] <= ang[2] and ang_cuerpo == ang)
    print(f"[3] invariante        orden de los paneles y de los ángulos: {ok}")
    if not ok:
        fallos.append("invariante: paneles o ángulos fuera de orden")
    print()
    if fallos:
        for x in fallos:
            print("FALLA:", x)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    codigo = selftest()
    if codigo or args.selftest:
        sys.exit(codigo)
    datos = leer(AQUI / SALIDA)
    dibujar(*datos, GRIS, str(AQUI / DESTINO), con_numeros=False)
    print(f"escrita {DESTINO}")
    dibujar(*datos, GRIS, str(AQUI / DESTINO_NUMEROS))
    print(f"escrita {DESTINO_NUMEROS}")


if __name__ == "__main__":
    main()
