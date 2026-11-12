#!/usr/bin/env python3
"""
Capítulo 6 — la muestra 36 con sus tramos copiados recuadrados (L24).

El «copiado» de una muestra es qué parte de sus palabras queda dentro de algún tramo que está tal
cual en los libros de entrenamiento. La figura enseña la muestra 36 entera, con cada tramo
sombreado y su largo, las tres palabras del comienzo marcadas, y la cuenta abajo: como los
dígitos recuadrados de la figura del reloj del capítulo 3.

Nada se calcula aquí: la muestra, los tramos y la cuenta se leen del bloque 2 de
`datos/salidas/copias_de_una_muestra.txt`.

Uso:
    python figura_copias_muestra.py --selftest
    python figura_copias_muestra.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/copias_de_una_muestra.txt"
DESTINO = "../figuras/copias_muestra.png"
ALTO = 3.0
TAM = 8.0                      # puntos
CARACTERES_POR_LINEA = 52
PALABRAS_ARRANQUE = 3

# ==========================================================

import argparse
import re
import sys
from pathlib import Path

from matplotlib.patches import Circle, Rectangle

from formato import leer_tablas
from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent
PT_POR_UNIDAD = 4.45 * 72 / 100          # un punto de letra, en unidades del lienzo
ANCHO_CHAR = 0.602 * TAM / PT_POR_UNIDAD  # DejaVu Sans Mono


def leer(ruta):
    """La muestra marcada y su cuenta, de la muestra y la tabla editoriales del bloque 2
    (L24, 9 de octubre)."""
    tablas = leer_tablas(Path(ruta).read_text(encoding="utf-8"))
    marcadas = [t for t in tablas if re.match(r"^La muestra \d+, tramo a tramo$", t)]
    assert len(marcadas) == 1, f"se esperaba una muestra marcada en {ruta}; hay {marcadas}"
    n_muestra = int(re.search(r"\d+", marcadas[0]).group())
    marcado = " ".join(l.strip() for l in tablas[marcadas[0]][1])
    palabras, cubiertas, dentro = [], [], False
    for w in marcado.split():
        empieza = w.startswith("[")
        m = re.match(r"^(.*)\](\d+)$", w)
        limpio = w.lstrip("[")
        if m:
            limpio = m.group(1).lstrip("[")
        if empieza:
            dentro = True
        palabras.append(limpio)
        cubiertas.append(dentro)
        if m:
            dentro = False
    notas = " ".join(tablas[f"Los tramos de la muestra {n_muestra}"][2])
    m = re.search(r"Palabras dentro de algún tramo: (\d+) de (\d+)", notas)
    p = re.search(r"\d+ entre \d+: (\d+,\d %)", notas)
    assert m and p, "no se encontró la cuenta del bloque 2"
    assert sum(cubiertas) == int(m.group(1)) and len(palabras) == int(m.group(2)), \
        f"los corchetes cubren {sum(cubiertas)} de {len(palabras)}; la salida dice {m.group(1)} de {m.group(2)}"
    return n_muestra, palabras, cubiertas, p.group(1)


def tramos(cubiertas):
    out, i = [], 0
    while i < len(cubiertas):
        if cubiertas[i]:
            j = i
            while j + 1 < len(cubiertas) and cubiertas[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def dibujar(n, palabras, cubiertas, porcentaje, paleta, ruta):
    L = Lienzo(f"¿Qué ha copiado? La muestra {n}",
               "Sombreado: un tramo que está tal cual, palabra por palabra, en los libros\n"
               "con que se entrenó la red (siete palabras seguidas o más).", paleta, alto=ALTO)
    p, ax = L.p, L.ax
    x0, y = 6.0, L.y - 3.5
    alto_linea = 5.6
    pos = []
    col = 0
    for w in palabras:
        if col and col + 1 + len(w) > CARACTERES_POR_LINEA:
            col = 0
            y -= alto_linea
        if col:
            col += 1
        pos.append((x0 + col * ANCHO_CHAR, y, len(w)))
        col += len(w)
    for i, (x, yy, k) in enumerate(pos):
        arr = i < PALABRAS_ARRANQUE
        L.texto(x, yy, palabras[i], tam=TAM, mono=True, color=p.suave if arr else p.tinta)
        if arr:
            ax.plot([x, x + k * ANCHO_CHAR], [yy - 1.9, yy - 1.9], color=p.suave, linewidth=0.8)
    cajas = []
    tr = tramos(cubiertas)
    for num, (d, h) in enumerate(tr, 1):
        por_linea = {}
        for i in range(d, h + 1):
            por_linea.setdefault(pos[i][1], []).append(i)
        filas = sorted(por_linea, reverse=True)
        for k, yy in enumerate(filas):
            idx = por_linea[yy]
            xa = pos[idx[0]][0] - 0.6
            xb = pos[idx[-1]][0] + pos[idx[-1]][2] * ANCHO_CHAR + 0.6
            ax.add_patch(Rectangle((xa, yy - 2.0), xb - xa, 4.0, facecolor=p.neutro,
                                   edgecolor="none", zorder=1))
            if k == 0:
                ax.add_patch(Circle((xa - 0.2, yy + 2.1), 1.25, facecolor=p.tinta, edgecolor="none",
                                    zorder=4))
                ax.text(xa - 0.2, yy + 2.1, str(num), ha="center", va="center", fontsize=6.2,
                        color="white", fontweight="bold", zorder=5)
            cajas.append((xa, xb, yy))
    yk = y - 9.0
    for num, (d, h) in enumerate(tr, 1):
        L.texto(6 + (num - 1) * 30, yk + 3.4, f"tramo {num}: {h - d + 1} palabras seguidas", tam=7.2,
                negrita=True)
    yk -= 1.5
    ax.plot([6, 10], [yk, yk], color=p.suave, linewidth=0.8)
    L.texto(11, yk + 0.3, "subrayado: las tres palabras que se le dieron; el resto lo escribió la red",
            tam=6.8, color=p.suave)
    # Orden 153: fuera la línea «32 de 45 palabras sombreadas: 71,1 %». La cuenta (treinta y dos palabras,
    # 71,1 %) la hace el anexo A6 con palabras; la figura enseña los dos tramos y su largo y nada más.
    # El lienzo (ALTO) no cambia. `porcentaje` se sigue leyendo de la salida (lo usa el selftest).
    L.guardar(ruta)
    return cajas


def selftest():
    fallos = []
    n, pal, cub, pct = leer(AQUI / SALIDA)
    # 1. TEST NULO — sin nada copiado no hay ningún recuadro.
    cajas = dibujar(n, pal, [False] * len(pal), "0,0 %", GRIS, "/dev/null")
    print(f"[1] test nulo         sin copias, recuadros: {len(cajas)}")
    if cajas:
        fallos.append("test nulo: dibuja recuadros sin copias")
    # 2. SEÑAL — los tramos de la salida: tantos tramos como corchetes, y ninguno sale de la caja.
    cajas = dibujar(n, pal, cub, pct, GRIS, "/dev/null")
    fuera = [c for c in cajas if c[0] < 0 or c[1] > 100]
    print(f"[2] señal             tramos {len(tramos(cub))}, recuadros {len(cajas)}, fuera de la caja {len(fuera)}")
    if not tramos(cub) or fuera or len(cajas) < len(tramos(cub)):
        fallos.append("señal: faltan recuadros o se salen de la página")
    # 3. INVARIANTE — la suma de los largos de los tramos es lo que la salida llama «dentro de
    #    algún tramo» (lo comprueba leer); y se dibuja en color y en gris.
    suma = sum(h - d + 1 for d, h in tramos(cub))
    dibujar(n, pal, cub, pct, COLOR, "/dev/null")
    print(f"[3] invariante        suma de los largos de los tramos: {suma} = cubiertas {sum(cub)}")
    if suma != sum(cub):
        fallos.append("invariante: los tramos no suman lo cubierto")
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
    dibujar(*leer(AQUI / SALIDA), GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
