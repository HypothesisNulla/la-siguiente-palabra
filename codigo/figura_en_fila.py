#!/usr/bin/env python3
"""
Capítulo 9 — en fila o a la vez, dibujado (L24).

La misma frase corta en las dos máquinas. Arriba, la que lee en orden: la cuenta de cada palabra
necesita el resumen que dejó la anterior, así que hay que dar un paso detrás de otro, tantos como
palabras. Abajo, la que mira todo a la vez: la cuenta de cada palabra solo necesita las listas de
las palabras, que ya están todas, así que todas se hacen en el mismo paso.

Los pasos NO se deciden aquí: se leen del apartado 3 de `datos/salidas/cuadricula_y_fila.txt`.
En gris: el libro se imprime en negro.

Uso:
    python figura_en_fila.py --selftest
    python figura_en_fila.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/cuadricula_y_fila.txt"
DESTINO = "../figuras/en_fila.png"
ALTO = 3.2                 # pulgadas
ALTO_PANEL = 17.5          # unidades del lienzo

# ==========================================================

import argparse
import re
import sys
from pathlib import Path

from matplotlib.patches import FancyArrowPatch

from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent
NUMEROS = {1: "una tanda", 2: "dos tandas", 3: "tres tandas", 4: "cuatro tandas", 5: "cinco tandas",
           6: "seis tandas", 7: "siete tandas"}


def leer(ruta):
    """Del apartado 3: la lista (paso, palabra) de la que lee en orden y la de la que mira a la vez."""
    texto = Path(ruta).read_text(encoding="utf-8")
    assert "3. CUÁNTAS TANDAS HAY QUE HACER" in texto, f"se esperaba el apartado 3 en {ruta}"
    tres = texto.split("3. CUÁNTAS TANDAS HAY QUE HACER", 1)[1]
    orden = [(int(m.group(1)), m.group(2))
             for m in re.finditer(r"^\s+tanda (\d+): «([^»]+)»(?:, con el resumen de «[^»]+»)?$", tres, re.M)]
    m = re.search(r"^\s+tanda (\d+): (.+), a la vez$", tres, re.M)
    assert m, "no está la línea de la que mira a la vez"
    a_la_vez = [(int(m.group(1)), w) for w in re.findall(r"«([^»]+)»", m.group(2))]
    assert [w for _, w in orden] == [w for _, w in a_la_vez], "las dos máquinas no leen la misma frase"
    return orden, a_la_vez


def fila(L, y, pasos, encadenar):
    p = L.p
    n = len(pasos)
    ancho = 84.0 / n
    xs = []
    for i, (paso, w) in enumerate(pasos):
        x = 8.0 + i * ancho
        L.ficha(x + 1.0, y, w, ancho - 5.0, alto=4.6, relleno="white", negrita=True, tam=8.4)
        L.texto(x + 1.0 + (ancho - 5.0) / 2, y - 4.4, f"tanda {paso}", ha="center", tam=7.4, color=p.suave)
        xs.append((x + 1.0, x + ancho - 4.0))
    if encadenar:
        for (a0, a1), (b0, b1) in zip(xs, xs[1:]):
            L.ax.add_patch(FancyArrowPatch((a1 + 0.3, y), (b0 - 0.3, y), arrowstyle="-|>",
                                           mutation_scale=7, color=p.acento, linewidth=1.0))


def dibujar(orden, a_la_vez, paleta, ruta):
    L = Lienzo("En fila, o a la vez",
               "La misma frase en las dos máquinas. La flecha es el resumen que cada palabra\n"
               "tiene que esperar de la anterior.", paleta, alto=ALTO)
    pasos = []
    for n, (titulo_base, datos, encadenar) in enumerate(
            (("La que lee en orden", orden, True), ("La que mira todo a la vez", a_la_vez, False)), 1):
        cuantos = len({p for p, _ in datos})
        pasos.append(cuantos)
        x0, ytop, _ = L.panel(n, f"{titulo_base}: {NUMEROS.get(cuantos, str(cuantos) + ' tandas')}",
                              ALTO_PANEL)
        fila(L, ytop - 1.6, datos, encadenar)
    L.pie("Cada recuadro es la cuenta de una palabra; debajo, en qué tanda se puede hacer. En una tanda\n"
          "van todas las cuentas que se pueden hacer a la vez.")
    L.guardar(ruta)
    return pasos


def selftest():
    fallos = []
    orden, a_la_vez = leer(AQUI / SALIDA)
    # 1. TEST NULO — una salida sin el apartado 3 no da figura.
    import os, tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write("nada\n")
    try:
        leer(fh.name)
        fallos.append("test nulo: leyó pasos de un texto que no los tiene")
    except (AssertionError, AttributeError):
        pass
    finally:
        os.unlink(fh.name)
    print("[1] test nulo         un texto sin el apartado 3 no da figura")
    # 2. SEÑAL — en orden, tantos pasos como palabras; a la vez, uno.
    pasos = dibujar(orden, a_la_vez, GRIS, "/dev/null")
    print(f"[2] señal             pasos dibujados: en orden {pasos[0]}, a la vez {pasos[1]} "
          f"({len(orden)} palabras)")
    if pasos != [len(orden), 1]:
        fallos.append("señal: los pasos dibujados no son los de la salida")
    # 3. INVARIANTE — en orden, cada palabra va un paso después de la anterior; y se dibuja en las
    #    dos paletas.
    seguidos = all(b[0] == a[0] + 1 for a, b in zip(orden, orden[1:]))
    for pal in (COLOR, GRIS):
        dibujar(orden, a_la_vez, pal, "/dev/null")
    print(f"[3] invariante        en orden, cada palabra un paso después de la anterior: "
          f"{'sí' if seguidos else 'NO'}")
    if not seguidos:
        fallos.append("invariante: la que lee en orden no va paso a paso")
    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
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
