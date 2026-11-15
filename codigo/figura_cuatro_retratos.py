#!/usr/bin/env python3
"""
Capítulo 10 — los cuatro retratos en fila, con las palabras de verdad subrayadas (L24, D04).

El capítulo enseña cuatro muestras separadas por párrafos, y el lector tenía que volver atrás,
ponerlas en fila en su cabeza y decidir él qué es palabra y qué no. Aquí van en fila, cada una
con el nombre de su peldaño, y subrayadas las palabras de verdad, con el mismo criterio que la
escalera del capítulo 1 (palabras del Quijote). La regla 4 del libro no deja números en las
muestras: el lector no cuenta, ve.

Qué es palabra NO se decide aquí: se lee del bloque 2 de `datos/salidas/cuatro_retratos.txt`,
donde van entre corchetes. El selftest comprueba que, quitadas las marcas, cada muestra es letra
a letra la que imprimió `en_que_orden_aprende.py`.

Uso:
    python figura_cuatro_retratos.py --selftest
    python figura_cuatro_retratos.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/cuatro_retratos.txt"
SALIDA_CAP10 = "../datos/salidas/en_que_orden_aprende.txt"
DESTINO = "../figuras/cuatro_retratos.png"
ALTO = 4.8                  # pulgadas
LETRAS_RENGLON = 64         # letras por renglón de muestra
TAM = 6.5                   # cuerpo de la letra de las muestras
RENGLON = 3.35              # unidades del lienzo entre renglones
PELDANO = {30: "el aspecto", 300: "los tríos de letras más frecuentes",
           3_000: "las palabras y sus terminaciones", 50_000: "la sintaxis"}

# ==========================================================

import argparse
import re
import sys

from matplotlib.patches import FancyBboxPatch

from infografia import COLOR, GRIS, Lienzo
from formato import miles


def leer(ruta=SALIDA):
    t = open(ruta, encoding="utf-8").read()
    assert "--- 2. PALABRAS DE VERDAD EN CADA RETRATO ---" in t, f"falta el bloque 2 en {ruta}"
    bloque = t.split("--- 2. PALABRAS DE VERDAD EN CADA RETRATO ---", 1)[1].split("--- 3.", 1)[0]
    pares = re.findall(r"\[tras ([\d.]+) pasos: \d+ de \d+ palabras\]\n(.*)\n", bloque)
    assert len(pares) == 4, f"se esperaban cuatro retratos; hay {len(pares)}"
    return [(int(p.replace(".", "")), m) for p, m in pares]


def desmarcar(marcada):
    """(texto limpio, lista de (inicio, fin, clase)) con clase «dada» o «palabra»."""
    limpio, tramos, i = "", [], 0
    while i < len(marcada):
        c = marcada[i]
        if c in "[{":
            cierre = "]" if c == "[" else "}"
            j = marcada.index(cierre, i)
            palabra = marcada[i + 1:j]
            tramos.append((len(limpio), len(limpio) + len(palabra), "palabra" if c == "[" else "dada"))
            limpio += palabra
            i = j + 1
        else:
            limpio += c
            i += 1
    return limpio, tramos


def partir(texto, n=LETRAS_RENGLON):
    """Renglones de n letras como mucho, cortando en un espacio; sin tocar los espacios de
    dentro del renglón (dos espacios seguidos son un dato de la muestra)."""
    renglones, inicio = [], 0
    while inicio < len(texto):
        if len(texto) - inicio <= n:
            renglones.append((inicio, texto[inicio:]))
            break
        corte = texto.rfind(" ", inicio, inicio + n + 1)
        assert corte > inicio, "un renglón sin espacios donde cortar"
        renglones.append((inicio, texto[inicio:corte]))
        inicio = corte + 1
    return renglones


def ancho_letra(L):
    """Lo que ocupa una letra de la fuente fija, en unidades del lienzo, sacado de la propia fuente:
    el avance de una letra de DejaVu Sans Mono, en fracción del cuerpo, por el cuerpo en puntos,
    entre los puntos que mide una unidad del lienzo. (Medirlo en pantalla no vale: a la resolución
    de la pantalla la fuente se ajusta a la rejilla de puntos y el avance sale distinto del de la
    imagen a 300 ppp, y los subrayados se corrían.)"""
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font
    f = FT2Font(font_manager.findfont("DejaVu Sans Mono"))
    f.set_size(100, 72)                           # cien puntos, a 72 ppp: un punto, un píxel
    g = f.load_char(ord("m"), flags=1)            # sin ajuste a la rejilla
    fraccion = g.linearHoriAdvance / 65536 / 100  # avance de una letra, en fracción del cuerpo
    assert 0.55 < fraccion < 0.65, f"avance inesperado para una fuente fija: {fraccion}"
    puntos_por_unidad = L.fig.get_size_inches()[0] * 72 / 100
    return fraccion * TAM / puntos_por_unidad


def dibujar(retratos, paleta, ruta):
    L = Lienzo("Cuatro retratos de la misma red",   # Orden 167: «red», como en el texto
               "Cada vez se le da «el» para empezar y escribe sola. Subrayado, lo que es\n"
               "una palabra de verdad; lo demás, invención.", paleta, alto=ALTO)
    p, ax = L.p, L.ax
    w = ancho_letra(L)
    # la clave
    y = L.y - 0.6
    x = 4
    L.texto(x, y, "subrayado: sale en el Quijote al menos dos veces (de una letra: a, e, o, u, y)",
            tam=6.6, color=p.suave)
    y -= 3.2
    ax.add_patch(FancyBboxPatch((x, y - 1.3), 3.2, 2.6, boxstyle="round,pad=0,rounding_size=0.4",
                                facecolor="none", edgecolor=p.suave, linewidth=0.7))
    L.texto(x + 4.2, y, "recuadrado: lo que se le dio para empezar", tam=6.6, color=p.suave)
    y -= 5.4
    subrayadas = {}
    for pasos, marcada in retratos:
        limpio, tramos = desmarcar(marcada)
        L.texto(4, y, f"Tras {miles(pasos)} pasos", tam=8.0, negrita=True)
        L.texto(24.5, y, f"peldaño: {PELDANO[pasos]}", tam=7.4, color=p.suave)
        y -= 3.6
        n = 0
        for inicio, renglon in partir(limpio):
            L.texto(6, y, renglon, tam=TAM, mono=True)
            fin = inicio + len(renglon)
            for a, b, clase in tramos:
                if a >= fin or b <= inicio:
                    continue
                xa, xb = 6 + (max(a, inicio) - inicio) * w, 6 + (min(b, fin) - inicio) * w
                if clase == "palabra":
                    ax.plot([xa + 0.1, xb - 0.1], [y - 1.35, y - 1.35], color=p.tinta, linewidth=0.9,
                            solid_capstyle="butt")
                    n += 1
                else:
                    ax.add_patch(FancyBboxPatch((xa - 0.3, y - 1.3), xb - xa + 0.6, 2.6,
                                                boxstyle="round,pad=0,rounding_size=0.4",
                                                facecolor="none", edgecolor=p.suave, linewidth=0.7))
            y -= RENGLON
        subrayadas[pasos] = n
        y -= 2.2
    assert y > 1.0, f"los retratos no caben en la figura (sobran {1.0 - y:.1f} unidades)"
    L.guardar(ruta)
    return subrayadas


def selftest():
    fallos = []
    retratos = leer()
    cap10 = open(SALIDA_CAP10, encoding="utf-8").read()

    # 1. TEST NULO — una muestra sin palabras marcadas no lleva ni un subrayado.
    sin = [(p, m.replace("[", "").replace("]", "")) for p, m in retratos]
    rec = dibujar(sin, GRIS, "/dev/null")
    print(f"[1] test nulo         sin marcas, subrayados: {sum(rec.values())}")
    if sum(rec.values()):
        fallos.append("test nulo: subraya palabras que no están marcadas")

    # 2. SEÑAL — se subrayan exactamente las palabras marcadas en la salida.
    rec = dibujar(retratos, GRIS, "/dev/null")
    esperados = {p: m.count("[") for p, m in retratos}
    print(f"[2] señal             subrayadas {rec}; marcadas en la salida {esperados}")
    # una palabra partida entre dos renglones se subraya en los dos: se admite, se cuenta
    if any(rec[p] < esperados[p] for p in rec):
        fallos.append("señal: se subrayan menos palabras de las marcadas")

    # 3. INVARIANTE — quitadas las marcas, cada muestra es la de en_que_orden_aprende.txt, y la
    #    figura se dibuja en las dos paletas.
    malas = []
    for p, m in retratos:
        limpio, _ = desmarcar(m)
        if limpio not in cap10:
            malas.append(p)
    for pal in (COLOR, GRIS):
        dibujar(retratos, pal, "/dev/null")
    print(f"[3] invariante        muestras que no son literales: {malas or 'ninguna'}")
    if malas:
        fallos.append(f"invariante: no son literales las de {malas}")
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
    dibujar(leer(), GRIS, DESTINO)
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
