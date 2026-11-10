#!/usr/bin/env python3
"""
Capítulo 4 — la red del colegio, con sus números (Orden 20, 18 oct 2026).

La red que resuelve el colegio (sí de 6 a 12 años), dibujada con lo que aprendió escrito encima: el
peso de cada línea y el listón de cada neurona. La edad entra tal cual, en años, por un cuadrado;
las dos de en medio le preguntan cosas distintas (una, «¿tiene seis o más?»; la otra, «¿tiene trece
o más?»), y la final las junta: sí cuando la primera dice sí y la segunda dice no.

Los números NO se deciden aquí: se leen de `datos/salidas/culpa_en_el_colegio_datos.csv`, que escribe
`culpa_en_el_colegio.py`. Las preguntas de los rótulos salen de esos números (el corte de cada
neurona es su listón entre su peso), no están escritas a mano.

Desde la Orden 148 (10 nov 2026, regla 11.9: una figura del cuerpo no lleva números que el cuerpo no explique)
salen DOS figuras:
  - `la_red_del_colegio.png`, la del cuerpo del capítulo 4: sin pesos ni listones y sin el texto de arriba; junto a
    cada neurona, su pregunta («¿tiene 6 o más?», «¿tiene 13 o más?», «¿entra en el colegio?»), que sale de sus
    números como siempre; debajo de la edad, «en años»;
  - `la_red_del_colegio_numeros.png`, la de antes, con sus números, para el anexo A4.

Uso:
    python figura_la_red_del_colegio.py --selftest
    python figura_la_red_del_colegio.py
"""

# ======================= CONSTANTES =======================

DATOS = "../datos/salidas/culpa_en_el_colegio_datos.csv"
DESTINO = "../figuras/la_red_del_colegio.png"                     # la del cuerpo, sin números
DESTINO_NUMEROS = "../figuras/la_red_del_colegio_numeros.png"      # la del anexo, con sus números
ALTO_CUERPO = 3.0              # pulgadas
ALTO = 3.7                     # pulgadas
GROSOR_MAXIMO = 3.2            # grosor de la línea del peso más grande
RADIO = 5.2                    # radio de cada neurona, en unidades del lienzo
X_ENTRADA, X_MEDIO, X_FINAL = 13.0, 50.0, 85.0
NOMBRES_MEDIO = ["la primera", "la segunda"]

# ==========================================================

import argparse
import csv
import math
import sys
from pathlib import Path

from matplotlib.patches import Circle, FancyBboxPatch

from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent


def leer_datos(ruta=None):
    """El fichero de datos entero: {(que, a, b): valor} con los números como float (los que lo son)."""
    d = {}
    with open(ruta or AQUI / DATOS, encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            try:
                d[(fila["que"], fila["a"], fila["b"])] = float(fila["valor"])
            except ValueError:
                pass
    assert d, "no hay datos"
    return d


def v(d, que, a="", b=""):
    return d[(que, str(a), str(b))]


def f(x, n=3, signo=True):
    s = f"{abs(x):.{n}f}".replace(".", ",")
    return (("+" if x >= 0 else "−") if signo else ("" if x >= 0 else "−")) + s


def red_de(d, cual="final"):
    """(pesos, listones): pesos[neurona][de] y listones[neurona] de la red `cual` («final» o «caso»)."""
    pesos = {"la primera": {"la edad": v(d, f"{cual}_peso_medio", 0, 0)},
             "la segunda": {"la edad": v(d, f"{cual}_peso_medio", 0, 1)},
             "la final": {"la primera": v(d, f"{cual}_peso_final", 0), "la segunda": v(d, f"{cual}_peso_final", 1)}}
    listones = {"la primera": v(d, f"{cual}_liston_medio", 0), "la segunda": v(d, f"{cual}_liston_medio", 1),
                "la final": v(d, f"{cual}_liston_final")}
    return pesos, listones


def desde_que_edad(peso, liston):
    """La primera edad entera a la que una neurona de en medio dice sí (le llega más que su listón),
    con la edad en años; None si no es «desde» (peso negativo)."""
    if peso <= 0:
        return None
    return math.floor(liston / peso) + 1


def que_dice_la_final(pesos, listones, a, b):
    """Con un sí o un no seco (1 o 0) de las dos de en medio: ¿pasa del listón la final?"""
    p = pesos["la final"]
    return a * p["la primera"] + b * p["la segunda"] > listones["la final"]


def dibujar_red(L, pesos, listones, y_arriba, y_abajo, rotulos_linea=None, texto_neurona=None,
                flechas_atras=False, texto_entrada=None, t_a_la_final=0.62, separacion_a_la_final=2.6):
    """La red: la edad, dos de en medio y la final. rotulos_linea[(neurona, de)] es el texto de cada
    línea; texto_neurona[neurona] lo que va junto a cada neurona."""
    p, ax = L.p, L.ax
    y_medio = (y_arriba + y_abajo) / 2
    pos = {"la edad": (X_ENTRADA, y_medio), "la primera": (X_MEDIO, y_arriba),
           "la segunda": (X_MEDIO, y_abajo), "la final": (X_FINAL, y_medio)}
    maximo = max(abs(w) for d in pesos.values() for w in d.values())
    for destino, entradas in pesos.items():
        for origen, peso in entradas.items():
            (x0, y0), (x1, y1) = pos[origen], pos[destino]
            ax.plot([x0 + RADIO, x1 - RADIO], [y0, y1], color=p.acento,
                    linestyle="-" if peso > 0 else (0, (1.2, 1.4)),
                    linewidth=0.6 + GROSOR_MAXIMO * abs(peso) / maximo, zorder=1, solid_capstyle="butt")
            if flechas_atras:
                t = 0.42
                xa, ya = x0 + RADIO + (x1 - x0 - 2 * RADIO) * t, y0 + (y1 - y0) * t
                ax.annotate("", xy=(xa - 2.2, ya - (y1 - y0) / (x1 - x0) * 2.2), xytext=(xa, ya),
                            arrowprops=dict(arrowstyle="-|>", color=p.tinta, lw=1.0), zorder=5)
            if rotulos_linea:
                # las dos líneas que llegan a la final se juntan en ella: sus rótulos, más atrás y más
                # separados (Orden 21), para que no queden pegados
                a_la_final = destino == "la final"
                t = t_a_la_final if a_la_final else 0.62
                sep = separacion_a_la_final if a_la_final else 2.6
                xr = x0 + RADIO + (x1 - x0 - 2 * RADIO) * t
                yr = y0 + (y1 - y0) * t + (sep if y1 > y0 else -sep)
                ax.text(xr, yr, rotulos_linea[(destino, origen)], ha="center", va="center", fontsize=7.6,
                        color=p.tinta, zorder=4,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))
    for nombre, (x, y) in pos.items():
        if nombre == "la edad":
            ax.add_patch(FancyBboxPatch((x - RADIO, y - RADIO), 2 * RADIO, 2 * RADIO,
                                        boxstyle="round,pad=0,rounding_size=1.0", facecolor=p.fondo,
                                        edgecolor=p.tinta, linewidth=1.2, zorder=3))
            ax.text(x, y, "la\nedad", ha="center", va="center", fontsize=6.6, color=p.tinta, zorder=4,
                    linespacing=1.0)
        else:
            ax.add_patch(Circle((x, y), RADIO, facecolor="white", edgecolor=p.tinta, linewidth=1.2, zorder=3))
            corto = {"la primera": "la\nprimera", "la segunda": "la\nsegunda", "la final": "la\nfinal"}[nombre]
            ax.text(x, y, corto, ha="center", va="center", fontsize=6.6, color=p.tinta, fontweight="bold",
                    zorder=4, linespacing=1.0)
        if nombre == "la edad" and texto_entrada:
            ax.text(x, y - RADIO - 1.2, texto_entrada, ha="center", va="top", fontsize=7.2, color=p.suave,
                    linespacing=1.15, zorder=4)
        if texto_neurona and nombre in texto_neurona:
            debajo = y < y_medio or nombre == "la final"
            ax.text(x, y - RADIO - 1.2 if debajo else y + RADIO + 1.2, texto_neurona[nombre], ha="center",
                    va="top" if debajo else "bottom", fontsize=7.2, color=p.tinta, linespacing=1.15,
                    zorder=4)
    return pos


def preguntas(pesos, listones):
    """«¿tiene N o más?» de cada neurona de en medio, de sus números."""
    out = []
    for n in NOMBRES_MEDIO:
        e = desde_que_edad(pesos[n]["la edad"], listones[n])
        out.append(f"«¿tiene {e} o más?»")
    return out


def dibujar_cuerpo(d, paleta, ruta):
    """La figura del cuerpo (Orden 148): la red sin pesos ni listones, con la pregunta de cada neurona."""
    pesos, listones = red_de(d)
    q = preguntas(pesos, listones)
    L = Lienzo("La red del colegio, entrenada", "", paleta, alto=ALTO_CUERPO)
    y_arriba, y_abajo = L.y - 9.0, L.y - 37.0
    txt = {"la primera": q[0], "la segunda": q[1], "la final": "«¿entra en\nel colegio?»"}
    dibujar_red(L, pesos, listones, y_arriba, y_abajo, None, txt, texto_entrada="en años")
    L.pie("Línea continua: suma; de puntos: resta. Cuanto más gruesa, más pesa.")
    L.guardar(ruta)
    return q


def dibujar(d, paleta, ruta):
    """La figura del anexo: la red con lo que aprendió escrito encima."""
    pesos, listones = red_de(d)
    L = Lienzo("La red del colegio, entrenada",
               "La edad en años, tal cual; la primera capa, con dos neuronas, y la final. Cada\n"
               "círculo es un comité: suma lo que le llega por sus líneas, cada una por su peso, y lo\n"
               "compara con su listón. Éstos son los números que aprendió.", paleta, alto=ALTO)
    p = L.p
    y_arriba, y_abajo = L.y - 13.0, L.y - 41.0
    q = preguntas(pesos, listones)
    rot = {(dst, o): f(w, 2) for dst, e in pesos.items() for o, w in e.items()}
    txt = {"la primera": f"listón {f(listones['la primera'], 2, False)}\n{q[0]}",
           "la segunda": f"listón {f(listones['la segunda'], 2, False)}\n{q[1]}",
           "la final": f"listón {f(listones['la final'], 2, False)}\n«¿entra en\nel colegio?»"}
    dibujar_red(L, pesos, listones, y_arriba, y_abajo, rot, txt, texto_entrada="en años,\ntal cual",
                t_a_la_final=0.40, separacion_a_la_final=3.4)
    L.pie("Línea continua: suma; de puntos: resta; cuanto más gruesa, más pesa. El número de cada línea\n"
          "es su peso; el de cada círculo, su listón.")
    L.guardar(ruta)


def selftest():
    fallos = []
    d = leer_datos()
    pesos, listones = red_de(d)

    # 1. TEST NULO — con un listón por las nubes, una neurona de en medio no dice sí a ninguna edad:
    #    no hay edad «desde» la que diga sí (el cálculo del rótulo no se inventa nada), y la final no
    #    dice sí con las dos de en medio diciendo no.
    alto = listones["la primera"] * 1e6
    nulo = desde_que_edad(pesos["la primera"]["la edad"], alto)
    final_no_no = que_dice_la_final(pesos, listones, 0, 0)
    print(f"[1] test nulo         con el listón por las nubes, la primera diría sí desde los {nulo} años; "
          f"con las dos en no, la final dice {'sí' if final_no_no else 'no'}")
    if nulo < 100 or final_no_no:
        fallos.append("test nulo: el rótulo o la final se inventan un sí")

    # 2. SEÑAL — los rótulos de la figura son lo que hacen sus números: la primera, «desde los 6 años»;
    #    la segunda, «desde los 13 años»; y la final dice sí solo con «la primera sí, la segunda no».
    a = desde_que_edad(pesos["la primera"]["la edad"], listones["la primera"])
    b = desde_que_edad(pesos["la segunda"]["la edad"], listones["la segunda"])
    si = [(x, y) for x in (0, 1) for y in (0, 1) if que_dice_la_final(pesos, listones, x, y)]
    print(f"[2] señal             la primera dice sí desde los {a} años; la segunda, desde los {b}; la final "
          f"dice sí con {si}")
    if (a, b) != (6, 13) or si != [(1, 0)]:
        fallos.append("señal: los rótulos de la figura no son lo que hacen sus números")

    # 3. INVARIANTE — las neuronas de en medio dicen lo que dice la tabla de la salida (los datos y la
    #    cuenta coinciden: peso por edad pasa del listón desde la edad del rótulo, para las 13 edades),
    #    y la figura se dibuja en las dos paletas.
    malas = 0
    for e in range(3, 16):
        for n, desde in zip(NOMBRES_MEDIO, (a, b)):
            si_pasa = e * pesos[n]["la edad"] > listones[n]
            dice = v(d, "final_dice_medio", e, NOMBRES_MEDIO.index(n)) > 0.5
            malas += int(si_pasa != dice or dice != (e >= desde))
    for pal in (COLOR, GRIS):
        dibujar(d, pal, "/dev/null")
        q = dibujar_cuerpo(d, pal, "/dev/null")
    # la figura del cuerpo lleva las dos preguntas que salen de los números, y ningún número más que ellas
    print(f"[3] invariante        edades en que la cuenta, lo que dicen y el rótulo no coinciden: {malas}; "
          f"preguntas de la figura del cuerpo: {q}")
    if malas:
        fallos.append("invariante: los datos, la cuenta y el rótulo no dicen lo mismo")
    if q != ["«¿tiene 6 o más?»", "«¿tiene 13 o más?»"]:
        fallos.append(f"invariante: las preguntas de la figura del cuerpo no son las de los números: {q}")
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
    d = leer_datos()
    dibujar_cuerpo(d, GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")
    dibujar(d, GRIS, str(AQUI / DESTINO_NUMEROS))
    print(f"escrita {DESTINO_NUMEROS}")


if __name__ == "__main__":
    main()
