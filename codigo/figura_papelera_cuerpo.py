#!/usr/bin/env python3
"""
Capítulo 8 — la papelera, para el CUERPO del capítulo (Orden 159, 12 nov 2026).

El capítulo 8 enseña, en el cuerpo, a quién presta atención el final de la frase del «alto»
con una sola barra por palabra y porcentajes enteros: la figura grande de `figura_el_reparto.py`
(las tres frases, con decimales) se queda en el anexo A8. Esta figura es la del cuerpo.

De dónde salen los números. De las mismas funciones y del mismo CSV que `figura_el_reparto.py`
(`../datos/salidas/reparto_una_a_una.csv`, escrito por `reparto_una_a_una.py`): este programa
**no carga ningún modelo** y no mide nada, solo importa `promedios` y `leer_filas` de aquel
y dibuja la media de las 336 miradas para la frase del «alto». Si alguien vuelve a medir,
cambian las dos figuras o no cambia ninguna.

Lleva solo lo que hace falta para leerla (regla 11.9): una barra por palabra, el entero al lado,
la llamada de «la papelera» y una clave corta. Los enteros se redondean aquí y suman cien (lo
comprueba el selftest); las frases que explican van en la prosa.

Uso:
    python figura_papelera_cuerpo.py --selftest
    python figura_papelera_cuerpo.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../figuras/papelera_cuerpo.png"
ANCHO_ALTO = (6.2, 4.6)          # pulgadas; entra a lo ancho de la caja, como la figura del anexo
PUNTOS = 200
FRASE = "alto"
ROTULO_PRIMERA = "El"            # la primera palabra, con mayúscula (es ortografía, no dato)
ROTULO_ULTIMA = "alto."          # la barra del final es el adjetivo con el punto que va pegado
TITULO = "A quién presta atención el final de la frase"
EJE = "de cada cien"
LLAMADA = "la papelera"
CLAVE = "En negro, las dos palabras entre las que había que elegir."
IMPLANTADA_VASO = 0.90           # selftest: el reparto que se le mete a «vaso»
TOL_SUMA = 0.01                  # puntos: lo que puede apartarse de cien la suma de los valores
# Lo que el capítulo dice: los enteros de la figura (si cambian los datos, esto falla, y el
# texto hay que mirarlo).
ENTEROS_DEL_CAPITULO = [53, 4, 1, 2, 1, 2, 3, 4, 2, 3, 25]

# ==========================================================

import argparse
import io
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figura_el_reparto as fr
from formato import coma


def enteros_de(valores):
    return [round(v) for v in valores]


def rotulos_de(palabras):
    r = list(palabras)
    r[0] = ROTULO_PRIMERA
    r[-1] = ROTULO_ULTIMA
    return r


def dibujar(banda, salida=SALIDA):
    palabras, valores = list(banda["palabras"]), banda["valores"]
    n = len(palabras)
    enteros = enteros_de(valores)
    fig, ax = plt.subplots(figsize=ANCHO_ALTO)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.89, bottom=0.20)
    ax.barh(range(n), valores, height=0.62, color=[fr.color_de(p) for p in palabras],
            edgecolor="black", linewidth=0.4, zorder=3)
    for k, (v, e) in enumerate(zip(valores, enteros)):
        ax.text(v + 0.9, k, f"{e}", va="center", ha="left", fontsize=10.5, color="0.15", zorder=4)
    ax.set_yticks(range(n))
    ax.set_yticklabels(rotulos_de(palabras), fontsize=11.5)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.invert_yaxis()
    ax.set_ylim(n - 0.4, -0.6)
    ax.set_xlim(0, 60)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax.xaxis.set_tick_params(labelsize=10.4)
    ax.set_xlabel(EJE, fontsize=10.8, labelpad=5)
    ax.grid(axis="x", color="0.88", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate(LLAMADA, xy=(40, 0.0), xytext=(33, 2.0), fontsize=10.5, color="0.15",
                ha="left", va="center", arrowprops=dict(arrowstyle="-", color="0.45", lw=0.8))
    fig.text(0.5, 0.975, TITULO, ha="center", va="top", fontsize=13)
    fig.text(0.5, 0.03, CLAVE, ha="center", va="bottom", fontsize=10, color="0.3")
    fig.savefig(salida, dpi=PUNTOS, format="png")
    plt.close(fig)
    return enteros


# ============================ SELFTEST ============================

def filas_inventadas(reparto):
    """Un CSV de mentira con las palabras de la frase del «alto» y las porciones que se le pidan."""
    palabras = fr.FRASES[FRASE].rstrip(".").split()
    return [{"adjetivo": FRASE, "capa": str(c), "mirada": str(m), "palabra": p.lower(),
             "porcion": f"{x:.6f}"}
            for c in (1, 2) for m in (1, 2) for p, x in zip(palabras, reparto)]


def selftest():
    fallos = []
    banda = fr.promedios(fr.leer_filas())[FRASE]
    n = len(banda["palabras"])

    # [1] Test nulo: un reparto plano, las once por igual, no puede dibujar una barra que destaque
    #     ni dar enteros distintos; si la figura adornara una barra por su cuenta, aquí se vería.
    plano = fr.promedios(filas_inventadas([1.0 / n] * n))[FRASE]
    e_plano = enteros_de(plano["valores"])
    destaca, _ = fr.destaca_la_primera(plano["valores"])
    print(f"[1] test nulo         reparto plano: las {n} barras dan el mismo entero "
          f"({sorted(set(e_plano))}); la primera no destaca: {'FALLA' if destaca else 'correcto'}")
    if len(set(e_plano)) != 1 or destaca:
        fallos.append("test nulo: con un reparto plano las barras tienen que ser iguales y ninguna destacar")

    # [2] Señal implantada: «vaso» con el 90 % tiene que salir la barra más larga y en negro.
    palabras_frase = [p.lower() for p in fr.FRASES[FRASE].rstrip(".").split()]
    reparto = [(1.0 - IMPLANTADA_VASO) / (n - 1)] * n
    reparto[palabras_frase.index("vaso")] = IMPLANTADA_VASO
    implantada = fr.promedios(filas_inventadas(reparto))[FRASE]
    k = max(range(n), key=lambda i: implantada["valores"][i])
    mayor, tinta = implantada["palabras"][k], fr.color_de(implantada["palabras"][k])
    print(f"[2] señal implantada  metido un {coma(100 * IMPLANTADA_VASO, 0)} % a «vaso»: la barra más larga "
          f"es «{mayor}» ({enteros_de(implantada['valores'])[k]}) y sale en "
          f"{'negro' if tinta == fr.NEGRO else 'gris'}")
    if mayor != "vaso" or tinta != fr.NEGRO:
        fallos.append("señal implantada: la barra de «vaso» tenía que ser la mayor y salir en negro")

    # [3] Invariante: un reparto es un pastel entero. Los valores suman cien, los enteros que se
    #     dibujan también, y son los que dice el capítulo.
    suma, enteros = sum(banda["valores"]), enteros_de(banda["valores"])
    print(f"[3] invariante        {n} barras: los valores suman {coma(suma, 3)}, los enteros "
          f"{enteros} suman {sum(enteros)}; los del capítulo: {'iguales' if enteros == ENTEROS_DEL_CAPITULO else 'DISTINTOS'}")
    if abs(suma - 100.0) > TOL_SUMA:
        fallos.append(f"invariante: los valores suman {coma(suma, 3)} y tendrían que sumar cien")
    if sum(enteros) != 100:
        fallos.append(f"invariante: los enteros suman {sum(enteros)} y tendrían que sumar cien")
    if enteros != ENTEROS_DEL_CAPITULO:
        fallos.append(f"invariante: los enteros son {enteros} y el capítulo dice {ENTEROS_DEL_CAPITULO}")
    dibujar(banda, io.BytesIO())

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    if p.parse_args().selftest:
        return selftest()
    enteros = dibujar(fr.promedios(fr.leer_filas())[FRASE])
    print(f"escrita {SALIDA}; enteros {enteros}, suma {sum(enteros)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
