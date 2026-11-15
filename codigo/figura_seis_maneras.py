#!/usr/bin/env python3
"""
Capítulo 11 — seis maneras de aparecer.

La tabla de las seis tareas (crecer.py), dibujada como seis cuadritos: en cada uno, cuántas de
las cinco preguntas de la tarea acierta cada uno de los cuatro tamaños. Lo que en la tabla hay
que sacar comparando filas se ve aquí de un golpe: cada tarea tiene su propia forma.

Los valores se leen de `datos/salidas/crecer.csv`.

Uso:
    python figura_seis_maneras.py --selftest
    python figura_seis_maneras.py
"""

# ======================= CONSTANTES =======================

CSV = "../datos/salidas/crecer.csv"
DESTINO = "../figuras/seis_maneras.png"
ALTO = 4.6
PREGUNTAS_POR_TAREA = 5
# El nombre redondo de cada modelo, el mismo que usa la tabla del libro (crecer.py,
# ETIQUETA_TAMANO). Se copia aquí para no cargar torch solo por cuatro rótulos; el selftest
# comprueba que el orden por número de pesos es el de los rótulos.
ROTULO = {"Qwen/Qwen2.5-0.5B": "500M", "Qwen/Qwen2.5-1.5B": "1.500M",
          "Qwen/Qwen2.5-3B": "3.000M", "Qwen/Qwen2.5-7B": "7.000M"}

# ==========================================================

import argparse
import csv
import sys

from infografia import COLOR, GRIS, Lienzo


def leer(ruta=CSV):
    """Devuelve (tamaños en orden, tareas en orden, {tarea: [aciertos de 5 por tamaño]})."""
    with open(ruta, newline="", encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    assert filas and list(filas[0]) == ["modelo", "numeros", "tarea", "acierto"], (
        f"Se esperaban las columnas modelo, numeros, tarea, acierto en {ruta}; "
        f"se encontraron {list(filas[0]) if filas else 'ninguna fila'}")
    pesos = {}
    tareas = []
    valor = {}
    for f in filas:
        assert f["modelo"] in ROTULO, f"Se esperaba un modelo de {list(ROTULO)}; hay {f['modelo']}"
        pesos[f["modelo"]] = int(f["numeros"])
        if f["tarea"] not in tareas:
            tareas.append(f["tarea"])
        a = float(f["acierto"]) * PREGUNTAS_POR_TAREA
        assert abs(a - round(a)) < 1e-6 and 0 <= round(a) <= PREGUNTAS_POR_TAREA, (
            f"Se esperaba un número entero de aciertos entre 0 y {PREGUNTAS_POR_TAREA} en "
            f"{f['modelo']}, {f['tarea']}; sale {a}")
        valor[(f["modelo"], f["tarea"])] = int(round(a))
    modelos = sorted(pesos, key=pesos.get)
    assert [ROTULO[m] for m in modelos] == list(ROTULO.values()), (
        f"Se esperaba que el orden por número de pesos fuera {list(ROTULO.values())}; "
        f"sale {[ROTULO[m] for m in modelos]}")
    assert len(valor) == len(modelos) * len(tareas), (
        f"Se esperaban {len(modelos) * len(tareas)} casillas; hay {len(valor)}")
    return ([ROTULO[m] for m in modelos], tareas,
            {t: [valor[(m, t)] for m in modelos] for t in tareas})


def dibujar(tamanos, tareas, datos, paleta, ruta):
    """Dibuja y devuelve, por tarea, los valores que ha puesto en el cuadrito."""
    L = Lienzo("Seis maneras de aparecer",
               "Cuántas de las cinco preguntas de cada tarea acierta cada tamaño.\n"
               "500M es la máquina de quinientos millones de pesos, y así las demás.",   # Orden 166: la tabla ya no está en el cuerpo (pasó al A11)
               paleta, alto=ALTO)
    p = paleta
    arriba = L.y / L.alto_u
    dibujados = {}
    columnas, filas = 3, 2
    ancho, alto = 0.80 / columnas, (arriba - 0.12) / filas
    x = list(range(len(tamanos)))
    for k, t in enumerate(tareas):
        c, r = k % columnas, k // columnas
        ax = L.fig.add_axes([0.08 + c * (ancho + 0.03), 0.10 + (filas - 1 - r) * (alto + 0.02),
                             ancho, alto - 0.07])
        y = datos[t]
        ax.plot(x, y, color=p.tinta, linewidth=1.6, marker="o", markersize=3.2)
        for xi, yi in zip(x, y):
            ax.annotate(str(yi), (xi, yi), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=6.4, color=p.tinta)
        ax.set_ylim(-0.4, PREGUNTAS_POR_TAREA + 1.2)
        ax.set_yticks([0, PREGUNTAS_POR_TAREA])
        ax.set_xticks(x)
        ax.set_xticklabels(tamanos, fontsize=5.8)
        ax.set_xlim(-0.4, len(x) - 0.6)
        ax.tick_params(labelsize=6.2, length=2.0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(t, fontsize=7.6, color=p.tinta, loc="left")
        if c == 0:
            ax.set_ylabel("aciertos (de 5)", fontsize=6.6)
        dibujados[t] = list(ax.get_lines()[0].get_ydata())
    L.guardar(ruta)
    return dibujados


def selftest():
    fallos = []
    tamanos, tareas, datos = leer()

    # 1. TEST NULO — con todas las casillas a cero, los seis cuadritos quedan planos en el suelo.
    ceros = {t: [0] * len(tamanos) for t in tareas}
    d = dibujar(tamanos, tareas, ceros, GRIS, "/dev/null")
    planos = all(max(v) == 0 for v in d.values())
    print(f"[1] test nulo         todo a cero: {'seis cuadritos planos' if planos else 'NO'}")
    if not planos:
        fallos.append("test nulo: con todo a cero algún cuadrito no queda plano")

    # 2. SEÑAL IMPLANTADA — lo dibujado es lo que dice el csv, punto a punto; y la traducción,
    #    la fila del salto, es 0, 4, 5, 5 como en la tabla del libro.
    d = dibujar(tamanos, tareas, datos, GRIS, "/dev/null")
    iguales = all(list(d[t]) == datos[t] for t in tareas)
    salto = datos.get("traducir del inglés") == [0, 4, 5, 5]
    print(f"[2] señal implantada  lo dibujado es el csv: {'sí' if iguales else 'NO'}; "
          f"traducir: {datos.get('traducir del inglés')}")
    if not (iguales and salto):
        fallos.append("señal: lo dibujado no es el csv, o la traducción no es 0, 4, 5, 5")

    # 3. INVARIANTE — seis tareas, cuatro tamaños, y los totales por tamaño son los de la
    #    tabla del libro: 4, 14, 19 y 19 de 30.
    for pal in (COLOR, GRIS):
        dibujar(tamanos, tareas, datos, pal, "/dev/null")
    totales = [sum(datos[t][i] for t in tareas) for i in range(len(tamanos))]
    ok = len(tareas) == 6 and len(tamanos) == 4 and totales == [4, 14, 19, 19]
    print(f"[3] invariante        {len(tareas)} tareas, {len(tamanos)} tamaños, "
          f"totales {totales}: {'bien' if ok else 'MAL'}")
    if not ok:
        fallos.append("invariante: no son 6 tareas por 4 tamaños, o los totales no son 4, 14, "
                      "19 y 19")
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
    dibujar(*leer(), GRIS, DESTINO)
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
