#!/usr/bin/env python3
"""
Capítulo 7 — la lista de probabilidades, leída como una encuesta (F7-3, aprobada en CLARIDAD.md).

La lista que da la máquina detrás de «La capital de Francia es»: los diez primeros trozos y «el
resto», como las barras de una encuesta electoral, con una clave para las barras y el sitio de
«Par», el primer trozo de «París».

Nada se calcula aquí: la lista se lee del bloque 3 de `datos/salidas/maquina_entera.txt` y el
puesto de «Par» del bloque 1 de `datos/salidas/banco_y_paris.txt`.

Uso:
    python figura_encuesta.py --selftest
    python figura_encuesta.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/maquina_entera.txt"
SALIDA_PARIS = "../datos/salidas/banco_y_paris.txt"
DESTINO = "../figuras/encuesta.png"
ALTO = 3.75
ANCHO_BARRAS = 50.0          # unidades del lienzo para el 40 %
TOPE = 0.40

# ==========================================================

import argparse
import re
import sys
from pathlib import Path

from matplotlib.patches import Rectangle

from formato import leer_tablas
from infografia import COLOR, GRIS, Lienzo
from lista_de_probabilidades import leer_lista

AQUI = Path(__file__).resolve().parent


def leer(ruta):
    """La lista, de maquina_entera.txt, y el puesto de «Par», de banco_y_paris.txt; de sus tablas
    editoriales (L24, 9 de octubre)."""
    filas = leer_lista(ruta)
    paris = leer_tablas(Path(AQUI / SALIDA_PARIS).read_text(encoding="utf-8"))
    assert "París, trozo a trozo" in paris, "no se encontró el puesto de «Par»"
    _, trozo_, prob, puesto = paris["París, trozo a trozo"][1][0]
    m = re.match(r"^(\d+) de ([\d.]+)$", puesto)
    assert trozo_ == "`_Par`" and m, "no se encontró el puesto de «Par»"
    lista = leer_tablas(Path(ruta).read_text(encoding="utf-8"))
    nota = " ".join(next(v for t, v in lista.items() if t.startswith("Qué viene después de «"))[2])
    r = re.search(r"El resto se reparte entre las otras ([\d.]+) posibilidades", nota)
    assert r, "no se encontró cuántos trozos junta «el resto»"
    return filas, (prob.replace(" %", ""), m.group(1), m.group(2), r.group(1))


def dibujar(filas, paris, paleta, ruta):
    L = Lienzo("¿Qué sigue a «La capital de Francia es»?",
               "Como una encuesta: cada trozo con su porcentaje, y al final «el resto»,\n"
               "que junta a todos los que no caben. Entre todos suman el 100 %.", paleta, alto=ALTO)
    p, ax = L.p, L.ax
    y = L.y - 2.5
    alto_barra, paso = 3.0, 4.35
    anchos = []
    for t, v in filas:
        resto = t == "(el resto)"
        w = ANCHO_BARRAS * v / TOPE
        anchos.append(w)
        L.texto(21, y, "el resto" if resto else t, tam=7.8, ha="right", negrita=resto,
                mono=not resto)
        ax.add_patch(Rectangle((23, y - alto_barra / 2), w, alto_barra,
                               facecolor=p.tinta if resto else p.contra, edgecolor="none"))
        # Orden 157 (12 nov 2026): porcentajes enteros, que es lo que explica el cuerpo (11.9). Antes: 17,46 %.
        L.texto(23 + w + 1.2, y, f"{round(v * 100)} %", tam=7.6, negrita=resto)
        y -= paso
    # Antes, dos líneas aquí con el puesto y el 0,0544 % de «Par»: ahora van en la prosa del capítulo y en
    # la tabla «Dónde sale «Par»», no dentro de la figura (11.9). `paris` se sigue leyendo para el selftest.
    L.pie("Largo de la barra: el porcentaje. El guion bajo marca que el trozo empieza con un\n"
          "espacio. «El resto» junta a todos los demás trozos de la caja.")
    L.guardar(ruta)
    return anchos


def selftest():
    fallos = []
    filas, paris = leer(AQUI / SALIDA)
    # 1. TEST NULO — una lista vacía de probabilidad no dibuja barras con largo.
    a = dibujar([(t, 0.0) for t, _ in filas], paris, GRIS, "/dev/null")
    print(f"[1] test nulo         todo a 0 -> barras de largo {max(a)}")
    if max(a) != 0:
        fallos.append("test nulo")
    # 2. SEÑAL — la barra más larga es «el resto» y la siguiente «_la», como en la salida.
    a = dibujar(filas, paris, GRIS, "/dev/null")
    orden = sorted(range(len(a)), key=lambda i: -a[i])
    ok = filas[orden[0]][0] == "(el resto)" and filas[orden[1]][0] == "_la" and paris[1] == "123"
    print(f"[2] señal             más larga «{filas[orden[0]][0]}», luego «{filas[orden[1]][0]}»; «Par» en el {paris[1]}: {ok}")
    if not ok:
        fallos.append("señal")
    # 3. INVARIANTE — largo proporcional al porcentaje, y todo suma el 100 %.
    suma = sum(v for _, v in filas)
    prop = all(abs(w - ANCHO_BARRAS * v / TOPE) < 1e-9 for w, (_, v) in zip(a, filas))
    dibujar(filas, paris, COLOR, "/dev/null")
    print(f"[3] invariante        suma {suma:.4f}; barras proporcionales: {prop}")
    if abs(suma - 1) > 0.001 or not prop:
        fallos.append("invariante")
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
