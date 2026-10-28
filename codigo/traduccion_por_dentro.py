#!/usr/bin/env python3
"""
Capítulo 11 — el acantilado de la traducción, mirado por dentro (L24, tercera vuelta).

En la batería del capítulo, la fila de la traducción sale «ninguna, cuatro, cinco, cinco»: parece
un acantilado. El trabajo de 2023 que cuenta el capítulo dice que esos acantilados los fabrica la
regla de corrección, y que para ver si hay un salto de verdad hay que mirar una medida continua:
cuánta probabilidad da la máquina a la respuesta buena. Esto hace este programa: para las cinco
preguntas de traducción de `crecer.py`, y a los cuatro tamaños, la probabilidad de la palabra
correcta entera (la de su primer trozo por la de los siguientes), justo donde la máquina tiene que
escribirla, y en qué puesto de su lista queda su primer trozo. Mismos modelos, mismo troceado,
mismo procesador y misma precisión que `crecer.py`; no toca su salida.

Uso:
    python traduccion_por_dentro.py --selftest
    python traduccion_por_dentro.py
"""

# ======================= CONSTANTES =======================

TAREA = "traducir del inglés"
CSV_TABLA = "../datos/salidas/crecer.csv"

# ==========================================================

import argparse
import datetime
import platform
import sys

import crecer
from crecer import ETIQUETA_TAMANO, MODELOS, TAREAS, TAREA_CONTROL, acierta
from crudo_o_adiestrado_una_a_una import lista, partes_palabra, visible
from formato import miles, pct, tabla_editorial, trozo


def medir(tok, modelo, items=None):
    """Para cada pregunta: (palabra, probabilidad de la palabra entera, puesto del primer trozo,
    el trozo que la máquina pone primero y su probabilidad)."""
    fuera = []
    for p, e in (items or TAREAS[TAREA]):
        ids = tok(p)["input_ids"]
        # La máquina puede escribir la palabra con un espacio delante («_agua») o sin él
        # («agua»), y la regla del capítulo da por buenas las dos: se suman, como las dos
        # maneras de escribir París en el capítulo 12. El puesto es el mejor de los dos.
        total, puesto = 0.0, None
        for forma in (e, " " + e):
            # prob_palabra desapareció de crudo_o_adiestrado_una_a_una.py el 8 oct 2026; la misma
            # cuenta sale de partes_palabra: la palabra entera y el puesto de su primer trozo.
            partes, v = partes_palabra(tok, modelo, ids, forma)
            pu = partes[0][2]
            total += v
            puesto = pu if puesto is None else min(puesto, pu)
        pl = lista(modelo, ids)
        i = int(pl.argmax())
        fuera.append((e, total, puesto, tok.decode([i]), float(pl[i])))
    return fuera


def bloque(usados, res):
    """Desde el 9 oct 2026, dos tablas de libro (formato.py; REGLAS 6 ter): la probabilidad de la
    palabra buena y, debajo, lo que cada tamaño pone primero en su lista."""
    et = [ETIQUETA_TAMANO[m] for m in usados]
    filas = []
    for k, (p, e) in enumerate(TAREAS[TAREA]):
        filas.append([e] + [pct(res[m][k][1], 1) for m in usados])
    medias = [sum(r[1] for r in res[m]) / len(res[m]) for m in usados]
    filas.append(["**media**"] + [f"**{pct(v, 1)}**" for v in medias])
    assert medias[0] < 0.01 and medias[1] > 0.5, "la nota dice «de menos de uno de cada cien a más de la mitad»"
    t1 = tabla_editorial(
        "La traducción por dentro", ["palabra"] + et, filas, "i" + "d" * len(et),
        [# Orden 97: nota para el lector. Que la palabra buena valga con espacio delante o sin él, y que
         # su probabilidad sea la del primer trozo por la de los siguientes, está en `medir`.
         "Cuánta probabilidad le da cada tamaño a la traducción buena, entera, justo donde tiene que "
         "escribirla. El salto está entre los dos más pequeños: de menos de uno de cada cien a más "
         "de la mitad."])
    filas = []
    for k, (p, e) in enumerate(TAREAS[TAREA]):
        for j, m in enumerate(usados):
            _, _, puesto, top, v = res[m][k]
            filas.append([f"«{p.split(chr(10))[-1].strip()}»" if j == 0 else "",
                          ETIQUETA_TAMANO[m], trozo(visible(top)), pct(v, 1), miles(puesto)])
    t2 = tabla_editorial(
        "Lo que pone primero cada tamaño en su lista",
        ["enunciado", "tamaño", "primero", "con cuánto", "puesto de la buena"], filas, "irirr"
        .replace("r", "d"),
        ["«_»: un espacio. Puesto de la buena: el del primer trozo de la palabra buena en la "
         "lista, de más a menos probable; con espacio delante o sin él, el mejor de los dos."])
    return [t1, t2], medias


def selftest():
    fallos = []
    tok, modelo = crecer.cargar(MODELOS[-1])
    items = TAREAS[TAREA]
    res = medir(tok, modelo)
    # 1. TEST NULO — la palabra de OTRA pregunta, en el mismo sitio, casi no tiene probabilidad.
    ajenas = [(p, items[(k + 1) % len(items)][1]) for k, (p, _) in enumerate(items)]
    nulo = sum(r[1] for r in medir(tok, modelo, ajenas)) / len(items)
    real = sum(r[1] for r in res) / len(items)
    print(f"[1] test nulo         palabra de otra pregunta: {pct(nulo, 2)}; la buena: {pct(real, 1)}")
    if nulo > 0.05 or nulo >= real:
        fallos.append("test nulo: la palabra de otra pregunta sale probable")
    # 2. SEÑAL — copiar la palabra de delante: casi toda la probabilidad.
    copia = medir(tok, modelo, TAREA_CONTROL)
    v = min(r[1] for r in copia)
    print(f"[2] señal implantada  copiar la palabra anterior: al menos {pct(v, 1)}")
    if v < 0.5:
        fallos.append("señal: copiar una palabra no sale probable")
    # 3. INVARIANTE — lo que pone primero la lista es lo que escribe la máquina al elegir lo más
    #    probable (crecer.continuar): la medida y la batería miran la misma máquina.
    malas = [e for (p, e), r in zip(items, res)
             if crecer.continuar(tok, modelo, p, 1) != r[3]]
    print(f"[3] invariante        primer trozo de la lista = primer trozo escrito: {'sí' if not malas else malas}")
    if malas:
        fallos.append(f"invariante: no coinciden en {malas}")
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
    if args.selftest:
        sys.exit(selftest())
    print(f"Medido el {datetime.date.today()} en {platform.platform()}; en procesador, como crecer.py.")
    res, usados = {}, []
    for m in MODELOS:
        tok, modelo = crecer.cargar(m)
        res[m] = medir(tok, modelo)
        # Los aciertos de la regla, con estas mismas máquinas, tienen que ser los de crecer.csv.
        aciertos = sum(acierta(crecer.continuar(tok, modelo, p), e) for p, e in TAREAS[TAREA])
        print(f"{ETIQUETA_TAMANO[m]}: {aciertos} de {len(TAREAS[TAREA])} con la regla del capítulo")
        usados.append(m)
        del modelo
    print()
    tablas, _ = bloque(usados, res)
    for t in tablas:
        print("\n".join(t))
        print()


if __name__ == "__main__":
    main()
