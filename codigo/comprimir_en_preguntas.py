#!/usr/bin/env python3
"""
Capítulo 13 — las dos tablas de la batería, en preguntas (L24).

Los capítulos 11 y 12 dan la batería de treinta preguntas en preguntas acertadas («2 de 5»,
«19 de 30»), no en porcentajes. El 13 tiene que hablar igual. Este programa no vuelve a pasar
ningún modelo: lee lo que ya midieron `comprimir.py` (datos/salidas/comprimir.csv) y
`crudo_o_adiestrado_una_a_una.py` (datos/salidas/crudo_o_adiestrado_una_a_una.txt, bloque del
modelo de 7.000M) y lo imprime en preguntas.

  1. La batería con tres máquinas: 7.000M sin comprimir, 7.000M comprimido, 32.000M comprimido.
  2. La columna «adiestrado» del capítulo 12 al lado de «7.000M sin comprimir» del 13: el mismo
     modelo, la misma batería, la misma regla, y otro programa que lo pone en marcha.

Uso:
    python comprimir_en_preguntas.py --selftest
    python comprimir_en_preguntas.py > ../datos/salidas/comprimir_en_preguntas.txt
"""

# ======================= CONSTANTES =======================

CSV_COMPRIMIR = "../datos/salidas/comprimir.csv"
TXT_CAP12 = "../datos/salidas/crudo_o_adiestrado_una_a_una.txt"
# Desde el 8 oct 2026 crudo_o_adiestrado_una_a_una.py imprime la batería como tabla de libro;
# se lee de ahí, por su título.
BLOQUE_CAP12 = "La batería del capítulo 11, en las tres columnas (7.000M)"
POR_TAREA = 5
# Orden 97: cifras que las notas escriben con letra; si cambia el parámetro, falla aquí y no en la página.
EN_LETRA = {5: "cinco", 3: "tres"}
TAREAS = ["sumar dos cifras", "plurales", "traducir del inglés", "datos del mundo",
          "seguir un patrón inventado", "razonar sobre una frase"]
COLUMNAS = ["7B sin comprimir", "7B comprimido", "32B comprimido"]

# ==========================================================

import argparse
import csv
import re
import sys
from pathlib import Path

from formato import coma, tabla_editorial

AQUI = Path(__file__).resolve().parent


def leer_comprimir(ruta):
    filas = {}
    with open(ruta, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["clave"] != "bateria" or r["tarea"] == "media":
                continue
            filas[r["tarea"]] = [float(r[c]) for c in COLUMNAS]
    assert list(filas) == TAREAS, f"se esperaban las seis tareas en orden; hay {list(filas)}"
    aciertos = {}
    for t, fr in filas.items():
        n = [f * POR_TAREA for f in fr]
        assert all(abs(x - round(x)) < 1e-6 and 0 <= round(x) <= POR_TAREA for x in n), \
            f"«{t}»: {fr} no son aciertos sobre {POR_TAREA} preguntas"
        aciertos[t] = [int(round(x)) for x in n]
    return aciertos


def leer_cap12(ruta):
    texto = Path(ruta).read_text(encoding="utf-8")
    assert texto.count(BLOQUE_CAP12) == 1, f"se esperaba el bloque «{BLOQUE_CAP12}» una vez en {ruta}"
    tramo = texto.split(BLOQUE_CAP12, 1)[1].split("\n:::\n", 1)[0]
    adiestrado = {}
    for t in TAREAS:
        m = re.search(rf"^\| {re.escape(t)} \| (\d) de 5 \| (\d) de 5 \| (\d) de 5 \|$", tramo,
                      re.M)
        assert m, f"no encuentro la fila «{t}» en el bloque del 7.000M de {ruta}"
        adiestrado[t] = int(m.group(2))            # columnas: crudo, adiestrado, con formato
    return adiestrado


# Desde el 8 oct 2026, tablas de libro (formato.py, tabla_editorial; REGLAS 6 ter). En la primera,
# «7.000M: sin comprimir» lo parte el filtro del libro en dos pisos: el tamaño arriba, abarcando
# sus columnas, y la copia debajo.
ROTULOS_1 = ["7.000M: sin comprimir", "7.000M: comprimido", "32.000M: comprimido"]
ROTULOS_2 = ["cap. 12, adiestrado", "cap. 13, sin comprimir", "¿cambia?"]


def bloques(a, c12):
    """Las dos tablas, con su título y su nota."""
    filas = [[t] + [f"{x} de 5" for x in a[t]] for t in TAREAS]
    tot = [sum(a[t][k] for t in TAREAS) for k in range(3)]
    filas.append(["**en total**"] + [f"**{x} de 30**" for x in tot])
    t1 = tabla_editorial(
        "La batería del capítulo 11 con tres máquinas", ["tarea"] + ROTULOS_1, filas, "iddd",
        [# Orden 97: nota para el lector. Que 7.000M y 32.000M sean millones de números (nominal) y
         # que «sin comprimir» sea tal como salió del entrenamiento está en el código y en cap. 13.
         f"Cuántas de las {EN_LETRA[POR_TAREA]} preguntas de cada tarea acierta cada máquina. Las "
         f"{EN_LETRA[len(COLUMNAS)]} están adiestradas; la comprimida guarda cada número con menos "
         "detalle."])
    filas = []
    for t in TAREAS:
        x, y = c12[t], a[t][0]
        filas.append([t, f"{x} de 5", f"{y} de 5", "sí" if x != y else "no"])
    t12, t13 = sum(c12.values()), tot[0]
    filas.append(["**en total**", f"**{t12} de 30**", f"**{t13} de 30**", ""])
    t2 = tabla_editorial(
        "La misma máquina en los capítulos 12 y 13", ["tarea"] + ROTULOS_2, filas, "iddc",
        ["Las dos columnas son la misma máquina, con las mismas preguntas y la misma regla de "
         "corrección. Solo cambia el programa que la pone en marcha: en el capítulo 12, uno de uso "
         "general; en éste, el de Apple."])
    return [t1, t2], tot, (t12, t13)


def selftest():
    fallos = []
    a = leer_comprimir(AQUI / CSV_COMPRIMIR)
    c12 = leer_cap12(AQUI / TXT_CAP12)
    # 1. TEST NULO — un csv con una tarea que no es un número entero de aciertos de 5 revienta.
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as fh:
        fh.write("clave,tarea," + ",".join(COLUMNAS) + "\n")
        for t in TAREAS:
            fh.write(f"bateria,{t},0.500,0.600,0.600\n")
    try:
        leer_comprimir(fh.name)
        fallos.append("test nulo: acepta 0,5 como aciertos sobre 5")
    except AssertionError:
        pass
    finally:
        os.unlink(fh.name)
    print("[1] test nulo         un 0,5 no se deja convertir en aciertos de 5")
    # 2. SEÑAL — el número que el capítulo comenta: el 32.000M acierta 0 de 5 en «saber cosas
    #    del mundo», y el 7.000M sin comprimir y el adiestrado del 12 difieren en dos filas.
    _, tot, (t12, t13) = bloques(a, c12)
    cambian = [t for t in TAREAS if c12[t] != a[t][0]]
    ok = a["datos del mundo"][2] == 0 and cambian == ["sumar dos cifras", "razonar sobre una frase"]
    print(f"[2] señal             32.000M en «datos del mundo»: {a['datos del mundo'][2]} de 5; "
          f"cambian: {cambian}")
    if not ok:
        fallos.append("señal: no salen el número del cero ni las dos filas que cambian")
    # 3. INVARIANTE — el total en preguntas es la media del csv por treinta.
    medias = {}
    with open(AQUI / CSV_COMPRIMIR, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["clave"] == "bateria" and r["tarea"] == "media":
                medias = [float(r[c]) for c in COLUMNAS]
    ok = all(abs(m * 30 - x) < 0.02 for m, x in zip(medias, tot))
    print(f"[3] invariante        totales {tot} = medias del csv {', '.join(coma(m, 3) for m in medias)} por 30: {'sí' if ok else 'NO'}")
    if not ok:
        fallos.append("invariante: los totales no son la media por treinta")
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
    print("--- selftest ---")
    codigo = selftest()
    if codigo or args.selftest:
        sys.exit(codigo)
    print()
    tablas, _, _ = bloques(leer_comprimir(AQUI / CSV_COMPRIMIR), leer_cap12(AQUI / TXT_CAP12))
    for t in tablas:
        print("\n".join(t))
        print()


if __name__ == "__main__":
    main()
