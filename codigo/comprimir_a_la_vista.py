#!/usr/bin/env python3
"""
Capítulo 13 — la tabla de la compresión, leída con lo que el lector necesita al lado (L24, E11,
E13 y E14).

No ejecuta ningún modelo: lee las salidas que ya dejaron `comprimir.py` (la tabla y las respuestas
una a una) y `crudo_o_adiestrado.py` (la tabla del capítulo 12), y las pone así:

  1. LA TABLA, CON LOS TAMAÑOS COMO LOS ESCRIBE EL CAPÍTULO 11. «7.000M» y «32.000M», en millones
     de números, en lugar de «7B» y «32B».
  2. LA MISMA MÁQUINA EN LOS CAPÍTULOS 12 Y 13. La columna «adiestrado» de la máquina de siete mil
     millones en el capítulo 12, al lado de la de «sin comprimir» del 13: la misma máquina, las
     mismas treinta preguntas y la misma regla de corrección; solo cambia el programa que la
     ejecuta. Qué filas cambian, y la media en preguntas.
  3. LAS PALABRAS AL REVÉS. El enunciado que recibe la máquina (los dos ejemplos del capítulo 11 y
     la palabra), y una fila por palabra: la palabra, cómo debía quedar al revés, y la primera
     línea que escribió cada máquina.

Uso:
    python comprimir_a_la_vista.py --selftest
    python comprimir_a_la_vista.py > ../datos/salidas/comprimir_a_la_vista.txt
"""

# ======================= CONSTANTES =======================

TABLA = "../datos/salidas/comprimir.csv"
DETALLE = "../datos/salidas/comprimir_detalle.txt"
CAP12 = "../datos/salidas/crudo_o_adiestrado.txt"
CAP12_TAMANO = "7.000M"                 # la sección del capítulo 12 que se compara
CAP12_COLUMNA = "adiestrado"            # su columna: enunciado tal cual, como en el capítulo 13
PREGUNTAS_POR_TAREA = 5
# Orden 97: cifra que la nota escribe con letra (MAX_NUEVOS de crecer.py); si cambia, falla aquí y no en la página.
EN_LETRA = {8: "ocho"}
TAREA_REVES = "seguir un patrón inventado"
TAREA_MUNDO = "datos del mundo"      # el cero de cinco de la de 32.000M
NOMBRES = {                              # como escribe los tamaños el capítulo 11
    "7B sin comprimir": "7.000M sin comprimir",
    "7B comprimido": "7.000M comprimido",
    "32B comprimido": "32.000M comprimido",
}
GRANDE, PEQUENO = "32B comprimido", "7B sin comprimir"   # las dos de la tabla de las palabras

# ==========================================================

import argparse
import csv
import re
import sys
from pathlib import Path

from formato import (ANCHO_CAJA, ANCHO_CAJA_CITA, comprobar_ancho, muestra_editorial, pct,
                     tabla_editorial)

AQUI = Path(__file__).resolve().parent


def tabla(ruta=AQUI / TABLA):
    with open(ruta, encoding="utf-8") as fh:
        filas = list(csv.reader(fh))
    cab = filas[0][2:]
    assert cab == list(NOMBRES), f"se esperaban las columnas {list(NOMBRES)}; hay {cab}"
    return cab, {f[1]: [float(x) for x in f[2:]] for f in filas[1:]}


def cap12(ruta=AQUI / CAP12):
    """La columna «adiestrado» de la batería, en la sección de 7.000M del capítulo 12."""
    t = Path(ruta).read_text(encoding="utf-8")
    seccion = t.split(f"TAMAÑO {CAP12_TAMANO}", 1)[1].split("3. La batería", 1)[1]
    lineas = seccion.splitlines()
    cab = next(l for l in lineas if l.strip().startswith("tarea"))
    columnas = re.split(r"\s{2,}", cab.strip())[1:]
    k = columnas.index(CAP12_COLUMNA)
    out = {}
    for l in lineas[lineas.index(cab) + 1:]:
        m = re.match(r"\s+(.+?)\s{2,}(\d+ %)\s+(\d+ %)\s+(\d+ %)\s*$", l)
        if not m:
            if out:
                break
            continue
        out[m.group(1)] = int(m.group(2 + k).split()[0]) / 100
    media = out.pop("media")
    assert abs(media - sum(out.values()) / len(out)) < 0.005, "la media del capítulo 12 no casa"
    assert f"tamaño del modelo en las tres columnas: {CAP12_TAMANO}" in seccion
    return out


def respuestas(ruta=AQUI / DETALLE, tarea=TAREA_REVES):
    """Para cada máquina, en una tarea (de entrada, la de las palabras al revés): (enunciado, esperada, primera
    línea de lo que escribió, lo que escribió entero)."""
    t = Path(ruta).read_text(encoding="utf-8")
    trozos = re.split(r"(?m)^={10,}\n(.+)\n={10,}$", t)
    out = {}
    for i in range(1, len(trozos), 2):
        seccion = trozos[i + 2 - 1]
        reves = seccion.split(f"--- {tarea} ---", 1)[1].split("\n---", 1)[0]
        items, actual = [], None
        for l in reves.splitlines():
            m = re.match(r"  \[(sí|NO)\] (.*)$", l)
            if m:
                actual = {"acierto": m.group(1) == "sí", "enunciado": [m.group(2)], "resp": []}
                items.append(actual)
                continue
            m = re.match(r"  esperada '(.*?)' -> (.*)$", l)
            if m:
                actual["esperada"], actual["resp"] = m.group(1), [m.group(2)]
                continue
            if actual is not None and l.strip():
                (actual["resp"] if "esperada" in actual else actual["enunciado"]).append(l.strip())
        out[trozos[i]] = items
    return out


def informe():
    cab, t = tabla()
    print("--- 1. LA TABLA, CON LOS TAMAÑOS EN MILLONES ---")
    print(f"  {'tarea':<28}" + "".join(f"{NOMBRES[c]:>24}" for c in cab))
    for tarea, v in t.items():
        print(f"  {tarea:<28}" + "".join(f"{pct(x, 0):>24}" for x in v))
    print()
    for l in comprobar_ancho([
            "  las cifras de la tabla son aciertos sobre 5 preguntas por tarea.",
            "  «7.000M», «32.000M»: millones de números (nominal).",
            "  «comprimido»: guardado con menos detalle en cada número;",
            "  «sin comprimir»: tal como salió del entrenamiento.",
            "  las tres máquinas están adiestradas (capítulo 12)."], ANCHO_CAJA):
        print(l)

    print("\n--- 2. LA MISMA MÁQUINA EN LOS CAPÍTULOS 12 Y 13 ---")
    doce = cap12()
    trece = {k: v[cab.index(PEQUENO)] for k, v in t.items() if k != "media"}
    assert set(doce) == set(trece), f"las tareas no casan: {sorted(doce)} / {sorted(trece)}"
    print(f"  {'tarea':<28}{'cap. 12, adiestrado':>22}{'cap. 13, sin comprimir':>25}{'¿cambia?':>10}")
    for tarea in trece:
        cambia = "sí" if doce[tarea] != trece[tarea] else "no"
        print(f"  {tarea:<28}{pct(doce[tarea], 0):>22}{pct(trece[tarea], 0):>25}{cambia:>10}")
    n = PREGUNTAS_POR_TAREA * len(trece)
    a12 = round(sum(doce.values()) * PREGUNTAS_POR_TAREA)
    a13 = round(sum(trece.values()) * PREGUNTAS_POR_TAREA)
    print(f"  {'media':<28}{pct(a12 / n, 0):>22}{pct(a13 / n, 0):>25}")
    print(f"  {'en preguntas':<28}{f'{a12} de {n}':>22}{f'{a13} de {n}':>25}")
    return t, doce, trece, a12, a13


def informe_reves():
    """Desde el 8 oct 2026, el enunciado como muestra de libro y las palabras como tabla de libro
    (formato.py; REGLAS 6 ter), con el veredicto de la regla en su propia columna."""
    r = respuestas()
    print(f"\n--- 3. LAS PALABRAS AL REVÉS ---\n")
    g, p = r[GRANDE], r[PEQUENO]
    print("\n".join(muestra_editorial(
        "Lo que recibe la máquina, la primera de las cinco", g[0]["enunciado"],
        ["El mismo enunciado del capítulo 11: dos ejemplos y la palabra."])))
    filas = []
    for a, b in zip(g, p):
        assert a["esperada"] == b["esperada"]
        palabra = a["enunciado"][-1].rstrip(" ->").strip()
        assert palabra[::-1] == a["esperada"], f"«{palabra}» al revés no es «{a['esperada']}»"
        filas.append([palabra, a["esperada"], a["resp"][0].strip(), "sí" if a["acierto"] else "no",
                      b["resp"][0].strip(), "sí" if b["acierto"] else "no"])
    print()
    print("\n".join(tabla_editorial(
        "Las palabras al revés",
        ["palabra", "al revés, como debía", "32.000M: escribe", "32.000M: ¿vale?",
         "7.000M: escribe", "7.000M: ¿vale?"], filas, "iiicic",
        ["La máquina grande, comprimida, y la de 7.000 millones, sin comprimir; las dos, "
         "adiestradas. «Escribe» es la primera línea de lo que escribió cada una, y «¿vale?», si "
         "la regla de corrección la da por buena."])))
    print("\nlo que escribió después de la primera línea, entero:")
    for nombre, items in ((GRANDE, g), (PEQUENO, p)):
        for x in items:
            print(f"  {NOMBRES[nombre]:<22} {x['esperada']:<6} -> " + " / ".join(x["resp"]))
    return r


def max_nuevos(ruta=AQUI / "crecer.py"):
    """Cuántos trozos de respuesta se le dejan escribir: MAX_NUEVOS de crecer.py, que es el que
    usa comprimir.py. Se lee sin importarlo, que cargaría la biblioteca de la tarjeta gráfica."""
    import ast
    arbol = ast.parse(Path(ruta).read_text(encoding="utf-8"))
    return next(ast.literal_eval(n.value) for n in arbol.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "MAX_NUEVOS")


def informe_mundo():
    """El cero de cinco de la de 32.000M en «datos del mundo», respuesta a respuesta."""
    g = respuestas(tarea=TAREA_MUNDO)[GRANDE]
    filas = [[x["esperada"], " / ".join(l.strip() for l in x["resp"]),
              "sí" if x["acierto"] else "no"] for x in g]
    print(f"\n--- 4. LAS CINCO RESPUESTAS DE LA DE 32.000M EN «{TAREA_MUNDO.upper()}» ---\n")
    print("\n".join(tabla_editorial(
        f"Las cinco respuestas de «{TAREA_MUNDO}», 32.000M comprimido",
        ["se esperaba", "lo que escribe", "¿vale?"], filas, "iic",
        ["«/» es un salto de línea. Lo que escribe, sin tocar, en los "
         f"{EN_LETRA[max_nuevos()]} trozos que se le dejan. «¿Vale?» dice si la regla de corrección "
         "la da por buena, y la regla exige que la respuesta empiece por la palabra que se esperaba."])))
    return g


def selftest():
    fallos = []
    cab, t = tabla()
    doce, r = cap12(), respuestas()

    # 1. TEST NULO — si todas las respuestas estuvieran suspendidas, el recuento tiene que dar
    #    cero aciertos: el recuento no pone aciertos que no están. Se prueba sobre una copia de
    #    las respuestas con todas las marcas cambiadas a «NO».
    import tempfile
    texto = (AQUI / DETALLE).read_text(encoding="utf-8").replace("  [sí] ", "  [NO] ")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write(texto)
    nulo = respuestas(fh.name)
    Path(fh.name).unlink()
    ceros = all(sum(x["acierto"] for x in v) == 0 for v in nulo.values())
    print(f"[1] test nulo         con todas las respuestas suspendidas, aciertos contados: "
          f"{'ninguno' if ceros else 'ALGUNO'}")
    if not ceros:
        fallos.append("test nulo: el recuento encuentra aciertos donde no los hay")

    # 2. SEÑAL IMPLANTADA — lo que el capítulo dice que hay: la de 32.000M acierta sol, pan y
    #    mar y falla libro y gato; y el capítulo 12 dio 57 % a la de 7.000M adiestrada.
    ok_g = [x["esperada"] for x in r[GRANDE] if x["acierto"]] == ["los", "nap", "ram"]
    media12 = sum(doce.values()) / len(doce)
    print(f"[2] señal implantada  aciertos de la de 32.000M: "
          f"{[x['esperada'] for x in r[GRANDE] if x['acierto']]}; media del capítulo 12: "
          f"{pct(media12, 0)}")
    if not ok_g or pct(media12, 0) != "57 %":
        fallos.append("señal: las respuestas o la tabla del capítulo 12 no son las que se esperan")

    # 3. INVARIANTE DEL DOMINIO — el acierto de la tarea en la tabla es el que sale de contar las
    #    respuestas una a una, para las tres máquinas.
    casan = all(abs(t[TAREA_REVES][cab.index(m)] - sum(x["acierto"] for x in r[m]) / len(r[m]))
                < 1e-9 for m in cab)
    print(f"[3] invariante        la tabla y las respuestas una a una cuentan lo mismo: "
          f"{'sí' if casan else 'NO'}")
    if not casan:
        fallos.append("invariante: la tabla y las respuestas una a una no casan")
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
    if ap.parse_args().selftest:
        sys.exit(selftest())
    informe()
    informe_reves()
    informe_mundo()


if __name__ == "__main__":
    main()
