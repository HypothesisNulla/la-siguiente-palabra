#!/usr/bin/env python3
"""
Capítulo 11 — la batería de treinta preguntas, pregunta a pregunta (L24, hallazgos D10, D11,
D12, D13 y D15).

`crecer.py` imprime la tabla del capítulo: cuántas preguntas de cada tarea acierta cada tamaño.
Lo que no imprime es lo que el lector necesita para leerla sin imaginar nada: qué se le
escribió a la máquina, qué contestó y por qué eso cuenta como acierto o como fallo. Este
programa vuelve a pasar la MISMA batería, con los MISMOS modelos y la MISMA regla (importa
`crecer.py`, no lo copia), guarda todas las respuestas y comprueba que salen exactamente los
aciertos de `crecer.csv`. Después imprime:

  1. la tabla del capítulo contada en preguntas («2 de 5») en vez de en porcentajes;
  2. una pregunta de cada tarea tal como la recibe la máquina, con lo que contesta cada tamaño;
  3. cómo le llega a la máquina, en trozos, el patrón inventado (la palabra al revés);
  4. la misma tabla puntuada con otra regla, por letras, para ver que la forma de una fila
     depende de cómo se corrige;
  5. cuántos de los números de cada modelo son la tabla de la primera estación (la lista de
     números de cada trozo), que es la parte que el estudio de 2020 dejó fuera de su recuento.

No toca `crecer.py` ni su salida, que el libro ya cita.

Uso:
    python bateria_una_a_una.py --selftest
    python bateria_una_a_una.py
"""

# ======================= CONSTANTES =======================

CSV_TABLA = "../datos/salidas/crecer.csv"            # lo que tiene que salir, pregunta a pregunta
SALIDA_TODAS = "../datos/salidas/bateria_una_a_una_todas.txt"
TAREA_PATRON = "seguir un patrón inventado"
PALABRAS_TABLA = ["libro", "sol"]    # de las cinco del patrón, las que enseña la tabla del troceado
ANCHO_RESPUESTA = 30         # caracteres de la respuesta que caben en la columna
ANCHO_ENUNCIADO = 54         # renglón del enunciado, antes de partirlo
TOLERANCIA = 1e-9

# ==========================================================

import argparse
import csv
import datetime
import platform
import sys
import textwrap

import torch

import crecer
from crecer import ETIQUETA_TAMANO, MODELOS, TAREAS, TAREA_CONTROL, acierta, normalizar
from formato import (barra, miles, muestra_editorial, pct,
                     tabla_editorial, trozo)

# Desde el 9 oct 2026 los cinco bloques salen como tablas y muestras de libro (formato.py;
# REGLAS 6 ter). Las respuestas y las cuentas no cambian.


def primera_linea(respuesta):
    """Lo que la regla corrige: el primer renglón de lo que escribe (crecer.acierta)."""
    return respuesta.split("\n")[0]


def por_letras(respuesta, esperada):
    """Otra regla, para comparar: qué parte de la palabra correcta escribe bien, letra a letra y
    desde el principio, antes de equivocarse. «libros» para «libros» vale 1; «lib» vale 0,5."""
    r, e = normalizar(primera_linea(respuesta)), normalizar(esperada)
    n = 0
    while n < min(len(r), len(e)) and r[n] == e[n]:
        n += 1
    return n / len(e)


def responder_todo(tok, modelo):
    """Todas las respuestas del modelo a la batería, en el orden de TAREAS."""
    return {t: [crecer.continuar(tok, modelo, p) for p, _ in items] for t, items in TAREAS.items()}


def leer_tabla(ruta=CSV_TABLA):
    with open(ruta, newline="", encoding="utf-8") as fh:
        filas = list(csv.reader(fh))
    assert filas[0] == ["modelo", "numeros", "tarea", "acierto"], f"cabecera inesperada en {ruta}"
    return {(m, t): float(a) for m, _, t, a in filas[1:]}


def comprobar_contra_tabla(nombre, respuestas, tabla):
    for t, items in TAREAS.items():
        a = sum(acierta(r, e) for r, (_, e) in zip(respuestas[t], items)) / len(items)
        assert abs(a - tabla[(nombre, t)]) < TOLERANCIA, (
            f"Se esperaba reproducir crecer.csv: {nombre}, «{t}» da {a:.3f} y la tabla "
            f"dice {tabla[(nombre, t)]:.3f}")


def trozos(tok, texto):
    return [tok.decode([i]) for i in tok(texto)["input_ids"]]


def visible(t):
    return t.replace(" ", "_")


def enunciado_en_renglones(p):
    """El enunciado tal cual, renglón a renglón; los largos, partidos para que quepan."""
    lin = []
    for r in p.split("\n"):
        partes = textwrap.wrap(r, ANCHO_ENUNCIADO) or [""]
        lin.append("    " + partes[0])
        lin += ["      " + x for x in partes[1:]]
    return lin


def recorte(s, n=ANCHO_RESPUESTA):
    s = s.replace("\t", " ")
    return s if len(s) <= n else s[:n - 1] + "…"


def elegida(t, veredictos):
    """La pregunta que se enseña de cada tarea: la primera en que no aciertan todos los tamaños
    por igual; si en todas aciertan o fallan todos a la vez, la primera."""
    for i in range(len(TAREAS[t])):
        if len({veredictos[m][t][i] for m in veredictos}) > 1:
            return i
    return 0


NOTA_REGLA = "Acierta si lo primero que escribe es la respuesta correcta."
# Orden 97: cifras que las notas escriben con letra; si cambia el parámetro, falla aquí y no en la página.
EN_LETRA = {5: "cinco", 500: "quinientos"}


def bloque_tabla(usados, resp, cuentas):
    etiquetas = [ETIQUETA_TAMANO[m] for m in usados]
    filas, totales, n_total = [], {m: 0 for m in usados}, 0
    preguntas = len(next(iter(TAREAS.values())))
    assert all(len(items) == preguntas for items in TAREAS.values()), "la nota dice «cinco preguntas de cada tarea»"
    for t, items in TAREAS.items():
        fila = [t]
        for m in usados:
            a = sum(acierta(r, e) for r, (_, e) in zip(resp[m][t], items))
            totales[m] += a
            fila.append(f"{a} de {len(items)}")
        n_total += len(items)
        filas.append(fila)
    filas.append(["**en total**"] + [f"**{totales[m]} de {n_total}**" for m in usados])
    out = ["--- 1. LA TABLA DEL CAPÍTULO, EN PREGUNTAS ---", ""]
    out += tabla_editorial(
        "La batería, en los cuatro tamaños", ["tarea"] + etiquetas, filas,
        "i" + "d" * len(etiquetas),
        [# Orden 97: nota para el lector. La regla de acierto (NOTA_REGLA) está en `acierta`.
         f"Cuántas de las {EN_LETRA[preguntas]} preguntas de cada tarea acierta cada tamaño. "
         f"{etiquetas[0]} quiere decir {EN_LETRA[int(etiquetas[0][:-1])]} millones de números, sus "
         "pesos: es la cifra redonda con que se publicó el modelo."])
    out += [""] + tabla_editorial(
        "Cuántos números tiene cada modelo", ["modelo", "números (pesos)"],
        [[ETIQUETA_TAMANO[m], miles(cuentas[m][0])] for m in usados], "dd",
        ["La suma de todos los números de sus tablas, con el modelo ya cargado."])
    return out


def bloque_ejemplos(usados, resp):
    out = ["--- 2. UNA PREGUNTA DE CADA TAREA, Y LO QUE CONTESTA ---"]
    ver = {m: {t: [acierta(r, e) for r, (_, e) in zip(resp[m][t], TAREAS[t])] for t in TAREAS}
           for m in usados}
    for t, items in TAREAS.items():
        i = elegida(t, ver)
        p, e = items[i]
        out += [""] + muestra_editorial(
            f"Lo que recibe la máquina: una pregunta de «{t}»",
            [l[4:] for l in enunciado_en_renglones(p)], [f"Respuesta correcta: «{e}»."])
        filas = [[ETIQUETA_TAMANO[m], f"«{recorte(primera_linea(resp[m][t][i]).strip())}»",
                  "acierta" if ver[m][t][i] else "falla"] for m in usados]
        out += [""] + tabla_editorial(
            f"Lo que contesta cada tamaño a esa pregunta de «{t}»",
            ["tamaño", "lo que escribe", "la regla"], filas, "dic",
            ["De cada respuesta va el primer renglón, que es lo único que mira la regla."])
    return out


def bloque_troceado(tok, usados, resp):
    items = TAREAS[TAREA_PATRON]
    p, e = items[0]
    out = ["--- 3. CÓMO LE LLEGA EL PATRÓN INVENTADO ---", ""]
    t = trozos(tok, p)
    renglones, renglon = [], []
    for x in t:
        assert "\n" not in x or x == "\n", f"se esperaba el salto de línea como trozo aparte: {x!r}"
        if x == "\n":
            renglones.append("|".join(visible(y) for y in renglon))
            renglon = []
        else:
            renglon.append(x)
    renglones.append("|".join(visible(y) for y in renglon))
    out += muestra_editorial(
        "El primer enunciado del patrón inventado, en trozos", renglones,
        ["«_» es un espacio, y «|», el corte entre dos trozos. Cada renglón acaba además en un "
         "trozo que no se ve: el salto de línea."])
    filas = []
    for pal, rev in [("casa", "asac"), ("mesa", "asem")] + \
            [(pp.split("\n")[-1].split(" ")[0], ee) for pp, ee in items
             if pp.split("\n")[-1].split(" ")[0] in PALABRAS_TABLA]:
        assert pal[::-1] == rev, f"se esperaba que «{rev}» fuera «{pal}» al revés"
        filas.append([pal, trozo("|".join(trozos(tok, pal))), rev,
                      trozo("|".join(trozos(tok, rev)))])
    notas = ["«|» marca el corte entre dos trozos."]
    # En el enunciado, lo que va detrás de «->» lleva un espacio delante, y el espacio cambia el
    # troceado: «asem» sola es «as|em», y « asem» es «_a|sem». Se dice, para que el lector no vea
    # dos troceados de la misma palabra sin saber por qué.
    for rev in ("asac", "asem"):
        solo, con = trozos(tok, rev), trozos(tok, " " + rev)
        if [visible(x) for x in con] != ["_" + solo[0]] + solo[1:]:
            notas[0] += (f" En el enunciado, «{rev}» va con un espacio delante, y así se parte de "
                         f"otra manera: {'|'.join(visible(x) for x in con)}.")
    out += [""] + tabla_editorial(
        "Las palabras y sus vueltas, en trozos",
        ["palabra", "en trozos", "al revés", "en trozos"], filas, "iiii", notas)
    pregunta = p.split(chr(10))[-1].strip()
    out += [""] + tabla_editorial(
        f"Lo que contesta cada tamaño a «{pregunta}»", ["tamaño", "lo que escribe"],
        [[ETIQUETA_TAMANO[m], f"«{recorte(primera_linea(resp[m][TAREA_PATRON][0]).strip())}»"]
         for m in usados], "di",
        [f"Lo que debía escribir es «{e}». "
         + ("Ninguno lo escribe." if not any(acierta(resp[m][TAREA_PATRON][0], e) for m in usados)
            else "Lo escribe " + ", ".join(ETIQUETA_TAMANO[m] for m in usados
                                           if acierta(resp[m][TAREA_PATRON][0], e)) + ".")])
    return out


def bloque_por_letras(usados, resp):
    etiquetas = [ETIQUETA_TAMANO[m] for m in usados]
    filas = []
    for t, items in TAREAS.items():
        filas.append([t] + [pct(sum(por_letras(r, e) for r, (_, e) in zip(resp[m][t], items))
                                / len(items), 0) for m in usados])
    return ["--- 4. LAS MISMAS RESPUESTAS, CORREGIDAS POR LETRAS ---", ""] + tabla_editorial(
        "Las mismas respuestas, corregidas por letras", ["tarea"] + etiquetas, filas,
        "i" + "d" * len(etiquetas),
        ["Cada número: de las letras de la respuesta correcta, qué parte escribe bien y en su "
         "sitio, desde la primera, antes de equivocarse; en promedio de las 5 preguntas.",
         "«libros» para «libros» vale 100 %; «lib», 50 %. Las mayúsculas no cuentan: a «¿cuál "
         "es la capital de Italia?», «rome» por «Roma» vale 75 %."])


def bloque_primera_estacion(usados, cuentas):
    filas = []
    for m in usados:
        total, tabla = cuentas[m]
        filas.append([ETIQUETA_TAMANO[m], miles(total), miles(tabla), pct(tabla / total, 1),
                      barra(tabla / total)])
    return ["--- 5. LAS LISTAS DE LA SEGUNDA ESTACIÓN ---", ""] + tabla_editorial(
        "Las listas de la segunda estación",
        ["tamaño", "números en total", "en las listas", "parte", ""], filas, "ddddi",
        ["Cuántos de los números de cada máquina están en las listas de los trozos, las que "
         "consulta al entrar el texto, y qué parte son del total."])


def selftest():
    fallos = []
    nombre = MODELOS[0]
    tok, modelo = crecer.cargar(nombre)
    resp = responder_todo(tok, modelo)

    # 1. TEST NULO — corregir cada respuesta contra la respuesta de OTRA pregunta de la misma
    #    tarea (girando la lista una posición): la regla no puede regalar aciertos.
    girado = {t: [e for _, e in items[1:] + items[:1]] for t, items in TAREAS.items()}
    nulo = sum(acierta(r, e) for t in TAREAS for r, e in zip(resp[t], girado[t]))
    real = sum(acierta(r, e) for t, items in TAREAS.items() for r, (_, e) in zip(resp[t], items))
    print(f"[1] test nulo         aciertos con las respuestas giradas {nulo}, con las de verdad {real}")
    if nulo >= real or nulo > 2:
        fallos.append(f"test nulo: con las respuestas giradas acierta {nulo}")

    # 2. SEÑAL IMPLANTADA — copiar una palabra que está delante: la regla la da por buena y
    #    la regla por letras le da la nota entera.
    ok = [acierta(crecer.continuar(tok, modelo, p), e) for p, e in TAREA_CONTROL]
    letras = [por_letras(crecer.continuar(tok, modelo, p), e) for p, e in TAREA_CONTROL]
    print(f"[2] señal implantada  copiar la palabra anterior: {sum(ok)} de {len(ok)}; por letras "
          f"{pct(sum(letras) / len(letras), 0)}")
    if not all(ok) or min(letras) < 1:
        fallos.append("señal implantada: copiar una palabra no sale como acierto")

    # 3. INVARIANTE — las respuestas de aquí dan exactamente los aciertos de crecer.csv.
    try:
        comprobar_contra_tabla(nombre, resp, leer_tabla())
        igual = True
    except AssertionError as e:
        igual = False
        fallos.append(f"invariante: {e}")
    print(f"[3] invariante        {ETIQUETA_TAMANO[nombre]} reproduce crecer.csv: {'sí' if igual else 'NO'}")
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

    print(f"Medido el {datetime.date.today()} en {platform.platform()}; generación determinista "
          "(do_sample=False) en procesador, como crecer.py.")
    tabla = leer_tabla()
    resp, cuentas, usados, tok_patron = {}, {}, [], None
    for nombre in MODELOS:
        tok, modelo = crecer.cargar(nombre)
        resp[nombre] = responder_todo(tok, modelo)
        comprobar_contra_tabla(nombre, resp[nombre], tabla)
        cuentas[nombre] = (sum(p.numel() for p in modelo.parameters()),
                           modelo.get_input_embeddings().weight.numel())
        usados.append(nombre)
        if tok_patron is None:
            tok_patron = tok
        else:
            assert trozos(tok, TAREAS[TAREA_PATRON][0][0]) == trozos(tok_patron, TAREAS[TAREA_PATRON][0][0]), \
                "Se esperaba el mismo troceador en los cuatro tamaños"
        del modelo
    print("Las respuestas reproducen crecer.csv en los cuatro tamaños.\n")

    for bloque in (bloque_tabla(usados, resp, cuentas), bloque_ejemplos(usados, resp),
                   bloque_troceado(tok_patron, usados, resp), bloque_por_letras(usados, resp),
                   bloque_primera_estacion(usados, cuentas)):
        print("\n".join(bloque))
        print()

    with open(SALIDA_TODAS, "w", encoding="utf-8") as fh:
        for t, items in TAREAS.items():
            for i, (p, e) in enumerate(items):
                fh.write(f"[{t}, pregunta {i + 1}] correcta: {e!r}\n{p!r}\n")
                for m in usados:
                    r = resp[m][t][i]
                    fh.write(f"  {ETIQUETA_TAMANO[m]:>6} {'acierta' if acierta(r, e) else 'falla  '} {r!r}\n")
                fh.write("\n")
    print(f"\nTodas las respuestas, en {SALIDA_TODAS}")


if __name__ == "__main__":
    main()
