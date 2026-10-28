#!/usr/bin/env python3
"""
Capítulo 12 — crudo o adiestrado, pregunta a pregunta (L24, hallazgos D17, D18 y D21).

`crudo_o_adiestrado.py` mide lo que cuenta el capítulo. Este programa, con los MISMOS modelos,
las MISMAS funciones (las importa, no las copia) y la MISMA regla, saca lo que el lector necesita
ver paso a paso, ya en tablas de libro (formato.py, tabla_editorial):

  1. La lista de «La capital de Francia es», en crudo y adiestrado, y dónde está París en cada
     una: el primer trozo, el segundo sabiendo el primero, la palabra entera (uno por otro) y la
     forma sin tilde, de un solo trozo.
  2. La pregunta sola, en crudo: la lista justo antes de escribir la ciudad, las tres ciudades
     enteras y lo que escribe. En la pequeña sale Nápoles.
  3. La batería del capítulo 11 en tres columnas (crudo, adiestrado, adiestrado con su formato),
     comprobada contra la de crudo_o_adiestrado.csv.
  4. Una pregunta de la batería en las tres columnas: lo que recibe cada una y lo que escribe.

Cambio del 8 de octubre de 2026 (Carlos: «volcamos directamente la salida del script; esas
líneas con todo seguido no ayudan»): las mismas cuentas, y además la probabilidad del segundo
trozo de «París», que antes no se imprimía; todo sale en tablas con título y nota.

Uso:
    python crudo_o_adiestrado_una_a_una.py --selftest
    python crudo_o_adiestrado_una_a_una.py
"""

# ======================= CONSTANTES =======================

CSV_TABLA = "crudo_o_adiestrado.csv"      # donde lo deja crudo_o_adiestrado.py
PARIS = [(" París", "«París», con tilde"), (" Paris", "«Paris», sin tilde")]
CIUDADES = [" París", " Nápoles", " Roma"]
EJEMPLOS = [("datos del mundo", 0), ("sumar dos cifras", 0)]   # (tarea, pregunta)
ANCHO_RESPUESTA = 34
ANCHO_RENGLON = 56     # un renglón más largo de lo que recibe la máquina se parte aquí
TOLERANCIA = 1e-9
NOMBRE = {"500M": "500 millones", "7.000M": "7.000 millones"}
# Orden 97: cifras que las notas escriben con letra; si cambia el parámetro, falla aquí y no en la página.
EN_LETRA = {6: "seis", 5: "cinco", 2: "dos"}
TAREA_NOTA = "sumar dos cifras"      # una tarea cualquiera: la nota solo cuenta cuántas preguntas tiene cada una
# Lo que dice la nota de la lista justo antes de la ciudad, según el tamaño (el borrador del capítulo 12
# las escribe distintas porque lo que enseña cada una es distinto).
NOTA_PREGUNTA_SOLA = {
    "7.000M": "La versión cruda de {nombre}, con la pregunta delante: la lista que tiene justo antes "
              "de escribir la ciudad. Acaba escribiendo «{entero}»",
    "500M": "La versión cruda pequeña, de {nombre}, con la pregunta delante, justo antes de escribir "
            "la ciudad. La «N» es el principio de Nápoles, y lo que acaba escribiendo es «{entero}»",
}

# ==========================================================

import argparse
import csv
import datetime
import platform
import sys
import textwrap

import torch

import crecer
import crudo_o_adiestrado as cap12
from crecer import ETIQUETA_TAMANO, TAREAS, acierta
from formato import barra, miles, muestra_editorial, pct, tabla_editorial, trozo


# --------------------------- las cuentas ---------------------------

def lista(modelo, ids):
    with torch.no_grad():
        logits = modelo(torch.tensor([ids])).logits[0, -1]
    p = torch.softmax(logits.float(), dim=-1)
    assert abs(float(p.sum()) - 1.0) < 1e-3, (
        f"Se esperaba una lista que sumara uno; suma {float(p.sum())}")
    return p


def puesto(p, i):
    return int((p > p[i]).sum()) + 1


def visible(t):
    return t.replace(" ", "_")


def partes_palabra(tok, modelo, ids, palabra):
    """La palabra detrás de `ids`, trozo a trozo: [(trozo, probabilidad sabiendo los de antes,
    puesto en su lista)], y la palabra entera, que es el producto."""
    partes, total = [], 1.0
    for i in tok(palabra)["input_ids"]:
        p = lista(modelo, ids)
        partes.append((tok.decode([i]), float(p[i]), puesto(p, i)))
        total *= float(p[i])
        ids = ids + [i]
    return partes, total


def primeros(tok, p, n=cap12.TOP_N):
    top = torch.topk(p, n)
    return [(tok.decode([int(i)]), float(v)) for v, i in zip(top.values, top.indices)]


def p_txt(x):
    """Un porcentaje para la tabla: un decimal, o «menos de 0,1 %» si no llega."""
    return pct(x, 1) if x >= 0.0005 else "menos de 0,1 %"


def rayas(pares):
    """Nota para los trozos con rayas bajas de verdad: «_» es como el libro dibuja un espacio,
    y sin la nota «___» se leería como tres espacios."""
    out = []
    for w, _ in pares:
        n = w.count("_")
        if n:
            ante = "un espacio seguido de " if w.startswith(" ") else ""
            out.append(f"{trozo(visible(w))} es {ante}{EN_LETRA.get(n, str(n))} "
                       f"raya{'s' if n > 1 else ''} baja{'s' if n > 1 else ''}.")
    return out


# --------------------------- 1. dónde está París ---------------------------

def lista_y_paris(tok, mod):
    ids = tok(cap12.FRASE_CAP7)["input_ids"]
    p = lista(mod, ids)
    pares = primeros(tok, p)
    paris = {w: partes_palabra(tok, mod, ids, w) for w, _ in PARIS}
    return pares, paris


def tabla_lista(pares, titulo, quien):
    filas = [[trozo(visible(w)), pct(v, 1), barra(v)] for w, v in pares]
    # Orden 97: una sola nota, para el lector: el «_», lo que son los trozos de la lista (`quien`, con
    # {n} por cuántos son) y, si hay trozos con rayas bajas de verdad, qué son.
    notas = [" ".join([f"{trozo('_')} marca el espacio que va pegado delante del trozo.",
                       quien.format(n=EN_LETRA[len(pares)])] + rayas(pares))]
    return tabla_editorial(titulo, ["trozo siguiente", "probabilidad", ""], filas, "idi", notas)


def tabla_paris(datos, etiqueta):
    """datos: [(rótulo de columna, {palabra: (partes, total)})]."""
    tilde, sin = PARIS[0][0], PARIS[1][0]
    cols = [d for _, d in datos]
    assert len(cols[0][tilde][0]) == 2 and len(cols[0][sin][0]) == 1, (
        "Se esperaba «París» en dos trozos y «Paris» en uno")
    (t1, _, _), (t2, _, _) = cols[0][tilde][0]
    filas = [
        [f"primer trozo de «París», {trozo(visible(t1))}"] + [p_txt(d[tilde][0][0][1]) for d in cols],
        ["su puesto en la lista"] + [miles(d[tilde][0][0][2]) for d in cols],
        [f"segundo trozo, {trozo(visible(t2))}, detrás del primero"]
        + [p_txt(d[tilde][0][1][1]) for d in cols],
        ["«París» entera: el primero por el segundo"] + [p_txt(d[tilde][1]) for d in cols],
        [f"«Paris» sin tilde, un solo trozo, {trozo(visible(cols[0][sin][0][0][0]))}"]
        + [p_txt(d[sin][1]) for d in cols],
        ["su puesto en la lista"] + [miles(d[sin][0][0][2]) for d in cols],
        ["**las dos maneras juntas**"] + [f"**{p_txt(d[tilde][1] + d[sin][1])}**" for d in cols],
    ]
    # Orden 97: «con tilde va en dos trozos; sin tilde, en uno» es lo que comprueba el assert de arriba.
    notas = ["La versión cruda y la adiestrada, con la frase sola. «París» con tilde va en dos "
             "trozos; sin tilde, en uno. El puesto es el que ocupa cada trozo en la lista entera, "
             "del más probable al menos."]
    return tabla_editorial("Dónde está «París» detrás de «La capital de Francia es»",
                           [""] + [r for r, _ in datos], filas, "idd", notas)


# --------------------------- 2. la pregunta sola ---------------------------

def pregunta_sola(tok, mod):
    """Escribe la respuesta a la pregunta sola y se para justo antes del trozo que empieza la
    ciudad. Devuelve (lo que lleva escrito, la lista ahí, las ciudades, lo que escribe)."""
    ids = tok(cap12.PREGUNTA)["input_ids"]
    with torch.no_grad():
        salida = mod.generate(torch.tensor([ids]), max_new_tokens=cap12.MAX_NUEVOS,
                              do_sample=False, pad_token_id=tok.eos_token_id)
    nuevos = salida[0, len(ids):].tolist()
    entero = tok.decode(nuevos, skip_special_tokens=True).strip()
    corte = None
    for k in range(1, len(nuevos)):
        if tok.decode(nuevos[:k]).strip().endswith("Francia es"):
            corte = k
            break
    assert corte is not None, f"Se esperaba «Francia es» en lo que escribe; escribe {entero!r}"
    delante = ids + nuevos[:corte]
    p = lista(mod, delante)
    ciudades = {c: partes_palabra(tok, mod, delante, c) for c in CIUDADES}
    return tok.decode(nuevos[:corte]).strip(), primeros(tok, p), ciudades, entero


def tablas_pregunta_sola(etiqueta, llevado, pares, ciudades, entero):
    t1 = tabla_editorial(
        f"La pregunta sola, en crudo: lo que puede venir detrás de «{llevado}»",
        ["trozo siguiente", "probabilidad", ""],
        [[trozo(visible(w)), pct(v, 1), barra(v)] for w, v in pares], "idi",
        [f"{trozo('_')} marca el espacio que va pegado delante del trozo."] + rayas(pares)
        + [NOTA_PREGUNTA_SOLA[etiqueta].format(nombre=NOMBRE[etiqueta], entero=entero)])
    if etiqueta == "500M":
        assert pares[0][0].strip() == "N" and "Nápoles" in entero, (
            "la nota dice que la «N» es el principio de Nápoles y que acaba escribiendo Nápoles")
    filas = []
    for c in CIUDADES:
        partes, total = ciudades[c]
        filas.append([c.strip(), " + ".join(trozo(visible(t)) for t, _, _ in partes),
                      p_txt(total), miles(partes[0][2])])
    gana_entera = max(CIUDADES, key=lambda c: ciudades[c][1])
    gana_trozo = min(CIUDADES, key=lambda c: ciudades[c][0][0][2])
    if gana_entera == gana_trozo:
        resultado = f"Gana {gana_entera.strip()}, como ciudad entera y como trozo."
    else:
        resultado = (f"Como ciudad entera gana {gana_entera.strip()}; como trozo, la "
                     f"«{ciudades[gana_trozo][0][0][0].strip()}» de {gana_trozo.strip()}.")
    t2 = tabla_editorial(
        f"Las tres ciudades enteras, en ese mismo punto ({etiqueta})",
        ["ciudad", "en trozos", "la palabra entera", "puesto del primer trozo"], filas, "iidd",
        ["La probabilidad de cada ciudad entera, con todos sus trozos, frente al puesto de su primer "
         f"trozo en la lista. {resultado}"])
    return t1, t2


# --------------------------- 3 y 4. la batería ---------------------------

def responder(tok, mod, con_formato):
    """Las respuestas a la batería, exactamente como las da crudo_o_adiestrado.py."""
    if con_formato:
        return {t: [cap12.continuar(tok, mod, cap12.con_formato(tok, p), 12) for p, _ in items]
                for t, items in TAREAS.items()}
    return {t: [crecer.continuar(tok, mod, p) for p, _ in items] for t, items in TAREAS.items()}


def leer_tabla(ruta=CSV_TABLA):
    with open(ruta, newline="", encoding="utf-8") as fh:
        filas = list(csv.reader(fh))
    return {(f[0], f[2]): [float(x) for x in f[3:6]] for f in filas[1:] if f[1] == "bateria"}


def aciertos(resp):
    return {t: sum(acierta(r, e) for r, (_, e) in zip(resp[t], items))
            for t, items in TAREAS.items()}


def comprobar(etiqueta, columnas, tabla):
    for j, resp in enumerate(columnas):
        for t, a in aciertos(resp).items():
            esperado = tabla[(etiqueta, t)][j]
            assert abs(a / len(TAREAS[t]) - esperado) < TOLERANCIA, (
                f"Se esperaba reproducir {CSV_TABLA}: {etiqueta}, «{t}», columna {j + 1}: "
                f"{a} de {len(TAREAS[t])} frente a {esperado:.3f}")


# La tercera se rotula «con su formato», no «adiestrado, con su formato»: el rótulo largo partía
# la cabecera de la tabla en dos renglones (revisión del PDF, 8 oct 2026). Que también es la
# adiestrada lo dice la nota de la tabla.
COLUMNAS = ["en crudo", "adiestrado", "con su formato"]


def tabla_bateria(etiqueta, columnas):
    filas, tot = [], [0, 0, 0]
    assert all(len(v) == len(TAREAS[TAREA_NOTA]) for v in TAREAS.values()), (
        "la nota dice «cinco preguntas de cada tarea»")
    for t in TAREAS:
        fila = [t]
        for j, resp in enumerate(columnas):
            a = aciertos(resp)[t]
            tot[j] += a
            fila.append(f"{a} de {len(TAREAS[t])}")
        filas.append(fila)
    n = sum(len(v) for v in TAREAS.values())
    filas.append(["**en total**"] + [f"**{a} de {n}**" for a in tot])
    return tabla_editorial(
        f"La batería del capítulo 11, en las tres columnas ({etiqueta})",
        ["tarea"] + COLUMNAS, filas, "iddd",
        [f"Cuántas de las {EN_LETRA[len(TAREAS[TAREA_NOTA])]} preguntas de cada tarea acierta el "
         f"mismo modelo de {NOMBRE[etiqueta]}: en crudo; adiestrado, con el enunciado tal cual; y "
         "adiestrado, con el enunciado dentro de su formato de conversación, que es como lo usas tú. "
         "La regla es la del capítulo 11: acierta si lo primero que escribe es la respuesta "
         "correcta."])


def renglones(texto):
    """El texto en renglones que caben; el que se parte sigue con dos espacios delante."""
    out = []
    for r in texto.rstrip("\n").split("\n"):
        partes = textwrap.wrap(r, ANCHO_RENGLON) or [""]
        out.append(partes[0])
        out += ["  " + x for x in partes[1:]]
    return out


def tablas_una_pregunta(tok_i, columnas, etiqueta):
    out = []
    for tarea, i in EJEMPLOS:
        p, e = TAREAS[tarea][i]
        filas = []
        # Celdas cortas: con «el enunciado tal cual» y parecidos, la columna se partía en tres
        # renglones por fila (revisión del PDF, 8 oct 2026). «el enunciado» pasa al rótulo.
        for j, recibe in enumerate(("tal cual", "tal cual", "dentro de su formato")):
            r = columnas[j][tarea][i]
            primera = r.split("\n")[0].strip()
            if len(primera) > ANCHO_RESPUESTA:
                primera = primera[:ANCHO_RESPUESTA - 1] + "…"
            filas.append([COLUMNAS[j], recibe, f"«{primera}»",
                          "acierta" if acierta(r, e) else "falla"])
        out.append(tabla_editorial(
            f"Una pregunta de «{tarea}» en las tres columnas ({etiqueta})",
            ["columna", "recibe el enunciado", "lo que escribe", "la regla"], filas, "iiii",
            [f"La respuesta correcta es «{e}», y la regla solo da por buena la respuesta que "
             "empieza por ella."]))
        out.append(muestra_editorial(
            f"El enunciado de «{tarea}», tal cual", renglones(p),
            ["Así lo reciben las dos primeras columnas."]))
        formato = renglones(cap12.con_formato(tok_i, p))
        out.append(muestra_editorial(
            f"El mismo enunciado, dentro del formato de conversación", formato,
            ["Así lo recibe la tercera columna. «<|im_start|>» y «<|im_end|>» abren y cierran "
             "cada parte, y «system», «user» y «assistant» dicen de quién es: la instrucción de "
             "serie, quien pregunta y la máquina. La última parte se queda abierta: es su turno."]))
    return out


# --------------------------- selftest ---------------------------

def cargar_pareja(base, instruido):
    tok_b, mod_b = cap12.cargar(base)
    tok_i, mod_i = cap12.cargar(instruido)
    return tok_b, mod_b, tok_i, mod_i


def selftest():
    fallos = []
    base, instruido = cap12.PAREJAS[0]
    tok_b, mod_b, tok_i, mod_i = cargar_pareja(base, instruido)

    # 1. TEST NULO — una palabra que no tiene nada que ver con la frase tiene una probabilidad
    #    minúscula, y la suma de las dos Parises no pasa del cien por cien.
    ids = tok_b(cap12.FRASE_CAP7)["input_ids"]
    _, nada = partes_palabra(tok_b, mod_b, ids, " hipopótamo")
    paris = sum(partes_palabra(tok_b, mod_b, ids, w)[1] for w, _ in PARIS)
    print(f"[1] test nulo         «hipopótamo» detrás de la frase: {pct(nada, 4)}; "
          f"París, las dos: {pct(paris, 2)}")
    if nada > 0.001 or paris > 1:
        fallos.append("test nulo: una palabra ajena sale probable, o las probabilidades pasan de uno")

    # 2. SEÑAL IMPLANTADA — detrás de «La capital de Francia es la ciudad de», el crudo pone
    #    París la primera (capítulo 7): la cuenta tiene que verlo, y la palabra entera tiene que
    #    ser exactamente el primer trozo por el segundo.
    ids2 = tok_b(cap12.FRASE_CAP7 + " la ciudad de")["input_ids"]
    partes, v = partes_palabra(tok_b, mod_b, ids2, " París")
    producto = 1.0
    for _, q, _ in partes:
        producto *= q
    print(f"[2] señal implantada  «París» tras «…la ciudad de»: {pct(v, 2)}, puesto "
          f"{partes[0][2]}; en {len(partes)} trozos, producto igual: "
          f"{'sí' if abs(producto - v) < TOLERANCIA else 'NO'}")
    if partes[0][2] != 1 or abs(producto - v) >= TOLERANCIA:
        fallos.append("señal implantada: París no sale la primera, o la palabra entera no es el producto")

    # 3. INVARIANTE — las respuestas de aquí reproducen la batería de crudo_o_adiestrado.csv.
    cols = [responder(tok_b, mod_b, False), responder(tok_i, mod_i, False),
            responder(tok_i, mod_i, True)]
    try:
        comprobar(ETIQUETA_TAMANO[base], cols, leer_tabla())
        igual = True
    except AssertionError as e:
        igual = False
        fallos.append(f"invariante: {e}")
    print(f"[3] invariante        {ETIQUETA_TAMANO[base]} reproduce {CSV_TABLA}: "
          f"{'sí' if igual else 'NO'}")
    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


# --------------------------- principal ---------------------------

def imprimir(bloques):
    for b in bloques:
        print("\n".join(b) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    print(f"Medido el {datetime.date.today()} en {platform.platform()}; generación determinista "
          "(do_sample=False), como crudo_o_adiestrado.py.\n")
    tabla = leer_tabla()
    for base, instruido in cap12.PAREJAS:
        etiqueta = ETIQUETA_TAMANO[base]
        print(f"{'=' * 60}\nTAMAÑO {etiqueta}\n{'=' * 60}\n")
        # Uno detrás de otro, no los dos a la vez: dos de siete mil millones no caben juntos
        # en la memoria del portátil con lo demás que hay en marcha.
        tok_b, mod_b = cap12.cargar(base)
        pares_b, paris_b = lista_y_paris(tok_b, mod_b)
        sola = pregunta_sola(tok_b, mod_b)
        cols = [responder(tok_b, mod_b, False)]
        del mod_b
        tok_i, mod_i = cap12.cargar(instruido)
        pares_i, paris_i = lista_y_paris(tok_i, mod_i)
        cols += [responder(tok_i, mod_i, False), responder(tok_i, mod_i, True)]
        del mod_i
        comprobar(etiqueta, cols, tabla)
        imprimir([
            tabla_lista(pares_b, "Lo que puede venir detrás de «La capital de Francia es», "
                                 "en crudo", "Los {n} trozos más probables de toda la lista, "
                                 "en la versión cruda, con la frase sola."),
            tabla_lista(pares_i, "Lo mismo, en la versión adiestrada",
                        "Los {n} más probables en la versión adiestrada, con la misma frase "
                        "sola, sin pregunta ni formato de conversación."),
            tabla_paris([("en crudo", paris_b), ("adiestrado", paris_i)], etiqueta),
            *tablas_pregunta_sola(etiqueta, *sola),
        ])
        print(f"(las respuestas reproducen la batería de {CSV_TABLA})\n")
        imprimir([tabla_bateria(etiqueta, cols), *tablas_una_pregunta(tok_i, cols, etiqueta)])


if __name__ == "__main__":
    main()
