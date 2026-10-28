#!/usr/bin/env python3
"""
Capítulo 7 — de dónde salen los trozos (Carlos, 4 de octubre de 2026: «no está muy claro de
dónde salió esa forma de trocear»).

La máquina del capítulo 7 parte el texto en trozos con un procedimiento que se puede hacer a mano:

  1. se empieza con las letras sueltas; cada palabra lleva pegado su espacio de delante;
  2. se cuentan, en todo el texto, las parejas de piezas que van seguidas dentro de una palabra;
  3. la pareja que más se repite se suelda en una pieza nueva, que entra en la caja;
  4. y vuelta al paso 2.

Esto lo hace con el Quijote, y enseña las primeras juntas y cómo queda troceada una palabra con
0, 10, 100, 1.000 y 10.000 juntas en la caja.

Supuestos (declarados): letras, no bytes —la máquina grande empieza por los bytes, y una letra con
tilde es ahí dos piezas al principio—; los saltos de línea cuentan como espacios; el texto se
parte antes en palabras con su espacio delante, números y signos aparte, como hace la máquina
grande; si dos parejas empatan, gana la primera por orden alfabético; se respetan mayúsculas y
tildes.

Uso:
    python trocear_a_mano.py --selftest
    python trocear_a_mano.py > ../datos/salidas/trocear_a_mano.txt
"""

# ======================= CONSTANTES =======================

JUNTAS = 10_000                       # cuántas piezas nuevas se sueldan
PRIMERAS = 12                         # cuántas juntas se enseñan una a una
MOMENTOS = [0, 10, 100, 1_000, 10_000]
PALABRAS = [" murciélago", " Francia"]   # con su espacio delante (" caballero" salió el 4 oct.: no cabía en la página)
SALIDA_CSV = "trocear_a_mano.csv"
SEMILLA = 20261004

# ==========================================================

import argparse
import csv
import heapq
import random
import re
import sys
from collections import Counter, defaultdict

from formato import tabla_editorial, trozo
from ngrama import cargar_corpus, CORPUS

PARTIR = re.compile(r" ?[^\W\d_]+| ?\d+| ?[^\s\w]+")


def a_la_vista(t):
    return t.replace(" ", "_")


def palabras(texto):
    """Las palabras del texto, cada una con su espacio de delante, contadas."""
    texto = re.sub(r"\s+", " ", texto)
    trozos = PARTIR.findall(texto)
    assert trozos, "Se esperaba algún texto que partir; no hay nada"
    assert "".join(trozos).replace(" ", "") == texto.replace(" ", ""), \
        "Se esperaba que las palabras, juntas, devolvieran el texto entero"
    return Counter(trozos)


def aprender(cuenta, juntas):
    """Las juntas, en orden: [(a, b, veces)]. Cuenta las parejas una vez y después solo actualiza
    las palabras que cambian."""
    tipos = [list(w) for w in cuenta]
    peso = [cuenta[w] for w in cuenta]
    parejas = Counter()
    donde = defaultdict(set)
    for i, s in enumerate(tipos):
        for p in zip(s, s[1:]):
            parejas[p] += peso[i]
            donde[p].add(i)
    monton = [(-n, p) for p, n in parejas.items()]
    heapq.heapify(monton)
    fuera = []
    while len(fuera) < juntas and monton:
        n, p = heapq.heappop(monton)
        if parejas.get(p, 0) != -n or n == 0:
            continue                                   # entrada vieja del montón
        a, b = p
        fuera.append((a, b, -n))
        cambiadas = set()
        for i in list(donde[p]):
            s = tipos[i]
            for q in zip(s, s[1:]):
                parejas[q] -= peso[i]
                donde[q].discard(i)
                cambiadas.add(q)
            nuevo, k = [], 0
            while k < len(s):
                if k + 1 < len(s) and s[k] == a and s[k + 1] == b:
                    nuevo.append(a + b); k += 2
                else:
                    nuevo.append(s[k]); k += 1
            tipos[i] = nuevo
            for q in zip(nuevo, nuevo[1:]):
                parejas[q] += peso[i]
                donde[q].add(i)
                cambiadas.add(q)
        for q in cambiadas:
            if parejas[q] > 0:
                heapq.heappush(monton, (-parejas[q], q))
        parejas.pop(p, None)
    return fuera


def trocear(palabra, juntas):
    """Trocea una palabra con las juntas que haya en la caja, en su orden."""
    rango = {(a, b): k for k, (a, b, _) in enumerate(juntas)}
    s = list(palabra)
    while len(s) > 1:
        cand = [(rango[p], i) for i, p in enumerate(zip(s, s[1:])) if p in rango]
        if not cand:
            break
        r, _ = min(cand)
        a, b, _ = juntas[r]
        nuevo, k = [], 0
        while k < len(s):
            if k + 1 < len(s) and s[k] == a and s[k + 1] == b:
                nuevo.append(a + b); k += 2
            else:
                nuevo.append(s[k]); k += 1
        s = nuevo
    return s


def miles(n):
    return f"{n:,}".replace(",", ".")


def tabla(juntas, letras):
    """Las cuatro tablas editoriales (L24, 9 de octubre; regla 6 ter). Las cuentas no cambian."""
    ver = lambda t: trozo(a_la_vista(t))
    out = tabla_editorial(
        "Las primeras juntas, con el Quijote", ["junta", "pareja", "pieza nueva", "veces"],
        [[str(k), f"{ver(a)} y {ver(b)}", ver(a + b), miles(n)]
         for k, (a, b, n) in enumerate(juntas[:PRIMERAS], 1)], "ciid",
        ["El guion bajo es un espacio. Cada junta suelda la pareja que más se repite."])
    total = sum(n for _, n in letras)
    out += [""] + tabla_editorial(
        "Las piezas de una letra con que se empieza", ["de qué clase", "piezas"],
        [[q, str(n)] for q, n in letras] + [["**en total**", f"**{total}**"]], "id",
        [f"Con {miles(JUNTAS)} juntas, la caja tiene {miles(total + JUNTAS)} piezas."])
    todas = juntas
    juntas = juntas[:JUNTAS]
    filas = []
    for w in PALABRAS:
        for k, m in enumerate(MOMENTOS):
            filas.append([ver(w) if k == 0 else "", miles(m),
                          " ".join(ver(t) for t in trocear(w, juntas[:m]))])
    out += [""] + tabla_editorial(
        "Una palabra, según cuántas juntas hay en la caja",
        ["la palabra", "juntas en la caja", "cómo queda cortada"], filas, "idi",
        ["Cada trozo va aparte, en letra de máquina."])
    out += [""] + tabla_editorial(
        f"«{a_la_vista(PALABRAS[0])}», de {miles(JUNTAS)} juntas en adelante",
        ["juntas en la caja", "cómo queda cortada"],
        [[miles(m), " ".join(ver(t) for t in s)] for m, s in cambios(PALABRAS[0], todas, JUNTAS)],
        "di",
        [f"Cada fila, el número de juntas en el que la palabra pierde un corte. Con "
         f"{miles(len(todas))} juntas, cada palabra del Quijote es ya una sola pieza, y no queda "
         "ninguna pareja que soldar."])
    return out


def cambios(w, juntas, desde):
    """Cada número de juntas, de `desde` en adelante, en el que la palabra pierde un corte."""
    fuera = [(desde, trocear(w, juntas[:desde]))]
    while len(fuera[-1][1]) > 1:
        objetivo = len(fuera[-1][1]) - 1
        if len(trocear(w, juntas)) > objetivo:
            break
        lo, hi = fuera[-1][0], len(juntas)
        while lo < hi:
            m = (lo + hi) // 2
            if len(trocear(w, juntas[:m])) <= objetivo:
                hi = m
            else:
                lo = m + 1
        fuera.append((lo, trocear(w, juntas[:lo])))
    return fuera


def composicion(cuenta):
    """Las piezas de una letra con que se empieza, por clases."""
    todas = {ch for w in cuenta for ch in w}
    clases = [("el espacio", lambda c: c == " "),
              ("minúsculas", lambda c: "a" <= c <= "z"),
              ("mayúsculas", lambda c: "A" <= c <= "Z"),
              ("letras con tilde, diéresis o eñe", lambda c: c.isalpha() and not c.isascii()),
              ("cifras", lambda c: c.isdigit()),
              ("signos", lambda c: True)]
    fuera, quedan = [], set(todas)
    for nombre, es in clases:
        estas = {c for c in quedan if es(c)}
        quedan -= estas
        fuera.append((nombre, len(estas)))
    assert not quedan and sum(n for _, n in fuera) == len(todas), "una letra se quedó sin clase"
    return fuera


def guardar(juntas, ruta):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["junta", "a", "b", "veces"])
        for k, (a, b, n) in enumerate(juntas, 1):
            w.writerow([k, a, b, n])


def leer(ruta):
    with open(ruta, encoding="utf-8", newline="") as f:
        return [(r["a"], r["b"], int(r["veces"])) for r in csv.DictReader(f)]


# ============================ SELFTEST ============================

def selftest():
    fallos = []
    texto = cargar_corpus(CORPUS)
    cuenta = palabras(texto)

    # [1] Test nulo: palabras de letras al azar. Ninguna pareja de letras puede destacar: la que
    #     más se repite no llega a vez y media la media de todas.
    rng = random.Random(SEMILLA)
    abc = "abcdefghijklmnopqrst"
    azar = " ".join("".join(rng.choice(abc) for _ in range(rng.randint(3, 8))) for _ in range(60_000))
    c = Counter()
    for w, n in palabras(azar).items():
        for p in zip(w, w[1:]):
            if " " not in p:
                c[p] += n
    razon = max(c.values()) / (sum(c.values()) / len(c))
    print(f"[1] test nulo         letras al azar: la pareja que más se repite, "
          f"{razon:.2f} veces la media (tiene que quedar por debajo de 1,5)")
    if razon >= 1.5:
        fallos.append("test nulo: con letras al azar hay una pareja que destaca")

    # [2] Señal implantada: una palabra inventada, repetida doscientas mil veces, sale soldada
    #     entera entre las primeras juntas.
    con = cuenta.copy()
    con[" zxqvk"] += 200_000
    j = aprender(con, 15)
    entera = trocear(" zxqvk", j) == [" zxqvk"]
    print(f"[2] señal implantada  «_zxqvk» doscientas mil veces: entera con 15 juntas: {entera}")
    if not entera:
        fallos.append("señal: la palabra implantada no sale soldada entre las primeras juntas")

    # [3] Invariante: pegar los trozos devuelve la palabra exacta, y con más juntas en la caja una
    #     palabra nunca tiene más cortes que con menos.
    juntas = aprender(cuenta, 2_000)
    muestra = rng.sample(sorted(cuenta), 2_000) + PALABRAS
    exacta = all("".join(trocear(w, juntas)) == w for w in muestra)
    mono = all(len(trocear(w, juntas[:a])) >= len(trocear(w, juntas[:b]))
               for w in PALABRAS for a, b in zip([0, 10, 100, 1000], [10, 100, 1000, 2000]))
    print(f"[3] invariante        los trozos pegados dan la palabra: {exacta}; "
          f"más juntas, nunca más cortes: {mono}")
    if not (exacta and mono):
        fallos.append("invariante: trocear rompe la palabra o le añade cortes")
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
    cuenta = palabras(cargar_corpus(CORPUS))
    letras = composicion(cuenta)
    juntas = aprender(cuenta, 10 ** 9)            # hasta que no quede ninguna pareja que soldar
    assert len(juntas) > JUNTAS, f"Se esperaban más de {JUNTAS} juntas; salieron {len(juntas)}"
    assert all(len(trocear(w, juntas)) == 1 for w in list(cuenta)[:3000]), \
        "Se esperaba que, sin parejas que soldar, cada palabra fuera una sola pieza"
    for l in tabla(juntas, letras):
        print(l)
    guardar(juntas, SALIDA_CSV)
    return 0


if __name__ == "__main__":
    sys.exit(main())
