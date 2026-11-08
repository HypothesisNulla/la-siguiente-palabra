#!/usr/bin/env python3
"""Capítulo 1 — cuánto crece la libreta con cada letra o palabra más de contexto, y dos parejas de palabras vistas por dentro.

Dos cosas que el capítulo 1 reescrito necesita y que ninguna salida guardada daba (Orden 138, 8 nov 2026;
notas de Carlos de las p. 21 y 23-27; guion del capítulo 1, puntos 5 y 7):

  A. «¿Por qué esta vía se acabó?» Una sola progresión: cuántas hojas tiene la libreta con el selector
     en 1, 2, 3, 4 y 5 letras y en 1 y 2 palabras, y cuántas de ellas tienen algo apuntado en el
     Quijote. Una hoja por cada contexto que podría existir (posibles) y una por cada contexto que
     de verdad sale en el libro con algo detrás (con algo apuntado).
  B. Dos ejemplos que sustituyen a la Tabla 7 (la de «veces / pueden venir / la que más sale»):
     una pareja de palabras que sale una sola vez —«sitio escogió»— y lo que va detrás, y la pareja
     más frecuente del Quijote, con cuántas palabras distintas van detrás y las cinco que más.
     «La más frecuente» se dice de dos maneras, porque la máquina cuenta los signos como palabras:
     la más frecuente de todas (puede llevar un signo) y las tres más frecuentes hechas de dos palabras.

Mismo Quijote, misma limpieza y mismo corte en trozos que `ngrama.py`, `casillas_vacias.py` y
`traza_escalera.py`, para que todos hablen del mismo texto. No usa el azar.

Lo que NO se mide aquí porque ya está medido en otra salida (y se comprueba que coincide, en el selftest):
las hojas con algo apuntado de 1, 2, 3 y 5 letras (`traza_escalera.txt`) y las de 2 palabras
(`casillas_vacias.txt`). Lo nuevo es el 4 letras, el 1 palabra, el 42 elevado a cada selector (las hojas
posibles de las letras) y las parejas de B.

POR QUÉ ESTA MEDICIÓN NO DEPENDE DE LA MÁQUINA. Se cuentan letras y palabras de un fichero de texto; sale lo
mismo en cualquier ordenador.

Uso:
    python la_libreta_que_crece.py --selftest
    python la_libreta_que_crece.py > ../datos/salidas/la_libreta_que_crece.txt
"""

# ======================= CONSTANTES =======================

CORPUS = "../datos/quijote.txt"
SALIDA_ESCALERA = "../datos/salidas/traza_escalera.txt"     # hojas con algo apuntado de 1, 2, 3 y 5 letras
SALIDA_CASILLAS = "../datos/salidas/casillas_vacias.txt"    # hojas posibles y con algo apuntado de 2 palabras

SELECTORES_LETRAS = [1, 2, 3, 4, 5]
SELECTORES_PALABRAS = [1, 2]

PAREJA_UNA_VEZ = ("sitio", "escogió")   # la pareja que sale una sola vez (Tabla 7 de hoy)
PAREJAS_DE_PALABRAS = 3                 # cuántas de las más frecuentes hechas de dos palabras se ven por dentro
DETRAS_QUE_MAS = 5                      # cuántas de las que van detrás se enseñan

# Selftest
TROZO_NULO = 400_000                    # letras del corpus del test nulo
MINI = "a b a b a c"                    # texto de seis trozos, contado con el dedo
MINI_IMPLANTADO = "x y z " * 7          # una pareja que siempre lleva lo mismo detrás
# ==========================================================

import platform
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import ngrama as N
from formato import ANCHO_CAJA, coma, comprobar_ancho, miles, pct, tabla_editorial

AQUI = Path(__file__).resolve().parent


# ---- cada cuenta en una función que se puede probar sola ---------------------------

def hojas_posibles(simbolos, selector):
    """Una hoja por cada contexto que podría existir: los símbolos distintos elevados al selector."""
    return simbolos ** selector


def hojas_con_algo(secuencia, selector):
    """Contextos de `selector` símbolos que salen en la secuencia con algo detrás (los que `ngrama.construir`
    convierte en una hoja). Vale para letras (una cadena) y para trozos (una lista)."""
    return len({tuple(secuencia[i:i + selector]) for i in range(len(secuencia) - selector)})


def detras(trozos, pareja):
    """Qué trozo va detrás de cada aparición de la pareja, y cuántas veces cada uno."""
    n = len(pareja)
    return Counter(trozos[i + n] for i in range(len(trozos) - n) if tuple(trozos[i:i + n]) == tuple(pareja))


def parejas_mas_frecuentes(trozos, solo_palabras=False):
    """Las parejas de trozos seguidos, de más a menos frecuente; con empate, la que sale antes en el libro.
    Con `solo_palabras`, ninguno de los dos trozos es un signo de puntuación."""
    signos = set(N.SIGNOS)
    cuenta, primera = Counter(), {}
    for i, p in enumerate(zip(trozos, trozos[1:])):
        if solo_palabras and (p[0] in signos or p[1] in signos):
            continue
        cuenta[p] += 1
        primera.setdefault(p, i)
    return sorted(cuenta.items(), key=lambda kv: (-kv[1], primera[kv[0]]))


def una_de_cada(posibles, con_algo):
    """«Una hoja con algo apuntado de cada…»: de cada cuántas hojas posibles tiene alguna algo."""
    x = posibles / con_algo
    return coma(x, 1) if x < 10 else miles(round(x))


def cuantas(n, singular, plural):
    return f"{miles(n)} {singular if n == 1 else plural}"


def nombre_selector(orden, unidad):
    return f"{orden} {unidad}" + ("s" if orden != 1 else "")


def tabla_progresion(secuencia, trozos):
    """Las dos filas de la progresión: letras y palabras. Devuelve (filas, datos) con los números sin formatear."""
    filas, datos = [], {}
    simbolos = len(N.ALFABETO)
    previo = None
    for o in SELECTORES_LETRAS:
        p, v = hojas_posibles(simbolos, o), hojas_con_algo(secuencia, o)
        datos[("letras", o)] = (p, v)
        filas.append([nombre_selector(o, "letra"), miles(p), miles(v), una_de_cada(p, v),
                      "—" if previo is None else f"{miles(p // previo)}"])
        previo = p
    previo = None
    vocabulario = len(set(trozos))
    for o in SELECTORES_PALABRAS:
        p, v = hojas_posibles(vocabulario, o), hojas_con_algo(trozos, o)
        datos[("palabras", o)] = (p, v)
        filas.append([nombre_selector(o, "palabra"), miles(p), miles(v), una_de_cada(p, v),
                      "—" if previo is None else f"{miles(p // previo)}"])
        previo = p
    return filas, datos


def tabla_detras(trozos, pareja, titulo, notas_extra=()):
    """Una tabla con lo que va detrás de una pareja: las cinco más frecuentes y cuántas distintas hay."""
    c = detras(trozos, pareja)
    total = sum(c.values())
    arriba = c.most_common(DETRAS_QUE_MAS)
    filas = [[f"«{t}»", miles(n), pct(n / total, 1)] for t, n in arriba]
    notas = [f"«{' '.join(pareja)}» sale {cuantas(total, 'vez', 'veces')} en el Quijote y detrás de ella "
             f"{'va' if len(c) == 1 else 'van'} {cuantas(len(c), 'palabra distinta', 'palabras distintas')}."]
    if len(c) > DETRAS_QUE_MAS:
        resto = total - sum(n for _, n in arriba)
        notas.append(f"Las otras {miles(len(c) - DETRAS_QUE_MAS)} suman {pct(resto / total, 1)}.")
    notas += list(notas_extra)
    return tabla_editorial(titulo, ["detrás va", "veces", "de cada cien veces"], filas, "idd", notas)


def leer_numero(ruta, patron):
    """El primer número (con puntos de millar) que sigue a `patron` en la salida guardada, como entero."""
    texto = Path(ruta).read_text(encoding="utf-8")
    m = re.search(patron + r"\s*\|?\s*([0-9][0-9.]*)", texto)
    assert m, f"Se esperaba encontrar «{patron}» seguido de un número en {ruta}; no está"
    return int(m.group(1).replace(".", ""))


# ---- selftest ---------------------------------------------------------------------

def selftest(secuencia, trozos):
    fallos = []

    # 1. TEST NULO — con las letras barajadas la libreta se llena mucho más que con el castellano: en el texto
    #    real mirar tres letras deja casi todas las hojas vacías. Y una pareja de palabras que sale en el libro no
    #    sale en las palabras barajadas.
    real = secuencia[:TROZO_NULO]
    baraja = list(real)
    random.Random(N.SEMILLA).shuffle(baraja)
    v_real, v_baraja = hojas_con_algo(real, 3), hojas_con_algo("".join(baraja), 3)
    print(f"[1] test nulo         hojas con algo mirando 3 letras: barajado={miles(v_baraja)}  real={miles(v_real)}")
    if not v_baraja > 2 * v_real:
        fallos.append(f"test nulo: se esperaba barajado > 2 × real; salió {v_baraja} y {v_real}")
    palabras_baraja = list(trozos[:200_000])
    random.Random(N.SEMILLA).shuffle(palabras_baraja)
    sale_baraja = sum(detras(palabras_baraja, ("el", "caballero")).values())
    sale_el_caballero = sum(detras(trozos[:200_000], ("el", "caballero")).values())
    print(f"                      «el caballero» en 200.000 trozos: barajado={sale_baraja}  real={sale_el_caballero}")
    if not sale_el_caballero > 5 * max(1, sale_baraja):     # al barajar quedan por azar unas pocas, no tantas
        fallos.append(f"test nulo: «el caballero» debería salir mucho más en el texto real; "
                      f"salió {sale_el_caballero} y {sale_baraja} al barajar")

    # 2. SEÑAL IMPLANTADA — un texto diminuto contado con el dedo. «a b a b a c»: vocabulario de 3 → 3 y 9 hojas
    #    posibles; con dos trozos de contexto, las parejas con algo detrás son «a b» y «b a» (2); detrás de «a b»
    #    va «a» las dos veces; detrás de «b a», «b» y «c». Y «x y z» repetido 7 veces: detrás de «x y» va siempre «z».
    mini = MINI.split()
    esperado = (hojas_posibles(len(set(mini)), 1), hojas_posibles(len(set(mini)), 2), hojas_con_algo(mini, 2),
                dict(detras(mini, ("a", "b"))), dict(detras(mini, ("b", "a"))),
                dict(detras(MINI_IMPLANTADO.split(), ("x", "y"))))
    deseado = (3, 9, 2, {"a": 2}, {"b": 1, "c": 1}, {"z": 7})
    print(f"[2] señal implantada  {esperado}")
    if esperado != deseado:
        fallos.append(f"señal implantada: se esperaba {deseado}; salió {esperado}")
    orden = [p for p, _ in parejas_mas_frecuentes(("a b a b a c".split()))][:2]
    if orden != [("a", "b"), ("b", "a")]:
        fallos.append(f"señal implantada: se esperaba (a, b) y (b, a) como las más frecuentes; salió {orden}")
    con_signo = parejas_mas_frecuentes([",", "y", ",", "y", "don", "quijote"], solo_palabras=True)
    if con_signo != [(("y", "don"), 1), (("don", "quijote"), 1)]:      # las tres parejas con la coma no cuentan
        fallos.append(f"señal implantada: sin signos solo debían quedar «y don» y «don quijote»; salió {con_signo}")

    # 3. INVARIANTES — lo que ya estaba medido en otra salida tiene que dar lo mismo aquí; las hojas con algo
    #    nunca pasan de las posibles; y lo que va detrás suma las veces que sale la pareja.
    letras_traza = {1: 41, 2: 808, 3: 6896, 5: 103082}
    letras_aqui = {o: hojas_con_algo(secuencia, o) for o in letras_traza}
    traza_ok = all(leer_numero(AQUI / SALIDA_ESCALERA, rf"\|\s*{o} letras?\s*\|") == letras_aqui[o] == letras_traza[o]
                   for o in letras_traza)
    pos2 = leer_numero(AQUI / SALIDA_CASILLAS, re.escape("hojas posibles (trozos distintos al cuadrado)"))
    vis2 = leer_numero(AQUI / SALIDA_CASILLAS, re.escape("hojas con algo apuntado (parejas vistas)"))
    casillas_ok = (pos2, vis2) == (hojas_posibles(len(set(trozos)), 2), hojas_con_algo(trozos, 2))
    print(f"[3] invariante        hojas con algo de 1, 2, 3 y 5 letras = las de traza_escalera.txt: {traza_ok}; "
          f"hojas posibles y con algo de 2 palabras = las de casillas_vacias.txt: {casillas_ok}")
    if not traza_ok:
        fallos.append(f"invariante: las hojas con algo de 1, 2, 3 y 5 letras ({letras_aqui}) no coinciden con "
                      f"{SALIDA_ESCALERA} ({letras_traza})")
    if not casillas_ok:
        fallos.append(f"invariante: las hojas de 2 palabras no coinciden con {SALIDA_CASILLAS}: "
                      f"{SALIDA_CASILLAS} dice {pos2} y {vis2}")
    sobra = [(k, p, v) for k, (p, v) in tabla_progresion(secuencia, trozos)[1].items() if v > p]
    c = detras(trozos, PAREJA_UNA_VEZ)
    veces_pareja = sum(1 for i in range(len(trozos) - 1) if (trozos[i], trozos[i + 1]) == PAREJA_UNA_VEZ)
    # la pareja sale veces_pareja veces; detrás de cada una va algo salvo si es el final del texto
    print(f"                      hojas con algo que pasen de las posibles: {len(sobra)}; «{' '.join(PAREJA_UNA_VEZ)}» sale "
          f"{veces_pareja} vez y detrás va {dict(c)}")
    if sobra:
        fallos.append(f"invariante: hay hojas con algo por encima de las posibles: {sobra}")
    if veces_pareja != 1 or sum(c.values()) != 1:
        fallos.append(f"invariante: «{' '.join(PAREJA_UNA_VEZ)}» debía salir una vez con una palabra detrás; "
                      f"sale {veces_pareja} y detrás va {dict(c)}")
    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


# ---- la salida ---------------------------------------------------------------------

def main():
    secuencia = N.normalizar(N.cargar_corpus(str(AQUI / CORPUS)), N.ALFABETO)
    trozos = N.trocear(secuencia)

    print("--- selftest ---")
    codigo = selftest(secuencia, trozos)
    if codigo or "--selftest" in sys.argv:
        sys.exit(codigo)

    L = ["", "########## capítulo 1: lo que crece la libreta ##########",
         f"fecha: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
         f"máquina: {platform.system()} {platform.machine()}. La medición no",
         "depende de la máquina: cuenta letras y palabras de un fichero",
         "corpus: datos/quijote.txt, limpiado y troceado como en ngrama.py",
         f"símbolos del alfabeto de las letras: {len(N.ALFABETO)} (letras, espacio y signos)",
         f"trozos distintos (palabras y {len(N.SIGNOS)} signos): {miles(len(set(trozos)))}"]
    E = ["", "A. LA LIBRETA, SEGÚN CUÁNTO CONTEXTO SE MIRE", ""]
    filas, datos = tabla_progresion(secuencia, trozos)
    E += tabla_editorial(
        "Hojas de la libreta según el selector",
        ["selector", "hojas posibles", "con algo apuntado", "una con algo de cada", "se multiplica por"],
        filas, "iddd" + "d",
        ["Posibles: una hoja por cada contexto que podría existir. Con algo apuntado: las que de verdad salen "
         "en el Quijote con algo detrás.",
         "Una con algo de cada: de cada cuántas hojas posibles, una tiene algo apuntado.",
         "Se multiplica por: lo que crece el número de hojas posibles al añadir una letra, o una palabra, al contexto."])

    # La tabla de las letras tal como la pone el anexo A1 del capítulo 1 (Orden 140.3: el programa sirve al libro):
    # mismos título y rótulos, solo letras.
    E += [""] + tabla_editorial(
        "Las hojas de la libreta de letras, según el selector",
        ["selector", "hojas posibles", "hojas con algo en el Quijote"],
        [[nombre_selector(o, "letra"), miles(datos[("letras", o)][0]), miles(datos[("letras", o)][1])]
         for o in SELECTORES_LETRAS], "idd")

    E += ["", "B. DOS PAREJAS DE PALABRAS, POR DENTRO", "",
          "Como en el resto del capítulo, los signos de puntuación cuentan como palabras.", ""]
    E += tabla_detras(trozos, PAREJA_UNA_VEZ, f"Lo que va detrás de «{' '.join(PAREJA_UNA_VEZ)}»") + [""]
    todas = parejas_mas_frecuentes(trozos)
    solo = parejas_mas_frecuentes(trozos, solo_palabras=True)[:PAREJAS_DE_PALABRAS]
    mejor, veces_mejor = todas[0]
    E += tabla_detras(trozos, mejor, f"La pareja más frecuente de todas: «{' '.join(mejor)}»",
                      ["Es una pareja con una coma: la más frecuente hecha solo de palabras es otra."]
                      if set(mejor) & set(N.SIGNOS) else ()) + [""]
    for pareja, _ in solo:
        E += tabla_detras(trozos, pareja, f"Una de las tres parejas de dos palabras más frecuentes: "
                                           f"«{' '.join(pareja)}»") + [""]
    E.append("Las parejas más frecuentes, de mayor a menor (sin contar signos de puntuación): " + "; ".join(
        f"«{' '.join(p)}» {miles(n)}" for p, n in parejas_mas_frecuentes(trozos, solo_palabras=True)[:8]) + ".")

    for l in comprobar_ancho([l.rstrip() for l in "\n".join(L).split("\n")], ANCHO_CAJA):
        print(l)
    for l in E:
        print(l)


if __name__ == "__main__":
    main()
