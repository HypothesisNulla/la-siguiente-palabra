#!/usr/bin/env python3
"""
Capítulo 4 — la red del colegio con la edad en años, tal cual: por qué se atascan y si tiene
arreglo (Orden 19, 18 oct 2026). Solo mide.

Con los ajustes de la Orden 15 (`el_colegio.py`: poquito 0,5, 20.000 pasos, pesos de arranque entre
−1 y 1, los mismos 200 arranques), con la edad en años solo la resuelven 54 de 200 redes. La
explicación que propuso Claude, sin medir: lo que le llega a cada neurona es el peso por un número
de 3 a 15, sale enseguida de la franja en que la rampa reacciona, la neurona se queda pegada a 0 o
a 1, y ahí la culpa es casi cero; además el peso se corrige en proporción a la edad y el listón
en proporción a 1, así que el peso corre mucho más que el listón. Este programa lo comprueba y
mira si, sin tocar la entrada, hay unos ajustes con los que lo aprendan casi todas:

  19.2  Por qué se atascan: neuronas pegadas (dicen menos de 0,01 o más de 0,99 a las 13 edades),
        qué edades fallan, y lo que se mueve el peso de la edad frente al listón en los primeros
        500 pasos; para las 146 que no lo resuelven en años, las 54 que sí y, como comparación,
        las 29 que no lo resuelven con la edad en decenas.
  19.3  Dieciséis combinaciones: poquito 0,5 / 0,1 / 0,05 / 0,01, pasos 20.000 / 100.000, pesos de
        arranque los de ahora o divididos entre 10: cuántas resuelven y cuántas acaban con alguna
        neurona pegada.
  19.4  Para la mejor combinación, lo que dio la Orden 15 en su apartado b): cuántas resuelven, las
        parejas, la red del capítulo, y qué edades falla esa red cada 500 pasos.

Reutiliza, importándolos y sin cambiarlos, `Red` de `retropropagacion.py`, los datos y las funciones
del colegio de `el_colegio.py` y el sorteo de arranques de `que_inventan_las_capas.py`. Las
semillas son las mismas 200 de la Orden 15.

Uso:
    python el_colegio_en_anos.py --selftest
    python el_colegio_en_anos.py > ../datos/salidas/el_colegio_en_anos.txt
"""

# ======================= CONSTANTES =======================

PEGADA_BAJO, PEGADA_ALTO = 0.01, 0.99     # una neurona pegada dice menos de 0,01 o más de 0,99 a las 13 edades
PASOS_DIAGNOSTICO = 500                   # los primeros pasos, para lo que se mueven pesos y listones
TASAS = [0.5, 0.1, 0.05, 0.01]            # el poquito: 0,5 es el de ahora
PASOS_LISTA = [20_000, 100_000]           # 20.000 es el de ahora
DIVISORES = [1.0, 10.0]                   # pesos de arranque: los de ahora, o divididos entre 10
CADA_CUANTOS = 500                        # cada cuántos pasos se mira qué edades falla la red del capítulo
MAX_PATRONES = 8                          # patrones de fallo que se enseñan
ANCHO_SALIDA = 62                         # líneas de no más de 62 caracteres, fuera de las tablas
EN_LETRA = {2: "dos"}                     # la nota de «Lo que acaban haciendo…» lo dice con letra (Orden 50)
# Los ajustes «de ahora» (poquito 0,5, 20.000 pasos, 200 arranques, semilla base 20260916, umbral 0,5)
# son los de `que_inventan_las_capas.py` y `el_colegio.py`, y se importan de allí.

# ==========================================================

import argparse
import platform
import statistics
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import date

import numpy as np

import el_colegio as C
from formato import comprobar_ancho, miles, tabla_editorial
from retropropagacion import Red

TASA_AHORA, PASOS_AHORA = C.TASA, C.PASOS
EDADES, DEBERIA = C.EDADES, C.DEBERIA
coma, c3, firmado, NEURONAS = C.coma, C.c3, C.firmado, C.NEURONAS


# ---------------------------------------------------------------- lo básico

def red_nueva(semilla, divisor=1.0):
    """La red 1-2-1 del colegio con los pesos de arranque de `Red` divididos entre `divisor` (los
    listones empiezan en cero, como siempre)."""
    red = Red([1, 2, 1], semilla=int(semilla))
    if divisor != 1.0:
        red.W = [w / divisor for w in red.W]
    return red


def paso(red, X, tasa):
    """Un paso de aprendizaje, exactamente el de `Red.entrenar` pero con el poquito que se pida."""
    gW, gb = red.gradiente_retro(X, DEBERIA)
    for k in range(len(red.W)):
        red.W[k] -= tasa * gW[k]
        red.b[k] -= tasa * gb[k]


def neuronas_pegadas(red, X):
    """Cuántas de las dos neuronas de en medio dicen lo mismo a las 13 edades: todas menos de
    PEGADA_BAJO, o todas más de PEGADA_ALTO."""
    medio = red.adelante(X)[1]
    return int(sum(bool((medio[:, j] < PEGADA_BAJO).all() or (medio[:, j] > PEGADA_ALTO).all())
                   for j in range(2)))


def edades_mal(red, X):
    salida = red.adelante(X)[-1].ravel()
    return [int(e) for e, s, d in zip(EDADES, salida, DEBERIA) if (s > C.UMBRAL) != (d > C.UMBRAL)]


def pareja_de(red, X):
    """La pareja de lo que hace cada neurona de en medio, ordenada como en `el_colegio.py`."""
    medio = red.adelante(X)[1]
    return tuple(sorted(C.sentido(medio[:, j]) for j in range(2)))


def foto(red, X):
    mal = edades_mal(red, X)
    return dict(ok=not mal, mal=mal, pegadas=neuronas_pegadas(red, X),
                pareja=None if mal else tuple(t for _, t in pareja_de(red, X)))


# ---------------------------------------------------------------- los trabajos (uno por arranque)

def trabajo_barrido(args):
    """Una red de las 200 de una combinación: se entrena hasta el máximo de PASOS_LISTA y se mira
    en cada uno de ellos (los 20.000 pasos son el comienzo de los 100.000: el mismo recorrido)."""
    semilla, tasa, divisor = args
    X = C.entrada(1.0)
    red = red_nueva(semilla, divisor)
    fotos = {}
    for n in range(1, max(PASOS_LISTA) + 1):
        paso(red, X, tasa)
        if n in PASOS_LISTA:
            fotos[n] = foto(red, X)
    return fotos


def trabajo_diagnostico(args):
    """Con los ajustes de la Orden 15, una red: pegadas al principio, a los 500 pasos y al final; lo
    que ha movido cada peso de la edad y cada listón en los primeros 500 pasos; y cómo acaba."""
    semilla, escala = args
    X = C.entrada(escala)
    red = red_nueva(semilla)
    w0, b0 = red.W[0][0].copy(), red.b[0].copy()
    r = {"pegadas0": neuronas_pegadas(red, X)}
    for n in range(1, PASOS_AHORA + 1):
        paso(red, X, TASA_AHORA)
        if n == PASOS_DIAGNOSTICO:
            r["pegadas500"] = neuronas_pegadas(red, X)
            dw, db = np.abs(red.W[0][0] - w0), np.abs(red.b[0] - b0)
            r["mov_peso"], r["mov_liston"] = dw.tolist(), db.tolist()
    r.update(foto(red, X))
    return r


def correr(trabajo, lista):
    with ProcessPoolExecutor() as pool:
        return list(pool.map(trabajo, lista, chunksize=5))


# ---------------------------------------------------------------- 19.2

def tabla_pegadas(grupos, emitir):
    """grupos: [(rótulo, lista de diagnósticos)]."""
    filas = []
    for k in (0, 1, 2):
        filas.append([f"{k} neuronas pegadas"] + [str(sum(1 for d in g if d["pegadas"] == k))
                                                  for _, g in grupos])
    filas.append(["**redes**"] + [f"**{len(g)}**" for _, g in grupos])
    emitir(tabla_editorial(
        "Neuronas de en medio pegadas, al acabar los 20.000 pasos",
        [""] + [r for r, _ in grupos], filas, "i" + "d" * len(grupos),
        ["Pegada: dice menos de 0,01 o más de 0,99 a las trece edades (dice lo mismo a todo el "
         "mundo).",
         "Los ajustes son los de la Orden 15: poquito 0,5 y 20.000 pasos."]))
    emitir([""])


def tabla_fallos(grupos, emitir):
    """Los patrones de fallo más frecuentes de las redes que no lo resuelven."""
    for rotulo, g in grupos:
        cuenta = Counter(tuple(d["mal"]) for d in g)
        filas = [[", ".join(map(str, p)), str(n)] for p, n in cuenta.most_common(MAX_PATRONES)]
        resto = len(g) - sum(n for _, n in cuenta.most_common(MAX_PATRONES))
        if resto:
            filas.append([f"otros {len(cuenta) - min(len(cuenta), MAX_PATRONES)} patrones", str(resto)])
        filas.append(["**redes que no lo resuelven**", f"**{len(g)}**"])
        emitir(tabla_editorial(
            f"Qué edades fallan las que no lo resuelven, {rotulo}", ["edades que fallan", "redes"],
            filas, "id", ["Fallar: dice sí a quien no entra, o no a quien entra (lado equivocado de "
                          "0,5)."]))
        emitir([""])


def mediana_cociente(diag):
    """La mediana, entre las neuronas de en medio de todas las redes, de lo que se ha movido el peso
    de la edad dividido por lo que se ha movido el listón de su neurona, en los primeros pasos."""
    razones = []
    for d in diag:
        for dw, db in zip(d["mov_peso"], d["mov_liston"]):
            assert db > 0, "El listón no se ha movido: no se puede dividir"
            razones.append(dw / db)
    q = statistics.quantiles(razones, n=4)
    return statistics.median(razones), q[0], q[2], min(razones), max(razones), len(razones)


def tabla_movimiento(anios, decenas, emitir):
    filas = []
    for rotulo, diag in (("la edad en años", anios), ("la edad en decenas", decenas)):
        m, q1, q3, lo, hi, n = mediana_cociente(diag)
        filas.append([rotulo, coma(m, 2), f"{coma(q1, 2)} a {coma(q3, 2)}", f"{coma(lo, 2)} a "
                                                                              f"{coma(hi, 2)}", str(n)])
    emitir(tabla_editorial(
        f"Lo que se mueve el peso de la edad frente al listón, en los primeros {PASOS_DIAGNOSTICO} pasos",
        ["", "mediana", "entre el cuarto y el tres cuartos", "de la menor a la mayor", "neuronas"],
        filas, "idddd",
        ["Por neurona de en medio: lo que se ha movido el peso de la línea de la edad, dividido por "
         "lo que se ha movido el listón de esa neurona.",
         "Las 200 redes, con las dos neuronas de cada una; los ajustes son los de la Orden 15."]))
    emitir([""])
    return filas


def tabla_pegadas_en_el_tiempo(anios, decenas, emitir):
    filas = []
    for rotulo, diag in (("la edad en años", anios), ("la edad en decenas", decenas)):
        filas.append([rotulo] + [str(sum(d[k] for d in diag)) for k in ("pegadas0", "pegadas500",
                                                                         "pegadas")])
    emitir(tabla_editorial(
        "Neuronas de en medio pegadas, de las 400, a lo largo del entrenamiento",
        ["", "al empezar", f"a los {PASOS_DIAGNOSTICO} pasos", "a los 20.000"], filas, "iddd",
        ["Doscientas redes por dos neuronas: 400. Los ajustes son los de la Orden 15."]))
    emitir([""])


def bloque_192(emitir):
    semillas = C.sortear()
    anios = correr(trabajo_diagnostico, [(s, 1.0) for s in semillas])
    decenas = correr(trabajo_diagnostico, [(s, C.DECENAS) for s in semillas])
    emitir(["--- POR QUÉ SE ATASCAN ---", ""])
    si_a = [d for d in anios if d["ok"]]
    no_a = [d for d in anios if not d["ok"]]
    no_d = [d for d in decenas if not d["ok"]]
    emitir([f"Con la edad en años: {len(si_a)} lo resuelven y {len(no_a)} no.",
            f"Con la edad en decenas: {200 - len(no_d)} lo resuelven y {len(no_d)} no.", ""])
    tabla_pegadas([("años: no lo resuelven", no_a), ("años: sí", si_a),
                   ("decenas: no lo resuelven", no_d)], emitir)
    tabla_pegadas_en_el_tiempo(anios, decenas, emitir)
    tabla_fallos([("con la edad en años", no_a), ("con la edad en decenas", no_d)], emitir)
    filas = tabla_movimiento(anios, decenas, emitir)
    return dict(si_a=si_a, no_a=no_a, no_d=no_d, anios=anios, decenas=decenas, mov=filas)


# ---------------------------------------------------------------- 19.3

def bloque_193(emitir):
    semillas = C.sortear()
    combos = [(t, d) for t in TASAS for d in DIVISORES]
    res = {}
    for t, d in combos:
        res[(t, d)] = correr(trabajo_barrido, [(s, t, d) for s in semillas])
    filas, tabla = [], {}
    for t, d in combos:
        for n in PASOS_LISTA:
            fs = [r[n] for r in res[(t, d)]]
            ok = sum(f["ok"] for f in fs)
            pe = sum(1 for f in fs if f["pegadas"] >= 1)
            tabla[(t, n, d)] = (ok, pe)
    for t in TASAS:
        for n in PASOS_LISTA:
            for d in DIVISORES:
                ok, pe = tabla[(t, n, d)]
                filas.append([coma(t, 2), miles(n), "los de ahora" if d == 1.0 else "entre 10",
                              str(ok), str(pe)])
    emitir(["--- ¿TIENE ARREGLO SIN CAMBIAR LA ENTRADA? ---", ""])
    emitir(tabla_editorial(
        "Cuántas de las 200 redes resuelven el colegio, con la edad en años",
        ["poquito", "pasos", "pesos de arranque", "lo resuelven", "con alguna neurona pegada"],
        filas, "iiidd",
        ["Pesos de arranque: «los de ahora» son los de la Orden 15 (entre −1 y 1); «entre 10», esos "
         "mismos divididos entre 10.",
         "Con alguna neurona pegada: redes (resuelvan o no) con al menos una de las dos neuronas de "
         "en medio diciendo lo mismo a las trece edades, al acabar.",
         "La fila 0,50 con 20.000 pasos y «los de ahora» es la de la Orden 15."]))
    emitir([""])
    return res, tabla


def mejor_combinacion(tabla):
    """La que más resuelve; si empatan, la de menos pasos; y después la de poquito más grande (y,
    si aún empatan, la de pesos de ahora)."""
    return max(tabla, key=lambda k: (tabla[k][0], -k[1], k[0], -k[2]))


# ---------------------------------------------------------------- 19.4

def bloque_194(res, tabla, emitir):
    t, n, d = mejor_combinacion(tabla)
    semillas = C.sortear()
    X = C.entrada(1.0)
    fs = [r[n] for r in res[(t, d)]]
    emitir(["--- LA MEJOR COMBINACIÓN ---", ""])
    emitir(tabla_editorial(
        "La mejor combinación", ["", "cuál"],
        [["poquito", coma(t, 2)], ["pasos", miles(n)],
         ["pesos de arranque", "los de ahora" if d == 1.0 else "entre 10"],
         ["redes que lo resuelven, de 200", str(tabla[(t, n, d)][0])],
         ["con alguna neurona pegada", str(tabla[(t, n, d)][1])]], "id",
        ["La que más resuelve; si empatan, la de menos pasos, y después la de poquito más grande."]))
    emitir([""])
    cuenta, primera = Counter(), {}
    for k, f in enumerate(fs, 1):
        if f["ok"]:
            cuenta[f["pareja"]] += 1
            primera.setdefault(f["pareja"], k)
    resueltas = sum(cuenta.values())
    orden = sorted(cuenta, key=lambda c: (-cuenta[c], primera[c]))
    emitir(tabla_editorial(
        "Cuántas de las 200 redes resuelven el colegio",
        ["", "redes"],
        [["arranques distintos", "200"],
         ["lo resuelven (las 13 edades del lado correcto de 0,5)", str(resueltas)],
         ["no lo resuelven", str(200 - resueltas)]], "id",
        [f"Las semillas, las mismas 200 de siempre; cada red, {miles(n)} pasos, poquito {coma(t, 2)}."]))
    emitir([""])
    filas = [[a, b, str(round(100 * cuenta[(a, b)] / resueltas)), str(cuenta[(a, b)])]
             for a, b in orden]
    emitir(tabla_editorial(
        "Lo que acaban haciendo las dos neuronas de la primera capa",
        ["una", "la otra", "de cada 100", "redes"], filas, "iidd",
        [f"De 200 redes entrenadas desde cero, cada una con su propio arranque, {resueltas} lo "
         f"resolvieron, y todas con una de estas {EN_LETRA.get(len(orden), len(orden))} parejas; "
         f"«de cada 100» se cuenta sobre esas {resueltas}. «Sí desde los 6 años» quiere decir que "
         "la neurona dice más de 0,5 de los seis años en adelante y menos a los más pequeños; «sí "
         "hasta los 5 años» es al revés."]))
    emitir([""])
    mas = cuenta[orden[0]]
    elegida = min((c for c in orden if cuenta[c] == mas), key=lambda c: primera[c])
    k = primera[elegida]
    semilla = semillas[k - 1]
    # la red del capítulo, entrenada otra vez, mirando qué edades falla cada 500 pasos
    red = red_nueva(semilla, d)
    mal = {}
    for i in range(1, n + 1):
        paso(red, X, t)
        if i % CADA_CUANTOS == 0:
            mal[i] = edades_mal(red, X)
    f_final = foto(red, X)
    assert f_final["ok"] and f_final["pareja"] == elegida, \
        "La red del capítulo, entrenada otra vez, no es la del barrido"
    a, b = elegida
    emitir(["LA RED DEL CAPÍTULO (con la edad en años)", ""])
    emitir(tabla_editorial(
        "Cuál es la red del capítulo", ["", "cuál"],
        [["la primera de la lista de arranques con la pareja más frecuente", f"la {k}.ª"],
         ["su semilla", str(semilla)], ["una neurona de en medio", a], ["la otra", b]], "id"))
    emitir([""])
    W0, b0, W1, b1 = red.W[0][0], red.b[0], red.W[1][:, 0], red.b[1][0]
    emitir(tabla_editorial(
        "Los pesos y los listones de la red del capítulo",
        ["neurona", "le llega la línea de", "peso", "listón"],
        [[NEURONAS[0], "la edad", coma(W0[0], 3), coma(-b0[0], 3)],
         [NEURONAS[1], "la edad", coma(W0[1], 3), coma(-b0[1], 3)],
         ["la final", NEURONAS[0], coma(W1[0], 3), coma(-b1, 3)],
         ["", NEURONAS[1], coma(W1[1], 3), ""]], "iidd",
        [f"Entrenada {miles(n)} pasos. El listón es menos el sesgo.",
         "Lo que le llega a cada neurona de en medio es el peso por la edad."]))
    emitir([""])
    act = red.adelante(X)
    medio, final = act[1], act[2].ravel()
    llega_medio = X @ red.W[0]
    emitir(tabla_editorial(
        "Lo que dice cada neurona de en medio, edad por edad",
        ["edad", "1.ª: le llega", "1.ª: dice", "2.ª: le llega", "2.ª: dice"],
        [[str(int(e)), coma(llega_medio[i, 0], 2), c3(medio[i, 0]), coma(llega_medio[i, 1], 2),
          c3(medio[i, 1])] for i, e in enumerate(EDADES)], "idddd",
        ["Le llega: el peso por la edad, sin el listón."]))
    emitir([""])
    llega_final = medio @ red.W[1][:, 0]
    emitir(tabla_editorial(
        "Lo que dice la final, edad por edad",
        ["edad", "le llega", "dice", "tendría que decir", "falla"],
        [[str(int(e)), coma(llega_final[i], 2), c3(final[i]), str(int(DEBERIA[i])),
          firmado(final[i] - DEBERIA[i], 3)] for i, e in enumerate(EDADES)], "idddd",
        ["Lo que falla: lo que dice menos lo que tendría que decir.",
         "Le llega: la suma de lo que dice cada una por el peso de su línea."]))
    emitir([""])
    secas = []
    for m1 in (0, 1):
        for m2 in (0, 1):
            llega = m1 * W1[0] + m2 * W1[1]
            pasa = llega - (-b1)
            secas.append([("sí" if m1 else "no"), ("sí" if m2 else "no"), coma(llega, 2),
                          coma(-b1, 2), firmado(pasa, 2), c3(float(C.rampa(pasa))),
                          "sí" if pasa > 0 else "no"])
    emitir(tabla_editorial(
        "La final, si las de en medio dijeran sí o no secos",
        ["1.ª", "2.ª", "le llega", "listón", "pasa por", "dice", "→"], secas, "iidddd" + "c",
        ["Un sí seco es un 1 y un no seco es un 0.",
         "Pasa por: lo que le llega menos el listón. Dice sí si es positivo."]))
    emitir([""])
    primera_limpia = min(p for p, m in mal.items() if not m)
    assert primera_limpia, "Se esperaba un múltiplo de 500 en que la red dejara de fallar"
    ultima_mala = max((p for p, m in mal.items() if m), default=0)
    hasta = ultima_mala + CADA_CUANTOS
    assert hasta in mal and not mal[hasta], "Se esperaba que la red dejara de fallar para siempre"
    filas = [[miles(p), str(len(m)), ", ".join(map(str, m)) or "ninguna"]
             for p, m in mal.items() if p <= hasta]
    notas = [f"La red del capítulo: semilla {semilla}.",
             f"Deja de fallar para siempre en el paso {miles(hasta)}: el último múltiplo de "
             f"{miles(CADA_CUANTOS)} con alguna mal es el {miles(ultima_mala)}."]
    if primera_limpia != hasta:
        notas.append(f"Antes, en el paso {miles(primera_limpia)}, ninguna fallaba, y después "
                     f"volvió a fallar alguna.")
    emitir(tabla_editorial(
        "Edades del lado equivocado de 0,5, cada 500 pasos", ["paso", "cuántas", "cuáles"],
        filas, "idi", notas))
    emitir([""])
    return dict(t=t, n=n, d=d, resueltas=resueltas, semilla=semilla, indice=k, hasta=hasta)


# ---------------------------------------------------------------- selftest

def selftest():
    fallos = []
    C.comprobar_entrada()
    X = C.entrada(1.0)
    semillas = C.sortear()[:20]

    # 1. TEST NULO — con poquito 0 no se aprende nada: de 20 redes, ninguna lo resuelve (y los
    #    pesos no se mueven).
    n_ok, quietas = 0, True
    for s in semillas:
        red = red_nueva(s)
        antes = [w.copy() for w in red.W]
        for _ in range(200):
            paso(red, X, 0.0)
        quietas = quietas and all(np.array_equal(a, b) for a, b in zip(antes, red.W))
        n_ok += int(not edades_mal(red, X))
    print(f"[1] test nulo         con poquito 0, resuelven {n_ok} de 20 y los pesos quedan quietos: "
          f"{'sí' if quietas else 'NO'}")
    if n_ok or not quietas:
        fallos.append("test nulo: sin aprender, algo resuelve o algo se mueve")

    # 2. SEÑAL IMPLANTADA — con los ajustes de la Orden 15 este programa da, red a red, lo mismo que
    #    `el_colegio.py` (resuelve o no, y la pareja), con la edad en años y en decenas, en las 20
    #    primeras semillas; y a los 20.000 pasos la red es la que entrena `Red.entrenar`.
    iguales = 0
    for escala in (1.0, C.DECENAS):
        for s in semillas:
            Xe = C.entrada(escala)
            red = red_nueva(s)
            for _ in range(PASOS_AHORA):
                paso(red, Xe, TASA_AHORA)
            ok_mio = not edades_mal(red, Xe)
            pareja_mia = (tuple(t for _, t in pareja_de(red, Xe)) if ok_mio else None)
            ok_su, pareja_su = C.un_arranque((s, escala))
            pareja_su = tuple(t for _, t in pareja_su) if ok_su else None
            iguales += int(ok_mio == ok_su and pareja_mia == pareja_su)
    de_verdad = Red([1, 2, 1], semilla=semillas[0]).entrenar(X, DEBERIA, TASA_AHORA, PASOS_AHORA)
    mia = red_nueva(semillas[0])
    for _ in range(PASOS_AHORA):
        paso(mia, X, TASA_AHORA)
    mismos = all(np.array_equal(a, b) for a, b in zip(mia.W + mia.b, de_verdad.W + de_verdad.b))
    print(f"[2] señal implantada  coincido con el_colegio.py en {iguales} de 40 redes (20 en años y 20 "
          f"en decenas); mismos pesos que Red.entrenar: {'sí' if mismos else 'NO'}")
    if iguales != 40 or not mismos:
        fallos.append("señal implantada: este programa no da lo mismo que el_colegio.py / Red.entrenar")

    # 3. INVARIANTE — la culpa por el atajo y a lo bruto coinciden en ocho decimales, con los pesos
    #    de siempre y divididos entre 10; todo lo que dice una neurona está entre 0 y 1; y una red
    #    con los pesos a mano pegados a 1 cuenta como pegada, y la de pesos pequeños, no.
    peor, fuera = 0.0, 0
    for divisor in DIVISORES:
        red = red_nueva(semillas[0], divisor)
        for _ in range(300):
            paso(red, X, 0.05)
        ga, _ = red.gradiente_retro(X, DEBERIA)
        gb = red.gradiente_fuerza_bruta(X, DEBERIA)
        peor = max(peor, max(float(np.abs(a - b).max()) for a, b in zip(ga, gb)))
        fuera += sum(int(((a < 0) | (a > 1)).sum()) for a in red.adelante(X)[1:])
    pegada = red_nueva(semillas[0]); pegada.W[0][0, :] = [5.0, 5.0]; pegada.b[0][:] = [-3.0, -3.0]
    suave = red_nueva(semillas[0], 10.0)
    print(f"[3] invariante        atajo contra fuerza bruta, diferencia máxima {peor:.1e}; respuestas "
          f"fuera de 0 a 1: {fuera}; pegadas: a mano {neuronas_pegadas(pegada, X)}, pesos entre 10 "
          f"{neuronas_pegadas(suave, X)}")
    if peor >= C.TOL_BRUTA or fuera or neuronas_pegadas(pegada, X) != 2 or neuronas_pegadas(suave, X):
        fallos.append("invariante: el gradiente, el rango o la cuenta de neuronas pegadas están mal")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    C.comprobar_entrada()
    if args.selftest:
        return selftest()

    dentro = [False]

    def emitir(lineas):
        for l in lineas:
            if l.startswith("::: "):
                dentro[0] = True
            if not dentro[0]:
                comprobar_ancho([l], ANCHO_SALIDA)
            if l == ":::":
                dentro[0] = False
            print(l)

    emitir(["######## el colegio con la edad en años (Orden 19) ########",
            f"fecha: {date.today().isoformat()}",
            f"máquina: {platform.system()} {platform.machine()}",
            f"python {platform.python_version()}, numpy {np.__version__}",
            "GPU: no se usa (todo en numpy). Nada de lo que sigue",
            "depende del ordenador: aciertos, recuentos y pesos.",
            "", "La edad entra en años, tal cual, salvo donde se dice.",
            "Las semillas, las 200 de la Orden 15.", ""])
    bloque_192(emitir)
    res, tabla = bloque_193(emitir)
    bloque_194(res, tabla, emitir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
