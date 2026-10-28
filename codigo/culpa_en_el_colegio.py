#!/usr/bin/env python3
"""
Capítulo 4 — la culpa, hecha a mano, sobre la red del colegio (Orden 20, 18 oct 2026).

Lo mismo que `culpa_hacia_atras.py` hace con el pasillo en sus bloques 1 a 5 y 8, con la red del
colegio: la que recibe la edad en años, tal cual, y aprende con la mejor combinación de
`el_colegio_en_anos.py` (poquito 0,5, 100.000 pasos y los pesos de arranque divididos entre 10).
La red del capítulo es la de semilla 318265456 («sí desde los 6 años» y «sí desde los 13 años»).
Los bloques 6 y 7 de `culpa_hacia_atras.py` (la rampa y el codo; la red de cinco capas) no son del
pasillo y se quedan donde están.

Imprime:
  a) la red entrenada: lo que dice, edad a edad; sus pesos y listones; la cuenta de cada neurona
     de en medio edad a edad; y la final si las de en medio dijeran un sí o un no secos;
  b) un ejemplo a medio entrenar, en el paso 1.000, hacia delante y hacia atrás: la edad que más
     falla, lo que falla cada una de las trece, los cuatro tramos, la comprobación a lo bruto de
     dos pesos, y dónde cae cada neurona en su rampa;
  c) el paso: la culpa de un peso edad a edad y su media, el paso de los cuatro pesos de las
     líneas y el de los tres listones, comprobados contra el entrenamiento de verdad;
  d) el terreno del error con un solo peso (el de la línea de la primera a la final);
  e) las que se atascan: de las 200, las que no lo resuelven, con 100.000 pasos más;
  f) los números para la cadena de «la culpa que se desvanece»;
  g) cuántos números tiene la red.

Reutiliza, importándolos y sin cambiarlos, `Red`, los datos y las funciones del colegio de
`el_colegio.py` (que no fallan con una neurona que dice 0,99998) y `el_colegio_en_anos.py`, y
`PELIN` de `culpa_hacia_atras.py`. Escribe también `culpa_en_el_colegio_datos.csv`, con los
números que dibujan las cinco figuras.

Uso:
    python culpa_en_el_colegio.py --selftest
    python culpa_en_el_colegio.py > ../datos/salidas/culpa_en_el_colegio.txt
"""

# ======================= CONSTANTES =======================

SEMILLA_RED = 318265456          # la red del capítulo (el_colegio_en_anos.txt: la 2.ª de la lista)
DIVISOR = 10.0                   # pesos de arranque: los de Red divididos entre 10
PASOS_ENTRENADO = 100_000        # la red entrenada
PASO_DEL_CASO = 1_000            # el ejemplo a medio entrenar: ya acierta de 3 a 12 y deja entrar a 13, 14 y 15
PASOS_EXTRA = 100_000            # a las que se atascan, cuántos pasos más
PASO_DESDE = 7_500               # «en qué pasos falla alguna edad después de ...»
CADA_CUANTOS_PASOS = 500         # cada cuántos pasos se mira qué edades falla la red del capítulo
PESO_DEL_TERRENO = (1, 0, 0)     # la línea de la primera a la final: (capa, de, a) en Red.W
# El terreno se mueve sobre la red ENTRENADA. De -5 a 30 se ve el llano de la izquierda (donde la
# culpa es casi cero), la bajada, el fondo (cerca de 14) y la subida suave de la derecha. La cuesta
# abajo sale de 4,0 (en la ladera, error 0,49): en 2.000 pasos baja hasta 13,2 y el error se queda
# en 0,0004, cerca del fondo; cada 100 pasos se ve la bajada entera, con los saltos grandes al
# principio y los cortos al final, que es lo que enseña.
TERRENO_DESDE, TERRENO_HASTA, TERRENO_PUNTOS = -5.0, 30.0, 176
INICIO_CUESTA_ABAJO = 4.0
PASOS_CUESTA_ABAJO = 2_000
CADA_CUANTOS = 100
TOL_BRUTA = 1e-8                 # atajo y a lo bruto: tienen que coincidir en ocho decimales
TOL_CUENTA = 1e-6                # la cuenta a mano del caso (con lo que deja pasar la rampa medido con un pelín) y el atajo
TOL_MANO = 1e-3                  # la cuenta a mano del error y el error medido, de las que se atascan
ANCHO_SALIDA = 62                # líneas de no más de 62 caracteres, fuera de las tablas
# Cómo dicen con letra los números que las notas de las tablas escriben así (Orden 50)
EN_LETRA = {7: "siete", 9: "nueve", 10: "diez", 13: "trece"}
EN_ORDINAL = {9: "noveno"}   # «hasta el noveno decimal» (Orden 97)

SALIDA_DATOS = "../datos/salidas/culpa_en_el_colegio_datos.csv"   # para las figuras
EL_COLEGIO_EN_ANOS = "../datos/salidas/el_colegio_en_anos.txt"    # donde la Orden 19 dio esta red

# ==========================================================

import argparse
import csv
import platform
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np

import el_colegio as C
import el_colegio_en_anos as EA
from culpa_hacia_atras import PELIN
from formato import comprobar_ancho, leer_tablas, miles, tabla_editorial
from retropropagacion import Red

AQUI = Path(__file__).resolve().parent
X = C.entrada(1.0)
Y = C.DEBERIA
EDADES = C.EDADES
POQUITO = C.TASA
coma, c3, firmado, NEURONAS = C.coma, C.c3, C.firmado, C.NEURONAS
r4, s4, dice_r, dice_s = C.r4, C.s4, C.dice_r, C.dice_s


# ---------------------------------------------------------------- la red y su entrenamiento

def red_en_el_paso(n, semilla=SEMILLA_RED):
    red = EA.red_nueva(semilla, DIVISOR)
    for _ in range(n):
        EA.paso(red, X, POQUITO)
    return red


def entrenada():
    return red_en_el_paso(PASOS_ENTRENADO)


def trayectoria(semilla=SEMILLA_RED):
    """Qué edades falla la red cada CADA_CUANTOS_PASOS pasos, hasta PASOS_ENTRENADO."""
    red = EA.red_nueva(semilla, DIVISOR)
    mal = {}
    for n in range(1, PASOS_ENTRENADO + 1):
        EA.paso(red, X, POQUITO)
        if n % CADA_CUANTOS_PASOS == 0:
            mal[n] = EA.edades_mal(red, X)
    return mal


def siguiente(red):
    otra = C.copia(red)
    EA.paso(otra, X, POQUITO)
    return otra


def culpa_de_un_ejemplo(red, p):
    return red.gradiente_retro(X[p:p + 1], Y[p:p + 1])


# ---------------------------------------------------------------- las que se atascan

def trabajo_atascada(semilla):
    """Una de las 200: entrenada PASOS_ENTRENADO; si no lo resuelve, PASOS_EXTRA más."""
    red = EA.red_nueva(semilla, DIVISOR)
    for _ in range(PASOS_ENTRENADO):
        EA.paso(red, X, POQUITO)
    antes = dict(error=red.error(X, Y), mal=EA.edades_mal(red, X), dice=red.adelante(X)[-1].ravel().tolist())
    if not antes["mal"]:
        return dict(semilla=semilla, ok=True, error=antes["error"])
    for _ in range(PASOS_EXTRA):
        EA.paso(red, X, POQUITO)
    despues = dict(error=red.error(X, Y), mal=EA.edades_mal(red, X), dice=red.adelante(X)[-1].ravel().tolist())
    return dict(semilla=semilla, ok=False, antes=antes, despues=despues)


def las_200():
    with ProcessPoolExecutor() as pool:
        return list(pool.map(trabajo_atascada, C.sortear(), chunksize=4))


DECIMALES_IMPRESOS = 10          # los de las tablas «La misma culpa, a lo bruto»


def decimales_iguales_impresos(a, b, decimales=DECIMALES_IMPRESOS):
    """Cuántos decimales coinciden tal como se imprimen: se escriben los dos con `decimales` cifras
    (como los imprimen las tablas) y se cuentan las cifras iguales desde la coma, hasta la primera que
    difiere. Si la parte entera o el signo ya difieren, 0. (Antes se comparaba la diferencia con
    media unidad de la cifra, y −0,5935879966 y −0,5935879967, que se imprimen distintos en la décima
    cifra, salían iguales en diez; Orden 21.)"""
    ea, da = f"{a:.{decimales}f}".split(".")
    eb, db = f"{b:.{decimales}f}".split(".")
    if ea != eb:
        return 0
    n = 0
    for x, y in zip(da, db):
        if x != y:
            break
        n += 1
    return n


def cuenta_con_lo_impreso(p, entran, fuera, decimales_p=3, decimales=4):
    """La cuenta del error de decir `p` a todas las edades, con cada fila hecha a partir de los números
    tal como se imprimen en la fila de encima, para que el lector la rehaga con una calculadora: lo que
    falla (1 − p, con `p` redondeado a `decimales_p`), por sí mismo (su cuadrado, a `decimales`), tantas
    veces como edades, y lo mismo para las que no entran; el error, la suma de las dos entre las trece.
    Devuelve las cifras de cada fila y el error. (Antes «por sí mismo» salía del 6/13 sin redondear y
    con el 0,462 que se veía daba 0,2134 y no 0,2130; Orden 23.)"""
    p = round(p, decimales_p)
    falla_si, falla_no = round(1 - p, decimales_p), p
    por_si, por_no = round(falla_si ** 2, decimales), round(falla_no ** 2, decimales)
    veces_si, veces_no = round(entran * por_si, decimales), round(fuera * por_no, decimales)
    error = (veces_si + veces_no) / (entran + fuera)
    return dict(p=p, falla_si=falla_si, por_si=por_si, veces_si=veces_si, falla_no=falla_no,
                por_no=por_no, veces_no=veces_no, error=error)


def error_a_mano(dice_por_edad):
    """El error de una red que dice lo mismo a un grupo de edades, construido a mano: para cada
    grupo (edades, lo que dice) la suma de lo que falla por sí mismo; entre las trece."""
    return sum((d - y) ** 2 for d, y in zip(dice_por_edad, Y)) / len(Y)


# ---------------------------------------------------------------- impresión

def letra(n):
    return EN_LETRA.get(n, str(n))


def tabla(*bloques):
    for b in bloques:
        EMITIR(b)
        EMITIR([""])


def bloque_a(red):
    act = red.adelante(X)
    medio, dice = act[1], act[2].ravel()
    W0, b0, W1, b1 = red.W[0][0], red.b[0], red.W[1][:, 0], red.b[1][0]
    EMITIR(["--- a) LA RED ENTRENADA ---", ""])
    tabla(
        tabla_editorial(
            f"Lo que dice la red entrenada, edad por edad",
            ["edad", "tendría que decir", "la red dice", "lo que falla"],
            [[str(int(e)), str(int(Y[i])), c3(dice[i]), firmado(dice[i] - Y[i], 3)]
             for i, e in enumerate(EDADES)], "iddd",
            [f"Lo que tendría que decir es 1 para los {letra(int(sum(Y)))} de primaria y 0 para los "
             "demás. La última columna, «lo que falla», es lo que dice la red menos lo que tendría "
             "que decir."]),
        tabla_editorial(
            "Los pesos y los listones de la red entrenada",
            ["neurona", "la línea que viene de", "peso", "listón"],
            [[NEURONAS[0], "la edad", firmado(W0[0], 3), coma(-b0[0], 3)],
             [NEURONAS[1], "la edad", firmado(W0[1], 3), coma(-b0[1], 3)],
             ["la final", NEURONAS[0], firmado(W1[0], 3), coma(-b1, 3)],
             ["", NEURONAS[1], firmado(W1[1], 3), ""]], "iidd",
            ["Peso de cada línea: positivo suma, negativo resta. El listón es menos el sesgo."]))
    nombres = [C.sentido(medio[:, j])[1] for j in range(2)]
    nota = ("Le llega: el peso por la edad. Dice: cerca de 1 si le llega más que su listón, cerca "
            "de 0 si le llega menos.")        # la de «La final, que escucha a las dos» (no cambia)
    T = []
    for j in range(2):
        llega = X[:, 0] * W0[j]
        T.append(tabla_editorial(
            f"{NEURONAS[j].capitalize()}: «{nombres[j]}»",
            ["edad", "le llega", "¿pasa del listón?", "dice"],
            [[str(int(e)), coma(llega[i], 2), "sí" if llega[i] - (-b0[j]) > 0 else "no", c3(medio[i, j])]
             for i, e in enumerate(EDADES)], "idcd",
            [f"{NEURONAS[j].capitalize()} neurona necesita que le lleguen más de {coma(-b0[j], 2)}, su "
             "listón. Lo que le llega es su peso por la edad. Lo que dice está cerca de 1 si le "
             "llega más que su listón, y cerca de 0 si le llega menos."]))
    llega_f = medio @ W1
    T.append(tabla_editorial(
        "La final, que escucha a las dos", ["edad", "le llega", "¿pasa del listón?", "dice"],
        [[str(int(e)), coma(llega_f[i], 2), "sí" if llega_f[i] - (-b1) > 0 else "no", c3(dice[i])]
         for i, e in enumerate(EDADES)], "idcd", [f"Listón: {coma(-b1, 2)}. " + nota]))
    tabla(*T)
    secas = []
    for m1 in (0, 1):
        for m2 in (0, 1):
            llega = m1 * W1[0] + m2 * W1[1]
            pasa = llega - (-b1)
            secas.append([("sí" if m1 else "no"), ("sí" if m2 else "no"), firmado(llega, 2),
                          "sí" if pasa > 0 else "no"])
    tabla(tabla_editorial(
        "La final, si las dos de la primera capa dijeran un sí o un no secos",
        ["la primera", "la segunda", "le llega", "¿pasa?"], secas, "ccdc",
        [f"Aquí «sí» vale 1 y «no» vale 0. Para pasar, a la final le tienen que llegar más de "
         f"{coma(-b1, 2)}, su listón."]))
    return nombres


def bloque_b(red, p):
    c = C.caso(red, X, p)
    m = C.cadena_a_mano(red, c)
    W1 = m["pesos_final"]
    deberia = int(c["deberia"])
    edad = int(EDADES[p])
    dice = red.adelante(X)[-1].ravel()
    EMITIR([f"--- b) UN EJEMPLO A MEDIO ENTRENAR: PASO {miles(PASO_DEL_CASO)} ---", ""])
    tabla(tabla_editorial(
        f"Lo que falla cada edad en el paso {miles(PASO_DEL_CASO)}",
        ["edad", "tendría que decir", "la red dice", "lo que falla", "¿del lado equivocado?"],
        [[str(int(e)), str(int(Y[i])), c3(dice[i]), firmado(dice[i] - Y[i], 3),
          "sí" if (dice[i] > C.UMBRAL) != (Y[i] > C.UMBRAL) else "no"] for i, e in enumerate(EDADES)],
        "idddc",
        [f"La del ejemplo es la que más falla: {edad} años, con {c3(abs(dice[p] - Y[p]))}."]))
    filas = []
    for j in range(2):
        filas.append([NEURONAS[j], coma(c["llega_medio"][j], 2), coma(c["liston_medio"][j], 2),
                      coma(c["pasa_medio"][j], 2), dice_s(c["dice_medio"][j])])
    filas.append(["la final", coma(c["llega_final"], 2), coma(c["liston_final"], 2),
                  coma(c["pasa_final"], 2), dice_s(c["dice_final"])])
    contexto = (f"Es el paso {miles(PASO_DEL_CASO)} de los {miles(PASOS_ENTRENADO)} del entrenamiento, "
                f"con la edad de {edad} años, que el colegio {'admite' if deberia else 'no admite'}: "
                f"la final tendría que decir {deberia}.")
    dos = [r4(m["dice_medio"][j] * W1[j]) for j in range(2)]
    casi_uno = [NEURONAS[j] for j in range(2) if dice_r(c["dice_medio"][j]) != r4(c["dice_medio"][j])]
    notas = [contexto]
    if casi_uno:
        notas.append("A cuatro cifras, lo que dice " + " y ".join(casi_uno) + " saldría 1 y lo que le "
                     "falta para 1 saldría 0: se redondea lo que le falta, y se resta.")
    tabla(
        tabla_editorial("Hacia delante", ["neurona", "le llega", "listón", "pasa del listón por", "dice"],
                        filas, "idddd", notas),
        tabla_editorial(
            "Lo que le llega a la final",
            ["viene de", "lo que dice", "por el peso de su línea", "da"],
            [[NEURONAS[j], dice_s(c["dice_medio"][j]), coma(W1[j], 2), s4(dos[j])] for j in range(2)]
            + [["**la suma**", "", "", f"**{coma(c['llega_final'], 2)}**"]], "iddd",
            ["La suma, redondeada a dos decimales, como en la tabla anterior."]),
        tabla_editorial(
            "Lo que falla, y el error", ["", "cuánto"],
            [["lo que dice la final", dice_s(c["dice_final"])],
             ["lo que tendría que decir", str(deberia)],
             ["lo que falla: lo que dice menos lo que tendría que decir", s4(m["falla"], True)],
             ["**el error con esta edad: lo que falla por sí mismo**", f"**{s4(r4(m['falla']) ** 2)}**"]],
            "id", ["Los números, redondeados a cuatro cifras, y cada cuenta con los redondeados."]))
    s = c["dice_final"]
    e_mas = (s + PELIN - c["deberia"]) ** 2
    dice_mas = float(C.rampa(c["pasa_final"] + PELIN))
    EMITIR(["HACIA ATRÁS", ""])
    tabla(
        tabla_editorial(
            "Tramo 1, medido: cuánto se mueven el error y lo que dice la final",
            ["qué se mide", "tal cual", "una millonésima más arriba", "sube, en millonésimas"],
            [["el error, si se mueve lo que dice", coma(c["error"], 9), coma(e_mas, 9),
              coma((e_mas - c["error"]) / PELIN, 3)],
             ["lo que dice, si se mueve lo que le llega", coma(c["dice_final"], 9), coma(dice_mas, 9),
              coma((dice_mas - c["dice_final"]) / PELIN, 4)]], "iddd",
            ["Lo que sube el error es el doble de lo que falla; lo que sube lo que dice es lo que "
             "deja pasar su rampa. La tabla siguiente lo saca con cuentas."]),
        tabla_editorial(
            "Tramo 1: la culpa de la final", ["", "cuánto"],
            [["lo que falla", s4(m["falla"])], ["el doble de lo que falla", s4(m["doble"])],
             ["lo que dice", dice_s(c["dice_final"])], ["lo que le falta para 1", s4(m["falta_final"])],
             ["lo que deja pasar su rampa: lo que dice, por lo que le falta para 1", s4(m["pasa_final"])],
             ["**la culpa de la final: el doble de lo que falla, por lo que deja pasar su rampa**",
              f"**{s4(m['culpa_final'])}**"]], "id",
            ["Cada cuenta está hecha con los números redondeados a cuatro cifras, para que se pueda "
             "repetir con el dedo."]),
        tabla_editorial(
            "Tramo 2: las dos líneas que llegan a la final",
            ["la línea que viene de", "la culpa de la final", "por lo que trajo esa línea", "da"],
            [[NEURONAS[j], s4(m["culpa_final"]), dice_s(c["dice_medio"][j]), s4(m["lineas_final"][j])]
             for j in range(2)], "iddd",
            ["Lo que trae cada línea es lo que dice la neurona de la que sale."]),
        tabla_editorial(
            "Tramo 3: cuánta le toca a cada una de las dos de la primera capa", ["", NEURONAS[0], NEURONAS[1]],
            [["la culpa de la final", s4(m["culpa_final"]), s4(m["culpa_final"])],
             ["por el peso de su línea", firmado(W1[0], 2), firmado(W1[1], 2)],
             ["le toca", s4(m["le_toca"][0]), s4(m["le_toca"][1])],
             ["lo que dice", dice_s(c["dice_medio"][0]), dice_s(c["dice_medio"][1])],
             ["lo que le falta para 1", s4(m["falta_medio"][0]), s4(m["falta_medio"][1])],
             ["lo que deja pasar su rampa", s4(m["pasa_medio"][0]), s4(m["pasa_medio"][1])],
             ["**su culpa: lo que le toca, por lo que deja pasar su rampa**",
              f"**{s4(m['culpa_medio'][0])}**", f"**{s4(m['culpa_medio'][1])}**"]], "idd",
            ["Lo que deja pasar la rampa de cada una es lo que dice, por lo que le falta para 1."]),
        tabla_editorial(
            "Tramo 4: las dos líneas de la primera capa",
            ["línea", "culpa de su neurona", "por lo que trajo", "da"],
            [[f"{NEURONAS[j]}, de la edad", s4(m["culpa_medio"][j]), str(edad), s4(m["lineas_primera"][j])]
             for j in range(2)], "iddd",
            [f"Lo que trae cada una de estas dos líneas es la edad, {edad}."]))
    # la comprobación a lo bruto de dos pesos, sin redondear
    gW, _ = culpa_de_un_ejemplo(red, p)
    filas, juntas = [], []
    for (capa, i, j, nombre) in ((0, 0, 1, "la segunda, de la edad"), (1, 1, 0, "la final, de la segunda")):
        mas, menos, bruta = C.culpa_a_lo_bruto(red, X, p, capa, i, j)
        atajo = float(gW[capa][i, j])
        assert abs(atajo - bruta) < TOL_BRUTA, f"atajo y fuerza bruta difieren en {abs(atajo - bruta)}"
        filas.append((nombre, atajo, bruta, mas, menos))
        juntas.append(decimales_iguales_impresos(atajo, bruta))
    tabla(tabla_editorial(
        "La misma culpa, a lo bruto", ["peso", "por el atajo", "a lo bruto"],
        [[n, coma(a, 10), coma(b, 10)] for n, a, b, _, _ in filas], "idd",
        ["La columna «a lo bruto» se calcula sin atajo: se mueve el peso una cienmilésima hacia "
         "arriba y otra hacia abajo, y se mira cuánto cambia el error. Coincide con la del atajo "
         f"hasta el {EN_ORDINAL[min(juntas)]} decimal."]))
    for n, _, _, mas, menos in filas:
        tabla(tabla_editorial(
            f"El error, con {n} movido", ["el peso", "el error"],
            [["una cienmilésima más arriba", coma(mas, 10)], ["una cienmilésima más abajo", coma(menos, 10)]],
            "id"))
    # dónde cae cada neurona en su rampa, para la figura de la rampa
    tabla(tabla_editorial(
        "Dónde cae cada neurona en su rampa, y lo que deja pasar",
        ["neurona", "pasa del listón por", "dice", "deja pasar"],
        [[NEURONAS[0], coma(c["pasa_medio"][0], 3), dice_s(c["dice_medio"][0]), s4(m["pasa_medio"][0])],
         [NEURONAS[1], coma(c["pasa_medio"][1], 3), dice_s(c["dice_medio"][1]), s4(m["pasa_medio"][1])],
         ["la final", coma(c["pasa_final"], 3), dice_s(c["dice_final"]), s4(m["pasa_final"])]], "iddd",
        ["Pasa del listón por: lo que le llega menos su listón. Deja pasar: lo que dice, por lo que le "
         "falta para 1."]))
    return c, m


def bloque_c(red, p):
    por_edad = [culpa_de_un_ejemplo(red, i)[0][0][0, 1] for i in range(len(EDADES))]
    gW, gb = red.gradiente_retro(X, Y)
    media = float(np.mean(por_edad))
    assert abs(media - gW[0][0, 1]) < 1e-12, "la media de las trece no es la culpa del programa"
    otra = siguiente(red)
    filas = [("la primera, de la edad", 0, 0, 0), ("la segunda, de la edad", 0, 0, 1),
             ("la final, de la primera", 1, 0, 0), ("la final, de la segunda", 1, 1, 0)]
    pasos = []
    for nombre, capa, i, j in filas:
        culpa = gW[capa][i, j]
        antes, despues = red.W[capa][i, j], otra.W[capa][i, j]
        assert abs((antes - POQUITO * culpa) - despues) < 1e-12, "el paso no es el del programa"
        pasos.append([nombre, firmado(culpa, 4), firmado(-POQUITO * culpa, 4), coma(antes, 4), coma(despues, 4)])
    EMITIR(["--- c) EL PASO: CADA PESO SE MUEVE CONTRA SU CULPA ---", ""])
    tabla(tabla_editorial(
        "La culpa de la segunda, de la edad, edad a edad", ["edad", "culpa"],
        [[str(int(e)), firmado(por_edad[i], 4)] for i, e in enumerate(EDADES)]
        + [["**la media**", f"**{firmado(media, 4)}**"]], "id",
        [f"La culpa con la que se mueve el peso es la media de las {letra(len(EDADES))}: se "
         f"suman y se divide entre {len(EDADES)}."]))
    tabla(tabla_editorial(
        "El paso: cada peso se mueve contra su culpa", ["peso", "culpa", "se mueve", "antes", "después"],
        pasos, "idddd",
        [("Cada peso se mueve la mitad de su culpa" if POQUITO == 0.5 else
          f"Cada peso se mueve {coma(POQUITO, 1)} veces su culpa")
         + ", en contra: si la culpa es negativa, sube; si es positiva, baja."]))
    listones = []
    for nombre, capa, j in (("la primera", 0, 0), ("la segunda", 0, 1), ("la final", 1, 0)):
        culpa = -gb[capa][j]            # cuánto sube el error por cada unidad que sube el listón
        antes, despues = -red.b[capa][j], -otra.b[capa][j]
        assert abs((antes - POQUITO * culpa) - despues) < 1e-12, "el paso del listón no es el del programa"
        listones.append([nombre, firmado(culpa, 4), firmado(-POQUITO * culpa, 4), coma(antes, 4), coma(despues, 4)])
    tabla(tabla_editorial(
        "El paso de los tres listones", ["listón", "culpa", "se mueve", "antes", "después"], listones,
        "idddd", ["La culpa de un listón: cuánto sube el error por cada unidad que sube el listón. "
                  "Se mueve igual, contra su culpa."]))
    return gW, gb, por_edad, media


def terreno(red):
    capa, i, j = PESO_DEL_TERRENO
    xs = np.linspace(TERRENO_DESDE, TERRENO_HASTA, TERRENO_PUNTOS)
    otra = C.copia(red)
    ys = []
    for v in xs:
        otra.W[capa][i, j] = v
        ys.append(otra.error(X, Y))
    otra = C.copia(red)
    otra.W[capa][i, j] = INICIO_CUESTA_ABAJO
    camino = [(float(otra.W[capa][i, j]), otra.error(X, Y))]
    for _ in range(PASOS_CUESTA_ABAJO):
        gW, _ = otra.gradiente_retro(X, Y)
        otra.W[capa][i, j] -= POQUITO * gW[capa][i, j]
        camino.append((float(otra.W[capa][i, j]), otra.error(X, Y)))
    return xs, np.array(ys), camino


def bloque_d(red, xs, ys, camino):
    capa, i, j = PESO_DEL_TERRENO
    k = int(np.argmin(ys))
    EMITIR(["--- d) EL TERRENO DEL ERROR, CON UN SOLO PESO ---", ""])
    filas = []
    for n in range(0, len(camino), CADA_CUANTOS):
        v, e = camino[n]
        movido = "" if n == 0 else coma(v - camino[n - CADA_CUANTOS][0], 2)
        filas.append([miles(n), coma(v, 2), coma(e, 4), movido])
    tabla(
        tabla_editorial(
            "El terreno del error, con un solo peso", ["dónde", "peso", "error"],
            [["donde lo dejó el entrenamiento", coma(red.W[capa][i, j], 2), coma(red.error(X, Y), 5)],
             ["el fondo de esta curva", coma(xs[k], 1), coma(ys[k], 5)],
             ["un extremo", firmado(xs[0], 0), coma(ys[0], 4)],
             ["el otro extremo", firmado(xs[-1], 0), coma(ys[-1], 4)]], "idd",
            ["La red entrenada; se mueve solo la línea de la primera a la final, y los demás pesos se "
             "quedan quietos. El error es la media de las trece edades."]),
        tabla_editorial(
            f"Cuesta abajo, moviendo solo ese peso, desde {coma(INICIO_CUESTA_ABAJO, 1)}",
            ["paso", "peso", "error", f"lo que se movió en esos {CADA_CUANTOS} pasos"], filas, "dddd",
            [f"Cada paso mueve el peso {coma(POQUITO, 1)} veces su culpa, en contra."]))
    return k


def bloque_e(resultados, red_capitulo_mal):
    malas = [r for r in resultados if not r["ok"]]
    buenas = [r for r in resultados if r["ok"]]
    siguen = [r for r in malas if r["despues"]["mal"]]
    salen = [r for r in malas if not r["despues"]["mal"]]
    rango = lambda xs: (coma(min(xs), 4) if abs(min(xs) - max(xs)) < 5e-5
                        else f"{coma(min(xs), 4)} a {coma(max(xs), 4)}")
    EMITIR(["--- e) LAS QUE SE ATASCAN ---", ""])
    tabla(tabla_editorial(
        f"Las que se atascan: {len(malas)} de {len(resultados)} arranques",
        ["", f"error tras {miles(PASOS_ENTRENADO)} pasos", f"con {miles(PASOS_EXTRA)} pasos más"],
        [[f"las {len(siguen)} que siguen", rango([r["antes"]["error"] for r in siguen]),
          rango([r["despues"]["error"] for r in siguen])],
         ["la que sale" if len(salen) == 1 else f"las {len(salen)} que salen",
          rango([r["antes"]["error"] for r in salen]), rango([r["despues"]["error"] for r in salen])],
         ["las que resuelven", rango([r["error"] for r in buenas]), ""]], "idd",
        [f"De las {len(malas)} que se atascaron, {len(siguen)} siguen sin resolverlo con los "
         f"{miles(PASOS_EXTRA)} pasos de más, y {len(salen)} "
         + ("sale" if len(salen) == 1 else "salen") + "."]))
    # qué edades fallan y qué dicen en ellas, las que siguen
    grupos = {}
    for r in siguen:
        grupos.setdefault(tuple(r["despues"]["mal"]), []).append(r)
    filas = []
    for mal, rs in sorted(grupos.items(), key=lambda t: -len(t[1])):
        dicen = [r["despues"]["dice"][int(e) - int(EDADES[0])] for r in rs for e in mal]
        todas = [v for r in rs for v in r["despues"]["dice"]]
        filas.append([", ".join(map(str, mal)), str(len(rs)), rango(dicen), rango([r["despues"]["error"]
                                                                                   for r in rs])])
    tabla(tabla_editorial(
        "Qué edades fallan las que siguen, y qué dicen en ellas",
        ["edades que fallan", "redes", "lo que dicen en ellas", "su error"], filas, "idd" + "d",
        [f"Las {len(siguen)} que siguen atascadas después de los {miles(PASOS_EXTRA)} pasos de "
         "más, agrupadas según las edades en las que fallan."]))
    # el estado en que se quedan casi todas, y el otro: cuenta a mano contra el error medido
    estados = []
    for mal, rs in sorted(grupos.items(), key=lambda t: -len(t[1])):
        d0 = rs[0]["despues"]["dice"]
        estados.append((mal, rs, d0))
    mal, rs, d0 = estados[0]
    medido = [r["despues"]["error"] for r in rs]
    # el estado: a los de 3 a 5 les dice 0; a los de 6 en adelante, la proporción de los que entran
    entre = [i for i, e in enumerate(EDADES) if e >= 6]
    p = float(np.mean([Y[i] for i in entre]))
    a_mano = error_a_mano([0.0 if e < 6 else p for e in EDADES])
    assert abs(a_mano - np.mean(medido)) < TOL_MANO, \
        f"la cuenta a mano ({a_mano}) no coincide con el error medido ({np.mean(medido)})"
    entran, fuera = int(sum(Y[i] for i in entre)), len(entre) - int(sum(Y[i] for i in entre))
    tabla(tabla_editorial(
        f"El error de decir {coma(p, 1)} a los de 6 en adelante y acertar los de 3 a 5",
        ["", "cuánto"],
        [[f"lo que falla en cada una de las {entran} que entran (dice {coma(p, 1)}, tendría que decir 1)",
          coma(1 - p, 1)],
         ["por sí mismo", coma((1 - p) ** 2, 2)],
         [f"{entran} veces", coma(entran * (1 - p) ** 2, 2)],
         [f"lo que falla en cada una de las {fuera} que no entran (dice {coma(p, 1)}, tendría que decir 0)",
          coma(p, 1)],
         ["por sí mismo", coma(p ** 2, 2)],
         [f"{fuera} veces", coma(fuera * p ** 2, 2)],
         ["en las de 3 a 5 acierta: no falla nada", "0"],
         ["**entre las trece: el error**", f"**{coma(a_mano, 4)}**"],
         ["el error medido, de las que se quedan así", rango(medido)]], "id",
        ["Es el error de la mejor respuesta que se puede dar sin distinguir una edad de otra entre "
         f"las de 6 en adelante: de esas {letra(len(entre))}, entran {letra(entran)}, y "
         f"{letra(entran)} entre {letra(len(entre))} da {coma(p, 1)}. Así se quedan {len(rs)} de las "
         f"{len(siguen)} redes que siguen atascadas."]))
    # el otro estado (si lo hay): todas dicen lo mismo
    for mal2, rs2, d2 in estados[1:]:
        medido2 = [r["despues"]["error"] for r in rs2]
        p2 = float(np.mean(Y))
        a_mano2 = error_a_mano([p2] * len(Y))
        if abs(a_mano2 - np.mean(medido2)) < TOL_MANO:
            e = int(sum(Y))
            n = int(len(Y) - sum(Y))
            c = cuenta_con_lo_impreso(p2, e, n)
            assert coma(c["error"], 4) == coma(a_mano2, 4), \
                f"la cuenta con lo impreso ({c['error']}) no da el error a mano ({a_mano2})"
            tabla(tabla_editorial(
                f"El error de decir {coma(c['p'], 3)} a todas las edades", ["", "cuánto"],
                [[f"lo que falla en cada una de las {e} que entran", coma(c["falla_si"], 3)],
                 ["por sí mismo", coma(c["por_si"], 4)],
                 [f"{e} veces", coma(c["veces_si"], 4)],
                 [f"lo que falla en cada una de las {n} que no entran", coma(c["falla_no"], 3)],
                 ["por sí mismo", coma(c["por_no"], 4)],
                 [f"{n} veces", coma(c["veces_no"], 4)],
                 ["**entre las trece: el error**", f"**{coma(c['error'], 4)}**"],
                 ["el error medido, de las que se quedan así", rango(medido2)]], "id",
                [f"Es el error de la mejor respuesta sin distinguir ninguna edad: la proporción de las "
                 f"que entran, {e} de {len(Y)}. Cada fila sale de los números de las de encima, tal como "
                 f"se ven; el error es la suma de las dos veces, entre {len(Y)}. {len(rs2)} de las "
                 f"{len(siguen)} se quedan así."]))
    # los pasos en que falla alguna edad, en la red del capítulo, después de 7.500
    despues = [(p_, m_) for p_, m_ in red_capitulo_mal.items() if p_ > PASO_DESDE and m_]
    tabla(tabla_editorial(
        f"Pasos en que falla alguna edad la red del capítulo, después del {miles(PASO_DESDE)}",
        ["paso", "cuántas", "cuáles"],
        [[miles(p_), str(len(m_)), ", ".join(map(str, m_))] for p_, m_ in despues], "idi",
        [f"Se mira cada {CADA_CUANTOS_PASOS} pasos, hasta el {miles(PASOS_ENTRENADO)}. El último paso con "
         f"alguna mal es el {miles(max(p_ for p_, _ in despues))}: desde el siguiente no vuelve a fallar "
         "ninguna."]))
    return dict(malas=malas, siguen=siguen, salen=salen, estados=estados, p_estado=p, a_mano=a_mano)


def bloque_f(c, m):
    EMITIR(["--- f) LOS NÚMEROS PARA LA CADENA ---", ""])
    tabla(tabla_editorial(
        "La culpa de la primera neurona, antes y después de su rampa", ["", "cuánto"],
        [["la culpa de la final", s4(m["culpa_final"])],
         ["por el peso de su línea", firmado(m["pesos_final"][0], 2)],
         ["lo que le toca a la primera", s4(m["le_toca"][0])],
         ["lo que deja pasar su rampa", s4(m["pasa_medio"][0])],
         ["su culpa, tras pasar por su rampa", s4(m["culpa_medio"][0])],
         ["sin redondear: lo que le toca", coma(c["le_toca_medio"][0], 6)],
         ["sin redondear: su culpa", coma(c["culpa_medio"][0], 6)]], "id",
        ["En el ejemplo del paso 1.000. La rampa deja pasar de lo que le toca solo una parte: "
         "su culpa es lo que le toca por esa parte."]))


def bloque_g(red):
    n_pesos = sum(w.size for w in red.W)
    n_listones = sum(b.size for b in red.b)
    EMITIR(["--- g) CUÁNTOS NÚMEROS TIENE LA RED ---", ""])
    tabla(tabla_editorial(
        "Cuántos números tiene la red", ["", "cuántos"],
        [["pesos de las líneas", str(n_pesos)], ["listones", str(n_listones)],
         ["**en total**", f"**{n_pesos + n_listones}**"]], "id",
        ["Dos líneas de la edad a las de en medio, dos de las de en medio a la final, y un listón por "
         "neurona."]))
    return n_pesos, n_listones


# ---------------------------------------------------------------- datos para las figuras

def guardar_datos(red, red_caso, c, m, p, xs, ys, camino, por_edad, media, resumen):
    with open(AQUI / SALIDA_DATOS, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["que", "a", "b", "valor"])
        w.writerow(["paso_del_caso", "", "", repr(PASO_DEL_CASO)])
        w.writerow(["edad_del_caso", "", "", repr(int(EDADES[p]))])
        w.writerow(["deberia_del_caso", "", "", repr(float(Y[p]))])
        for nombre, r in (("final", red), ("caso", red_caso)):
            w.writerow([f"{nombre}_peso_medio", 0, 0, repr(float(r.W[0][0, 0]))])
            w.writerow([f"{nombre}_peso_medio", 0, 1, repr(float(r.W[0][0, 1]))])
            for j in range(2):
                w.writerow([f"{nombre}_liston_medio", j, "", repr(float(-r.b[0][j]))])
                w.writerow([f"{nombre}_peso_final", j, "", repr(float(r.W[1][j, 0]))])
            w.writerow([f"{nombre}_liston_final", "", "", repr(float(-r.b[1][0]))])
        act = red.adelante(X)
        for i, e in enumerate(EDADES):
            for j in range(2):
                w.writerow(["final_dice_medio", int(e), j, repr(float(act[1][i, j]))])
            w.writerow(["final_dice", int(e), "", repr(float(act[2][i, 0]))])
            w.writerow(["deberia", int(e), "", repr(float(Y[i]))])
        for clave in ("culpa_final", "dice_final", "falla", "pasa_final", "pasa_rampa_final"):
            w.writerow([f"caso_{clave}", "", "", repr(float(c[clave]))])
        for j in range(2):
            for clave in ("dice_medio", "le_toca_medio", "culpa_medio", "pasa_rampa_medio", "pasa_medio"):
                w.writerow([f"caso_{clave}", j, "", repr(float(c[clave][j]))])
            w.writerow(["caso_culpa_lineas_final", j, "", repr(float(c["culpa_lineas_final"][j]))])
            w.writerow(["caso_culpa_peso_medio", 0, j, repr(float(c["culpa_pesos_medio"][0, j]))])
        # la cadena tal como se imprime (cuatro cifras): es la que dibuja la figura
        for clave in ("dice_final", "falla", "culpa_final", "pasa_final"):
            w.writerow([f"mano_{clave}", "", "", repr(m[clave])])
        for j in range(2):
            for clave in ("dice_medio", "le_toca", "pasa_medio", "culpa_medio", "lineas_final"):
                w.writerow([f"mano_{clave}", j, "", repr(m[clave][j])])
            w.writerow(["mano_lineas_primera", 0, j, repr(m["lineas_primera"][j])])
        w.writerow(["media_culpa_edad_segunda", "", "", repr(float(media))])
        for x, y in zip(xs, ys):
            w.writerow(["terreno", repr(float(x)), "", repr(float(y))])
        for n, (v, e) in enumerate(camino):
            w.writerow(["camino", n, repr(v), repr(e)])
        w.writerow(["cada_cuantos_camino", "", "", repr(CADA_CUANTOS)])
        w.writerow(["atascadas_siguen", "", "", repr(len(resumen["siguen"]))])


# ---------------------------------------------------------------- selftest

def leer_de_la_orden_19():
    T = leer_tablas((AQUI / EL_COLEGIO_EN_ANOS).read_text(encoding="utf-8"))
    return T


def selftest():
    fallos = []
    red1 = red_en_el_paso(PASO_DEL_CASO)

    # [1] TEST NULO — si la red ya dice lo que tendría que decir, no hay culpa que repartir: todos
    #     los pesos y listones tienen culpa cero. Se fabrica cambiando lo que «tendría que decir» por
    #     lo que ya dice.
    s = red1.adelante(X[3:4])[-1].ravel()
    gW, gb = red1.gradiente_retro(X[3:4], s)
    nulo = max(float(np.abs(g).max()) for g in gW + gb)
    print(f"[1] test nulo         si no falla, la mayor culpa de un peso es {nulo:.1e} (tiene que ser 0)")
    if nulo != 0.0:
        fallos.append(f"test nulo: sin fallo sale culpa {nulo}")

    # [2] SEÑAL IMPLANTADA — la red entrenada aquí es la de la Orden 19: los pesos, los listones y lo
    #     que dice la final edad a edad, escritos en `el_colegio_en_anos.txt`; y la cuenta a mano del
    #     caso da lo mismo que el atajo del programa.
    T = leer_de_la_orden_19()
    red = entrenada()
    W0, b0, W1, b1 = red.W[0][0], red.b[0], red.W[1][:, 0], red.b[1][0]
    esperado = T["Los pesos y los listones de la red del capítulo"][1]
    mios = [[coma(W0[0], 3), coma(-b0[0], 3)], [coma(W0[1], 3), coma(-b0[1], 3)],
            [coma(W1[0], 3), coma(-b1, 3)], [coma(W1[1], 3), ""]]
    iguales_pesos = [f[2:] for f in esperado] == mios
    dice = red.adelante(X)[-1].ravel()
    suyos = [f[2] for f in T["Lo que dice la final, edad por edad"][1]]
    iguales_dice = suyos == [c3(v) for v in dice]
    p = int(np.argmax(np.abs(red1.adelante(X)[-1].ravel() - Y)))
    c = C.caso(red1, X, p)
    gW1, _ = culpa_de_un_ejemplo(red1, p)
    dif = max(float(np.abs(gW1[0] - c["culpa_pesos_medio"]).max()),
              float(np.abs(gW1[1][:, 0] - c["culpa_lineas_final"]).max()))
    C.cadena_a_mano(red1, c)                # revienta si la cadena a mano se aleja más del 1 %
    print(f"[2] señal implantada  pesos y listones como en la Orden 19: {'sí' if iguales_pesos else 'NO'}; "
          f"lo que dice la final, también: {'sí' if iguales_dice else 'NO'}; la cuenta a mano del caso y "
          f"el atajo difieren en {dif:.1e}")
    if not (iguales_pesos and iguales_dice) or dif > TOL_CUENTA:
        fallos.append("señal implantada: la red no es la de la Orden 19, o la cuenta a mano no es el atajo")

    # [3] INVARIANTE — la culpa por el atajo es la de mover el peso un pelín (los siete números del
    #     caso, a lo bruto), la culpa de un peso es la media de la de las trece edades, el error de la
    #     curva del terreno en el peso de la red es el de la red, y el paso es el del entrenamiento.
    peor = 0.0
    for capa, i, j in ((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)):
        _, _, bruta = C.culpa_a_lo_bruto(red1, X, p, capa, i, j)
        peor = max(peor, abs(bruta - gW1[capa][i, j]))
    por_edad = [culpa_de_un_ejemplo(red1, i)[0][1][0, 0] for i in range(len(EDADES))]
    gW_t, _ = red1.gradiente_retro(X, Y)
    dif_media = abs(float(np.mean(por_edad)) - gW_t[1][0, 0])
    otra = C.copia(red)
    otra.W[1][0, 0] = red.W[1][0, 0]
    dif_terreno = abs(otra.error(X, Y) - red.error(X, Y))
    c538 = cuenta_con_lo_impreso(7 / 13, 7, 6)
    rehecha = round(round(7 * round(0.462 ** 2, 4), 4) + round(6 * round(0.538 ** 2, 4), 4), 4) / 13
    print(f"[3] invariante        la cuenta de 0,538 con lo impreso: {coma(c538['falla_si'], 3)}, "
          f"{coma(c538['por_si'], 4)}, {coma(c538['veces_si'], 4)}; {coma(c538['falla_no'], 3)}, "
          f"{coma(c538['por_no'], 4)}, {coma(c538['veces_no'], 4)}; error {coma(c538['error'], 4)}, y "
          f"rehecha a mano {coma(rehecha, 4)}")
    if (coma(c538["falla_si"], 3), coma(c538["por_si"], 4), coma(c538["veces_si"], 4)) != ("0,462", "0,2134", "1,4938") \
            or (coma(c538["por_no"], 4), coma(c538["veces_no"], 4)) != ("0,2894", "1,7364") \
            or coma(c538["error"], 4) != "0,2485" or coma(rehecha, 4) != coma(c538["error"], 4):
        fallos.append("invariante: la cuenta del 0,538 con lo impreso no sale como se rehace a mano")
    cuentan = (decimales_iguales_impresos(-0.5935879966, -0.5935879967),
               decimales_iguales_impresos(0.2266538661, 0.2266538661),
               decimales_iguales_impresos(-2.8720379865, -2.8720379841),
               decimales_iguales_impresos(1.5, -1.5))
    print(f"[3] invariante        los decimales iguales, tal como se imprimen: {cuentan[0]} (de -0,5935879966 "
          f"y -0,5935879967), {cuentan[1]} (de dos iguales), {cuentan[2]}, {cuentan[3]} (de 1,5 y -1,5)")
    if cuentan != (9, 10, 8, 0):
        fallos.append(f"invariante: los decimales iguales salen {cuentan} y deberían salir (9, 10, 8, 0)")
    print(f"[3] invariante        atajo contra fuerza bruta, diferencia máxima {peor:.1e}; la media de las "
          f"trece edades y la culpa del programa difieren en {dif_media:.1e}; el terreno en el peso de la "
          f"red da el error de la red: {'sí' if dif_terreno == 0 else 'NO'}")
    if peor > TOL_BRUTA or dif_media > 1e-12 or dif_terreno != 0:
        fallos.append("invariante: atajo y fuerza bruta, media de las trece o terreno no cuadran")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


# ---------------------------------------------------------------- main

EMITIR = None


def main():
    global EMITIR
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

    EMITIR = emitir
    emitir(["######## la culpa, en la red del colegio (Orden 20) ########",
            f"fecha: {date.today().isoformat()}",
            f"máquina: {platform.system()} {platform.machine()}",
            f"python {platform.python_version()}, numpy {np.__version__}",
            "GPU: no se usa (todo en numpy). Nada de lo que sigue",
            "depende del ordenador: aciertos, recuentos y pesos.", "",
            f"La red del colegio: semilla {SEMILLA_RED}, la edad en años,",
            f"{miles(PASOS_ENTRENADO)} pasos, poquito {coma(POQUITO, 1)}, pesos de arranque entre 10.", ""])
    red = entrenada()
    bloque_a(red)
    red1 = red_en_el_paso(PASO_DEL_CASO)
    p = int(np.argmax(np.abs(red1.adelante(X)[-1].ravel() - Y)))
    emitir([f"(la edad del ejemplo es la que más falla en ese paso: {int(EDADES[p])} años)", ""])
    c, m = bloque_b(red1, p)
    gW, gb, por_edad, media = bloque_c(red1, p)
    xs, ys, camino = terreno(red)
    bloque_d(red, xs, ys, camino)
    mal = trayectoria()
    resultados = las_200()
    resumen = bloque_e(resultados, mal)
    bloque_f(c, m)
    bloque_g(red)
    guardar_datos(red, red1, c, m, p, xs, ys, camino, por_edad, media, resumen)
    print(f"escrito {SALIDA_DATOS}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
