#!/usr/bin/env python3
"""
Capítulo 4 — la culpa, hecha a mano, sobre la red de la figura del pasillo (L24).

El capítulo contaba el reparto de la culpa hacia atrás con palabras y sin un solo número
(hallazgos A11 a A22 de `notas/auditoria/L24-IMAGINAR.md`). Este programa hace el caso
entero, con números, sobre la misma red que dibuja la figura del capítulo: la que resuelve
el o exclusivo con una neurona que pregunta «¿hay al menos uno subido?» y otra que pregunta
«¿están los dos subidos?» (`figura_apilar.red_representativa`, semilla y pasos en sus
constantes y en las de `que_inventan_las_capas.py`).

Imprime, por bloques:
  1. lo que dice la red entrenada en las cuatro posiciones, y lo que falla;
  2. los pesos y los listones que aprendió;
  3. la cuenta de cada neurona en cada posición (lo que le llega, su listón, lo que dice);
  4. un ejemplo a medio entrenar, hacia delante y hacia atrás: lo que falla la final, su
     culpa, cuánta le toca a cada una de las dos de en medio y por qué, la culpa de cada
     peso, y la comprobación a lo bruto de uno de ellos;
  5. el paso que aprende: la culpa de cada peso con las cuatro posiciones, el poquito, y el
     peso antes y después (comprobado contra el entrenamiento de verdad);
  6. lo que deja pasar la rampa corta y lo que deja pasar el codo (L24: antes «la que corta»);
  7. la culpa capa a capa en la red de cinco capas en medio (la de «como en los 80»);
  8. el terreno del error: un peso movido solo, y las redes que se atascan.

Las palabras, en el libro:
  - «lo que falla»: lo que dice la red menos lo que tendría que decir;
  - «el error» de un ejemplo: lo que falla, multiplicado por sí mismo; el de la red, la media
    de las cuatro posiciones (es el que minimiza `retropropagacion.Red.entrenar`);
  - «la culpa» de un peso: cuánto sube el error por cada unidad que sube ese peso; se mide
    moviéndolo un pelín, y el atajo la da sin moverlo;
  - «lo que deja pasar» la rampa: cuánto se mueve lo que dice una neurona por cada unidad que
    se mueve lo que le llega.

Uso:
    python culpa_hacia_atras.py
    python culpa_hacia_atras.py --selftest
"""

# ======================= CONSTANTES =======================

SEMILLA_RED = 318265456      # la que elige figura_apilar.red_representativa (el selftest lo comprueba)
PASO_DEL_CASO = 2_500        # a media bajada: el error ya ha empezado a caer y la red aún falla las cuatro
PELIN = 1e-6                 # lo que se sube lo que entra para medir lo que deja pasar la rampa
PELIN_BRUTO = 1e-5           # lo que se mueve un peso a lo bruto: el epsilon de retropropagacion.py
PUNTOS_RAMPA = [-6, -4, -2, -0.5, 0, 0.5, 2, 4, 6]   # «cuánto pasa del listón», para la tabla de las dos
PROFUNDIDAD_CADENA = 5       # capas en medio de la red en la que se sigue la culpa capa a capa
EN_LETRA = {6: "seis"}       # la nota de la tabla de la cadena dice «la red de seis capas» (Orden 97)
CAPA_DE_LA_SUMA = 3          # en esa red, la capa de en medio de la neurona cuya culpa se suma a mano
CUANTAS_SE_ENSENAN = 5       # de las dieciséis de encima, cuántas se enseñan una a una
KILOMETRO_EN_MM = 1_000_000  # para poner a escala el factor de veinte capas
# El terreno se dibuja sobre la red ENTRENADA: moviendo un solo peso, el fondo es el sitio
# donde el entrenamiento lo dejó, y se ve entero. (En la red del paso 2.500 la curva de
# cualquier peso suelto es casi plana alrededor de donde está: no enseña nada.)
PESO_DEL_TERRENO = (1, 0, 0) # la línea de la primera a la final: (capa, de, a) en Red.W
TERRENO_DESDE, TERRENO_HASTA, TERRENO_PUNTOS = -6.0, 16.0, 221
INICIO_CUESTA_ABAJO = 2.0    # se saca el peso de su sitio y se pone aquí, en la ladera
PASOS_CUESTA_ABAJO = 200     # pasos moviendo solo ese peso
CADA_CUANTOS = 25            # cada cuántos pasos se imprime (y se dibuja) dónde va
PESOS_DEL_MAPA = ((1, 0, 0), (1, 1, 0))   # las dos líneas que llegan a la final
MAPA_DESDE, MAPA_HASTA, MAPA_PUNTOS = -16.0, 16.0, 129
PASOS_EXTRA_ATASCADAS = 100_000   # a las que se atascan, cuántos pasos más se les dan
TOL_BRUTA = 1e-6             # culpa por el atajo y a lo bruto: tienen que coincidir hasta aquí

POSICIONES = ["ninguno subido", "solo el de arriba", "solo el de abajo", "los dos subidos"]
ENTRADAS = ["el de abajo", "el de arriba"]          # el orden de las columnas de TABLA_XOR
NEURONAS = ["la primera", "la segunda"]
PREGUNTAS = ["¿hay al menos uno subido?", "¿están los dos subidos?"]

SALIDA_DATOS = "../datos/salidas/culpa_hacia_atras_datos.csv"   # para las figuras

# ==========================================================

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from formato import coma, miles, tabla_editorial
from retropropagacion import Red, TABLA_XOR, Y_XOR, sigmoide, SEMILLA, ANCHO
from que_inventan_las_capas import SEMILLA_BASE, TASA, PASOS, OCULTAS, ARRANQUES

AQUI = Path(__file__).resolve().parent
POQUITO = TASA               # lo que el entrenamiento resta de cada peso por cada unidad de culpa


def comprobar_entrada():
    assert [tuple(f) for f in TABLA_XOR] == [(0., 0.), (0., 1.), (1., 0.), (1., 1.)], \
        f"Se esperaba el orden ninguno / solo el de arriba / solo el de abajo / los dos; hay {TABLA_XOR}"
    assert list(Y_XOR) == [0., 1., 1., 0.], f"Se esperaba el o exclusivo; hay {list(Y_XOR)}"
    assert OCULTAS == 2, f"Se esperaban dos neuronas en medio; hay {OCULTAS}"
    assert 0 < PASO_DEL_CASO < PASOS, f"El paso del caso tiene que estar dentro del entrenamiento"


# ---------------------------------------------------------------- la red y su entrenamiento

def un_paso(red):
    """Un paso de aprendizaje, exactamente el de Red.entrenar."""
    gW, gb = red.gradiente_retro(TABLA_XOR, Y_XOR)
    for k in range(len(red.W)):
        red.W[k] -= TASA * gW[k]
        red.b[k] -= TASA * gb[k]


def red_en_el_paso(n, semilla=SEMILLA_RED):
    red = Red([2, OCULTAS, 1], semilla=semilla)
    for _ in range(n):
        un_paso(red)
    return red


def copia(red):
    otra = Red(list(red.tamanos), semilla=0)
    otra.W = [w.copy() for w in red.W]
    otra.b = [b.copy() for b in red.b]
    return otra


def rampa(x):
    return sigmoide(x)


def el_codo(x):
    """Lo que pasa del listón, tal cual; si no llega, cero. (Se llamaba «la que corta»; chocaba
    con «la rampa corta», y se cambió en la segunda vuelta de L24.)"""
    return np.maximum(0.0, x)


def deja_pasar(f, x, pelin=PELIN):
    """Cuánto se mueve lo que sale por cada unidad que se mueve lo que entra: se sube lo que
    entra un pelín y se mira cuánto sube lo que sale. A lo bruto, sin ninguna cuenta."""
    return float((f(x + pelin) - f(x)) / pelin)


# ---------------------------------------------------------------- el caso, a mano

def caso(red, p):
    """Un ejemplo, hacia delante y hacia atrás, con cada número con su nombre."""
    x = TABLA_XOR[p]
    W0, b0, W1, b1 = red.W[0], red.b[0], red.W[1][:, 0], red.b[1][0]
    c = {"x": x, "deberia": Y_XOR[p]}
    c["llega_medio"] = x @ W0                   # lo que le llega a cada una de en medio
    c["liston_medio"] = -b0                     # su listón: el total tiene que pasar de ahí
    c["pasa_medio"] = c["llega_medio"] - c["liston_medio"]
    c["dice_medio"] = rampa(c["pasa_medio"])
    c["llega_final"] = c["dice_medio"] @ W1
    c["liston_final"] = -b1
    c["pasa_final"] = c["llega_final"] - c["liston_final"]
    c["dice_final"] = float(rampa(c["pasa_final"]))
    c["falla"] = c["dice_final"] - c["deberia"]
    c["error"] = c["falla"] ** 2
    # hacia atrás
    c["error_por_salida"] = 2 * c["falla"]      # si lo que dice sube un pelín, el error sube esto por pelín
    c["pasa_rampa_final"] = deja_pasar(rampa, c["pasa_final"])
    c["culpa_final"] = c["error_por_salida"] * c["pasa_rampa_final"]
    c["culpa_lineas_final"] = c["culpa_final"] * c["dice_medio"]
    c["le_toca_medio"] = c["culpa_final"] * W1                    # por el peso de su línea
    c["pasa_rampa_medio"] = np.array([deja_pasar(rampa, z) for z in c["pasa_medio"]])
    c["culpa_medio"] = c["le_toca_medio"] * c["pasa_rampa_medio"]
    c["culpa_pesos_medio"] = np.outer(x, c["culpa_medio"])       # [entrada, neurona]
    return c


def error_de_un_ejemplo(red, p):
    return float((red.adelante(TABLA_XOR[p:p + 1])[-1].ravel()[0] - Y_XOR[p]) ** 2)


def culpa_a_lo_bruto(red, p, capa, i, j, pelin=PELIN_BRUTO):
    """Mover un peso un pelín hacia arriba y hacia abajo y ver cuánto cambia el error de ese
    ejemplo: lo mismo que hace `retropropagacion.Red.gradiente_fuerza_bruta`, peso a peso."""
    otra = copia(red)
    otra.W[capa][i, j] += pelin
    mas = error_de_un_ejemplo(otra, p)
    otra.W[capa][i, j] -= 2 * pelin
    menos = error_de_un_ejemplo(otra, p)
    return mas, menos, (mas - menos) / (2 * pelin)


def culpas_por_posicion(red):
    """La culpa de cada peso en cada una de las cuatro posiciones, por el atajo."""
    salida = []
    for p in range(4):
        gW, _ = red.gradiente_retro(TABLA_XOR[p:p + 1], Y_XOR[p:p + 1])
        salida.append(gW)
    return salida


# ---------------------------------------------------------------- impresión

def tabla(*bloques):
    """Imprime tablas de libro (formato.py, tabla_editorial; REGLAS 6 ter). Desde el 9 oct 2026
    todas las cuentas de este capítulo salen así: cada número en su casilla, con su rótulo, en vez
    de frases con varias cifras seguidas. Las cuentas y las comprobaciones no cambian."""
    for b in bloques:
        print("\n".join(b))
        print()


def titulo(texto):
    print(texto)
    print()


def c3(x):
    """Tres decimales, con signo si es negativo."""
    return coma(x, 3)


def firmado(x, d=2):
    return ("+" if x >= 0 else "-") + coma(abs(x), d)


def bloque_1(red):
    act = red.adelante(TABLA_XOR)
    dice = act[-1].ravel()
    titulo("1. LO QUE DICE LA RED ENTRENADA EN LAS CUATRO POSICIONES")
    tabla(tabla_editorial(
        "Lo que dice la red entrenada en las cuatro posiciones",
        ["posición", "la luz tiene que estar", "la red dice", "lo que falla"],
        [[POSICIONES[p], "encendida" if Y_XOR[p] else "apagada", c3(dice[p]),
          firmado(dice[p] - Y_XOR[p], 3)] for p in range(4)], "iidd",
        ["La red dice un número de 0 a 1: cerca de 0 es «no» y cerca de 1 es «sí».",
         "Lo que falla: lo que dice menos lo que tendría que decir."]))
    return dice


def bloque_2(red):
    W0, b0, W1, b1 = red.W[0], red.b[0], red.W[1][:, 0], red.b[1][0]
    filas = []
    for j in range(2):
        for i in range(2):
            filas.append([NEURONAS[j] if i == 0 else "", ENTRADAS[i], firmado(W0[i, j]),
                          coma(-b0[j], 2) if i == 0 else ""])
    for j in range(2):
        filas.append(["la final" if j == 0 else "", NEURONAS[j], firmado(W1[j]),
                      coma(-b1, 2) if j == 0 else ""])
    titulo("2. LOS PESOS Y LOS LISTONES QUE APRENDIÓ")
    tabla(tabla_editorial(
        "Los pesos y los listones que aprendió",
        ["neurona", "la línea que viene de", "peso", "listón"], filas, "iidd",
        ["Peso de cada línea: positivo suma, negativo resta."]))


def bloque_3(red):
    titulo("3. LA CUENTA DE CADA NEURONA, POSICIÓN A POSICIÓN")
    nota = ("Le llega: la suma de lo que trae cada línea por su peso. Dice: cerca de 1 si le llega "
            "más que su listón, cerca de 0 si le llega menos.")
    T = []
    for j in range(2):
        filas = []
        for p in range(4):
            c = caso(red, p)
            filas.append([POSICIONES[p], coma(c["llega_medio"][j], 2),
                          "sí" if c["pasa_medio"][j] > 0 else "no", coma(c["dice_medio"][j], 2)])
        T.append(tabla_editorial(
            f"{NEURONAS[j].capitalize()}, «{PREGUNTAS[j]}»",
            ["posición", "le llega", "¿pasa del listón?", "dice"], filas, "idcd",
            [f"Listón: {coma(-red.b[0][j], 2)}. " + nota]))
    filas = []
    for p in range(4):
        c = caso(red, p)
        filas.append([POSICIONES[p], coma(c["llega_final"], 2),
                      "sí" if c["pasa_final"] > 0 else "no", coma(c["dice_final"], 2)])
    T.append(tabla_editorial(
        "La final, que escucha a las dos", ["posición", "le llega", "¿pasa del listón?", "dice"],
        filas, "idcd", [f"Listón: {coma(-red.b[1][0], 2)}. " + nota]))
    tabla(*T)
    # la regla de la final, en palabras, comprobada en las cuatro esquinas del cuadrado nuevo
    W1, b1 = red.W[1][:, 0], red.b[1][0]
    esquinas = {(a, b): (a * W1[0] + b * W1[1] + b1) > 0 for a in (0, 1) for b in (0, 1)}
    regla = [k for k, v in esquinas.items() if v]
    nombre = {0: "no", 1: "sí"}
    tabla(tabla_editorial(
        "La final, si las dos de la primera capa dijeran un sí o un no secos",
        ["la primera", "la segunda", "le llega", "¿pasa?"],
        [[nombre[a], nombre[b], firmado(a * W1[0] + b * W1[1], 2), "sí" if v else "no"]
         for (a, b), v in esquinas.items()], "ccdc",
        [f"Aquí «sí» vale 1 y «no» vale 0. Para pasar, a la final le tienen que llegar más de "
         f"{coma(-b1, 2)}, su listón."]))
    return regla


def r4(x):
    """Cuatro cifras significativas: el número tal como se imprime."""
    s = f"{x:.4g}"
    assert "e" not in s, f"número demasiado pequeño para imprimirlo sin exponente: {s}"
    return float(s)


def s4(x, signo=False):
    s = f"{abs(r4(x)):.4g}".replace(".", ",")
    if r4(x) < 0:
        return "-" + s
    return ("+" + s) if signo else s


def cadena_a_mano(red, c):
    """El caso hacia atrás, hecho como lo haría el lector con los números que se imprimen.

    Cada cuenta se hace con los números ya redondeados a cuatro cifras, y su resultado se
    redondea igual: así cualquier multiplicación del libro se puede repetir con el dedo y da lo
    mismo (hallazgo 19 de la verificación de L24). El selftest y los asserts de aquí comprueban
    que cada número de la cadena está a menos del 1 % del que da el programa sin redondear.
    «Lo que deja pasar la rampa» se calcula con la regla de la rampa, que es de multiplicar:
    lo que dice, por lo que le falta para 1."""
    W1 = [round(float(w), 2) for w in red.W[1][:, 0]]          # como en la tabla hacia delante
    m = {"pesos_final": W1}
    m["dice_final"] = r4(c["dice_final"])
    m["falla"] = r4(m["dice_final"] - c["deberia"])
    m["doble"] = r4(2 * m["falla"])
    m["falta_final"] = r4(1 - m["dice_final"])
    m["pasa_final"] = r4(m["dice_final"] * m["falta_final"])
    m["culpa_final"] = r4(m["doble"] * m["pasa_final"])
    m["dice_medio"] = [r4(v) for v in c["dice_medio"]]
    m["lineas_final"] = [r4(m["culpa_final"] * d) for d in m["dice_medio"]]
    m["le_toca"] = [r4(m["culpa_final"] * w) for w in W1]
    m["falta_medio"] = [r4(1 - d) for d in m["dice_medio"]]
    m["pasa_medio"] = [r4(d * f) for d, f in zip(m["dice_medio"], m["falta_medio"])]
    m["culpa_medio"] = [r4(a * b) for a, b in zip(m["le_toca"], m["pasa_medio"])]
    m["lineas_primera"] = [[r4(m["culpa_medio"][j] * c["x"][i]) for j in range(2)] for i in range(2)]
    exactos = [(m["culpa_final"], c["culpa_final"]), (m["pasa_final"], c["pasa_rampa_final"])]
    exactos += list(zip(m["lineas_final"], c["culpa_lineas_final"]))
    exactos += list(zip(m["le_toca"], c["le_toca_medio"]))
    exactos += list(zip(m["pasa_medio"], c["pasa_rampa_medio"]))
    exactos += list(zip(m["culpa_medio"], c["culpa_medio"]))
    for a, b in exactos:
        assert abs(a - b) <= 0.01 * abs(b) + 1e-9, f"la cadena a mano se aleja del programa: {a} y {b}"
    return m


def bloque_4(red, p):
    c = caso(red, p)
    m = cadena_a_mano(red, c)
    W1 = m["pesos_final"]
    deberia = int(c["deberia"])
    print(f"4. UN EJEMPLO A MEDIO ENTRENAR: EL PASO {miles(PASO_DEL_CASO)} DE {miles(PASOS)}")
    print()
    filas = []
    for j in range(2):
        filas.append([NEURONAS[j], coma(c["llega_medio"][j], 2), coma(c["liston_medio"][j], 2),
                      coma(c["pasa_medio"][j], 2), s4(m["dice_medio"][j])])
    filas.append(["la final", coma(c["llega_final"], 2), coma(c["liston_final"], 2),
                  coma(c["pasa_final"], 2), s4(m["dice_final"])])
    contexto = (f"Paso {miles(PASO_DEL_CASO)} de {miles(PASOS)}. Posición: {POSICIONES[p]}; la luz "
                f"tiene que estar {'encendida' if c['deberia'] else 'apagada'}, así que la final "
                f"tendría que decir {deberia}.")
    dos = [r4(m["dice_medio"][j] * W1[j]) for j in range(2)]
    tabla(
        tabla_editorial(
            "Hacia delante", ["neurona", "le llega", "listón", "pasa del listón por", "dice"],
            filas, "idddd", [contexto]),
        tabla_editorial(
            "Lo que le llega a la final",
            ["viene de", "lo que dice", "por el peso de su línea", "da"],
            [[NEURONAS[j], s4(m["dice_medio"][j]), coma(W1[j], 2), s4(dos[j])] for j in range(2)]
            + [["**la suma**", "", "", f"**{coma(c['llega_final'], 2)}**"]], "iddd",
            ["La suma, redondeada a dos decimales, como en la tabla anterior."]),
        tabla_editorial(
            "Lo que falla, y el error", ["", "cuánto"],
            [["lo que dice la final", s4(m["dice_final"])],
             ["lo que tendría que decir", str(deberia)],
             ["lo que falla: lo que dice menos lo que tendría que decir", s4(m["falla"], True)],
             ["**el error en esta posición: lo que falla por sí mismo**",
              f"**{s4(r4(m['falla']) ** 2)}**"]], "id",
            ["Los números, redondeados a cuatro cifras, y cada cuenta hecha con los números "
             "redondeados."]))

    s = c["dice_final"]
    e_mas = (s + PELIN - c["deberia"]) ** 2
    dice_mas = float(rampa(c["pasa_final"] + PELIN))
    print("HACIA ATRÁS")
    print()
    tabla(
        tabla_editorial(
            "Tramo 1, medido: cuánto se mueven el error y lo que dice la final",
            ["qué se mide", "tal cual", "una millonésima más arriba", "sube, en millonésimas"],
            [["el error, si se mueve lo que dice", coma(c["error"], 9), coma(e_mas, 9),
              coma((e_mas - c["error"]) / PELIN, 3)],
             ["lo que dice, si se mueve lo que le llega", coma(c["dice_final"], 9),
              coma(dice_mas, 9), coma((dice_mas - c["dice_final"]) / PELIN, 4)]], "iddd",
            ["Lo que sube el error es el doble de lo que falla; lo que sube lo que dice es lo que "
             "deja pasar su rampa. La tabla siguiente lo saca con cuentas."]),
        tabla_editorial(
            "Tramo 1: la culpa de la final", ["", "cuánto"],
            [["lo que falla", s4(m["falla"])],
             ["el doble de lo que falla", s4(m["doble"])],
             ["lo que dice", s4(m["dice_final"])],
             ["lo que le falta para 1", s4(m["falta_final"])],
             ["lo que deja pasar su rampa: lo que dice, por lo que le falta para 1",
              s4(m["pasa_final"])],
             ["**la culpa de la final: el doble de lo que falla, por lo que deja pasar su rampa**",
              f"**{s4(m['culpa_final'])}**"]], "id",
            ["Cada cuenta está hecha con los números redondeados a cuatro cifras, para que se "
             "pueda repetir con el dedo."]),
        tabla_editorial(
            "Tramo 2: las dos líneas que llegan a la final",
            ["la línea que viene de", "la culpa de la final", "por lo que trajo esa línea", "da"],
            [[NEURONAS[j], s4(m["culpa_final"]), s4(m["dice_medio"][j]),
              s4(m["lineas_final"][j])] for j in range(2)], "iddd",
            ["Lo que trae cada línea es lo que dice la neurona de la que sale."]),
        tabla_editorial(
            "Tramo 3: cuánta le toca a cada una de las dos de la primera capa",
            ["", NEURONAS[0], NEURONAS[1]],
            [["la culpa de la final", s4(m["culpa_final"]), s4(m["culpa_final"])],
             ["por el peso de su línea", firmado(W1[0], 2), firmado(W1[1], 2)],
             ["le toca", s4(m["le_toca"][0]), s4(m["le_toca"][1])],
             ["lo que dice", s4(m["dice_medio"][0]), s4(m["dice_medio"][1])],
             ["lo que le falta para 1", s4(m["falta_medio"][0]), s4(m["falta_medio"][1])],
             ["lo que deja pasar su rampa", s4(m["pasa_medio"][0]), s4(m["pasa_medio"][1])],
             ["**su culpa: lo que le toca, por lo que deja pasar su rampa**",
              f"**{s4(m['culpa_medio'][0])}**", f"**{s4(m['culpa_medio'][1])}**"]], "idd",
            ["Lo que deja pasar la rampa de cada una es lo que dice, por lo que le falta para 1."]),
        tabla_editorial(
            "Tramo 4: las cuatro líneas de la primera capa",
            ["línea", "culpa de su neurona", "por lo que trajo", "da"],
            [[f"{NEURONAS[j]}, del {ENTRADAS[i][3:]}", s4(m["culpa_medio"][j]),
              str(int(c["x"][i])), s4(m["lineas_primera"][i][j])]
             for j in range(2) for i in range(2)], "iddd",
            ["Lo que trajo la línea: 1 si el interruptor está subido, 0 si está bajado."]))
    c["mano"] = m
    return c


def bloque_4_bruto(red, p, c):
    """La comprobación a lo bruto de los pesos: moverlos un pelín y ver cuánto cambia el error."""
    gW, _ = red.gradiente_retro(TABLA_XOR[p:p + 1], Y_XOR[p:p + 1])
    filas = []
    for (capa, i, j, nombre) in ((0, 1, 1, "la segunda, del de arriba"),
                                 (1, 1, 0, "la final, de la segunda")):
        mas, menos, bruta = culpa_a_lo_bruto(red, p, capa, i, j)
        atajo = float(gW[capa][i, j])
        filas.append((nombre, atajo, bruta, mas, menos))
        # Diez decimales y no más: de la undécima cifra en adelante, lo que sale a lo bruto
        # cambia de un ordenador a otro (redondeo), y el libro no imprime nada que dependa
        # de la máquina (regla 10).
        assert abs(atajo - bruta) < 1e-10, f"atajo y fuerza bruta difieren en {abs(atajo - bruta)}"
    nombre, atajo, bruta, mas, menos = filas[0]
    tabla(
        tabla_editorial(
            "La misma culpa, a lo bruto", ["peso", "por el atajo", "a lo bruto"],
            [[n, coma(a, 10), coma(b, 10)] for n, a, b, _, _ in filas], "idd",
            ["«A lo bruto» quiere decir sin atajo: se sube el peso una cienmilésima y se mira el "
             "error, se baja una cienmilésima y se mira otra vez, y la culpa es lo que va de un "
             "error a otro, dividido entre lo que se ha movido el peso. Las dos columnas "
             "coinciden en sus diez primeros decimales."]),
        tabla_editorial(
            f"El error, con {nombre} movido", ["el peso", "el error"],
            [["una cienmilésima más arriba", coma(mas, 10)],
             ["una cienmilésima más abajo", coma(menos, 10)]], "id",
            [f"El redondeo, en una calculadora de seis decimales: un tercio es "
             f"{coma(round(1/3, 6), 6)}, y por tres da {coma(3 * round(1/3, 6), 6)}."]))
    return filas


def bloque_5(red):
    """El paso de aprendizaje: con las cuatro posiciones, lo que hace de verdad el programa."""
    por_pos = culpas_por_posicion(red)
    gW, gb = red.gradiente_retro(TABLA_XOR, Y_XOR)
    siguiente = copia(red)
    un_paso(siguiente)
    media = float(np.mean([por_pos[p][0][1, 1] for p in range(4)]))
    assert abs(media - gW[0][1, 1]) < 1e-12, "la media de las cuatro no es la culpa del programa"
    filas = []
    for j in range(2):
        for i in range(2):
            filas.append((f"{NEURONAS[j]}, del {ENTRADAS[i][3:]}", 0, i, j))
    for j in range(2):
        filas.append((f"la final, de {NEURONAS[j]}", 1, j, 0))
    pasos = []
    for nombre, capa, i, j in filas:
        culpa = gW[capa][i, j]
        antes, despues = red.W[capa][i, j], siguiente.W[capa][i, j]
        assert abs((antes - POQUITO * culpa) - despues) < 1e-12, "el paso no es el del programa"
        pasos.append([nombre, firmado(culpa, 4), firmado(-POQUITO * culpa, 4), coma(antes, 4),
                      coma(despues, 4)])
    titulo("5. EL PASO: CADA PESO SE MUEVE CONTRA SU CULPA")
    tabla(
        tabla_editorial(
            "La culpa de la segunda, del de arriba, posición a posición", ["posición", "culpa"],
            [[POSICIONES[p], coma(por_pos[p][0][1, 1], 4)] for p in range(4)]
            + [["**la media**", f"**{coma(media, 4)}**"]], "id",
            ["La culpa de cada peso es la media de su culpa en las cuatro posiciones."]),
        tabla_editorial(
            "El paso: cada peso se mueve contra su culpa",
            ["peso", "culpa", "se mueve", "antes", "después"], pasos, "idddd",
            [("Cada peso se mueve la mitad de su culpa" if POQUITO == 0.5 else
              f"Cada peso se mueve {coma(POQUITO, 1)} veces su culpa")
             + ", en contra: si la culpa es negativa, sube; si es positiva, baja."]))
    return gW


def bloque_6():
    filas = []
    for x in PUNTOS_RAMPA:
        etiqueta = "0" if x == 0 else ("+" if x > 0 else "-") + coma(abs(x), 1 if x % 1 else 0)
        if x == 0:
            # justo en el listón el codo se dobla: por debajo no deja pasar nada y por encima,
            # todo. No se imprime un número que dependa de hacia qué lado se mide.
            codo_dice, codo_pasa = coma(0.0, 1), "(el codo)"
        else:
            codo_dice, codo_pasa = coma(float(el_codo(x)), 1), coma(deja_pasar(el_codo, x), 3)
        filas.append([etiqueta, coma(float(rampa(x)), 3), coma(deja_pasar(rampa, x), 3),
                      codo_dice, codo_pasa])
    xs = np.linspace(-10, 10, 20001)
    maximo = max(deja_pasar(rampa, float(x)) for x in xs)
    titulo("6. LO QUE DEJA PASAR CADA UNA")
    tabla(tabla_editorial(
        "Lo que deja pasar cada una",
        ["pasa del listón por", "la rampa corta: dice", "la rampa corta: deja pasar",
         "el codo: dice", "el codo: deja pasar"], filas, "ddddd",
        ["Deja pasar: cuánto se mueve lo que dice la neurona por cada unidad que se mueve lo que "
         "le llega; medido subiendo lo que le llega una millonésima.",
         f"Lo más que deja pasar la rampa corta, en cualquier sitio: {coma(maximo, 3)}, en el "
         "centro, donde pasa del listón por 0."]))
    return maximo


def bloque_7():
    """La culpa capa a capa, en la red de cinco capas en medio que imprime retropropagacion.py."""
    rng = np.random.default_rng(SEMILLA)
    red = None
    for prof in [2, 5, 10, 20]:            # el mismo orden de tiradas que medir_desvanecimiento
        r = Red([ANCHO] + [ANCHO] * prof + [1])
        X = rng.normal(size=(64, ANCHO)); y = (rng.random(64) > 0.5).astype(float)
        if prof == PROFUNDIDAD_CADENA:
            red, Xc, yc = r, X, y
    gW, _ = red.gradiente_retro(Xc, yc)
    m = [float(np.abs(g).mean()) for g in gW]
    n = len(m)
    filas = []
    for k in range(n - 1, -1, -1):
        nombre = (f"la {k + 1}.ª, la última" if k == n - 1 else
                  "la 1.ª, la primera" if k == 0 else f"la {k + 1}.ª")
        factor = "" if k == n - 1 else coma(m[k] / m[k + 1], 3)
        filas.append([nombre, coma(m[k], 7), factor])
    veces = m[-1] / m[0]
    factores = [r4(float(f"{m[k] / m[k + 1]:.3f}")) for k in range(n - 2, -1, -1)]
    producto = 1.0
    for f in factores:
        producto *= f
    titulo(f"7. LA CULPA, CAPA A CAPA, EN LA RED DE {PROFUNDIDAD_CADENA + 1} CAPAS")
    tabla(
        tabla_editorial(
            f"La culpa, capa a capa, en la red de {PROFUNDIDAD_CADENA + 1} capas",
            ["capa de líneas, de la final hacia la entrada", "culpa",
             "de cada 1 de la de encima, le llega"], filas, "idd",
            [f"La red de {EN_LETRA[PROFUNDIDAD_CADENA + 1]} capas, recién arrancada y armada como "
             "en los 80. La culpa de cada capa de líneas es la media de la de sus pesos, sin mirar "
             "el signo, y la última columna dice qué parte de la culpa de la capa de encima le "
             "llega a ésta."]),
        tabla_editorial(
            "De la última capa de líneas a la primera", ["", "cuánto"],
            [["la culpa de la última", coma(m[-1], 7)],
             ["la culpa de la primera", coma(m[0], 7)],
             ["**la última entre la primera: cuántas veces menos le llega**",
              f"**{miles(round(veces))}**"],
             ["los cinco números de la última columna, multiplicados", s4(producto)],
             ["uno entre ese producto", miles(round(1 / producto))]], "id",
            [f"Uno entre el producto da casi el {miles(round(veces))}: lo que falta es el "
             "redondeo de los cinco números."]))
    suma_de_encima(red, Xc[:1], yc[:1])
    return m


def suma_de_encima(red, X, y):
    """La culpa que le llega a una neurona desde las dieciséis de la capa de encima: cada una por
    el peso de su línea, sumadas, y luego por lo que deja pasar su rampa. Con un ejemplo, y la
    neurona de la 3.ª capa de en medio a la que más culpa le llega. Las cuentas, con los números
    impresos (cuatro cifras), como en el caso del pasillo."""
    act = red.adelante(X)
    d = (act[-1] - y.reshape(-1, 1)) * 2 * act[-1] * (1 - act[-1])
    deltas = [None] * len(red.W)
    deltas[-1] = d
    for k in range(len(red.W) - 1, 0, -1):
        d = (d @ red.W[k].T) * act[k] * (1 - act[k])
        deltas[k - 1] = d
    capa = CAPA_DE_LA_SUMA
    j = int(np.argmax(np.abs(deltas[capa - 1][0])))
    encima = deltas[capa][0]
    pesos = red.W[capa][j, :]
    orden = list(np.argsort(-np.abs(encima * pesos)))
    vistas, resto = orden[:CUANTAS_SE_ENSENAN], orden[CUANTAS_SE_ENSENAN:]
    filas = []
    total = 0.0
    for k in vistas:
        c, w = r4(float(encima[k])), round(float(pesos[k]), 3)
        v = r4(c * w)
        total += v
        filas.append([f"la {k + 1}.ª", s4(c), firmado(w, 3), s4(v)])
    otras = r4(float((encima[resto] * pesos[resto]).sum()))
    total += otras
    total = r4(total)
    dice = r4(float(act[capa][0, j]))
    pasa = r4(dice * r4(1 - dice))
    culpa = r4(total * pasa)
    exacto = float(deltas[capa - 1][0, j])
    assert abs(culpa - exacto) <= 0.02 * abs(exacto), f"la suma a mano se aleja: {culpa} y {exacto}"
    media = float(np.abs(encima).mean())
    filas.append([f"las otras {len(resto)}, juntas", "", "", s4(otras)])
    filas.append(["**la suma**", "", "", f"**{s4(total)}**"])
    tabla(
        tabla_editorial(
            f"Una neurona de la {capa}.ª capa, con un ejemplo",
            ["de la de encima", "su culpa", "por el peso", "da"], filas, "iddd",
            [f"La de su capa a la que más culpa le llega. Recibe culpa de las {len(encima)} "
             "neuronas de la capa de encima, cada una por el peso de la línea que las une."]),
        tabla_editorial(
            "Y la suma pasa por su rampa", ["", "cuánto"],
            [["la suma", s4(total)], ["lo que dice", s4(dice)],
             ["lo que le falta para 1", s4(r4(1 - dice))],
             ["lo que deja pasar su rampa: lo que dice, por lo que le falta", s4(pasa)],
             ["**su culpa: la suma, por lo que deja pasar**", f"**{s4(culpa)}**"],
             [f"culpa media de las {len(encima)} de encima, sin mirar el signo", s4(media)]],
            "id",
            ["Su culpa es más que la media de las de encima, porque las que llegan por sus "
             "líneas se suman."]))


def bloque_7_escala(factor_20):
    mm = KILOMETRO_EN_MM / factor_20
    tabla(tabla_editorial(
        "A escala", ["la culpa de la capa de líneas", "con 21 capas"],
        [["la última", "1 kilómetro"], ["la primera", f"{coma(mm, 3)} mm"]], "id",
        ["Si la culpa que le llega a la última capa de líneas fuera un kilómetro, esto es lo que "
         "le llegaría a la primera, en una red armada como en los 80."]))


def terreno(red):
    capa, i, j = PESO_DEL_TERRENO
    xs = np.linspace(TERRENO_DESDE, TERRENO_HASTA, TERRENO_PUNTOS)
    otra = copia(red)
    ys = []
    for v in xs:
        otra.W[capa][i, j] = v
        ys.append(otra.error(TABLA_XOR, Y_XOR))
    # los pasos cuesta abajo moviendo SOLO ese peso, desde un sitio de la ladera
    otra = copia(red)
    otra.W[capa][i, j] = INICIO_CUESTA_ABAJO
    camino = [(float(otra.W[capa][i, j]), otra.error(TABLA_XOR, Y_XOR))]
    for n in range(1, PASOS_CUESTA_ABAJO + 1):
        gW, _ = otra.gradiente_retro(TABLA_XOR, Y_XOR)
        otra.W[capa][i, j] -= POQUITO * gW[capa][i, j]
        camino.append((float(otra.W[capa][i, j]), otra.error(TABLA_XOR, Y_XOR)))
    # el mapa de dos pesos
    (c1, i1, j1), (c2, i2, j2) = PESOS_DEL_MAPA
    malla = np.linspace(MAPA_DESDE, MAPA_HASTA, MAPA_PUNTOS)
    mapa = np.zeros((MAPA_PUNTOS, MAPA_PUNTOS))
    otra = copia(red)
    for a, va in enumerate(malla):
        for b, vb in enumerate(malla):
            otra.W[c1][i1, j1] = va
            otra.W[c2][i2, j2] = vb
            mapa[b, a] = otra.error(TABLA_XOR, Y_XOR)
    return xs, np.array(ys), camino, malla, mapa


def atascadas():
    """Las redes de los 200 arranques que no resuelven el o exclusivo: qué error tienen, y qué
    pasa si se les dan PASOS_EXTRA_ATASCADAS pasos más."""
    semillas = [int(s) for s in
                np.random.default_rng(SEMILLA_BASE).integers(1, 2**31 - 1, size=ARRANQUES)]
    # Cada arranque es independiente y tiene su semilla: repartirlos entre los núcleos del
    # ordenador no cambia ningún número, solo lo que se tarda.
    from multiprocessing import Pool
    with Pool() as pool:
        filas = pool.map(un_arranque, semillas)
    buenas = [f[1] for f in filas if f[0] == "buena"]
    malas = [f[1:] for f in filas if f[0] == "mala"]
    return buenas, malas


def un_arranque(s):
    red = Red([2, OCULTAS, 1], semilla=s).entrenar(TABLA_XOR, Y_XOR, TASA, PASOS)
    sal = red.adelante(TABLA_XOR)[-1].ravel()
    if ((sal > 0.5) == (Y_XOR > 0.5)).all():
        return ("buena", red.error(TABLA_XOR, Y_XOR))
    antes = red.error(TABLA_XOR, Y_XOR)
    red.entrenar(TABLA_XOR, Y_XOR, TASA, PASOS_EXTRA_ATASCADAS)
    sal2 = red.adelante(TABLA_XOR)[-1].ravel()
    resuelta = bool(((sal2 > 0.5) == (Y_XOR > 0.5)).all())
    dudas = int((np.abs(sal2 - 0.5) < 0.05).sum())
    return ("mala", antes, red.error(TABLA_XOR, Y_XOR), resuelta, dudas)


def bloque_8(red, xs, ys, camino, malas, buenas):
    capa, i, j = PESO_DEL_TERRENO
    k = int(np.argmin(ys))
    titulo("8. EL TERRENO DEL ERROR, CON UN SOLO PESO")
    filas = []
    for n in range(0, len(camino), CADA_CUANTOS):
        v, e = camino[n]
        movido = "" if n == 0 else coma(v - camino[n - CADA_CUANTOS][0], 2)
        filas.append([str(n), coma(v, 2), coma(e, 4), movido])
    tabla(
        tabla_editorial(
            "El terreno del error, con un solo peso", ["dónde", "peso", "error"],
            [["donde lo dejó el entrenamiento", coma(red.W[capa][i, j], 2),
              coma(red.error(TABLA_XOR, Y_XOR), 4)],
             ["el fondo de esta curva", coma(xs[k], 1), coma(ys[k], 4)],
             ["un extremo", firmado(xs[0], 0), coma(ys[0], 4)],
             ["el otro extremo", firmado(xs[-1], 0), coma(ys[-1], 4)]], "idd",
            ["La red entrenada; se mueve solo la línea de la primera a la final, y los demás "
             "pesos se quedan quietos. El error es la media de las cuatro posiciones."]),
        tabla_editorial(
            f"Cuesta abajo, moviendo solo ese peso, desde {coma(INICIO_CUESTA_ABAJO, 1)}",
            ["paso", "peso", "error", f"lo que se movió en esos {CADA_CUANTOS} pasos"], filas,
            "dddd", [f"Cada paso mueve el peso {coma(POQUITO, 1)} veces su culpa, en contra."]))
    siguen = [m for m in malas if not m[2]]
    salen = [m for m in malas if m[2]]
    rango = lambda xs: (coma(min(xs), 4) if abs(min(xs) - max(xs)) < 5e-5
                        else f"{coma(min(xs), 4)} a {coma(max(xs), 4)}")
    assert salen, "se esperaba que alguna saliera con los pasos de más"
    tabla(
        tabla_editorial(
            f"Las que se atascan: {len(malas)} de {len(malas) + len(buenas)} arranques",
            ["", f"error tras {miles(PASOS)} pasos", f"con {miles(PASOS_EXTRA_ATASCADAS)} pasos más"],
            [[f"las {len(siguen)} que siguen", rango([m[0] for m in siguen]),
              rango([m[1] for m in siguen])],
             ["la que sale" if len(salen) == 1 else f"las {len(salen)} que salen",
              rango([m[0] for m in salen]), rango([m[1] for m in salen])],
             ["las que resuelven", rango(buenas), ""]], "idd",
            [f"Las que dicen entre 0,45 y 0,55, «no lo tengo claro», en dos de las cuatro "
             f"posiciones: {sum(1 for m in siguen if m[3] == 2)} de las {len(siguen)} que siguen."]),
        tabla_editorial(
            "El error de decir 0,5 en dos posiciones y acertar las otras dos", ["", "cuánto"],
            [["lo que falla en cada una de las dos", "0,5"],
             ["por sí mismo", coma(0.25, 2)],
             ["dos veces", coma(0.5, 1)],
             ["**entre cuatro posiciones: el error**", f"**{coma(0.125, 3)}**"]], "id",
            ["En las otras dos acierta: no falla nada."]))


def guardar_datos(red_final, red_caso, c, xs, ys, camino, malla, mapa):
    """Lo que necesitan las figuras, en un fichero de datos al lado de la salida."""
    with open(AQUI / SALIDA_DATOS, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["que", "a", "b", "valor"])
        for nombre, red in (("final", red_final), ("caso", red_caso)):
            for i in range(2):
                for j in range(2):
                    w.writerow([f"{nombre}_peso_medio", i, j, repr(float(red.W[0][i, j]))])
            for j in range(2):
                w.writerow([f"{nombre}_liston_medio", j, "", repr(float(-red.b[0][j]))])
                w.writerow([f"{nombre}_peso_final", j, "", repr(float(red.W[1][j, 0]))])
            w.writerow([f"{nombre}_liston_final", "", "", repr(float(-red.b[1][0]))])
        act = red_final.adelante(TABLA_XOR)
        for p in range(4):
            for j in range(2):
                w.writerow(["final_dice_medio", p, j, repr(float(act[1][p, j]))])
            w.writerow(["final_dice", p, "", repr(float(act[2][p, 0]))])
        for clave in ("culpa_final", "dice_final", "falla"):
            w.writerow([f"caso_{clave}", "", "", repr(float(c[clave]))])
        for j in range(2):
            for clave in ("dice_medio", "le_toca_medio", "culpa_medio", "pasa_rampa_medio"):
                w.writerow([f"caso_{clave}", j, "", repr(float(c[clave][j]))])
            w.writerow(["caso_culpa_lineas_final", j, "", repr(float(c["culpa_lineas_final"][j]))])
            for i in range(2):
                w.writerow(["caso_culpa_pesos_medio", i, j, repr(float(c["culpa_pesos_medio"][i, j]))])
        w.writerow(["caso_pasa_rampa_final", "", "", repr(float(c["pasa_rampa_final"]))])
        # la cadena tal como se imprime (cuatro cifras): es la que dibuja la figura
        m = c["mano"]
        for clave in ("dice_final", "falla", "culpa_final", "pasa_final"):
            w.writerow([f"mano_{clave}", "", "", repr(m[clave])])
        for j in range(2):
            for clave in ("dice_medio", "le_toca", "pasa_medio", "culpa_medio", "lineas_final"):
                w.writerow([f"mano_{clave}", j, "", repr(m[clave][j])])
            for i in range(2):
                w.writerow(["mano_lineas_primera", i, j, repr(m["lineas_primera"][i][j])])
        for x, y in zip(xs, ys):
            w.writerow(["terreno", repr(float(x)), "", repr(float(y))])
        for n, (v, e) in enumerate(camino):
            w.writerow(["camino", n, repr(v), repr(e)])
        for a, va in enumerate(malla):
            for b, vb in enumerate(malla):
                w.writerow(["mapa", repr(float(va)), repr(float(vb)), repr(float(mapa[b, a]))])


# ============================ SELFTEST ============================

def selftest():
    fallos = []
    comprobar_entrada()

    # [1] TEST NULO — en una posición que la red ya acierta del todo (lo que falla es cero), no
    #     hay culpa que repartir: todos los pesos tienen que salir con culpa cero. Se fabrica
    #     cambiando lo que «tendría que decir» por lo que ya dice.
    red = red_en_el_paso(PASO_DEL_CASO)
    s = red.adelante(TABLA_XOR[3:4])[-1].ravel()
    gW, gb = red.gradiente_retro(TABLA_XOR[3:4], s)
    nulo = max(float(np.abs(g).max()) for g in gW + gb)
    print(f"[1] test nulo         si no falla, la mayor culpa de un peso es {nulo:.1e} (tiene que ser 0)")
    if nulo != 0.0:
        fallos.append(f"test nulo: sin fallo sale culpa {nulo}")

    # [2] SEÑAL IMPLANTADA — la red es la de la figura: la que figura_apilar elige, con su
    #     reparto, y la cuenta a mano del caso da lo mismo que el atajo del programa.
    from figura_apilar import red_representativa
    rr, _, reglas, semilla = red_representativa()
    print(f"[2] señal implantada  la red de la figura es la de la semilla {semilla}: "
          f"{'sí' if semilla == SEMILLA_RED else 'NO'}")
    if semilla != SEMILLA_RED:
        fallos.append(f"señal: la figura usa la semilla {semilla} y este programa {SEMILLA_RED}")
    final = red_en_el_paso(PASOS)
    if not all(np.allclose(a, b) for a, b in zip(final.W + final.b, rr.W + rr.b)):
        fallos.append("señal: la red entrenada aquí no es la de la figura")
    p = int(np.argmax(np.abs(red.adelante(TABLA_XOR)[-1].ravel() - Y_XOR)))
    c = caso(red, p)
    gW, _ = red.gradiente_retro(TABLA_XOR[p:p + 1], Y_XOR[p:p + 1])
    dif = max(float(np.abs(gW[0] - c["culpa_pesos_medio"]).max()),
              float(np.abs(gW[1][:, 0] - c["culpa_lineas_final"]).max()))
    print(f"[2] señal implantada  la cuenta a mano del caso y el atajo del programa difieren en "
          f"{dif:.1e}")
    if dif > TOL_BRUTA:
        fallos.append(f"señal: la cuenta a mano y el atajo difieren en {dif}")

    # [3] INVARIANTE DEL DOMINIO — la culpa por el atajo es la de mover el peso un pelín, en
    #     los seis pesos del caso; y lo que deja pasar la rampa no pasa nunca de una cuarta parte.
    peor = 0.0
    for capa, i, j in [(0, a, b) for a in range(2) for b in range(2)] + [(1, 0, 0), (1, 1, 0)]:
        otra = copia(red)
        otra.W[capa][i, j] += PELIN
        mas = error_de_un_ejemplo(otra, p)
        otra.W[capa][i, j] -= 2 * PELIN
        menos = error_de_un_ejemplo(otra, p)
        peor = max(peor, abs((mas - menos) / (2 * PELIN) - gW[capa][i, j]))
    maximo = max(deja_pasar(rampa, float(x)) for x in np.linspace(-10, 10, 2001))
    print(f"[3] invariante        atajo contra fuerza bruta, diferencia máxima {peor:.1e}; "
          f"la rampa deja pasar como mucho {coma(maximo, 4)}")
    if peor > TOL_BRUTA:
        fallos.append(f"invariante: atajo y fuerza bruta difieren en {peor}")
    if maximo > 0.25 + 1e-6:
        fallos.append(f"invariante: la rampa deja pasar {maximo}, más de una cuarta parte")

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
    comprobar_entrada()
    if args.selftest:
        return selftest()
    print("########## capítulo 4: la culpa, hecha a mano (L24) ##########")
    print(f"la red de la figura del pasillo: semilla {SEMILLA_RED}, {miles(PASOS)} pasos,")
    print(f"poquito {coma(POQUITO, 1)}; nada de lo que sigue depende del ordenador")
    print()
    final = red_en_el_paso(PASOS)
    bloque_1(final)
    bloque_2(final)
    regla = bloque_3(final)
    assert regla == [(1, 0)], f"la final debería encender solo con «la primera sí, la segunda no»: {regla}"
    red = red_en_el_paso(PASO_DEL_CASO)
    p = int(np.argmax(np.abs(red.adelante(TABLA_XOR)[-1].ravel() - Y_XOR)))
    print(f"(la posición del ejemplo es la que más falla en ese paso)")
    print()
    c = bloque_4(red, p)
    bloque_4_bruto(red, p, c)
    bloque_5(red)
    bloque_6()
    bloque_7()
    from culpa_que_se_desvanece import factores, MONTAJES
    f20 = [f for f in factores(MONTAJES[0]) if f["capas"] == 20][0]["veces"]
    bloque_7_escala(f20)
    xs, ys, camino, malla, mapa = terreno(final)
    buenas, malas = atascadas()
    bloque_8(final, xs, ys, camino, malas, buenas)
    guardar_datos(final, red, c, xs, ys, camino, malla, mapa)
    print(f"escrito {SALIDA_DATOS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
