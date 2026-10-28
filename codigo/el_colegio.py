#!/usr/bin/env python3
"""
Capítulos 2, 3 y 4 — el colegio de primaria: un solo listón dice «de aquí para arriba» o «de aquí
para abajo»; «de aquí a aquí» pide dos (Orden 15, 18 oct 2026).

Un colegio admite de 6 a 12 años. Los datos son las edades de 3 a 15 años, una por año (13
casos): la respuesta correcta es sí de 6 a 12 (7 casos) y no en 3, 4, 5, 13, 14 y 15 (6 casos).
La entrada es la edad en años, tal cual. Este programa solo mide; el libro cuenta qué salió.

  a) EL PERCEPTRÓN, un comité con un miembro que mira la edad, con la regla de Rosenblatt de
     `perceptron.py` y un plazo de 100.000 correcciones, como con el pasillo: con «sí de 6 a 12»,
     si se queda quieto, cuántas edades acierta al acabar y cuántas como máximo en algún momento;
     con «sí de 6 en adelante» y con «sí hasta 12», si se queda quieto y con cuántas correcciones.
  b) LA RED de una entrada, dos neuronas en medio y una final, con los ajustes y la manera de
     sortear el arranque de la red del pasillo (se importan de `que_inventan_las_capas.py`):
     200 arranques, cuántos lo resuelven, qué acaba haciendo cada neurona de en medio, y «la red
     del capítulo» (la primera de la lista que lo resuelve con la pareja más frecuente). Si
     menos de 100 de los 200 lo resuelven con la edad en años, se repite con la edad en decenas
     (la edad entre 10) y se dan las dos salidas.
  c) LA CULPA HACIA ATRÁS, para la red del capítulo, como `culpa_hacia_atras.py` con el pasillo.
  d) LOS TRES COMITÉS A MANO (caps. 2 y 3): «¿tiene al menos seis?», «¿tiene como mucho doce?»
     y «¿los dos dicen sí?», con pesos y listones enteros, comprobados contra las 13 edades.

Las palabras, como en `culpa_hacia_atras.py`: «lo que falla» es lo que dice la red menos lo que
tendría que decir; «el error» de un ejemplo es lo que falla por sí mismo; «la culpa» de un peso es
cuánto sube el error por cada unidad que sube ese peso.

Cómo se cuenta «se queda quieto» y «acierta»: una edad está bien cuando la cuenta del perceptrón
(peso por edad, más el sesgo) tiene el signo de la respuesta y no es cero, que es la condición con
la que `perceptron.entrenar` corrige. Para contar cuántas acierta en cada momento hace falta ver
cada corrección, y `perceptron.entrenar` solo devuelve el final: aquí se repite su bucle, línea
por línea, con la misma semilla, y el selftest comprueba que acaba en los mismos pesos.

Uso:
    python el_colegio.py --selftest
    python el_colegio.py > ../datos/salidas/el_colegio.txt
"""

# ======================= CONSTANTES =======================

EDAD_MIN, EDAD_MAX = 3, 15          # las edades de los datos, una por año
DESDE, HASTA = 6, 12                # el colegio admite de 6 a 12 años, los dos incluidos
PLAZO = 100_000                     # correcciones antes de rendirse, como con el pasillo
MINIMO_EN_ANIOS = 100               # si menos de estas lo resuelven con la edad en años, se repite en decenas
DECENAS = 10.0                      # para la edad en decenas: la edad entre esto
PASO_DEL_CASO = 2_500               # a media bajada, como en culpa_hacia_atras.py
CADA_CUANTOS = 500                  # si en ese paso nada falla, se busca entre sus múltiplos
TOL_BRUTA = 1e-8                    # atajo y a lo bruto: tienen que coincidir en ocho decimales
# Los ajustes de la red (TASA = 0,5, PASOS = 20.000, UMBRAL = 0,5), el número de arranques (200) y
# SEMILLA_BASE de la que salen las semillas se importan de que_inventan_las_capas.py: son los
# de la red del pasillo, y la manera de sortear el arranque es la misma.

# Los tres comités a mano: (pregunta, de dónde le llega, pesos, listón). Dice sí si lo que le llega
# pasa del listón (más que el listón, no igual): es la misma condición con la que el perceptrón
# da una edad por buena.
COMITES = [
    ("¿tiene al menos seis?", "la edad", (1,), 5),
    ("¿tiene como mucho doce?", "la edad", (-1,), -13),
    ("¿los dos dicen sí?", "los otros dos", (1, 1), 1),
]

NEURONAS = ["la primera", "la segunda"]
EN_LETRA = {2: "dos", 7: "siete", 10: "diez", 13: "trece", 9: "nueve"}   # las notas lo dicen con letra (Orden 50)
ANCHO_SALIDA = 62                   # líneas de no más de 62 caracteres, fuera de las tablas (que se componen
                                    # con su propio ancho, como en las demás salidas)

# ==========================================================

import argparse
import platform
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from decimal import Decimal

import numpy as np

import perceptron
from culpa_hacia_atras import PELIN, PELIN_BRUTO, copia, deja_pasar, rampa
from formato import coma as _coma, comprobar_ancho, miles, tabla_editorial
from que_inventan_las_capas import ARRANQUES, PASOS, SEMILLA_BASE, TASA, UMBRAL
from retropropagacion import Red

EDADES = np.arange(EDAD_MIN, EDAD_MAX + 1, dtype=float)       # 13 edades
DEBERIA = ((EDADES >= DESDE) & (EDADES <= HASTA)).astype(float)   # 1 si el colegio la admite


def comprobar_entrada():
    assert len(EDADES) == 13, f"Se esperaban 13 edades; hay {len(EDADES)}"
    assert list(EDADES) == [float(e) for e in range(3, 16)], f"Se esperaban las edades 3 a 15; hay {EDADES}"
    assert int(DEBERIA.sum()) == 7 and int((1 - DEBERIA).sum()) == 6, \
        f"Se esperaban 7 síes y 6 noes; hay {int(DEBERIA.sum())} y {int((1 - DEBERIA).sum())}"
    assert [int(e) for e in EDADES[DEBERIA == 1]] == list(range(6, 13)), "Los síes tienen que ser de 6 a 12"
    assert set(np.unique(DEBERIA)) == {0.0, 1.0}, "Se esperaban solo ceros y unos como respuesta"
    assert 0 < PASO_DEL_CASO < PASOS and PASO_DEL_CASO % CADA_CUANTOS == 0, "El paso del caso está mal"
    assert ARRANQUES == 200 and PASOS == 20_000 and TASA == 0.5 and UMBRAL == 0.5, \
        f"Se esperaban los ajustes del pasillo; hay {ARRANQUES}, {PASOS}, {TASA}, {UMBRAL}"
    assert 0 < MINIMO_EN_ANIOS <= ARRANQUES, "El mínimo de redes resueltas tiene que caber en los arranques"


def coma(x, decimales=1):
    """Como formato.coma, pero un número que se redondea a cero no lleva signo («-0,00» no)."""
    t = _coma(x, decimales)
    return t[1:] if t.startswith("-") and set(t[1:]) <= set("0,") else t


def entrada(escala):
    """La columna de entrada de la red: la edad entre `escala` (1 = en años; 10 = en decenas)."""
    return (EDADES / escala).reshape(-1, 1)


# ---------------------------------------------------------------- a) el perceptrón

def etiquetas(regla):
    """Respuestas en {-1, 1} para el perceptrón. `regla`: 'de 6 a 12', 'de 6 en adelante', 'hasta 12'."""
    if regla == "de 6 a 12":
        si = (EDADES >= DESDE) & (EDADES <= HASTA)
    elif regla == "de 6 en adelante":
        si = EDADES >= DESDE
    elif regla == "hasta 12":
        si = EDADES <= HASTA
    else:
        raise AssertionError(f"Regla desconocida: {regla}")
    return np.where(si, 1, -1)


def entrenar_con_rastro(X, y, plazo=PLAZO, semilla=perceptron.SEMILLA):
    """El bucle de `perceptron.entrenar`, repetido línea por línea para ver cada corrección.
    Devuelve (w, b, correcciones o None, acierta_tras_cada_correccion, edades_mal_al_final)."""
    rng = np.random.default_rng(semilla)
    w = np.zeros(X.shape[1])
    b = 0.0
    orden = np.arange(len(X))
    pasos = 0
    aciertos = [int((y * (X @ w + b) > 0).sum())]          # antes de corregir nada
    for _ in range(plazo):
        rng.shuffle(orden)
        errores = 0
        for i in orden:
            if y[i] * (X[i] @ w + b) <= 0:
                w += perceptron.TASA * y[i] * X[i]
                b += perceptron.TASA * y[i]
                errores += 1
                pasos += 1
                aciertos.append(int((y * (X @ w + b) > 0).sum()))
                if pasos >= plazo:
                    return w, b, None, aciertos
        if errores == 0:
            return w, b, pasos, aciertos
    return w, b, None, aciertos


def medir_perceptron():
    X = EDADES.reshape(-1, 1)
    res = {}
    for regla in ("de 6 a 12", "de 6 en adelante", "hasta 12"):
        y = etiquetas(regla)
        w, b, pasos, aciertos = entrenar_con_rastro(X, y)
        w0, b0, pasos0 = perceptron.entrenar(X, y, pasos_max=PLAZO)      # el de verdad, para comparar
        assert pasos == pasos0 and np.array_equal(w, w0) and b == b0, \
            f"La repetición del bucle no acaba donde perceptron.entrenar: {(w, b, pasos)} y {(w0, b0, pasos0)}"
        mal = [int(e) for e, yi in zip(EDADES, y) if not yi * (e * w[0] + b) > 0]
        res[regla] = dict(quieto=pasos is not None, pasos=pasos, w=float(w[0]), b=float(b),
                          final=aciertos[-1], maximo=max(aciertos),
                          primera_vez=aciertos.index(max(aciertos)), mal=mal)
    return res


def imprimir_perceptron(res, emitir):
    emitir(["--- a) EL PERCEPTRÓN, CON UN MIEMBRO QUE MIRA LA EDAD ---", ""])
    filas = []
    for regla in ("de 6 a 12", "de 6 en adelante", "hasta 12"):
        r = res[regla]
        filas.append([f"sí {regla}", "sí" if r["quieto"] else "no",
                      miles(r["pasos"]) if r["quieto"] else f"{miles(PLAZO)} (se acaba el plazo)"])
    emitir(tabla_editorial(
        "Cuántas correcciones necesita el perceptrón con la edad",
        ["lo que tiene que decir", "¿se queda quieto?", "correcciones"], filas, "icd",
        ["Se queda quieto cuando da una vuelta entera por las edades sin tener que corregir nada. "
         f"Con el colegio no lo consigue nunca: a las {miles(PLAZO)} correcciones se le acaba el "
         "plazo."]))
    emitir([""])
    r = res["de 6 a 12"]
    mal = ", ".join(str(e) for e in r["mal"])
    emitir(tabla_editorial(
        "Dónde acaba el perceptrón con «sí de 6 a 12»",
        ["", "cuánto"],
        [["acierta al acabar el plazo", f"{r['final']} de 13"],
         ["acierta como máximo, en algún momento", f"{r['maximo']} de 13"],
         ["llega a ese máximo tras", f"{miles(r['primera_vez'])} correcciones"],
         ["edades mal al final", mal],
         ["su peso al acabar", coma(r["w"], 1)],
         ["su listón al acabar", coma(-r["b"] / r["w"], 2) if r["w"] else "ninguno"]], "id",
        ["Una edad está bien cuando la cuenta tiene el signo de la respuesta, sin ser cero.",
         "El listón es la edad en la que la cuenta pasa de no a sí."]))
    emitir([""])
    filas = []
    for regla in ("de 6 en adelante", "hasta 12"):
        r = res[regla]
        filas.append([f"sí {regla}", coma(r["w"], 1), coma(-r["b"] / r["w"], 2)])
    emitir(tabla_editorial(
        "Lo que aprende cuando se queda quieto",
        ["lo que tiene que decir", "peso", "listón"], filas, "idd",
        ["Sí cuando el peso por la edad pasa del listón (con peso negativo, al revés)."]))
    emitir([""])


# ---------------------------------------------------------------- b) la red

def sentido(dice):
    """Lo que hace una neurona de en medio con las 13 edades, dicho como lo diría una persona.
    Devuelve (clave para ordenar, texto). `dice`: las 13 respuestas, entre 0 y 1."""
    assert dice.shape == (13,) and ((dice >= 0) & (dice <= 1)).all(), \
        f"Se esperaban 13 respuestas entre 0 y 1; hay {dice.shape}"
    si = dice > UMBRAL
    if si.all():
        return (2, 0), "siempre"
    if not si.any():
        return (3, 0), "nunca"
    edades = EDADES[si]
    if si[-1]:                                    # sí desde una edad hasta el final
        assert si[si.argmax():].all(), f"Se esperaba un sí que empieza y sigue: {si}"
        n = int(edades[0])
        return (1, n), f"sí desde los {n} años"
    assert si[0] and si[:si.sum()].all(), f"Se esperaba un sí que empieza al principio y acaba: {si}"
    n = int(edades[-1])
    return (0, n), f"sí hasta los {n} años"


def resuelve(red, X):
    salida = red.adelante(X)[-1].ravel()
    return bool(((salida > UMBRAL) == (DEBERIA > UMBRAL)).all())


def un_arranque(args):
    semilla, escala = args
    X = entrada(escala)
    red = Red([1, 2, 1], semilla=int(semilla)).entrenar(X, DEBERIA, TASA, PASOS)
    ok = resuelve(red, X)
    pareja = tuple(sorted(sentido(red.adelante(X)[1][:, j]) for j in range(2))) if ok else None
    return ok, pareja


def sortear():
    return [int(s) for s in np.random.default_rng(SEMILLA_BASE).integers(1, 2**31 - 1, size=ARRANQUES)]


def medir_red(escala):
    semillas = sortear()
    with ProcessPoolExecutor() as pool:
        resultados = list(pool.map(un_arranque, [(s, escala) for s in semillas], chunksize=5))
    cuenta = Counter()
    primera = {}                       # pareja -> índice (desde 1) del primer arranque que la tiene
    for k, (ok, pareja) in enumerate(resultados, 1):
        if ok:
            clave = tuple(t for _, t in pareja)
            cuenta[clave] += 1
            primera.setdefault(clave, k)
    resueltas = sum(cuenta.values())
    assert resueltas == sum(1 for ok, _ in resultados if ok)
    orden = sorted(cuenta, key=lambda c: (-cuenta[c], primera[c]))
    res = dict(escala=escala, semillas=semillas, resueltas=resueltas, cuenta=cuenta, primera=primera,
               orden=orden, resultados=resultados, capitulo=None)
    if resueltas:
        mas = cuenta[orden[0]]
        empatadas = [c for c in orden if cuenta[c] == mas]
        elegida = min(empatadas, key=lambda c: primera[c])
        k = primera[elegida]
        res["capitulo"] = dict(pareja=elegida, indice=k, semilla=semillas[k - 1], empatadas=empatadas)
    return res


def red_entrenada(semilla, escala, pasos=PASOS):
    X = entrada(escala)
    red = Red([1, 2, 1], semilla=semilla)
    for _ in range(pasos):
        un_paso(red, X)
    return red


def un_paso(red, X):
    """Un paso de aprendizaje, exactamente el de Red.entrenar."""
    gW, gb = red.gradiente_retro(X, DEBERIA)
    for k in range(len(red.W)):
        red.W[k] -= TASA * gW[k]
        red.b[k] -= TASA * gb[k]


def c3(x):
    return coma(x, 3)


def firmado(x, d=2):
    return ("+" if x >= 0 else "-") + coma(abs(x), d)


def nombre_escala(escala):
    return "en años" if escala == 1.0 else "en decenas"


def imprimir_red(res, emitir):
    escala = res["escala"]
    cap = res["capitulo"]
    X = entrada(escala)
    emitir([f"--- b) LA RED, CON LA EDAD {nombre_escala(escala).upper()} ---", ""])
    if escala != 1.0:
        emitir(["La entrada es la edad entre 10. Todo lo demás, igual.", ""])
    emitir(tabla_editorial(
        "Cuántas de las 200 redes resuelven el colegio",
        ["", "redes"],
        [["arranques distintos", str(ARRANQUES)],
         ["lo resuelven (las 13 edades del lado correcto de 0,5)", str(res["resueltas"])],
         ["no lo resuelven", str(ARRANQUES - res["resueltas"])]], "id",
        [f"Las semillas salen de la semilla base {SEMILLA_BASE}, como en el pasillo.",
         f"Cada red: {miles(PASOS)} pasos, poquito {coma(TASA, 1)}."]))
    emitir([""])
    if not res["resueltas"]:
        emitir(["Ninguna red lo resuelve: no hay parejas ni red del capítulo.", ""])
        return
    filas = [[a, b, str(round(100 * res["cuenta"][(a, b)] / res["resueltas"])),
              str(res["cuenta"][(a, b)])] for a, b in res["orden"]]
    emitir(tabla_editorial(
        "Lo que acaban haciendo las dos neuronas de la primera capa",
        ["una", "la otra", "de cada 100", "redes"], filas, "iidd",
        [f"Se entrenaron {ARRANQUES} redes, cada una desde su propio arranque, y {res['resueltas']} "
         f"lo resolvieron. Todas esas {res['resueltas']} acabaron haciendo una de estas "
         f"{EN_LETRA.get(len(res['orden']), len(res['orden']))} parejas de preguntas. «Sí desde los "
         "6 años» quiere decir que esa neurona dice que sí (más de 0,5) a los de seis años en "
         "adelante, y que no a los más pequeños; «sí hasta los 5 años» es al revés. La columna "
         f"«de cada 100» cuenta solo las {res['resueltas']} que lo resolvieron."]))
    emitir([""])
    # --- la red del capítulo
    red = red_entrenada(cap["semilla"], escala)
    act = red.adelante(X)
    medio, final = act[1], act[2].ravel()
    a, b = cap["pareja"]
    emitir([f"LA RED DEL CAPÍTULO ({nombre_escala(escala)})", ""])
    nota_empate = ([] if len(cap["empatadas"]) == 1 else
                   [f"Hay {len(cap['empatadas'])} parejas con las mismas redes: se toma la de "
                    f"la red que sale antes."])
    emitir(tabla_editorial(
        "Cuál es la red del capítulo",
        ["", "cuál"],
        [["la primera de la lista de arranques con la pareja más frecuente", f"la {cap['indice']}.ª"],
         ["su semilla", str(cap["semilla"])],
         ["una neurona de en medio", a], ["la otra", b]], "id", nota_empate))
    emitir([""])
    W0, b0, W1, b1 = red.W[0][0], red.b[0], red.W[1][:, 0], red.b[1][0]
    emitir(tabla_editorial(
        "Los pesos y los listones de la red del capítulo",
        ["neurona", "le llega la línea de", "peso", "listón"],
        [[NEURONAS[0], "la edad", coma(W0[0], 3), coma(-b0[0], 3)],
         [NEURONAS[1], "la edad", coma(W0[1], 3), coma(-b0[1], 3)],
         ["la final", NEURONAS[0], coma(W1[0], 3), coma(-b1, 3)],
         ["", NEURONAS[1], coma(W1[1], 3), ""]], "iidd",
        [f"Entrenada {miles(PASOS)} pasos. El listón es menos el sesgo.",
         ("Lo que le llega a cada neurona de en medio es el peso por la edad."
          if escala == 1.0 else
          "Lo que le llega a cada neurona de en medio es el peso por la edad entre 10.")]))
    emitir([""])
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
    filas = []
    for m1 in (0, 1):
        for m2 in (0, 1):
            llega = m1 * W1[0] + m2 * W1[1]
            pasa = llega - (-b1)
            filas.append([("sí" if m1 else "no"), ("sí" if m2 else "no"), coma(llega, 2),
                          coma(-b1, 2), firmado(pasa, 2), c3(float(rampa(pasa))),
                          "sí" if pasa > 0 else "no"])
    emitir(tabla_editorial(
        "La final, si las de en medio dijeran sí o no secos",
        ["1.ª", "2.ª", "le llega", "listón", "pasa por", "dice", "→"], filas, "iidddd" + "c",
        ["Un sí seco es un 1 y un no seco es un 0.",
         "Pasa por: lo que le llega menos el listón. Dice sí si es positivo."]))
    emitir([""])


# ---------------------------------------------------------------- c) la culpa hacia atrás

def edades_mal(red, X):
    salida = red.adelante(X)[-1].ravel()
    return [int(e) for e, s, d in zip(EDADES, salida, DEBERIA) if (s > UMBRAL) != (d > UMBRAL)]


def elegir_paso(semilla, escala):
    """El paso del caso: 2.500 si en ese paso alguna edad queda del lado equivocado; si no, el
    último múltiplo de 500 en que todavía queda alguna. Devuelve (paso, {múltiplo: edades mal})."""
    X = entrada(escala)
    red = Red([1, 2, 1], semilla=semilla)
    mal = {}
    for paso in range(1, PASOS + 1):
        un_paso(red, X)
        if paso % CADA_CUANTOS == 0:
            mal[paso] = edades_mal(red, X)
    if mal[PASO_DEL_CASO]:
        return PASO_DEL_CASO, mal
    con_fallo = [p for p, m in mal.items() if m]
    assert con_fallo, "Se esperaba algún múltiplo de 500 con una edad mal; no hay ninguno"
    return max(con_fallo), mal


def r4(x):
    """Cuatro cifras significativas, como en `culpa_hacia_atras.r4`. Aquí no se prohíbe el
    exponente: una neurona que dice 0,99998 deja 0,00001995 de lo que le falta para 1."""
    return float(f"{x:.4g}")


def pos(v):
    """Un número ya redondeado, escrito con coma y sin exponente (0,00001995; 0,99998005)."""
    t = format(Decimal(repr(round(float(v), 12))), "f")
    if "." in t:
        t = t.rstrip("0").rstrip(".")
    return t.replace(".", ",")


def s4(x, signo=False):
    """r4(x) escrito: coma decimal, con signo si es negativo (o siempre, con `signo`)."""
    v = r4(x)
    t = pos(abs(v))
    if v < 0:
        return "-" + t
    return ("+" + t) if signo else t


def dice_r(x):
    """Lo que dice una neurona, a cuatro cifras. Si a cuatro cifras sale justo 1 (una neurona que dice
    0,99998), se redondea lo que le falta para 1 y se resta de 1: 0,99998005 y no 1, porque con 1 la
    cuenta «lo que le falta para 1» saldría 0. (En `culpa_hacia_atras.py` no pasa nunca.)"""
    v = r4(x)
    if v == 1.0 and x < 1.0:
        return round(1.0 - r4(1.0 - x), 12)
    return v


def dice_s(x):
    return pos(dice_r(x))


def caso(red, X, p):
    """Un ejemplo, hacia delante y hacia atrás, con cada número con su nombre."""
    x = X[p]
    W0, b0, W1, b1 = red.W[0], red.b[0], red.W[1][:, 0], red.b[1][0]
    c = {"x": x, "deberia": DEBERIA[p]}
    c["llega_medio"] = x @ W0
    c["liston_medio"] = -b0
    c["pasa_medio"] = c["llega_medio"] - c["liston_medio"]
    c["dice_medio"] = rampa(c["pasa_medio"])
    c["llega_final"] = c["dice_medio"] @ W1
    c["liston_final"] = -b1
    c["pasa_final"] = c["llega_final"] - c["liston_final"]
    c["dice_final"] = float(rampa(c["pasa_final"]))
    c["falla"] = c["dice_final"] - c["deberia"]
    c["error"] = c["falla"] ** 2
    c["error_por_salida"] = 2 * c["falla"]
    c["pasa_rampa_final"] = deja_pasar(rampa, c["pasa_final"])
    c["culpa_final"] = c["error_por_salida"] * c["pasa_rampa_final"]
    c["culpa_lineas_final"] = c["culpa_final"] * c["dice_medio"]
    c["le_toca_medio"] = c["culpa_final"] * W1
    c["pasa_rampa_medio"] = np.array([deja_pasar(rampa, z) for z in c["pasa_medio"]])
    c["culpa_medio"] = c["le_toca_medio"] * c["pasa_rampa_medio"]
    c["culpa_pesos_medio"] = np.outer(x, c["culpa_medio"])
    return c


def cadena_a_mano(red, c):
    """El caso hacia atrás con los números ya redondeados a cuatro cifras, cada cuenta con los
    redondeados y su resultado redondeado, como en `culpa_hacia_atras.cadena_a_mano`. Cada número
    tiene que estar a menos del 1 % del que da el programa sin redondear."""
    W1 = [round(float(w), 2) for w in red.W[1][:, 0]]
    m = {"pesos_final": W1}
    m["dice_final"] = dice_r(c["dice_final"])
    m["falla"] = r4(m["dice_final"] - c["deberia"])
    m["doble"] = r4(2 * m["falla"])
    m["falta_final"] = r4(1 - c["dice_final"])
    m["pasa_final"] = r4(m["dice_final"] * m["falta_final"])
    m["culpa_final"] = r4(m["doble"] * m["pasa_final"])
    m["dice_medio"] = [dice_r(v) for v in c["dice_medio"]]
    m["lineas_final"] = [r4(m["culpa_final"] * d) for d in m["dice_medio"]]
    m["le_toca"] = [r4(m["culpa_final"] * w) for w in W1]
    m["falta_medio"] = [r4(1 - d) for d in c["dice_medio"]]
    m["pasa_medio"] = [r4(d * f) for d, f in zip(m["dice_medio"], m["falta_medio"])]
    m["culpa_medio"] = [r4(a * b) for a, b in zip(m["le_toca"], m["pasa_medio"])]
    m["lineas_primera"] = [r4(m["culpa_medio"][j] * float(c["x"][0])) for j in range(2)]
    exactos = [(m["culpa_final"], c["culpa_final"]), (m["pasa_final"], c["pasa_rampa_final"])]
    exactos += list(zip(m["lineas_final"], c["culpa_lineas_final"]))
    exactos += list(zip(m["le_toca"], c["le_toca_medio"]))
    exactos += list(zip(m["pasa_medio"], c["pasa_rampa_medio"]))
    exactos += list(zip(m["culpa_medio"], c["culpa_medio"]))
    exactos += list(zip(m["lineas_primera"], c["culpa_pesos_medio"][0]))
    for a, b in exactos:
        assert abs(a - b) <= 0.01 * abs(b) + 1e-9, f"la cadena a mano se aleja del programa: {a} y {b}"
    return m


def error_de_un_ejemplo(red, X, p):
    return float((red.adelante(X[p:p + 1])[-1].ravel()[0] - DEBERIA[p]) ** 2)


def culpa_a_lo_bruto(red, X, p, capa, i, j, pelin=PELIN_BRUTO):
    otra = copia(red)
    otra.W[capa][i, j] += pelin
    mas = error_de_un_ejemplo(otra, X, p)
    otra.W[capa][i, j] -= 2 * pelin
    menos = error_de_un_ejemplo(otra, X, p)
    return mas, menos, (mas - menos) / (2 * pelin)


def decimales_iguales(a, b):
    """Cuántos decimales coinciden, como mucho 10."""
    d = abs(a - b)
    n = 0
    while n < 10 and d < 0.5 * 10.0 ** -(n + 1):
        n += 1
    return n


def imprimir_culpa(res, emitir):
    escala = res["escala"]
    cap = res["capitulo"]
    X = entrada(escala)
    emitir([f"--- c) LA CULPA HACIA ATRÁS, CON LA EDAD {nombre_escala(escala).upper()} ---", ""])
    if cap is None:
        emitir(["No hay red del capítulo: no se hace.", ""])
        return
    paso, mal = elegir_paso(cap["semilla"], escala)
    filas = [[miles(p), str(len(m)), ", ".join(str(e) for e in m) or "ninguna"]
             for p, m in mal.items() if p <= max(paso, PASO_DEL_CASO)]
    emitir(tabla_editorial(
        "Edades del lado equivocado de 0,5, cada 500 pasos",
        ["paso", "cuántas", "cuáles"], filas, "idi",
        [f"La red del capítulo: semilla {cap['semilla']}.",
         (f"En el paso {miles(PASO_DEL_CASO)} todavía falla alguna: es el paso del caso."
          if paso == PASO_DEL_CASO else
          f"En el paso {miles(PASO_DEL_CASO)} ya no falla ninguna; el último múltiplo de "
          f"{miles(CADA_CUANTOS)} con alguna mal es el {miles(paso)}: es el paso del caso.")]))
    emitir([""])
    red = red_entrenada(cap["semilla"], escala, paso)
    dice_ahora = red.adelante(X)[-1].ravel()
    p = int(np.argmax(np.abs(dice_ahora - DEBERIA)))
    edad = int(EDADES[p])
    emitir([f"La edad del ejemplo es la que más falla en ese paso: {edad} años.", ""])
    c = caso(red, X, p)
    m = cadena_a_mano(red, c)
    W1 = m["pesos_final"]
    deberia = int(c["deberia"])
    emitir([f"UN EJEMPLO A MEDIO ENTRENAR: EL PASO {miles(paso)} DE {miles(PASOS)}", ""])
    x_txt = str(edad) if escala == 1.0 else coma(float(c["x"][0]), 1)
    filas = []
    for j in range(2):
        filas.append([NEURONAS[j], coma(c["llega_medio"][j], 2), coma(c["liston_medio"][j], 2),
                      coma(c["pasa_medio"][j], 2), dice_s(c["dice_medio"][j])])
    filas.append(["la final", coma(c["llega_final"], 2), coma(c["liston_final"], 2),
                  coma(c["pasa_final"], 2), dice_s(c["dice_final"])])
    contexto = (f"Es el paso {miles(paso)} de los {miles(PASOS)} del entrenamiento, con la edad de "
                f"{edad} años, que el colegio {'admite' if deberia else 'no admite'}: la final "
                f"tendría que decir {deberia}.")
    dos = [r4(m["dice_medio"][j] * W1[j]) for j in range(2)]
    notas_entrada = ([] if escala == 1.0 else [f"La entrada es la edad entre 10: {x_txt}."])
    casi_uno = [NEURONAS[j] for j in range(2) if dice_r(c["dice_medio"][j]) != r4(c["dice_medio"][j])]
    if dice_r(c["dice_final"]) != r4(c["dice_final"]):
        casi_uno.append("la final")
    if casi_uno:
        notas_entrada += ["A cuatro cifras, lo que dice " + " y ".join(casi_uno) + " saldría 1 y lo que "
                          "le falta para 1 saldría 0: se redondea lo que le falta, y se resta."]
    emitir(tabla_editorial(
        "Hacia delante", ["neurona", "le llega", "listón", "pasa del listón por", "dice"],
        filas, "idddd", [contexto] + notas_entrada))
    emitir([""])
    emitir(tabla_editorial(
        "Lo que le llega a la final",
        ["viene de", "lo que dice", "por el peso de su línea", "da"],
        [[NEURONAS[j], dice_s(c["dice_medio"][j]), coma(W1[j], 2), s4(dos[j])] for j in range(2)]
        + [["**la suma**", "", "", f"**{coma(c['llega_final'], 2)}**"]], "iddd",
        ["La suma, redondeada a dos decimales, como en la tabla anterior."]))
    emitir([""])
    emitir(tabla_editorial(
        "Lo que falla, y el error", ["", "cuánto"],
        [["lo que dice la final", dice_s(c["dice_final"])],
         ["lo que tendría que decir", str(deberia)],
         ["lo que falla: lo que dice menos lo que tendría que decir", s4(m["falla"], True)],
         ["**el error con esta edad: lo que falla por sí mismo**",
          f"**{s4(r4(m['falla']) ** 2)}**"]], "id",
        ["Los números, redondeados a cuatro cifras, y cada cuenta con los redondeados."]))
    emitir([""])
    s = c["dice_final"]
    e_mas = (s + PELIN - c["deberia"]) ** 2
    dice_mas = float(rampa(c["pasa_final"] + PELIN))
    emitir(["HACIA ATRÁS", ""])
    emitir(tabla_editorial(
        "Tramo 1, medido: cuánto se mueven el error y lo que dice la final",
        ["qué se mide", "tal cual", "una millonésima más arriba", "sube, en millonésimas"],
        [["el error, si se mueve lo que dice", coma(c["error"], 9), coma(e_mas, 9),
          coma((e_mas - c["error"]) / PELIN, 3)],
         ["lo que dice, si se mueve lo que le llega", coma(c["dice_final"], 9),
          coma(dice_mas, 9), coma((dice_mas - c["dice_final"]) / PELIN, 4)]], "iddd",
        ["Lo que sube el error es el doble de lo que falla; lo que sube lo que dice es lo que "
         "deja pasar su rampa. La tabla siguiente lo saca con cuentas."]))
    emitir([""])
    emitir(tabla_editorial(
        "Tramo 1: la culpa de la final", ["", "cuánto"],
        [["lo que falla", s4(m["falla"])],
         ["el doble de lo que falla", s4(m["doble"])],
         ["lo que dice", dice_s(c["dice_final"])],
         ["lo que le falta para 1", s4(m["falta_final"])],
         ["lo que deja pasar su rampa: lo que dice, por lo que le falta para 1",
          s4(m["pasa_final"])],
         ["**la culpa de la final: el doble de lo que falla, por lo que deja pasar su rampa**",
          f"**{s4(m['culpa_final'])}**"]], "id",
        ["Cada cuenta está hecha con los números redondeados a cuatro cifras, para que se pueda "
         "repetir con el dedo."]))
    emitir([""])
    emitir(tabla_editorial(
        "Tramo 2: las dos líneas que llegan a la final",
        ["la línea que viene de", "la culpa de la final", "por lo que trajo esa línea", "da"],
        [[NEURONAS[j], s4(m["culpa_final"]), dice_s(c["dice_medio"][j]), s4(m["lineas_final"][j])]
         for j in range(2)], "iddd",
        ["Lo que trae cada línea es lo que dice la neurona de la que sale."]))
    emitir([""])
    emitir(tabla_editorial(
        "Tramo 3: cuánta le toca a cada una de las dos de la primera capa",
        ["", NEURONAS[0], NEURONAS[1]],
        [["la culpa de la final", s4(m["culpa_final"]), s4(m["culpa_final"])],
         ["por el peso de su línea", firmado(W1[0], 2), firmado(W1[1], 2)],
         ["le toca", s4(m["le_toca"][0]), s4(m["le_toca"][1])],
         ["lo que dice", dice_s(c["dice_medio"][0]), dice_s(c["dice_medio"][1])],
         ["lo que le falta para 1", s4(m["falta_medio"][0]), s4(m["falta_medio"][1])],
         ["lo que deja pasar su rampa", s4(m["pasa_medio"][0]), s4(m["pasa_medio"][1])],
         ["**su culpa: lo que le toca, por lo que deja pasar su rampa**",
          f"**{s4(m['culpa_medio'][0])}**", f"**{s4(m['culpa_medio'][1])}**"]], "idd",
        ["Lo que deja pasar la rampa de cada una es lo que dice, por lo que le falta para 1."]))
    emitir([""])
    emitir(tabla_editorial(
        "Tramo 4: las dos líneas de la primera capa",
        ["línea", "culpa de su neurona", "por lo que trajo", "da"],
        [[f"{NEURONAS[j]}, de la edad", s4(m["culpa_medio"][j]), x_txt, s4(m["lineas_primera"][j])]
         for j in range(2)], "iddd",
        [f"Lo que trae cada una de estas dos líneas es la edad{'' if escala == 1.0 else ' entre 10'}, "
         f"{x_txt}."]))
    emitir([""])
    # la comprobación a lo bruto de dos pesos, sin redondear
    gW, _ = red.gradiente_retro(X[p:p + 1], DEBERIA[p:p + 1])
    filas, juntas = [], []
    for (capa, i, j, nombre) in ((0, 0, 1, "la segunda, de la edad"),
                                 (1, 1, 0, "la final, de la segunda")):
        mas, menos, bruta = culpa_a_lo_bruto(red, X, p, capa, i, j)
        atajo = float(gW[capa][i, j])
        assert abs(atajo - bruta) < TOL_BRUTA, f"atajo y fuerza bruta difieren en {abs(atajo - bruta)}"
        filas.append((nombre, atajo, bruta, mas, menos))
        juntas.append(decimales_iguales(atajo, bruta))
    emitir(tabla_editorial(
        "La misma culpa, a lo bruto", ["peso", "por el atajo", "a lo bruto"],
        [[n, coma(a, 10), coma(b, 10)] for n, a, b, _, _ in filas], "idd",
        ["«A lo bruto» quiere decir sin atajo: se sube el peso una cienmilésima y se mira el "
         "error, se baja una cienmilésima y se mira otra vez, y la culpa es lo que va de un "
         "error a otro, dividido entre lo que se ha movido el peso. Las dos columnas coinciden en "
         f"sus {EN_LETRA.get(min(juntas), min(juntas))} primeros decimales."]))
    emitir([""])
    for n, _, _, mas, menos in filas:
        emitir(tabla_editorial(
            f"El error, con {n} movido", ["el peso", "el error"],
            [["una cienmilésima más arriba", coma(mas, 10)],
             ["una cienmilésima más abajo", coma(menos, 10)]], "id"))
        emitir([""])


# ---------------------------------------------------------------- d) los tres comités a mano

def dice_comite(pesos, liston, llegan):
    """Lo que dice un comité a mano: sí (1) si lo que le llega pasa del listón."""
    llega = sum(p * v for p, v in zip(pesos, llegan))
    return llega, int(llega > liston)


def comites_a_mano():
    """Las 13 edades por los tres comités. Devuelve las filas y revienta si alguna no acierta."""
    (_, _, p1, l1), (_, _, p2, l2), (_, _, p3, l3) = COMITES
    filas = []
    for e, debe in zip(EDADES, DEBERIA):
        _, a = dice_comite(p1, l1, (e,))
        _, b = dice_comite(p2, l2, (e,))
        llega3, c = dice_comite(p3, l3, (a, b))
        filas.append((int(e), a, b, llega3, c, int(debe)))
    fallos = [f for f in filas if f[4] != f[5]]
    assert not fallos, f"Los comités a mano fallan en {fallos}"
    return filas


def tramos_de_edades(filas):
    """Las edades seguidas con las mismas tres respuestas, juntas en una fila: [(primera edad,
    última edad, (respuesta 1, respuesta 2, respuesta 3))]. Revienta si no salen exactamente tres."""
    tramos = []
    for e, a, b, _, c, _ in filas:
        r = tuple("sí" if x else "no" for x in (a, b, c))
        if tramos and tramos[-1][2] == r and tramos[-1][1] == e - 1:
            tramos[-1] = (tramos[-1][0], e, r)
        else:
            tramos.append((e, e, r))
    assert len(tramos) == 3, f"Se esperaban exactamente tres filas de edades; hay {len(tramos)}: {tramos}"
    return tramos


def imprimir_comites(emitir):
    filas = comites_a_mano()
    emitir(["--- d) LOS TRES COMITÉS A MANO ---", ""])
    emitir(tabla_editorial(
        "Los tres comités, con pesos y listones enteros",
        ["comité", "mira a", "pesos", "listón"],
        [[q, de, " y ".join(str(p) for p in ps), str(l)] for q, de, ps, l in COMITES], "iiid",
        ["Dice sí si lo que le llega pasa del listón (más que el listón, no igual).",
         "Lo que le llega: cada voto por su peso, y todo sumado.",
         "Un sí vale 1 y un no vale 0."]))
    emitir([""])
    emitir(tabla_editorial(
        "Las 13 edades por los tres comités",
        ["edad", "¿al menos seis?", "¿como mucho doce?", "le llega al 3.º", "¿los dos?",
         "tendría que decir"],
        [[str(e), "sí" if a else "no", "sí" if b else "no", str(ll), "sí" if c else "no",
          "sí" if d else "no"] for e, a, b, ll, c, d in filas], "idddcc",
        [f"El programa comprueba las 13: acierta {sum(f[4] == f[5] for f in filas)} de 13."]))
    emitir([""])
    emitir(tabla_editorial(
        "Los tres comités del colegio",
        ["la edad", "¿al menos seis?", "¿como mucho doce?", "¿entra?"],
        [[f"de {t[0]} a {t[1]}", *t[2]] for t in tramos_de_edades(filas)], "iccc",
        ["El tercero dice que sí si los dos primeros dicen que sí."]))
    emitir([""])


# ---------------------------------------------------------------- selftest

def selftest():
    fallos = []
    comprobar_entrada()
    X1 = EDADES.reshape(-1, 1)

    # [1] TEST NULO — con «sí de 6 en adelante» y con «sí hasta 12» (un solo listón basta) el
    #     perceptrón se queda quieto: no hay imposibilidad donde no la hay.
    quietos = {}
    for regla in ("de 6 en adelante", "hasta 12"):
        w, b, pasos = perceptron.entrenar(X1, etiquetas(regla), pasos_max=PLAZO)
        quietos[regla] = pasos
    print(f"[1] test nulo         se queda quieto con «de 6 en adelante» tras {quietos['de 6 en adelante']} "
          f"correcciones y con «hasta 12» tras {quietos['hasta 12']}")
    if any(p is None for p in quietos.values()):
        fallos.append(f"test nulo: con un solo listón el perceptrón no se queda quieto: {quietos}")

    # [2] SEÑAL IMPLANTADA — con «sí de 6 a 12» no se queda quieto en 100.000 correcciones, y la
    #     repetición del bucle acaba en los mismos pesos que perceptron.entrenar.
    w, b, pasos, aciertos = entrenar_con_rastro(X1, etiquetas("de 6 a 12"))
    w0, b0, pasos0 = perceptron.entrenar(X1, etiquetas("de 6 a 12"), pasos_max=PLAZO)
    igual = pasos is None and pasos0 is None and np.array_equal(w, w0) and b == b0
    print(f"[2] señal implantada  con «de 6 a 12» no se queda quieto en {miles(PLAZO)} correcciones "
          f"({'sí' if pasos is None else 'NO'}); acaba en los pesos de perceptron.entrenar: "
          f"{'sí' if igual else 'NO'}; acierta {aciertos[-1]} de 13, como mucho {max(aciertos)}")
    if pasos is not None or not igual or max(aciertos) >= 13:
        fallos.append("señal implantada: con «de 6 a 12» el perceptrón se queda quieto, "
                      "o la repetición del bucle no coincide con perceptron.entrenar")

    # [3] INVARIANTE — la culpa por el atajo y a lo bruto coinciden en ocho decimales (en una red
    #     sin entrenar y en una a medio entrenar, con la edad en años y en decenas), todo lo que dice
    #     una neurona está entre 0 y 1, y los tres comités a mano aciertan las 13.
    peor, fuera = 0.0, 0
    semilla = sortear()[0]
    for escala in (1.0, DECENAS):
        X = entrada(escala)
        for pasos_red in (0, 500):
            red = red_entrenada(semilla, escala, pasos_red)
            gW_atajo, _ = red.gradiente_retro(X, DEBERIA)
            gW_bruto = red.gradiente_fuerza_bruta(X, DEBERIA)
            peor = max(peor, max(float(np.abs(a - b).max()) for a, b in zip(gW_atajo, gW_bruto)))
            for a in red.adelante(X)[1:]:
                fuera += int(((a < 0) | (a > 1)).sum())
    filas = comites_a_mano()
    print(f"[3] invariante        atajo contra fuerza bruta, diferencia máxima {peor:.1e}; "
          f"respuestas fuera de 0 a 1: {fuera}; comités a mano: {sum(f[4] == f[5] for f in filas)} de 13")
    if peor >= TOL_BRUTA:
        fallos.append(f"invariante: atajo y fuerza bruta difieren en {peor}")
    if fuera:
        fallos.append(f"invariante: {fuera} respuestas fuera de 0 a 1")

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
    comprobar_entrada()
    if args.selftest:
        return selftest()

    dentro = [False]            # dentro de un bloque «::: tabla», que se compone con ancho propio

    def emitir(lineas):
        for l in lineas:
            if l.startswith("::: "):
                dentro[0] = True
            if not dentro[0]:
                comprobar_ancho([l], ANCHO_SALIDA)
            if l == ":::":
                dentro[0] = False
            print(l)

    emitir(["########## el colegio: de 6 a 12 años (Orden 15) ##########",
            f"fecha: {date.today().isoformat()}",
            f"máquina: {platform.system()} {platform.machine()}",
            f"python {platform.python_version()}, numpy {np.__version__}",
            "GPU: no se usa (todo en numpy). Nada de lo que sigue",
            "depende del ordenador: aciertos, recuentos y pesos.",
            "", f"edades de {EDAD_MIN} a {EDAD_MAX}: sí de {DESDE} a {HASTA} (7 casos),",
            "no en 3, 4, 5, 13, 14 y 15 (6 casos)", ""])
    imprimir_perceptron(medir_perceptron(), emitir)
    en_anios = medir_red(1.0)
    imprimir_red(en_anios, emitir)
    imprimir_culpa(en_anios, emitir)
    if en_anios["resueltas"] < MINIMO_EN_ANIOS:
        emitir([f"Con la edad en años lo resuelven {en_anios['resueltas']} de {ARRANQUES}, menos de "
                f"{MINIMO_EN_ANIOS}:", "se repite todo con la edad en decenas, y se dan las dos.", ""])
        en_decenas = medir_red(DECENAS)
        imprimir_red(en_decenas, emitir)
        imprimir_culpa(en_decenas, emitir)
    imprimir_comites(emitir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
