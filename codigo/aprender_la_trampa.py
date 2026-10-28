#!/usr/bin/env python3
"""
Capítulo 15 — se refuerza lo que se premia: una tabla de números que aprende a hacer trampa.

Pedido por Carlos el 11 de octubre de 2026, para el apartado sobre las máquinas que «se saltan las
reglas». La idea que tiene que enseñar: para que aparezca la trampa no hace falta nadie dentro. Basta
con una nota que se pueda subir por un camino que nadie quería.

No hay modelo de lenguaje. Hay un tablero de 7 × 7 casillas, una casilla de salida, una meta y un
letrero que pone «meta conseguida». Quien se mueve por el tablero es una tabla: para cada casilla y
cada uno de los cuatro movimientos, un número que dice cuánto premio cabe esperar si se hace ese
movimiento ahí. Al principio son todos cero. Tras cada movimiento, el número se ajusta hacia el premio
que vino después (aprendizaje por refuerzo con tabla, Q-learning, Watkins 1989). Es la tercera fase
del adiestramiento del capítulo 12, sin texto: «estudiar para el juez».

La partida acaba al pisar la meta o el letrero, o tras PASOS_MAX movimientos. Al acabar, un juez pone
la nota (1 o 0). Hay dos jueces:

  - el que solo lee: pone un 1 si al final se lee «meta conseguida», lo diga la meta o el letrero;
  - el que comprueba: pone un 1 solo si se ha llegado a la meta.

Cuatro variantes del tablero (no «montajes»: en el libro, «montaje» es la palabra del capítulo 2
para otra cosa, y la regla 5 ter no deja que una palabra nombre dos):

  A. juez que solo lee, letrero cerca de la salida (a 2 pasos; la meta está a 6);
  B. juez que comprueba, el mismo tablero (test nulo: el atajo no da nota);
  C. juez que solo lee, letrero lejos (a 12 pasos, más lejos que la meta);
  D. juez que solo lee, letrero lejos, y la meta rodeada de muros: no se puede alcanzar. Es la
     «tarea imposible» del informe de OpenAI (el 93 % de lo que se discutía en el tablón de los
     agentes salía de las tareas que ningún modelo había resuelto nunca). Añadido el mismo día,
     antes de ejecutarlo.

Se mide, para SEMILLAS tablas que aprenden cada una con su sorteo, qué hace la tabla en distintos
momentos del aprendizaje si se la deja elegir siempre el número más alto: llegar al letrero, llegar
a la meta o ninguna de las dos. El aprendizaje se acota por partidas, no por reloj (regla 10).

Predicciones, escritas ANTES de ejecutar:
  - A: al final, 18 o más de 20 tablas van al letrero.
  - B: ninguna va al letrero.
  - C: 18 o más de 20 van a la meta (el letrero solo es «trampa» si es el camino más corto).
  - D: 18 o más de 20 van al letrero, aunque esté a 12 pasos.

Uso:
    python aprender_la_trampa.py > ../datos/salidas/aprender_la_trampa.txt
    python aprender_la_trampa.py --selftest
"""

# ======================= CONSTANTES =======================

LADO = 7                          # el tablero es de LADO × LADO casillas
SALIDA = (0, 0)
META = (3, 3)                     # a 6 pasos de la salida
LETRERO_CERCA = (0, 2)            # a 2 pasos
LETRERO_LEJOS = (6, 6)            # a 12 pasos
MUROS_META = {(2, 3), (4, 3), (3, 2), (3, 4)}      # rodean la meta en el montaje D
MOVIMIENTOS = [(0, 1), (1, 0), (0, -1), (-1, 0)]   # derecha, abajo, izquierda, arriba
NOMBRES_MOV = ["derecha", "abajo", "izquierda", "arriba"]

PARTIDAS = 2000                   # partidas de aprendizaje por tabla
# Orden 97: cifra que la nota escribe con letra; si cambia el parámetro, falla aquí y no en la página.
EN_LETRA = {20: "veinte"}
PASOS_MAX = 40                    # movimientos por partida, como mucho
ALFA = 0.5                        # cuánto se mueve un número hacia lo que vino después
GAMMA = 0.95                      # cuánto vale un premio que llega un paso más tarde
EPSILON = 0.2                     # una de cada cinco veces, la tabla prueba un movimiento al azar
SEMILLAS = list(range(20))        # veinte tablas, cada una con su sorteo
MOMENTOS = [0, 10, 25, 50, 100, 200, 400, 1000, 2000]   # partidas aprendidas al mirar

# Otros valores de ALFA y GAMMA, para ver si el resultado depende de los elegidos (Carlos, 11 oct:
# «lo que quedaría por explicar es por qué se hacen esos cálculos»). Predicción, escrita ANTES de
# ejecutar: las cuatro variantes dan lo mismo (A y D al letrero, B y C a la meta) en las nueve
# combinaciones, salvo quizá con ALFA 0,1, que aprende más despacio y en la D podría no llegar.
OTROS_ALFA = [0.1, 0.5, 0.9]
OTROS_GAMMA = [0.9, 0.95, 0.99]
PARTIDAS_RASTRO = 10              # las primeras partidas de la primera tabla, una a una

# selftest
UMBRAL_SENAL = 0.9                # fracción de tablas que tienen que aprender el atajo en A
TOL = 1e-9

# ==========================================================

from formato import coma, miles, tabla_editorial

import argparse
import random
import sys
from collections import deque

LEE, COMPRUEBA = "solo lee", "comprueba"
SIN_MUROS = frozenset()
MONTAJES = {
    "A": ("juez que solo lee, letrero cerca", LEE, LETRERO_CERCA, SIN_MUROS),
    "B": ("juez que comprueba, letrero cerca", COMPRUEBA, LETRERO_CERCA, SIN_MUROS),
    "C": ("juez que solo lee, letrero lejos", LEE, LETRERO_LEJOS, SIN_MUROS),
    "D": ("juez que solo lee, letrero lejos, meta tapiada", LEE, LETRERO_LEJOS,
          frozenset(MUROS_META)),
}


def dentro(c):
    return 0 <= c[0] < LADO and 0 <= c[1] < LADO


def validar():
    for nombre, c in [("salida", SALIDA), ("meta", META), ("letrero cerca", LETRERO_CERCA),
                      ("letrero lejos", LETRERO_LEJOS)]:
        assert dentro(c), f"Se esperaba la casilla {nombre} dentro del tablero; es {c}"
    assert len({SALIDA, META, LETRERO_CERCA, LETRERO_LEJOS}) == 4, \
        "Se esperaban cuatro casillas distintas"
    assert distancia(SALIDA, LETRERO_CERCA) < distancia(SALIDA, META) < \
        distancia(SALIDA, LETRERO_LEJOS), (
        "Se esperaba el letrero cerca más cerca que la meta, y el lejos más lejos; distancias "
        f"{distancia(SALIDA, LETRERO_CERCA)}, {distancia(SALIDA, META)}, "
        f"{distancia(SALIDA, LETRERO_LEJOS)}")
    assert not MUROS_META & {SALIDA, META, LETRERO_CERCA, LETRERO_LEJOS}, \
        "Se esperaban los muros fuera de la salida, la meta y los letreros"
    assert distancia(SALIDA, META, MUROS_META) is None, \
        "Se esperaba la meta inalcanzable con los muros del montaje D"
    assert distancia(SALIDA, LETRERO_LEJOS, MUROS_META) == distancia(SALIDA, LETRERO_LEJOS), \
        "Se esperaba que los muros no alargaran el camino al letrero lejos"
    assert 0 < ALFA <= 1 and 0 < GAMMA < 1 and 0 <= EPSILON < 1, "Parámetros fuera de rango"
    assert MOMENTOS == sorted(MOMENTOS) and MOMENTOS[-1] == PARTIDAS, \
        "Se esperaban los momentos en orden y el último igual a PARTIDAS"


def mover(c, m, muros=SIN_MUROS):
    """Un movimiento; contra el borde o contra un muro, la ficha se queda donde está."""
    n = (c[0] + MOVIMIENTOS[m][0], c[1] + MOVIMIENTOS[m][1])
    return n if dentro(n) and n not in muros else c


def distancia(a, b, muros=SIN_MUROS):
    """Pasos del camino más corto (en este tablero sin muros, la distancia en cuadrícula, pero se
    calcula recorriendo el tablero para no suponerlo)."""
    vistos, cola = {a: 0}, deque([a])
    while cola:
        c = cola.popleft()
        if c == b:
            return vistos[c]
        for m in range(len(MOVIMIENTOS)):
            n = mover(c, m, muros)
            if n not in vistos:
                vistos[n] = vistos[c] + 1
                cola.append(n)
    return None                     # no se puede llegar


def nota(juez, final):
    """La nota del juez al acabar la partida: `final` es «meta», «letrero» o «nada»."""
    if juez == LEE:
        return 1.0 if final in ("meta", "letrero") else 0.0
    return 1.0 if final == "meta" else 0.0


def donde(c, letrero):
    return "meta" if c == META else "letrero" if c == letrero else None


def mejores(tabla, c):
    """Los movimientos empatados en el número más alto de la casilla."""
    fila = tabla[c]
    tope = max(fila)
    return [m for m in range(len(fila)) if fila[m] >= tope - TOL]


def mejor(tabla, c, rng):
    """El movimiento de número más alto; un empate se decide a sorteo. Sin sorteo, con la tabla
    a cero se elegiría siempre «derecha», y el letrero cerca está a la derecha de la salida: el
    atajo saldría del orden de la lista, no de lo aprendido (visto antes de ejecutar)."""
    return rng.choice(mejores(tabla, c))


def tabla_nueva():
    return {(i, j): [0.0] * len(MOVIMIENTOS) for i in range(LADO) for j in range(LADO)}


def partida(tabla, rng, juez, letrero, muros=SIN_MUROS, alfa=None, gamma=None):
    """Una partida aprendiendo. Devuelve cómo acabó, la nota y cuántos movimientos hizo."""
    alfa = ALFA if alfa is None else alfa
    gamma = GAMMA if gamma is None else gamma
    c = SALIDA
    for k in range(1, PASOS_MAX + 1):
        m = rng.randrange(len(MOVIMIENTOS)) if rng.random() < EPSILON else mejor(tabla, c, rng)
        n = mover(c, m, muros)
        final = donde(n, letrero)
        if final:
            r, siguiente = nota(juez, final), 0.0          # se acaba: no hay nada después
        else:
            r, siguiente = 0.0, max(tabla[n])
        tabla[c][m] += alfa * (r + gamma * siguiente - tabla[c][m])
        if final:
            return final, r, k
        c = n
    return "nada", nota(juez, "nada"), PASOS_MAX


def a_la_vista(tabla, letrero, muros=SIN_MUROS):
    """Qué hace la tabla si se la deja elegir siempre el número más alto, sin sorteo. Si en
    alguna casilla del camino todo sigue a cero, la tabla todavía no ha aprendido nada ahí: se
    devuelve «sin decidir» en vez de desempatar por el orden de la lista. Un empate por encima de
    cero es otra cosa: hay varios caminos igual de cortos hacia lo mismo (a la meta se llega de
    muchas maneras en 6 pasos), y cualquiera da el mismo final; se coge el primero. (La primera
    versión trataba todo empate como «sin decidir», y daba por indecisas tablas que llegaban a la
    meta siempre: lo cazó la nota media, 0,94 y 1, contra «ni una cosa ni otra» en 19 de 20.)
    Si vuelve a una casilla por la que ya pasó, da vueltas: «nada»."""
    c, pasos, vistos = SALIDA, 0, {SALIDA}
    while pasos < PASOS_MAX:
        ms = mejores(tabla, c)
        if max(tabla[c]) <= TOL and len(ms) > 1:
            return "sin decidir", pasos
        n = mover(c, ms[0], muros)
        pasos += 1
        final = donde(n, letrero)
        if final:
            return final, pasos
        if n in vistos:
            return "nada", pasos
        vistos.add(n)
        c = n
    return "nada", pasos


def camino(tabla, letrero, muros=SIN_MUROS):
    """Las casillas por las que pasa la tabla a la vista, de la salida al final. Con la misma
    regla que a_la_vista(); el selftest comprueba que las dos acaban en el mismo sitio."""
    c, cs = SALIDA, [SALIDA]
    while len(cs) <= PASOS_MAX:
        ms = mejores(tabla, c)
        if max(tabla[c]) <= TOL and len(ms) > 1:
            return cs
        n = mover(c, ms[0], muros)
        if n in cs:
            return cs
        cs.append(n)
        if donde(n, letrero):
            return cs
        c = n
    return cs


def aprender(semilla, juez, letrero, muros=SIN_MUROS, alfa=None, gamma=None):
    """Una tabla aprendiendo PARTIDAS partidas; qué hace a la vista en cada momento."""
    rng = random.Random(semilla)
    tabla, fotos, notas = tabla_nueva(), {}, []
    if 0 in MOMENTOS:
        fotos[0] = a_la_vista(tabla, letrero, muros)
    for k in range(1, PARTIDAS + 1):
        _, r, _ = partida(tabla, rng, juez, letrero, muros, alfa, gamma)
        notas.append(r)
        if k in MOMENTOS:
            fotos[k] = a_la_vista(tabla, letrero, muros)
    return tabla, fotos, notas


def ejecutar(clave, alfa=None, gamma=None):
    _, juez, letrero, muros = MONTAJES[clave]
    return [aprender(s, juez, letrero, muros, alfa, gamma) for s in SEMILLAS]


ESPERADO = {"A": "letrero", "B": "meta", "C": "meta", "D": "letrero"}


def otros_valores():
    """Para cada combinación de OTROS_ALFA y OTROS_GAMMA, cuántas de las tablas de cada variante
    acaban donde acaban con los valores elegidos (A y D en el letrero, B y C en la meta)."""
    filas = []
    for alfa in OTROS_ALFA:
        for gamma in OTROS_GAMMA:
            cuenta = {k: recuento(ejecutar(k, alfa, gamma), PARTIDAS, ESPERADO[k])
                      for k in MONTAJES}
            filas.append((alfa, gamma, cuenta))
    return filas


def recuento(resultados, momento, final):
    return sum(1 for _, fotos, _ in resultados if fotos[momento][0] == final)


# ------------------------------------------------------------------------------ la salida

def bloque_montaje(clave, resultados):
    titulo, _, letrero, muros = MONTAJES[clave]
    d_meta = distancia(SALIDA, META, muros)
    meta = "no se puede llegar" if d_meta is None else f"a {d_meta}"
    filas = [[f"{k:,}".replace(",", "."), str(recuento(resultados, k, "letrero")),
              str(recuento(resultados, k, "meta")),
              str(len(SEMILLAS) - recuento(resultados, k, "letrero")
                  - recuento(resultados, k, "meta"))]
             for k in MOMENTOS]
    return tabla_editorial(
        f"Variante {clave}: {titulo}",
        ["partidas aprendidas", "al letrero", "a la meta", "ni una cosa ni otra"], filas, "dddd",
        [f"De {len(SEMILLAS)} tablas, qué hace cada una si elige siempre el número más alto. "
         f"El letrero está a {distancia(SALIDA, letrero, muros)} pasos de la salida; a la meta, "
         f"{meta}."])


def bloque_dentro(resultados_a):
    tabla, fotos, _ = resultados_a[0]
    fila = tabla[SALIDA]
    filas = [[NOMBRES_MOV[m], coma(fila[m], 3)] for m in range(len(MOVIMIENTOS))]
    return tabla_editorial(
        "Lo que hay dentro, en la casilla de salida (variante A, primera tabla)",
        ["movimiento", "número"], filas, "id",
        [f"Toda la tabla son {LADO * LADO * len(MOVIMIENTOS)} números como estos: "
         f"{LADO * LADO} casillas por {len(MOVIMIENTOS)} movimientos. Al acabar, esta tabla va "
         f"{'al letrero' if fotos[PARTIDAS][0] == 'letrero' else 'a la meta'} en "
         f"{fotos[PARTIDAS][1]} pasos."])


def bloque_resumen(resultados):
    """La tabla del libro: las cuatro variantes, y qué hacen las tablas al terminar."""
    filas = []
    for clave, (_, juez, letrero, muros) in MONTAJES.items():
        d_meta = distancia(SALIDA, META, muros)
        filas.append([clave, "solo lee" if juez == LEE else "comprueba",
                       str(distancia(SALIDA, letrero, muros)),
                       "no se llega" if d_meta is None else str(d_meta),
                       str(recuento(resultados[clave], PARTIDAS, "letrero")),
                       str(recuento(resultados[clave], PARTIDAS, "meta"))])
    return tabla_editorial(
        "Lo que aprenden veinte tablas, según el juez y el tablero",
        ["", "el juez", "letrero, a", "meta, a", "al letrero", "a la meta"], filas, "iidddd",
        ["«Letrero, a» y «meta, a» dicen a cuántos pasos de la salida está cada uno; las dos "
         f"últimas columnas, cuántas de las {EN_LETRA[len(SEMILLAS)]} tablas acaban yendo ahí "
         f"después de {miles(PARTIDAS)} partidas. El juez que solo lee pone la nota si al final se "
         "lee «meta conseguida»; el que comprueba, solo si se ha llegado de verdad a la meta."])


def rastro(semilla=SEMILLAS[0], clave="A", partidas=PARTIDAS_RASTRO):
    """Las primeras partidas de una tabla, una a una: cómo acabó, cuántos movimientos, la nota, y
    los dos números del camino corto al letrero (el de «derecha» en la casilla de al lado del
    letrero y el de «derecha» en la salida), además de cuántos números han dejado de ser cero.
    Es la misma tabla que aprender(semilla, …): mismo sorteo, mismas partidas."""
    _, juez, letrero, muros = MONTAJES[clave]
    rng, tabla, filas = random.Random(semilla), tabla_nueva(), []
    al_lado = (letrero[0], letrero[1] - 1)
    assert mover(al_lado, 0, muros) == letrero and mover(SALIDA, 0, muros) == al_lado, (
        "El rastro supone que el camino corto es «derecha, derecha»; con este tablero no lo es")
    for k in range(1, partidas + 1):
        final, r, movs = partida(tabla, rng, juez, letrero, muros)
        filas.append({"partida": k, "final": final, "movimientos": movs, "nota": r,
                      "al_lado": tabla[al_lado][0], "salida": tabla[SALIDA][0],
                      "no_cero": sum(1 for f in tabla.values() for v in f if v > TOL)})
    return filas


def bloque_rastro(filas):
    nombre = {"letrero": "el letrero", "meta": "la meta", "nada": "nada"}
    return tabla_editorial(
        "Las diez primeras partidas de la primera tabla (variante A)",
        ["partida", "acaba en", "movimientos", "junto al letrero", "en la salida"],
        [[str(f["partida"]), nombre[f["final"]], str(f["movimientos"]),
          coma(f["al_lado"], 4), coma(f["salida"], 4)] for f in filas],
        "iiddd",
        ["Las dos últimas columnas son el número de «derecha» en dos casillas al acabar cada "
         "partida: la que está justo a la izquierda del letrero y la salida. Una partida acaba al "
         f"pisar la meta o el letrero, o a los {PASOS_MAX} movimientos; «nada» quiere decir que "
         f"llegó a los {PASOS_MAX} sin pisar ninguno."])


def bloque_longitudes():
    """Lo más que puede valer el primer movimiento de un camino de n pasos: la nota, 1, rebajada
    un GAMMA por cada paso que falta después del primero."""
    filas = []
    for nombre, destino in [("el letrero cerca", LETRERO_CERCA), ("la meta", META),
                            ("el letrero lejos", LETRERO_LEJOS)]:
        n = distancia(SALIDA, destino)
        filas.append([nombre, str(n), coma(GAMMA ** (n - 1), 3)])
    return tabla_editorial(
        "Lo más que puede valer el primer paso, según lo lejos que esté la nota",
        ["hacia", "pasos", "el primer paso vale, como mucho"], filas, "idd",
        [f"Cada paso de más multiplica por {coma(GAMMA, 2)}. Con un camino de 2 pasos queda una "
         f"rebaja; con uno de 6, cinco."])


def bloque_otros_valores(filas):
    def pon(x):
        return coma(x, 2).rstrip("0").rstrip(",") if x != 0.95 else "0,95"
    return tabla_editorial(
        "Lo mismo, con otros valores en las dos cuentas",
        ["parte del camino", "un paso más lejos", "A: al letrero", "B: a la meta",
         "C: a la meta", "D: al letrero"],
        [[pon(a), pon(g)] + [str(c[k]) for k in MONTAJES] for a, g, c in filas], "dddddd",
        [f"Parte del camino: cuánto sube o baja el número hacia lo que vino detrás (0,5 es «a "
         f"medio camino»; 0,1, una décima parte). Un paso más lejos: por cuánto se multiplica. Cada número, cuántas de las "
         f"{len(SEMILLAS)} tablas acaban ahí, tras {miles(PARTIDAS)} partidas.",
         f"En la fila de {pon(ALFA)} y {pon(GAMMA)} están los valores del resto del apartado."])


def nota_final(resultados, ultimas=100):
    notas = [r for _, _, ns in resultados for r in ns[-ultimas:]]
    return sum(notas) / len(notas)


# ------------------------------------------------------------------------------ selftest

def selftest():
    fallos = []
    a, b = ejecutar("A"), ejecutar("B")
    n = len(SEMILLAS)

    # 1. TEST NULO — con el juez que comprueba, el letrero no da nota: ninguna tabla puede
    #    acabar yendo a él. Y para que este nulo pueda fallar en la dirección del resultado, el
    #    mismo tablero con el juez que solo lee tiene que dar letreros (lo mira la prueba 2).
    b_letrero = recuento(b, PARTIDAS, "letrero")
    print(f"[1] test nulo         juez que comprueba: {b_letrero} de {n} tablas al letrero")
    if b_letrero != 0:
        fallos.append(f"test nulo: con el juez que comprueba, {b_letrero} tablas van al letrero")

    # 2. SEÑAL IMPLANTADA — con el juez que solo lee y el letrero a 2 pasos, el atajo tiene que
    #    aprenderse en la mayoría, y por el camino más corto: en el sitio y con la medida justas.
    a_letrero = recuento(a, PARTIDAS, "letrero")
    corto = distancia(SALIDA, LETRERO_CERCA)
    largos = [f[PARTIDAS][1] for _, f, _ in a if f[PARTIDAS][0] == "letrero"]
    print(f"[2] señal implantada  juez que solo lee: {a_letrero} de {n} al letrero; pasos del "
          f"camino aprendido: {sorted(set(largos))} (el más corto es {corto})")
    if a_letrero < UMBRAL_SENAL * n:
        fallos.append(f"señal implantada: se esperaban al menos {UMBRAL_SENAL * n:.0f} tablas "
                      f"al letrero; van {a_letrero}")
    if any(x != corto for x in largos):
        fallos.append(f"señal implantada: algún camino al letrero no es el más corto: {largos}")

    # 3. INVARIANTE DEL DOMINIO — la nota es 0 o 1 y nada se descuenta, así que ningún número de
    #    ninguna tabla puede salir de [0, 1]; y la nota de cada partida es 0 o 1.
    fuera = [v for res in (a, b) for t, _, _ in res for fila in t.values() for v in fila
             if v < -TOL or v > 1 + TOL]
    notas_raras = [r for res in (a, b) for _, _, ns in res for r in ns if r not in (0.0, 1.0)]
    print(f"[3] invariante        números fuera de [0, 1]: {len(fuera)}; "
          f"notas que no son 0 ni 1: {len(notas_raras)}")
    if fuera:
        fallos.append(f"invariante: {len(fuera)} números fuera de [0, 1]")
    if notas_raras:
        fallos.append(f"invariante: {len(notas_raras)} notas que no son 0 ni 1")

    # 4. EL RASTRO Y LOS CAMINOS — el rastro es la misma tabla que la primera de A (mismas notas
    #    en las mismas partidas); antes de la primera nota no cambia ningún número, y la primera
    #    nota cambia exactamente uno, a ALFA; y el camino que se dibuja acaba donde dice
    #    a_la_vista(), con tantos pasos como él.
    filas = rastro()
    _, _, notas_a0 = a[0]
    iguales = [f["nota"] for f in filas] == notas_a0[:len(filas)]
    primera = next((i for i, f in enumerate(filas) if f["nota"] > 0), None)
    antes = all(f["no_cero"] == 0 for f in filas[:primera]) if primera is not None else False
    justo = primera is not None and filas[primera]["no_cero"] == 1 and \
        abs(filas[primera]["al_lado"] - ALFA) < TOL
    malos = []
    for clave in MONTAJES:
        _, _, letrero, muros = MONTAJES[clave]
        for t_, fotos, _ in (ejecutar(clave) if clave not in "AB" else (a if clave == "A" else b)):
            cs = camino(t_, letrero, muros)
            final, pasos = fotos[PARTIDAS]
            if (donde(cs[-1], letrero) or "nada") != final or len(cs) - 1 != pasos:
                malos.append(clave)
    print(f"[4] rastro y caminos  mismas notas que la primera tabla de A: {iguales}; primera nota "
          f"en la partida {None if primera is None else primera + 1}, cambia un solo número, a "
          f"{coma(ALFA, 1)}: {antes and justo}; caminos que no cuadran: {len(malos)}")
    if not iguales:
        fallos.append("rastro: no da las mismas notas que la primera tabla de A")
    if not (antes and justo):
        fallos.append("rastro: la primera nota no cambia exactamente un número a ALFA")
    if malos:
        fallos.append(f"caminos: {len(malos)} no acaban donde dice a_la_vista ({set(malos)})")

    # 5. OTROS VALORES — que la tabla de otros valores sale de verdad de esos valores, y no de
    #    los de siempre: con ALFA 0,1, la primera nota de la primera tabla deja el número en 0,1
    #    (y no en 0,5); con GAMMA 0,9, el número de la salida en A acaba cerca de 0,9 y no de
    #    0,95. Y la fila de los valores de siempre da lo mismo que la tabla resumen.
    t01 = tabla_nueva()
    rng = random.Random(SEMILLAS[0])
    _, juez_a, letrero_a, muros_a = MONTAJES["A"]
    primera01 = None
    for _ in range(PARTIDAS_RASTRO):
        _, r, _ = partida(t01, rng, juez_a, letrero_a, muros_a, alfa=0.1)
        if r > 0:
            primera01 = max(v for f in t01.values() for v in f)
            break
    t09, _, _ = aprender(SEMILLAS[0], juez_a, letrero_a, muros_a, gamma=0.9)
    salida09 = t09[SALIDA][0]
    filas = otros_valores()
    de_siempre = [c for a_, g_, c in filas if a_ == ALFA and g_ == GAMMA]
    resumen = {k: recuento(r_, PARTIDAS, ESPERADO[k])
               for k, r_ in [("A", a), ("B", b)]}
    cuadra = bool(de_siempre) and all(de_siempre[0][k] == v for k, v in resumen.items())
    print(f"[5] otros valores     con alfa 0,1 la primera nota deja {coma(primera01 or 0, 3)}; con "
          f"gamma 0,9 la salida de A vale {coma(salida09, 3)}; la fila de siempre cuadra con el "
          f"resumen: {cuadra}")
    if primera01 is None or abs(primera01 - 0.1) > TOL:
        fallos.append("otros valores: con alfa 0,1 la primera nota no deja el número en 0,1")
    if abs(salida09 - 0.9) > 0.01:
        fallos.append(f"otros valores: con gamma 0,9 la salida de A vale {salida09}, no ≈ 0,9")
    if not cuadra:
        fallos.append("otros valores: la fila de los valores de siempre no cuadra con el resumen")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las cinco pruebas pasan.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    validar()
    if args.selftest:
        sys.exit(selftest())

    print(f"tablero {LADO} × {LADO}; {PARTIDAS} partidas por tabla; {len(SEMILLAS)} tablas por "
          f"variante; alfa {coma(ALFA, 2)}, gamma {coma(GAMMA, 2)}, epsilon {coma(EPSILON, 2)}")
    print()
    resultados = {}
    for clave in MONTAJES:
        resultados[clave] = ejecutar(clave)
        print("\n".join(bloque_montaje(clave, resultados[clave])))
        print(f"nota media en las 100 últimas partidas: "
              f"{coma(nota_final(resultados[clave]), 3)}")
        print()
    print("\n".join(bloque_dentro(resultados["A"])))
    print()
    print("\n".join(bloque_resumen(resultados)))
    print()
    print("\n".join(bloque_rastro(rastro())))
    print()
    print("\n".join(bloque_longitudes()))
    print()
    print("\n".join(bloque_otros_valores(otros_valores())))


if __name__ == "__main__":
    main()
