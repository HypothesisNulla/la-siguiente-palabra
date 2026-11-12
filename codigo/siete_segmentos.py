#!/usr/bin/env python3
"""
Capítulo 3 — por qué «¿es par?» es difícil escrito a mano, y fácil en un reloj digital.

El capítulo necesita justificar que «¿este dígito es par?» no se resuelve con una raya.
La justificación que estaba escrita era una intuición: «no hay ninguna forma que compartan
el cero, el dos, el cuatro, el seis y el ocho». Esto la pone a prueba en el sitio donde la
intuición es más fuerte —un reloj digital, donde los dígitos están hechos de siete piezas
fijas— y la intuición FALLA: ahí una sola raya sí separa los pares de los impares.

Ese fallo es el contenido del apartado. En un reloj hay siete piezas siempre en el mismo
sitio, y con eso casi cualquier división de los diez dígitos se resuelve con una raya.
Escritos a mano no hay piezas: hay tinta en sitios distintos cada vez, y ahí una sola raya
se queda en el 90 %. La dificultad no viene de «par», y tampoco viene solo de la escritura
a mano: el capítulo anterior reconoce dígitos manuscritos sueltos sin problema. Viene de las
dos cosas juntas: cinco formas distintas, y a mano.

Uso:
    python siete_segmentos.py
    python siete_segmentos.py --selftest
"""

# ======================= CONSTANTES =======================

SEMILLA = 20260917
PASOS_MAX = 20_000          # correcciones antes de dar una partición por no separable

# Los siete segmentos de un reloj digital, con el nombre que les pondría cualquiera.
# El orden de esta lista es el orden de las filas de la tabla.
SEGMENTOS = [
    "el de arriba",
    "el de arriba izquierda",
    "el de arriba derecha",
    "el del medio",
    "el de abajo izquierda",
    "el de abajo derecha",
    "el de abajo",
]

# Qué segmentos enciende cada dígito. Es el dibujo de toda la vida de un reloj de horno,
# de microondas o de despertador; se puede comprobar mirando uno.
ENCENDIDOS = {
    0: "el de arriba, el de arriba izquierda, el de arriba derecha, "
       "el de abajo izquierda, el de abajo derecha, el de abajo",
    1: "el de arriba derecha, el de abajo derecha",
    2: "el de arriba, el de arriba derecha, el del medio, "
       "el de abajo izquierda, el de abajo",
    3: "el de arriba, el de arriba derecha, el del medio, el de abajo derecha, "
       "el de abajo",
    4: "el de arriba izquierda, el de arriba derecha, el del medio, "
       "el de abajo derecha",
    5: "el de arriba, el de arriba izquierda, el del medio, el de abajo derecha, "
       "el de abajo",
    6: "el de arriba, el de arriba izquierda, el del medio, el de abajo izquierda, "
       "el de abajo derecha, el de abajo",
    7: "el de arriba, el de arriba derecha, el de abajo derecha",
    8: "el de arriba, el de arriba izquierda, el de arriba derecha, el del medio, "
       "el de abajo izquierda, el de abajo derecha, el de abajo",
    9: "el de arriba, el de arriba izquierda, el de arriba derecha, el del medio, "
       "el de abajo derecha, el de abajo",
}

# Una partición de los diez dígitos que NINGUNA raya separa en este espacio: el seis y el
# nueve a un lado y los otros ocho al otro. No está elegida a ojo —la primera que probé,
# «solo el cero a un lado», resultó SER separable y el test nulo la cazó—: sale del
# recuento de las 1.024 particiones, de las que 160 no se separan con una raya.
PARTICION_IMPOSIBLE = (0, 0, 0, 0, 0, 0, 1, 0, 0, 1)   # el 6 y el 9 contra los demás

SALIDA_CSV = "../datos/salidas/siete_segmentos.csv"

# ==========================================================

import argparse
import csv
import itertools
import sys

from formato import miles, tabla_editorial
import numpy as np


def tabla_de_segmentos():
    """Los diez dígitos como filas de ceros y unos, un número por segmento."""
    filas = []
    for d in range(10):
        enciende = [s.strip() for s in ENCENDIDOS[d].split(",")]
        for s in enciende:
            assert s in SEGMENTOS, f"«{s}» no es uno de los siete segmentos"
        filas.append([1.0 if s in enciende else 0.0 for s in SEGMENTOS])
    return np.array(filas)


def entrenar(X, y, pasos_max=PASOS_MAX, semilla=SEMILLA):
    """El perceptrón del capítulo 2, sin un solo añadido. Devuelve (pesos, listón) si
    encuentra la raya, y (None, None) si se rinde."""
    rng = np.random.default_rng(semilla)
    w = np.zeros(X.shape[1])
    b = 0.0
    orden = np.arange(len(X))
    for _ in range(pasos_max):
        rng.shuffle(orden)
        errores = 0
        for i in orden:
            if y[i] * (X[i] @ w + b) <= 0:
                w += y[i] * X[i]
                b += y[i]
                errores += 1
        if errores == 0:
            return w, b
    return None, None


def separable(X, bits):
    y = np.array([1.0 if b else -1.0 for b in bits])
    if len(set(bits)) == 1:            # todos al mismo lado: no hay nada que separar
        return True
    w, _ = entrenar(X, y)
    return w is not None


def un_solo_segmento_basta(X):
    """¿Hay UN segmento encendido en los cinco pares y apagado en los cinco impares (o al
    revés)? Es la versión ingenua de la pregunta, y la respuesta es que no."""
    for j, nombre in enumerate(SEGMENTOS):
        pares, impares = X[::2, j], X[1::2, j]
        if (pares.all() and not impares.any()) or (impares.all() and not pares.any()):
            return nombre
    return None


# ---- Los bloques que cita el capítulo ----------------------------------------------
# Desde el 9 oct 2026, tablas de libro (formato.py; REGLAS 6 ter). Los dígitos van con los cinco
# pares primero y los cinco impares después, con «pares» e «impares» encima: así se ve de un
# vistazo si un segmento está encendido en un grupo y apagado en el otro, que es la pregunta.

ORDEN_DIGITOS = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]


def tabla_digitos(titulo, filas, notas, primera="el segmento"):
    """Una tabla de «sí» y «no» por dígito. `filas`: [(rótulo, [valor del 0, del 1, …])]."""
    rotulos = [primera] + [f"{'pares' if d % 2 == 0 else 'impares'}: {d}" for d in ORDEN_DIGITOS]
    return tabla_editorial(titulo, rotulos,
                           [[nombre] + [valores[d] for d in ORDEN_DIGITOS]
                            for nombre, valores in filas],
                           "i" + "c" * len(ORDEN_DIGITOS), notas)


def bloque_tabla(X, nombre_solo):
    return tabla_digitos(
        "Los diez dígitos de un reloj digital, segmento a segmento",
        [(nombre, ["sí" if X[d, j] else "no" for d in range(10)])
         for j, nombre in enumerate(SEGMENTOS)],
        ["Un «sí» quiere decir que ese segmento está encendido en ese dígito.",
         (f"Un solo segmento está encendido en los cinco pares y apagado en los cinco impares: "
            f"{nombre_solo}." if nombre_solo else
            "No hay ningún segmento que esté en «sí» en las cinco columnas de los pares y en «no» en las "
            "cinco de los impares.")])


def texto_liston(liston):
    # El listón puede salir cero, y «-0» no lo escribe nadie.
    return f"{liston:+.0f}" if abs(liston) > 0.5 else "0"


def bloque_comite(X, w, b):
    pesos = tabla_editorial(
        "El comité que sí lo consigue, con los siete segmentos",
        ["el segmento", "puntos que suma"],
        [[nombre, f"{peso:+.0f}"] for nombre, peso in zip(SEGMENTOS, w)], "id",
        ["Estos pesos los ha aprendido una máquina con la regla del capítulo 2, empezando de cero. "
         f"Para decir «par», el total tiene que pasar de {texto_liston(-b)}."])
    filas = []
    for d in range(10):
        total = X[d] @ w + b
        filas.append([str(d), f"{total:+.0f}", "par" if total > 0 else "impar",
                      "sí" if (total > 0) == (d % 2 == 0) else "NO"])
    cuenta = tabla_editorial(
        "La cuenta de los diez dígitos con ese comité",
        ["dígito", "total", "dice", "¿acierta?"], filas, "cdcc",
        ["Total: los puntos de los segmentos encendidos, menos el listón."])
    return pesos, cuenta


def selftest():
    fallos = []
    X = tabla_de_segmentos()

    # 1. TEST NULO — una partición que ninguna raya separa. Si el montaje dijera que sí,
    #    estaría diciendo que sí a todo y la medición no valdría nada.
    imposible = separable(X, PARTICION_IMPOSIBLE)
    print(f"[1] test nulo         una partición que no se puede separar: "
          f"{'la separa, MAL' if imposible else 'no la separa, bien'}")
    if imposible:
        fallos.append("test nulo: dice separar una partición que no es separable")

    # 2. SEÑAL IMPLANTADA — par contra impar tiene que salir, y con los diez bien.
    y = np.array([1.0 if d % 2 == 0 else -1.0 for d in range(10)])
    w, b = entrenar(X, y)
    aciertos = 0 if w is None else int(sum((X[d] @ w + b > 0) == (d % 2 == 0)
                                          for d in range(10)))
    print(f"[2] señal implantada  par contra impar: "
          f"{'no converge' if w is None else f'converge y acierta {aciertos} de 10'}")
    if w is None or aciertos != 10:
        fallos.append("señal implantada: par contra impar debería separarse con los diez bien")

    # 3. INVARIANTE DEL DOMINIO — una raya separa dos grupos; si se cambia de lado a los
    #    dos grupos, la misma raya sirve girada. Si una partición se separa y su contraria
    #    no, el comprobador no está midiendo una raya.
    malas = []
    for bits in [(0,1,0,1,0,1,0,1,0,1), (1,1,0,0,0,0,0,0,0,0), PARTICION_IMPOSIBLE]:
        contraria = tuple(1 - x for x in bits)
        if separable(X, bits) != separable(X, contraria):
            malas.append(bits)
    print(f"[3] invariante        una partición y su contraria salen igual: "
          f"{'sí' if not malas else f'NO en {malas}'}")
    if malas:
        fallos.append(f"invariante: {malas} no coincide con su contraria")

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
    ap.add_argument("--recuento", action="store_true",
                    help="cuenta cuántas de las 1.024 particiones separa una raya (tarda)")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    X = tabla_de_segmentos()
    nombre = un_solo_segmento_basta(X)
    print("\n".join(bloque_tabla(X, nombre)))

    y = np.array([1.0 if d % 2 == 0 else -1.0 for d in range(10)])
    w, b = entrenar(X, y)
    assert w is not None, "no encontró la raya; con estos siete segmentos debería encontrarla"
    for t in bloque_comite(X, w, b):
        print()
        print("\n".join(t))

    filas = [["segmento", "peso"]] + [[n, f"{p:.0f}"] for n, p in zip(SEGMENTOS, w)]
    filas.append(["liston", f"{-b:.0f}"])

    if args.recuento:
        n = sum(1 for bits in itertools.product([0, 1], repeat=10) if separable(X, bits))
        print()
        print("\n".join(tabla_editorial(
            "Cuántas maneras de partir los diez dígitos separa una raya", ["", "cuántas"],
            [["maneras de partir los diez dígitos en dos grupos", miles(2 ** 10)],
             ["las que separa una sola raya", miles(n)]], "id",
            ["Cada partición se prueba con el perceptrón del capítulo 2; si en "
             f"{miles(PASOS_MAX)} pasadas por los diez dígitos no encuentra la raya, se da por no separable."])))
        filas.append(["particiones_separables", str(n)])

    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(filas)
    print(f"\nEscrito {SALIDA_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
