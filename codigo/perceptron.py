#!/usr/bin/env python3
"""
Capítulo 2 — el perceptrón: qué aprende y dónde se estrella.

Tres mediciones:
  1. Tareas de visión al estilo de las de Rosenblatt (1960), con dígitos manuscritos reales.
  2. Los dieciséis montajes posibles de la lámpara del pasillo: cuántos aprende y cuántos no.
  3. El XOR: cien mil correcciones y qué pasa.

Uso:
    python perceptron.py
    python perceptron.py --selftest
"""

# ======================= CONSTANTES =======================

SEMILLA = 20260914
PASOS_MAX = 100_000          # correcciones antes de rendirse
TASA = 1.0                   # el perceptrón original corrige con paso fijo
FRACCION_PRUEBA = 0.3        # parte de los datos reservada para evaluar

PAREJAS_DIGITOS = [(0, 1), (1, 7), (3, 5), (4, 9)]
UMBRAL_SELFTEST = 0.95       # acierto mínimo exigido en la señal implantada

SALIDA_CSV = "perceptron.csv"

# ==========================================================

import argparse
import csv
import itertools
import sys

from formato import ANCHO_CAJA, comprobar_ancho, miles, coma, tabla_editorial
import numpy as np


def entrenar(X, y, pasos_max=PASOS_MAX, semilla=SEMILLA):
    """Perceptrón de Rosenblatt: si se equivoca, corrige; si acierta, no toca nada.
    Devuelve (pesos, sesgo, pasos_hasta_converger o None)."""
    assert X.ndim == 2, f"Se esperaba una matriz de ejemplos de 2 ejes; se encontraron {X.ndim}"
    assert set(np.unique(y)) <= {-1, 1}, \
        f"Se esperaban etiquetas en {{-1, 1}}; se encontraron {sorted(set(np.unique(y)))}"
    assert len(X) == len(y), \
        f"Se esperaban tantas etiquetas como ejemplos; se encontraron {len(y)} para {len(X)}"

    rng = np.random.default_rng(semilla)
    w = np.zeros(X.shape[1])
    b = 0.0
    orden = np.arange(len(X))
    pasos = 0
    for _ in range(pasos_max):
        rng.shuffle(orden)
        errores = 0
        for i in orden:
            if y[i] * (X[i] @ w + b) <= 0:
                w += TASA * y[i] * X[i]
                b += TASA * y[i]
                errores += 1
                pasos += 1
                if pasos >= pasos_max:
                    return w, b, None
        if errores == 0:
            return w, b, pasos
    return w, b, None


def acierto(X, y, w, b):
    pred = np.where(X @ w + b > 0, 1, -1)
    return float((pred == y).mean())


def cargar_digitos(enteros=False):
    """Los dígitos manuscritos de scikit-learn: 1.797 dibujos de 8 por 8 puntos.

    Por omisión la tinta va de 0 a 1 (se divide entre 16), que es lo que usan las redes que se
    entrenan de otra manera. Con `enteros=True` es la tinta tal como la da el conjunto: enteros de
    0 a 16. Orden 38 (23 oct 2026): lo que se entrena con la regla de Rosenblatt (la tabla «Dos
    dígitos», «¿es un cuatro?» y el comité del capítulo 3) usa la tinta entera, para que la regla
    sea literalmente la del capítulo 2: si no reconoce un cuatro, se suma a cada miembro la tinta
    que ve y el listón baja un punto; si toma un nueve por un cuatro, se resta y el listón sube un
    punto. Antes se dividía entre 16 y los pesos salían en múltiplos de 1/32 (y las cuentas del
    capítulo 3 había que multiplicarlas por 16, por 32 y por 512 para que salieran enteras)."""
    from sklearn.datasets import load_digits
    d = load_digits()
    if enteros:
        assert np.array_equal(d.data, np.round(d.data)) and d.data.min() == 0 and d.data.max() == 16, \
            "se esperaba tinta entera de 0 a 16; el conjunto de dígitos ha cambiado"
        return d.data, d.target
    return d.data / 16.0, d.target


def tarea_digitos(X, t, a, b_dig, rng):
    mask = (t == a) | (t == b_dig)
    Xs, ys = X[mask], np.where(t[mask] == a, 1, -1)
    idx = rng.permutation(len(Xs))
    corte = int(len(Xs) * (1 - FRACCION_PRUEBA))
    tr, te = idx[:corte], idx[corte:]
    w, sesgo, pasos = entrenar(Xs[tr], ys[tr])
    return {
        "ejemplos_entrenamiento": len(tr),
        "acierto_prueba": acierto(Xs[te], ys[te], w, sesgo),
        "converge": pasos is not None,
        "correcciones": pasos,
    }


TABLA_DOS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)


def reglas_de_dos_interruptores():
    """Los dieciséis montajes posibles de la lámpara del pasillo. Devuelve (aprendidas, fallidas)."""
    aprendidas, fallidas, constantes = [], [], []
    for bits in itertools.product([0, 1], repeat=4):
        y = np.where(np.array(bits) == 1, 1, -1)
        if len(set(bits)) == 1:                       # montaje constante: sin nada que separar
            aprendidas.append(bits)
            constantes.append(bits)                   # se cuentan, pero no se entrenan: hay que decirlo
            continue
        w, b, pasos = entrenar(TABLA_DOS, y, pasos_max=20_000)
        (aprendidas if pasos is not None else fallidas).append(bits)
    return aprendidas, fallidas, constantes


# Los dieciséis montajes de la lámpara del pasillo, nombrados en castellano corriente.
# Una sola lámpara, dieciséis maneras de montarla: por eso son montajes y no lámparas.
# Vive aquí, en el capítulo 2, y de aquí la importa el capítulo 3: una cosa, un nombre.
# Los dos interruptores del libro son los del pasillo: uno abajo y otro arriba. Las
# cuatro casillas de cada montaje van en el orden de TABLA_DOS: los dos bajados, solo
# el de arriba subido, solo el de abajo subido, los dos subidos.
# Cada nombre completa la frase «la luz se enciende…», que es como los presenta la figura
# de los dieciséis montajes; por eso casi todos empiezan por «con» o «solo con».
NOMBRES = {
    (0, 0, 0, 0): "nunca",
    (0, 0, 0, 1): "con los dos subidos",
    (0, 0, 1, 0): "solo con el de abajo subido",
    (0, 0, 1, 1): "con el de abajo subido",
    (0, 1, 0, 0): "solo con el de arriba subido",
    (0, 1, 0, 1): "con el de arriba subido",
    (0, 1, 1, 0): "en posiciones distintas",
    (0, 1, 1, 1): "con al menos uno subido",
    (1, 0, 0, 0): "con los dos bajados",
    (1, 0, 0, 1): "en la misma posición",
    (1, 0, 1, 0): "con el de arriba bajado",
    (1, 0, 1, 1): "salvo si solo el de arriba está subido",
    (1, 1, 0, 0): "con el de abajo bajado",
    (1, 1, 0, 1): "salvo si solo el de abajo está subido",
    (1, 1, 1, 0): "con al menos uno bajado",
    (1, 1, 1, 1): "siempre",
}

# ---- El aspecto del bloque de las dieciséis -------------------------------------
TITULO_TABLA = "LAS DIECISÉIS MANERAS DE MONTAR LA LÁMPARA DEL PASILLO"
FILA_ABAJO   = "el interruptor de abajo"
FILA_ARRIBA  = "el interruptor de arriba"
FILA_NOMBRES = "¿cuándo se enciende?"
MARCA_FALLO  = " *"
NOTA_FALLO   = "* con estos dos el perceptrón no puede"
# ---------------------------------------------------------------------------------

ESQUINAS_PERCEPTRON = [(0, 0), (0, 1), (1, 0), (1, 1)]   # el mismo orden que TABLA_DOS
REJILLA_RAYAS = 241                                       # finura de la búsqueda a lo bruto


def hay_raya_que_separa(encienden, esquinas=None):
    """¿Existe una raya que deje las esquinas encendidas a un lado y las apagadas al otro?

    Con cuatro esquinas se puede comprobar a lo bruto: se prueban muchas rayas y se mira si
    alguna lo consigue. No es una demostración, es una comprobación; la demostración está en
    el dibujo.

    Vive aquí, en el capítulo 2, porque aquí es donde se usa para lo importante: comprobar
    que los montajes que el perceptrón aprende son exactamente los que una raya separa."""
    esquinas = ESQUINAS_PERCEPTRON if esquinas is None else esquinas
    rejilla = np.linspace(-6, 6, REJILLA_RAYAS)
    for a in rejilla:
        for b in rejilla:
            for c in rejilla[::4]:
                lados = {(x, y): (a * x + b * y + c) > 0 for x, y in esquinas}
                if all(lados[p] for p in encienden) and \
                   not any(lados[p] for p in esquinas if p not in encienden):
                    return True
    return False


def reglas_que_una_raya_separa():
    """Los montajes de la lámpara del pasillo que se pueden resolver con una sola raya, buscados
    sobre la geometría, sin entrenar nada. Es el otro lado del puente."""
    separables = []
    for bits in itertools.product([0, 1], repeat=4):
        encienden = [e for e, b in zip(ESQUINAS_PERCEPTRON, bits) if b]
        if not encienden or len(encienden) == 4:      # las constantes: no hay nada que separar
            separables.append(bits)
            continue
        if hay_raya_que_separa(encienden):
            separables.append(bits)
    return separables


def nombre_regla(bits):
    return NOMBRES[tuple(bits)]


def tabla_de_las_dieciseis(aprendidas, fallidas, constantes):
    """Las dieciséis, enteras, para que el lector las cuente con el dedo en vez de
    creerse que son dieciséis. El libro cita este bloque literal (regla 6).

    El bloque tiene que explicarse solo: el lector puede encontrárselo al volver una
    página, sin el párrafo que lo presenta delante. Por eso las dos primeras filas
    dicen, con las palabras completas, qué interruptor es y en qué posición está en
    cada columna. Un rótulo como «las cuatro situaciones» no vale: no dice de qué.

    Y tiene que caber en la caja de texto de un libro de 6 por 9 pulgadas. Medido,
    no supuesto: ver ANCHO_CAJA."""
    ancho = max(
        max(len(n) for n in NOMBRES.values()),
        len(FILA_ABAJO), len(FILA_ARRIBA), len(FILA_NOMBRES),
    ) + 2
    col = 7
    abajo  = ["bajado", "bajado", "subido", "subido"]   # en el orden de TABLA_DOS
    arriba = ["bajado", "subido", "bajado", "subido"]

    def fila(etiqueta, celdas):
        return f"{etiqueta:<{ancho}}" + "".join(f"{c:^{col}}" for c in celdas)

    lineas = [
        TITULO_TABLA,
        "",
        fila(FILA_ABAJO, abajo).rstrip(),
        fila(FILA_ARRIBA, arriba).rstrip(),
        "",
        fila(FILA_NOMBRES, ["-" * (col - 1)] * 4).rstrip(),
    ]
    for bits in sorted(NOMBRES, key=lambda b: (b[0], b[1], b[2], b[3])):
        # La marca va pegada al nombre, dentro de su columna, para que la fila marcada mida
        # lo mismo que las demás y quepa en la caja; la nota de abajo dice qué significa.
        nombre = NOMBRES[bits] + (MARCA_FALLO if bits in fallidas else "")
        lineas.append(fila(nombre, ["sí" if x else "no" for x in bits]).rstrip())
    if fallidas:
        lineas += ["", NOTA_FALLO]

    # INVARIANTE DEL FORMATO: si una línea se sale de la caja, el libro la imprime
    # partida o pisando el margen, y el lector ve una tabla rota. Que reviente aquí.
    return comprobar_ancho(lineas)


def medir_xor():
    y = np.where(np.array([0, 1, 1, 0]) == 1, 1, -1)
    w, b, pasos = entrenar(TABLA_DOS, y)
    return {
        "converge": pasos is not None,
        "correcciones": PASOS_MAX if pasos is None else pasos,
        "acierto_final": acierto(TABLA_DOS, y, w, b),
    }


def selftest():
    fallos = []
    rng = np.random.default_rng(SEMILLA)
    X, t = cargar_digitos(enteros=True)       # con la tinta con la que se entrena en main()

    # 1. TEST NULO — etiquetas permutadas al azar: no hay nada que aprender, el acierto en
    #    datos no vistos debe quedarse cerca del azar.
    mask = (t == 0) | (t == 1)
    Xs = X[mask]
    ys = rng.permutation(np.where(t[mask] == 0, 1, -1))
    idx = rng.permutation(len(Xs)); corte = int(len(Xs) * 0.7)
    w, b, _ = entrenar(Xs[idx[:corte]], ys[idx[:corte]], pasos_max=20_000)
    a_nulo = acierto(Xs[idx[corte:]], ys[idx[corte:]], w, b)
    print(f"[1] test nulo         etiquetas permutadas: acierto {a_nulo:.3f} (azar = 0,5)")
    if a_nulo > 0.65:
        fallos.append(f"test nulo: {a_nulo:.3f} de acierto con etiquetas al azar; algo filtra información")

    # 2. SEÑAL IMPLANTADA — una regla lineal evidente debe recuperarse por completo.
    Xi = rng.normal(size=(400, 5))
    yi = np.where(Xi[:, 2] > 0, 1, -1)          # solo la tercera columna decide
    w, b, pasos = entrenar(Xi, yi, pasos_max=20_000)
    a_imp = acierto(Xi, yi, w, b)
    dominante = int(np.argmax(np.abs(w)))
    print(f"[2] señal implantada  acierto {a_imp:.3f}; la entrada que más pesa es la {dominante} (implantada: 2)")
    if a_imp < UMBRAL_SELFTEST or dominante != 2:
        fallos.append(
            f"señal implantada: se esperaba acierto >= {UMBRAL_SELFTEST} con la entrada 2 dominante; "
            f"se obtuvo {a_imp:.3f} con la {dominante}"
        )

    # 3. INVARIANTE DEL DOMINIO — teorema de convergencia: sobre datos separables el
    #    perceptrón converge y deja CERO errores sobre lo que ha entrenado.
    w, b, pasos = entrenar(Xi, yi, pasos_max=50_000)
    err = int((np.where(Xi @ w + b > 0, 1, -1) != yi).sum())
    print(f"[3] invariante        converge en {pasos} correcciones y deja {err} errores de entrenamiento")
    if pasos is None or err != 0:
        fallos.append(f"invariante: sobre datos separables debe converger con 0 errores; {pasos=} {err=}")

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

    rng = np.random.default_rng(SEMILLA)
    X, t = cargar_digitos(enteros=True)       # tinta de 0 a 16, tal como viene (Orden 38)
    print(f"Dígitos manuscritos: {len(X)} imágenes de 8 por 8 puntos.\n")

    filas = []
    print("--- 1. DISTINGUIR DOS DÍGITOS ---")
    # Las etiquetas son las que lleva la tabla del libro, y el acierto va en porcentaje,
    # para que la tabla del capítulo sea literalmente esto y el verificador pueda
    # comprobarla fila por fila. Lo que el libro enseña y lo que el programa imprime
    # tienen que ser la misma cadena de caracteres.
    # L24 (9 de octubre de 2026): en tabla editorial. La columna «converge» pasa a una nota,
    # porque el capítulo no la enseña y en las cuatro dice lo mismo; si alguna no converge, la
    # nota lo dice y la tabla lo marca.
    editorial = []
    for a, b_dig in PAREJAS_DIGITOS:
        r = tarea_digitos(X, t, a, b_dig, rng)
        pct = f"{100 * r['acierto_prueba']:.0f} %"
        # Las correcciones son la otra mitad de la historia: el capítulo las compara con las
        # cien mil del o exclusivo, y un número que el libro compara tiene que estar impreso.
        editorial.append([f"distinguir un {a} de un {b_dig}", str(r["ejemplos_entrenamiento"]),
                          pct, miles(r["correcciones"]) if r["converge"] else "no se queda quieto"])
        filas.append(["digitos", f"{a}v{b_dig}", r["ejemplos_entrenamiento"],
                      f"{r['acierto_prueba']:.4f}", r["converge"]])
    convergen = sum(1 for f in filas if f[0] == "digitos" and f[4])
    # La clave de las columnas, debajo de la tabla y impresa por el programa (regla 9). Un
    # rótulo que dice lo que mide —«ejemplos para aprender»— no cabe en una columna de una
    # caja de 68 caracteres, y apilarlo en dos renglones no se puede verificar: al apilar por
    # columnas, los renglones entrelazan palabras de columnas distintas. Así que el rótulo va
    # corto arriba y lo que significa va entero aquí abajo, en líneas que el libro copia como
    # copia cualquier otra. Antes esto lo escribía el libro por su cuenta: fallo 4.32.
    for l in tabla_editorial(
            "Dos dígitos, con la regla de Rosenblatt",
            ["tarea", "ejemplos", "acierto", "correcciones"], editorial, "iddd",
            ["Cada fila es una prueba distinta. Los ejemplos son los dibujos con los que aprendió; "
             "el acierto se mide sobre otros que no vio mientras aprendía, y las correcciones son "
             "las veces que hubo que retocarle los pesos.",
             f"Deja de corregirse solo en {convergen} de las {len(PAREJAS_DIGITOS)} tareas."]):
        print(l)

    print("\n--- 2. LAS DIECISÉIS MANERAS DE MONTAR LA LÁMPARA ---")
    aprendidas, fallidas, constantes = reglas_de_dos_interruptores()
    for linea in tabla_de_las_dieciseis(aprendidas, fallidas, constantes):
        print(linea)
    entrenadas = len(aprendidas) - len(constantes)
    print()
    # L24 (9 de octubre de 2026): el resumen, en tabla editorial; antes, dos renglones que el
    # capítulo copiaba en una cita.
    assert len(aprendidas) + len(fallidas) == 16, "los montajes no suman dieciséis"
    for l in tabla_editorial(
            "Lo que hace el perceptrón con los dieciséis montajes",
            ["", "montajes"],
            [["los aprende entrenando", str(entrenadas)],
             ["los resuelve sin nada que aprender", str(len(constantes))],
             ["no puede", str(len(fallidas))],
             ["**en total**", "**16**"]], "id"):
        print(l)
    filas.append(["reglas_dos", "aprendidas", len(aprendidas), "", ""])
    filas.append(["reglas_dos", "entrenadas", entrenadas, "", ""])
    filas.append(["reglas_dos", "constantes", len(constantes), "", ""])
    filas.append(["reglas_dos", "fallidas", len(fallidas), "", ""])

    print("\n--- 3. EL PUENTE: LO QUE APRENDE Y LO QUE UNA RAYA SEPARA ---")
    separables = reglas_que_una_raya_separa()
    iguales = set(separables) == set(aprendidas)
    # Ancho fijo para las etiquetas: si cada una lleva el suyo, los números quedan en
    # tres columnas distintas y las dos cifras que hay que comparar dejan de estar una
    # encima de la otra, que es lo único que este bloque tiene que enseñar.
    # «entrenando» aquí era falso: este 14 es todo lo que el perceptrón resuelve, constantes
    # incluidas. Entrenando aprende 12, y así lo dice la tabla de arriba. (L24, 9 de octubre de
    # 2026: en tabla editorial.)
    for l in tabla_editorial(
            "Lo que resuelve el perceptrón y lo que separa una raya",
            ["", "montajes"],
            [["los que el perceptrón resuelve", str(len(aprendidas))],
             ["los que una sola raya puede separar", str(len(separables))],
             ["¿son exactamente los mismos?", "sí" if iguales else "no"]], "id"):
        print(l)
    filas.append(["puente", "aprendidas", len(aprendidas), "", ""])
    filas.append(["puente", "separables", len(separables), "", ""])
    filas.append(["puente", "coinciden", iguales, "", ""])

    print("\n--- 4. EL O EXCLUSIVO ---")
    r = medir_xor()
    print(f"correcciones: {miles(r['correcciones'])}   "
          f"converge: {'sí' if r['converge'] else 'NO'}   "
          f"acierta {round(4 * r['acierto_final'])} de las 4 posiciones")
    filas.append(["xor", "", r["correcciones"], f"{r['acierto_final']:.4f}", r["converge"]])

    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["tarea", "detalle", "n", "acierto", "converge"]] + filas)
    print(f"\nEscrito {SALIDA_CSV}")


if __name__ == "__main__":
    main()
