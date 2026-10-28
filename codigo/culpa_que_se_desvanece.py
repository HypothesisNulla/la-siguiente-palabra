#!/usr/bin/env python3
"""
Capítulo 4 — ¿de qué depende que la culpa se desvanezca?

El capítulo dice que el desvanecimiento «es una consecuencia directa del atajo»: que la
culpa se evapora porque al retroceder una capa se multiplica por un número menor que uno,
y que eso viene de repartir la culpa hacia atrás. Ocho líneas más abajo dice que lo que
resolvió el problema fueron mejores maneras de repartir los números iniciales y funciones
distintas dentro de cada neurona. Las dos cosas no pueden ser verdad a la vez: si el
desvanecimiento saliera del atajo, cambiar lo de dentro de la neurona no podría arreglarlo,
porque el atajo sigue siendo exactamente el mismo.

Esto lo mide. La retropropagación es LA MISMA en los cuatro montajes —el mismo código, la
misma pasada hacia delante y la misma pasada hacia atrás—; lo único que cambia es cómo se
reparten los números iniciales y qué función hay dentro de cada neurona de en medio.

Lo que sale: con los números iniciales de los ochenta intactos y cambiando SOLO la función
de dentro, el factor de veinte capas pasa de quince millones a once. No es el atajo.

Uso:
    python culpa_que_se_desvanece.py
    python culpa_que_se_desvanece.py --selftest
"""

# ======================= CONSTANTES =======================

SEMILLA = 20260914          # la misma que retropropagacion.py, para poder comparar
PROFUNDIDADES = [2, 5, 10, 20]
ANCHO = 16
EJEMPLOS = 64

# Los cuatro montajes. «reparto» es por cuánto se multiplican los números iniciales:
#   uno         los números tal como salen, uniformes entre -1 y +1 (lo de los ochenta)
#   por entrada se encogen según cuántas entradas tiene la neurona, que es lo que se hizo
#               después para que la señal no crezca ni se apague al atravesar capas
MONTAJES = [
    {"clave": "ochenta",   "reparto": "uno",         "ganancia": 1.0, "dentro": "sigmoide"},
    {"clave": "repartida", "reparto": "por entrada", "ganancia": 1.0, "dentro": "sigmoide"},
    {"clave": "funcion",   "reparto": "uno",         "ganancia": 1.0, "dentro": "rampa"},
    {"clave": "las_dos",   "reparto": "por entrada", "ganancia": 2.0, "dentro": "rampa"},
]

# Lo que ya imprime retropropagacion.py para el montaje clásico. Si esto deja de salir
# igual, este programa está midiendo otra cosa y no vale para comparar con el capítulo.
FACTORES_DEL_CAPITULO = [65, 777, 7192, 15048831]
TOL_GRADIENTE = 1e-6

SALIDA_CSV = "culpa_que_se_desvanece.csv"

# ---- El aspecto del bloque. Los rótulos están aquí, con nombre, porque son texto que se
# ---- lee solo: el que se encuentre esta tabla al volver una página tiene que poder saber
# ---- qué son sus filas y sus columnas sin haber leído el párrafo de antes (regla 9).
TITULO = "¿De qué depende que la culpa se desvanezca?"
# L24 (28 de septiembre): el primer renglón dice qué es cada número («cuántas VECES menos»,
# hallazgo A19); las filas ya no se llaman «otro reparto» ni «otra función» (A20, A21: «reparto»
# era la tercera cosa con ese nombre en ocho páginas, y «función» es una palabra que el lector no
# tiene); «uniformes» se dice con palabras, y los pesos de arranque llevan su tamaño.
# L24 (9 de octubre): tabla editorial (regla 6 ter); el subtítulo y el pie pasan a notas.
CAB_CAPAS = "capas"
CAB_MONTAJE = "cómo se arma"
CAB = {"ochenta": "como en los 80", "repartida": "más pequeños",
       "funcion": "el codo", "las_dos": "las dos cosas"}


def _entre(montaje, entradas=ANCHO):
    """Entre qué valores quedan los pesos de arranque de una neurona con tantas entradas."""
    if montaje["reparto"] == "uno":
        return 1.0
    return float(np.sqrt(montaje["ganancia"] / entradas))


def notas():
    """Las notas de debajo de la tabla, en dos párrafos (Orden 50): qué es cada número y qué es
    cada fila. Los tamaños de los pesos de arranque los calcula el programa."""
    chico = _entre(MONTAJES[1])
    doble = _entre(MONTAJES[3])
    c = lambda x: f"{x:.2f}".replace(".", ",")
    return [
        "Cada número es cuántas veces menos culpa le llega a la primera capa de líneas que a la "
        "última; si es menor que 1, a la primera le llega más. En las cuatro filas la culpa se "
        "reparte hacia atrás exactamente igual: solo cambian los pesos de arranque y lo que hay "
        "dentro de cada neurona, menos la final.",
        "La primera fila, «como en los 80», es la de antes: los pesos de arranque, al azar entre "
        "−1 y +1, y la rampa corta. La segunda encoge esos pesos según cuántas entradas tiene "
        f"cada neurona: aquí quedan entre −{c(chico)} y +{c(chico)}. La tercera deja los pesos de "
        "los 80 y cambia la rampa por el codo. La cuarta hace las dos cosas, con los pesos entre "
        f"−{c(doble)} y +{c(doble)}."]

# ==========================================================

import argparse
import csv
import sys

import numpy as np

from formato import tabla_editorial, miles, coma
from retropropagacion import sigmoide


class RedDeMontaje:
    """La misma red de `retropropagacion.py`, con dos piezas sueltas: por cuánto se
    multiplican los números iniciales y qué función hay dentro de cada neurona de en medio.

    Los números iniciales se SACAN igual en los cuatro montajes —misma semilla, mismo orden
    de tiradas— y después se encogen. Así los cuatro parten de la misma red y la única
    diferencia es la que se quiere medir. La última capa lleva siempre la función de siempre,
    porque su respuesta es un sí o un no entre 0 y 1 y el error se mide contra eso: si
    cambiara también ahí, los cuatro montajes no estarían resolviendo el mismo problema."""

    def __init__(self, tamanos, montaje, semilla=SEMILLA):
        rng = np.random.default_rng(semilla)
        self.tamanos = tamanos
        self.dentro = montaje["dentro"]
        W = [rng.uniform(-1, 1, (a, b)) for a, b in zip(tamanos[:-1], tamanos[1:])]
        if montaje["reparto"] == "por entrada":
            W = [w * np.sqrt(montaje["ganancia"] / w.shape[0]) for w in W]
        elif montaje["reparto"] != "uno":
            raise SystemExit(f"reparto desconocido: {montaje['reparto']}")
        self.W = W
        self.b = [np.zeros(b) for b in tamanos[1:]]

    def _fuera(self, z, ultima):
        if ultima or self.dentro == "sigmoide":
            return sigmoide(z)
        return np.maximum(0.0, z)

    def _pendiente(self, a, ultima):
        if ultima or self.dentro == "sigmoide":
            return a * (1 - a)
        return (a > 0).astype(float)

    def adelante(self, X):
        act = [X]
        ultimo = len(self.W) - 1
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            act.append(self._fuera(act[-1] @ W + b, k == ultimo))
        return act

    def error(self, X, y):
        return float(np.mean((self.adelante(X)[-1].ravel() - y) ** 2))

    def gradiente_retro(self, X, y):
        """El atajo, sin tocar: una pasada adelante, una atrás, y la culpa de todos los pesos."""
        act = self.adelante(X)
        n = len(X)
        ultimo = len(self.W) - 1
        delta = (act[-1] - y.reshape(-1, 1)) * self._pendiente(act[-1], True) * (2.0 / n)
        gW = [None] * len(self.W)
        for k in range(ultimo, -1, -1):
            gW[k] = act[k].T @ delta
            if k > 0:
                delta = (delta @ self.W[k].T) * self._pendiente(act[k], False)
        return gW

    def gradiente_fuerza_bruta(self, X, y, epsilon=1e-5):
        """Mover cada peso y ver cuánto cambia el error. Dos pasadas por peso."""
        gW = [np.zeros_like(W) for W in self.W]
        for k, W in enumerate(self.W):
            for i in np.ndindex(W.shape):
                original = W[i]
                W[i] = original + epsilon
                mas = self.error(X, y)
                W[i] = original - epsilon
                menos = self.error(X, y)
                W[i] = original
                gW[k][i] = (mas - menos) / (2 * epsilon)
        return gW


def escribir_veces(v):
    """El factor, en castellano.

    Por encima de diez va entero, como lo escribe `retropropagacion.py` y como lo imprime
    el libro: así el 65 de la primera fila es el mismo 65 que el lector ya ha visto, y no
    un «64,7» que le haga dudar de si son dos medidas distintas. Por debajo de diez va con
    un decimal, porque redondear 0,4 a «0» diría que a la primera capa no le llega culpa, y
    lo que ese 0,4 dice es lo contrario: que le llega MÁS que a la última."""
    # L24: por encima del billón, en billones. Con diecisiete cifras, las últimas eran ruido del
    # ordenador —la fila de veinte capas de «más pequeños» salía ...505.184 en una máquina y
    # ...505.312 en otra—, y además el lector no sabía leer el número (hallazgo A22).
    if v >= 1e12:
        return f"{miles(round(v / 1e12))} billones"
    return coma(v, 1) if v < 10 else miles(round(v))


def datos(rng, ejemplos=EJEMPLOS, ancho=ANCHO):
    return rng.normal(size=(ejemplos, ancho)), (rng.random(ejemplos) > 0.5).astype(float)


def factores(montaje, profundidades=PROFUNDIDADES):
    """Cuántas veces menos culpa le llega a la primera capa que a la última, capa por capa.

    Se mide igual que en `retropropagacion.py`: el promedio del valor absoluto de la culpa
    de los pesos de cada capa, y el cociente entre la última y la primera."""
    rng = np.random.default_rng(SEMILLA)
    salida = []
    for prof in profundidades:
        red = RedDeMontaje([ANCHO] + [ANCHO] * prof + [1], montaje)
        X, y = datos(rng)
        magnitudes = [float(np.abs(g).mean()) for g in red.gradiente_retro(X, y)]
        primera, ultima = magnitudes[0], magnitudes[-1]
        salida.append({"capas": prof, "primera": primera, "ultima": ultima,
                       "veces": ultima / primera if primera > 0 else float("inf")})
    return salida


def nota_de_una_fila():
    """La nota de la tabla de una sola fila (Orden 50, escrita por Claude y aprobada por Carlos)."""
    return ("Cada número es cuántas veces menos culpa le llega a la primera capa de líneas, la que "
            "sale de la entrada, que a la última, la que llega a la final. Las redes están recién "
            "arrancadas, sin entrenar, y armadas como en los 80: los pesos de arranque, al azar "
            "entre −1 y +1, y la rampa corta en cada neurona.")


def imprimir(tabla):
    """Las filas son las cuatro maneras de armar la red y las columnas la profundidad, con un
    rótulo de grupo («capas en medio») encima de las cuatro.

    Los rótulos y las notas no son adorno: quien se encuentre esta tabla sin haber leído nada
    tiene que poder saber qué son sus números y qué cambia de una fila a otra (regla 9)."""
    filas = [[CAB[m["clave"]]] + [escribir_veces(f["veces"]) for f in tabla[m["clave"]]]
             for m in MONTAJES]
    rotulos = [CAB_MONTAJE] + [f"{CAB_CAPAS}: {p + 1}" for p in PROFUNDIDADES]   # N capas en medio + la final
    # Orden 51: primero, la tabla de una sola fila («como en los 80»), la que el capítulo 4 enseña
    # antes de las cuatro; las cifras son las de la primera fila de la de abajo, no se recalculan.
    print("\n".join(tabla_editorial(TITULO, rotulos, filas[:1], "i" + "d" * len(PROFUNDIDADES),
                                    [nota_de_una_fila()])))
    print()
    print("\n".join(tabla_editorial(TITULO, rotulos, filas, "i" + "d" * len(PROFUNDIDADES),
                                    notas())))


def guardar(tabla):
    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["montaje", "capas_en_medio", "culpa_primera_capa",
                    "culpa_ultima_capa", "veces_menor"])
        for m in MONTAJES:
            for f in tabla[m["clave"]]:
                w.writerow([m["clave"], f["capas"], f"{f['primera']:.3e}",
                            f"{f['ultima']:.3e}", f"{f['veces']:.1f}"])


# ============================ SELFTEST ============================

def selftest():
    fallos = []

    # [1] TEST NULO — una red SIN capas de en medio: la primera capa ES la última, así que
    #     no hay ni un sitio por donde la culpa pueda desvanecerse. El factor tiene que
    #     salir exactamente 1 en los tres montajes. Si aquí ya sale algo distinto de 1, lo
    #     que este programa mide no es el camino de la culpa.
    unos = []
    for m in MONTAJES:
        red = RedDeMontaje([ANCHO, 1], m)
        X, y = datos(np.random.default_rng(SEMILLA))
        g = [float(np.abs(x).mean()) for x in red.gradiente_retro(X, y)]
        unos.append(g[-1] / g[0])
    print(f"[1] test nulo         sin capas de en medio el factor sale: "
          + ", ".join(coma(u, 3) for u in unos) + " (tiene que ser 1)")
    if any(abs(u - 1.0) > 1e-12 for u in unos):
        fallos.append(f"test nulo: sin capas de en medio el factor tiene que ser 1 y salió {unos}")

    # [2] SEÑAL IMPLANTADA — el montaje clásico tiene que reproducir EXACTAMENTE los cuatro
    #     factores que ya imprime retropropagacion.py y que están en el libro. Si no los
    #     reproduce, este programa mide otra cosa y no sirve para compararlo con el capítulo.
    clasico = [round(f["veces"]) for f in factores(MONTAJES[0])]
    print(f"[2] señal implantada  el montaje clásico da {clasico}; el capítulo dice "
          f"{FACTORES_DEL_CAPITULO}")
    if clasico != FACTORES_DEL_CAPITULO:
        fallos.append(f"señal implantada: esperaba {FACTORES_DEL_CAPITULO}; encontré {clasico}")

    # [3] INVARIANTE DEL DOMINIO — en los cuatro montajes, la culpa que sale de la pasada hacia
    #     atrás tiene que ser la misma que la de la fuerza bruta. Si no lo es, lo que se está
    #     midiendo no es la culpa, y toda la tabla de arriba es ruido con formato de tabla.
    peores = []
    for m in MONTAJES:
        red = RedDeMontaje([4, 5, 5, 1], m)
        r = np.random.default_rng(SEMILLA + 7)
        X = r.normal(size=(16, 4)); y = (r.random(16) > 0.5).astype(float)
        dif = max(float(np.abs(a - b).max())
                  for a, b in zip(red.gradiente_retro(X, y), red.gradiente_fuerza_bruta(X, y)))
        peores.append(dif)
    print(f"[3] invariante        atajo contra fuerza bruta, diferencia máxima en los cuatro "
          f"montajes: " + ", ".join(f"{d:.1e}".replace(".", ",") for d in peores))
    if any(d > TOL_GRADIENTE for d in peores):
        fallos.append(f"invariante: los dos métodos difieren en {peores}, por encima de {TOL_GRADIENTE:.0e}")

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
    args = p.parse_args()
    if args.selftest:
        return selftest()
    tabla = {m["clave"]: factores(m) for m in MONTAJES}
    imprimir(tabla)
    guardar(tabla)
    print(f"\nEscrito {SALIDA_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
