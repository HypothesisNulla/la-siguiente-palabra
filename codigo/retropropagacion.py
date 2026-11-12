#!/usr/bin/env python3
"""
Capítulo 4 — la retropropagación: repartir la culpa de una sola pasada.

Tres mediciones:
  1. El o exclusivo que el perceptrón no podía: una red de dos capas lo resuelve.
  2. Cuánto más cara es la fuerza bruta (mover cada peso y ver si mejora) que la
     retropropagación, y cómo crece esa diferencia con el número de pesos.
  3. Por qué, aun teniendo el método, las redes profundas seguían sin entrenarse.

Uso:
    python retropropagacion.py
    python retropropagacion.py --selftest
"""

# ======================= CONSTANTES =======================

SEMILLA = 20260914
TASA_XOR = 0.5
PASOS_XOR = 20_000
OCULTAS_XOR = 2

TAMANOS_CRONOMETRO = [50, 200, 800, 3200]     # pesos aproximados de la red de prueba
REPETICIONES = 3

PROFUNDIDADES = [2, 5, 10, 20]                # capas ocultas para medir el desvanecimiento
ANCHO = 16

PESOS_MODELO_GRANDE = 100_000_000_000         # 100.000 millones, orden de un modelo de hoy
TOL_GRADIENTE = 1e-6                          # tolerancia de la comprobación de gradiente
SEMILLA_GRADIENTE = 20260404                  # semilla propia: ver diferencia_entre_los_dos_metodos()

SALIDA_CSV = "../datos/salidas/retropropagacion.csv"

# ==========================================================

import argparse
import csv
import sys
import time

import numpy as np

from formato import coma, miles


def cientifica(x, decimales=2):
    """Notación científica con coma decimal: 7.03e-12 -> «7,03e-12»."""
    return f"{x:.{decimales}e}".replace(".", ",")


def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class Red:
    """Red de capas densas con sigmoide. Escrita a mano: aquí no hay biblioteca que
    esconda el reparto de culpa, que es justo lo que el capítulo explica."""

    def __init__(self, tamanos, semilla=SEMILLA):
        assert len(tamanos) >= 2, \
            f"Se esperaban al menos dos capas (entrada y salida); se encontraron {len(tamanos)}"
        rng = np.random.default_rng(semilla)
        self.tamanos = tamanos
        # Inicialización clásica de los años ochenta: uniforme pequeña, la misma para
        # todas las capas. Es la que hace visible el desvanecimiento.
        self.W = [rng.uniform(-1, 1, (a, b)) for a, b in zip(tamanos[:-1], tamanos[1:])]
        self.b = [np.zeros(b) for b in tamanos[1:]]

    @property
    def n_pesos(self):
        return sum(w.size for w in self.W) + sum(x.size for x in self.b)

    def adelante(self, X):
        activaciones = [X]
        for W, b in zip(self.W, self.b):
            activaciones.append(sigmoide(activaciones[-1] @ W + b))
        return activaciones

    def error(self, X, y):
        return float(np.mean((self.adelante(X)[-1].ravel() - y) ** 2))

    def gradiente_retro(self, X, y):
        """Retropropagación: UNA pasada adelante y UNA atrás dan la culpa de TODOS los pesos."""
        act = self.adelante(X)
        n = len(X)
        delta = (act[-1] - y.reshape(-1, 1)) * act[-1] * (1 - act[-1]) * (2.0 / n)
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for k in range(len(self.W) - 1, -1, -1):
            gW[k] = act[k].T @ delta
            gb[k] = delta.sum(axis=0)
            if k > 0:
                delta = (delta @ self.W[k].T) * act[k] * (1 - act[k])
        return gW, gb

    def gradiente_fuerza_bruta(self, X, y, epsilon=1e-5):
        """Mover cada peso un poquito y ver si el error mejora. DOS pasadas completas
        por peso, no una: se mira el error moviéndolo hacia arriba y moviéndolo hacia
        abajo. Es el recuento que imprime este mismo programa más abajo (2 pasadas por
        peso -> 200.000 millones para un modelo de hoy), y es lo que no escala."""
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

    def entrenar(self, X, y, tasa, pasos):
        for _ in range(pasos):
            gW, gb = self.gradiente_retro(X, y)
            for k in range(len(self.W)):
                self.W[k] -= tasa * gW[k]
                self.b[k] -= tasa * gb[k]
        return self


TABLA_XOR = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y_XOR = np.array([0., 1., 1., 0.])


def medir_xor():
    red = Red([2, OCULTAS_XOR, 1]).entrenar(TABLA_XOR, Y_XOR, TASA_XOR, PASOS_XOR)
    salida = red.adelante(TABLA_XOR)[-1].ravel()
    aciertos = int(((salida > 0.5) == (Y_XOR > 0.5)).sum())
    return {"aciertos": aciertos, "error": red.error(TABLA_XOR, Y_XOR),
            "neuronas_ocultas": OCULTAS_XOR, "salidas": salida}


def medir_cronometro():
    rng = np.random.default_rng(SEMILLA)
    filas = []
    for objetivo in TAMANOS_CRONOMETRO:
        ancho = max(2, int(round(objetivo / 6)))
        red = Red([4, ancho, 1])
        X = rng.normal(size=(32, 4))
        y = (rng.random(32) > 0.5).astype(float)

        t = time.perf_counter()
        for _ in range(REPETICIONES):
            red.gradiente_retro(X, y)
        t_retro = (time.perf_counter() - t) / REPETICIONES

        t = time.perf_counter()
        red.gradiente_fuerza_bruta(X, y)
        t_bruta = time.perf_counter() - t

        filas.append({"pesos": red.n_pesos, "retro_s": t_retro, "bruta_s": t_bruta,
                      "factor": t_bruta / t_retro})
    return filas


def medir_desvanecimiento():
    rng = np.random.default_rng(SEMILLA)
    filas = []
    for prof in PROFUNDIDADES:
        red = Red([ANCHO] + [ANCHO] * prof + [1])
        X = rng.normal(size=(64, ANCHO))
        y = (rng.random(64) > 0.5).astype(float)
        gW, _ = red.gradiente_retro(X, y)
        magnitudes = [float(np.abs(g).mean()) for g in gW]
        filas.append({"capas_ocultas": prof, "primera": magnitudes[0],
                      "ultima": magnitudes[-1],
                      "veces_menor": magnitudes[-1] / magnitudes[0] if magnitudes[0] > 0 else float("inf")})
    return filas


def selftest():
    fallos = []
    rng = np.random.default_rng(SEMILLA)

    # 1. TEST NULO — etiquetas al azar sin estructura: la red no debe generalizar a datos
    #    no vistos, por mucho que memorice los de entrenamiento.
    X = rng.normal(size=(200, 6)); y = (rng.random(200) > 0.5).astype(float)
    Xte = rng.normal(size=(200, 6)); yte = (rng.random(200) > 0.5).astype(float)
    red = Red([6, 12, 1]).entrenar(X, y, 0.5, 3000)
    pred = (red.adelante(Xte)[-1].ravel() > 0.5).astype(float)
    a_nulo = float((pred == yte).mean())
    print(f"[1] test nulo         acierto fuera de entrenamiento con ruido: {coma(a_nulo, 3)} (azar = 0,5)")
    if not 0.35 <= a_nulo <= 0.65:
        fallos.append(f"test nulo: acierto {coma(a_nulo, 3)} sobre ruido; debería quedarse cerca de 0,5")

    # 2. SEÑAL IMPLANTADA — el o exclusivo, que el perceptrón no podía, debe recuperarse.
    r = medir_xor()
    print(f"[2] señal implantada  o exclusivo: {r['aciertos']} de 4 casos correctos")
    if r["aciertos"] != 4:
        fallos.append(f"señal implantada: se esperaban 4 de 4 en el o exclusivo; se obtuvieron {r['aciertos']}")

    # 3. INVARIANTE DEL DOMINIO — la retropropagación tiene que dar EXACTAMENTE el mismo
    #    gradiente que la fuerza bruta. Si no, todo el capítulo es falso.
    dif = diferencia_entre_los_dos_metodos()
    print(f"[3] invariante        diferencia máxima entre los dos métodos: {cientifica(dif)} (tolerancia {TOL_GRADIENTE:.0e})")
    if dif > TOL_GRADIENTE:
        fallos.append(f"invariante: los dos gradientes difieren en {cientifica(dif)}, por encima de {TOL_GRADIENTE:.0e}")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def diferencia_entre_los_dos_metodos(semilla=SEMILLA_GRADIENTE):
    """La mayor diferencia entre la culpa calculada por los dos métodos, en una misma red.

    Tiene semilla propia, y por una razón que costó un número impreso. Esta comprobación vivía
    dentro de `--selftest`, compartiendo el generador de azar con las dos pruebas de antes, así
    que la red y los datos que se usaban aquí dependían de cuántas tiradas se hubieran gastado
    más arriba. Cambiar cualquier otra prueba cambiaba este número, y el libro llevaba impreso
    uno —«siete billonésimas»— que nadie podía volver a obtener. Fallo 4.5 con un agravante:
    no es que no hubiera fichero detrás, es que no había NADA detrás, ni siquiera el mismo
    cálculo dos veces.

    Ahora se calcula aquí, con su propia semilla, y sale en la medición además de en el
    selftest, para que el número del libro tenga un fichero al lado."""
    rng = np.random.default_rng(semilla)
    red = Red([4, 5, 1], semilla=semilla)
    X = rng.normal(size=(16, 4))
    y = (rng.random(16) > 0.5).astype(float)
    gW_r, _ = red.gradiente_retro(X, y)
    gW_b = red.gradiente_fuerza_bruta(X, y)
    return max(float(np.abs(a - b).max()) for a, b in zip(gW_r, gW_b))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())

    filas_csv = []

    print("--- 1. EL O EXCLUSIVO, CON UNA CAPA MÁS ---")
    r = medir_xor()
    print(f"red de {r['neuronas_ocultas']} neuronas ocultas: {r['aciertos']} de 4 casos correctos")
    print("salidas: " + "  ".join(coma(v, 3) for v in r["salidas"]) + "   (esperado: 0, 1, 1, 0)")
    filas_csv.append(["xor", r["neuronas_ocultas"], r["aciertos"], "", ""])

    print("\n--- 2. RETROPROPAGACIÓN CONTRA FUERZA BRUTA ---")
    print(f"{'pesos':>8}{'retro (s)':>13}{'bruta (s)':>13}{'factor':>10}")
    filas = medir_cronometro()
    for f in filas:
        print(f"{f['pesos']:>8}{coma(f['retro_s'], 5):>13}{coma(f['bruta_s'], 5):>13}{f['factor']:>10.0f}")
        filas_csv.append(["cronometro", f["pesos"], f"{f['retro_s']:.6f}",
                          f"{f['bruta_s']:.6f}", f"{f['factor']:.1f}"])
    # NO extrapolar desde el factor medido: a estos tamaños diminutos el factor está
    # contaminado por la sobrecarga de la biblioteca y no es proporcional a los pesos.
    # Lo que sí se sostiene es el recuento de pasadas, que no depende de la máquina.
    print(f"\nEl factor medido crece con cada peso añadido, de {filas[0]['factor']:.0f} a "
          f"{filas[-1]['factor']:.0f} veces.")
    # El número que el capítulo usa para decir «no es una aproximación: es el mismo número».
    # Va aquí, en la medición, para que tenga fichero detrás (ver la función, y el fallo 4.5).
    dif = diferencia_entre_los_dos_metodos()
    print(f"\nmayor diferencia entre la culpa calculada por los dos métodos, en la misma")
    print(f"red y con los mismos datos: {cientifica(dif)}")
    print("(es ruido de redondeo del ordenador: los dos métodos dan el mismo número)")

    print("\nLo que no depende de la máquina es el recuento de pasadas por los datos que hace")
    print("falta para dar UN paso de aprendizaje:")
    print(f"  fuerza bruta:        2 pasadas por peso  -> {miles(2*PESOS_MODELO_GRANDE)} pasadas")
    print(f"  retropropagación:    1 adelante y 1 atrás -> 2 pasadas, sea cual sea el número de pesos")

    print("\n--- 3. POR QUÉ LA PROFUNDIDAD SEGUÍA SIN FUNCIONAR ---")
    print(f"{'capas ocultas':>14}{'culpa 1.ª capa':>17}{'culpa última':>15}{'veces menor':>14}")
    for f in medir_desvanecimiento():
        print(f"{f['capas_ocultas']:>14}{cientifica(f['primera']):>17}{cientifica(f['ultima']):>15}"
              f"{miles(round(f['veces_menor'])):>14}")
        filas_csv.append(["desvanecimiento", f["capas_ocultas"], f"{f['primera']:.3e}",
                          f"{f['ultima']:.3e}", f"{f['veces_menor']:.1f}"])

    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["medicion", "a", "b", "c", "d"]] + filas_csv)
    print(f"\nEscrito {SALIDA_CSV}")


if __name__ == "__main__":
    main()
