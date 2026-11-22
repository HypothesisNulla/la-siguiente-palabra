#!/usr/bin/env python3
"""
El golf en el terreno del colegio (Orden 198, 22 nov 2026): qué pasa con poquitos grandes.

Es el apartado d) de `culpa_en_el_colegio.py` (un solo peso, el de la línea de la primera a la final;
los demás, quietos; la red entrenada del colegio; salida en 4,0; 2.000 pasos) repetido con una
rejilla de poquitos. Solo cambia el poquito. La red, el peso, el arranque y la función del error son
los de `culpa_en_el_colegio.py`, importados sin copiar nada.

Por cada poquito apunta: la posición del peso y el error en los pasos 1 a 20, en el 100 y en el 2.000;
cuántas veces cruza el fondo de la curva; dónde acaba; el paso más largo que da, y hasta dónde llega
el peso (y si se sale del tramo que dibuja la figura del terreno, de -5 a 30).

Cómo se decide «dónde acaba» (fijado antes de ver ningún número):
  - «en el fondo»: el error final es menor que UMBRAL_FONDO (0,01);
  - si no, «yendo y viniendo»: en los últimos ULTIMOS_PASOS pasos cruza el fondo al menos
    CRUCES_PARA_IR_Y_VENIR veces;
  - si no, «en el llano de la izquierda» si el peso final está a la izquierda del fondo, y «en el llano
    de la derecha» si está a la derecha.
(El peso y el error que no son números finitos se dicen tal cual: no se recortan ni se tapan.)

Mediciones independientes de la máquina (numpy, sin GPU, sin tiempos): con la misma semilla dan lo
mismo en cualquier ordenador.

Uso:
    python golf_en_el_colegio.py --selftest
    python golf_en_el_colegio.py > ../datos/salidas/golf_en_el_colegio.txt
"""

# ======================= CONSTANTES =======================

PODRIAS = (0.5, 2, 5, 20, 50, 100, 200, 500, 1000)   # la rejilla de poquitos (Orden 198; no se toca)
# Lo demás (el peso, el arranque 4,0, los 2.000 pasos, la red y su semilla) sale de culpa_en_el_colegio.py.
PRIMEROS_PASOS = 20              # el detalle: pasos 1 a 20
OTROS_PASOS = (100, 2_000)       # y estos dos
UMBRAL_FONDO = 0.01              # «acaba en el fondo» si el error final es menor
ULTIMOS_PASOS = 100              # para «yendo y viniendo»: los últimos pasos que se miran
CRUCES_PARA_IR_Y_VENIR = 2       # y cuántos cruces del fondo hacen falta en ellos
TRAMO_DESDE, TRAMO_HASTA = -5.0, 30.0    # el tramo que dibuja la figura del terreno (culpa_en_el_colegio.py)
FINO_DESDE, FINO_HASTA, FINO_PUNTOS = 13.0, 15.0, 2001   # solo para la columna aparte «cruces del fondo fino»
SALIDA_CSV = "../datos/salidas/golf_en_el_colegio_datos.csv"
# El invariante: lo que ya dice el anexo (poquito 0,5): de 4 a 8,19 a los cien pasos y 13,12 a los dos mil.
ESPERADO_A_100, ESPERADO_A_2000 = 8.19, 13.12

# ==========================================================

import argparse
import csv
import platform
import sys
from datetime import date
from pathlib import Path

import numpy as np

import culpa_en_el_colegio as K
import el_colegio as C

AQUI = Path(__file__).resolve().parent
X, Y = K.X, K.Y


def f(v, decimales=2):
    """Un número con coma decimal; los que no son finitos, tal cual."""
    if not np.isfinite(v):
        return str(v)
    return C.coma(v, decimales)


def poq(p):
    return f"{p:g}".replace(".", ",")


# ---------------------------------------------------------------- el recorrido

def recorrer(gradiente, w0, poquito, pasos):
    """Posiciones del peso en los pasos 0..pasos: w <- w - poquito * gradiente(w)."""
    w = float(w0)
    camino = [w]
    with np.errstate(all="ignore"):
        for _ in range(pasos):
            w = w - poquito * gradiente(w)
            camino.append(float(w))
    return np.array(camino)


def gradiente_del_colegio(red):
    capa, i, j = K.PESO_DEL_TERRENO
    otra = C.copia(red)

    def g(w):
        otra.W[capa][i, j] = w
        gW, _ = otra.gradiente_retro(X, Y)
        return float(gW[capa][i, j])
    return g


def error_del_colegio(red):
    capa, i, j = K.PESO_DEL_TERRENO
    otra = C.copia(red)

    def e(w):
        otra.W[capa][i, j] = w
        with np.errstate(all="ignore"):
            return float(otra.error(X, Y))
    return e


def cruces(camino, fondo):
    """Cuántas veces el peso pasa de un lado del fondo al otro entre dos pasos seguidos."""
    lado = np.sign(camino - fondo)
    n = 0
    for a, b in zip(lado[:-1], lado[1:]):
        if a != 0 and b != 0 and a != b:
            n += 1
    return n


def donde_acaba(camino, error_final, fondo):
    if np.isfinite(error_final) and error_final < UMBRAL_FONDO:
        return "en el fondo"
    if cruces(camino[-(ULTIMOS_PASOS + 1):], fondo) >= CRUCES_PARA_IR_Y_VENIR:
        return "yendo y viniendo"
    if np.isfinite(camino[-1]) and camino[-1] < fondo:
        return "en el llano de la izquierda"
    if np.isfinite(camino[-1]) and camino[-1] > fondo:
        return "en el llano de la derecha"
    return "sin número (no finito)"


def medir(red):
    """Todo lo de la orden: para cada poquito de la rejilla, el recorrido y sus resúmenes."""
    xs, ys, _ = K.terreno(red)
    fondo = float(xs[int(np.argmin(ys))])
    g, e = gradiente_del_colegio(red), error_del_colegio(red)
    finos = np.linspace(FINO_DESDE, FINO_HASTA, FINO_PUNTOS)
    fondo_fino = float(finos[int(np.argmin([e(w) for w in finos]))])
    filas = []
    for p in PODRIAS:
        camino = recorrer(g, K.INICIO_CUESTA_ABAJO, p, K.PASOS_CUESTA_ABAJO)
        errores = np.array([e(w) for w in camino])
        saltos = np.abs(np.diff(camino))
        fin = float(errores[-1])
        fuera = bool(np.any(camino < TRAMO_DESDE) or np.any(camino > TRAMO_HASTA) or not np.all(np.isfinite(camino)))
        filas.append({
            "poquito": p, "camino": camino, "errores": errores, "cruces": cruces(camino, fondo), "cruces_fino": cruces(camino, fondo_fino),
            "acaba": donde_acaba(camino, fin, fondo), "error_final": fin,
            "paso_largo": float(np.nanmax(saltos)) if np.any(np.isfinite(saltos)) else float("nan"),
            "minimo": float(np.nanmin(camino)), "maximo": float(np.nanmax(camino)), "fuera": fuera})
    return (fondo, fondo_fino), filas


# ---------------------------------------------------------------- la salida

def imprimir(fondos, filas, red):
    fondo, fondo_fino = fondos
    print("######## el golf en el terreno del colegio (Orden 198) ########")
    print(f"fecha: {date.today().isoformat()}")
    print(f"máquina: {platform.system()} {platform.machine()}")
    print(f"python {platform.python_version()}, numpy {np.__version__}")
    print("GPU: no se usa (todo en numpy). Nada de lo que sigue")
    print("depende del ordenador: pesos, errores y recuentos.")
    print()
    print(f"La red del colegio de culpa_en_el_colegio.py (semilla")
    print(f"{K.SEMILLA_RED}), entrenada {K.miles(K.PASOS_ENTRENADO)} pasos. Se mueve solo el peso")
    print("de la línea de la primera a la final, que el")
    print("entrenamiento dejó en " + f(float(red.W[K.PESO_DEL_TERRENO[0]][K.PESO_DEL_TERRENO[1], K.PESO_DEL_TERRENO[2]]), 2) + ";")
    print("los demás pesos, quietos. Sale de " + f(K.INICIO_CUESTA_ABAJO, 1) + " y da " + str(K.PASOS_CUESTA_ABAJO) + " pasos.")
    print(f"El fondo de la curva (el mínimo medido en la rejilla")
    print(f"del apartado d): {f(fondo, 1)}.")
    print(f"El fondo mirado con más finura (de 13 a 15, de milésima")
    print(f"en milésima): {f(fondo_fino, 3)}.")
    print(f"El tramo que dibuja la figura del terreno: de {f(TRAMO_DESDE, 0)} a {f(TRAMO_HASTA, 0)}.")
    print("Rejilla de poquitos (fijada de antemano):")
    print("  " + ", ".join(poq(p) for p in PODRIAS) + ".")
    print()
    print("--- RESUMEN ---")
    print()
    print(f"cruces = veces que cruza {f(fondo, 1)}; fino = veces que cruza {f(fondo_fino, 3)}.")
    print()
    cab = f"{'poquito':>8} {'cruces':>7} {'fino':>5} {'error final':>12} {'paso más largo':>15}  dónde acaba"
    print(cab)
    for r in filas:
        print(f"{poq(r['poquito']):>8} {r['cruces']:>7} {r['cruces_fino']:>5} {f(r['error_final'], 5):>12} {f(r['paso_largo'], 2):>15}  {r['acaba']}")
    print()
    print("--- HASTA DÓNDE LLEGA EL PESO (el tramo de la figura es de -5 a 30) ---")
    print()
    print(f"{'poquito':>8} {'el más bajo':>14} {'el más alto':>14}  ¿se sale?")
    for r in filas:
        print(f"{poq(r['poquito']):>8} {f(r['minimo'], 2):>14} {f(r['maximo'], 2):>14}  {'SÍ' if r['fuera'] else 'no'}")
    print()
    print("--- EL DETALLE: pasos 1 a 20, el 100 y el 2.000 ---")
    for r in filas:
        print()
        print(f"poquito {poq(r['poquito'])}")
        print(f"{'paso':>6} {'peso':>14} {'error':>12}")
        pasos = [0] + list(range(1, PRIMEROS_PASOS + 1)) + list(OTROS_PASOS)
        for n in pasos:
            print(f"{n:>6} {f(r['camino'][n], 3):>14} {f(r['errores'][n], 5):>12}")
    print()
    print("(el paso 0 es la salida: 4,0)")


def guardar_csv(filas):
    with open(AQUI / SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["poquito", "paso", "peso", "error"])
        for r in filas:
            for n, (v, e) in enumerate(zip(r["camino"], r["errores"])):
                w.writerow([repr(r["poquito"]), n, repr(float(v)), repr(float(e))])


# ---------------------------------------------------------------- selftest

def selftest():
    print("=== SELFTEST de golf_en_el_colegio.py ===")
    ok = True

    # [1] NULO: con poquito 0 el peso no se mueve (en la red del colegio y en una parábola)
    red = K.entrenada()
    g = gradiente_del_colegio(red)
    quieto = recorrer(g, K.INICIO_CUESTA_ABAJO, 0.0, 50)
    parabola = recorrer(lambda w: 3.0 * (w - 2.0), 7.0, 0.0, 50)
    n1 = bool(np.all(quieto == K.INICIO_CUESTA_ABAJO) and np.all(parabola == 7.0))
    print(f"[1] nulo: con poquito 0, el peso se queda en {quieto[-1]} tras 50 pasos: {'OK' if n1 else 'MAL'}")
    ok &= n1

    # [2] SEÑAL IMPLANTADA: parábola de curvatura conocida c, e(w) = c (w - w0)^2 / 2, gradiente c (w - w0).
    # Cada paso multiplica la distancia al fondo por (1 - poquito c). Cruza el fondo en cada paso
    # cuando poquito c > 1; no se asienta (la distancia crece) cuando poquito c > 2.
    c, w0 = 0.7, 2.0
    cruza, crece = {}, {}
    for pc in (0.5, 0.9, 0.99, 1.01, 1.5, 1.99, 2.01, 3.0):
        cam = recorrer(lambda w: c * (w - w0), 8.0, pc / c, 60)
        cruza[pc] = cruces(cam[:6], w0)   # los 5 primeros pasos: si el factor es casi 0 la distancia llega a 0 exacto enseguida
        crece[pc] = bool(abs(cam[-1] - w0) > abs(cam[0] - w0))
    cruza_bien = all(cruza[pc] == 0 for pc in (0.5, 0.9, 0.99)) and all(cruza[pc] == 5 for pc in (1.01, 1.5, 1.99, 2.01, 3.0))
    crece_bien = all(not crece[pc] for pc in (0.5, 0.9, 0.99, 1.01, 1.5, 1.99)) and all(crece[pc] for pc in (2.01, 3.0))
    n2 = cruza_bien and crece_bien
    print("[2] señal implantada (parábola, curvatura 0,7), poquito por curvatura = 0,5; 0,9; 0,99; 1,01; 1,5; 1,99; 2,01; 3:")
    print("    cruces del fondo en los 5 primeros pasos: " + ", ".join(f"{cruza[pc]}" for pc in cruza))
    print("    ¿se aleja del fondo?:        " + ", ".join("sí" if crece[pc] else "no" for pc in crece))
    print(f"    cruza el fondo cuando poquito×curvatura pasa de 1, y se aleja cuando pasa de 2, y no antes: {'OK' if n2 else 'MAL'}")
    ok &= n2

    # [3] INVARIANTE: con poquito 0,5 sale lo del anexo (de 4 a 8,19 a los cien pasos; 13,12 a los dos mil)
    # y el recorrido es el mismo, número a número, que el del apartado d de culpa_en_el_colegio.py
    _, _, camino_k = K.terreno(red)
    mio = recorrer(g, K.INICIO_CUESTA_ABAJO, 0.5, K.PASOS_CUESTA_ABAJO)
    iguales = max(abs(a - b) for a, b in zip(mio, [v for v, _ in camino_k])) < 1e-12
    a100, a2000 = round(float(mio[100]), 2), round(float(mio[2000]), 2)
    n3 = bool(iguales and a100 == ESPERADO_A_100 and a2000 == ESPERADO_A_2000 and mio[0] == 4.0)
    print(f"[3] invariante: poquito 0,5: paso 100 = {f(mio[100], 2)} (esperado {f(ESPERADO_A_100, 2)}), "
          f"paso 2.000 = {f(mio[2000], 2)} (esperado {f(ESPERADO_A_2000, 2)}); "
          f"igual que el apartado d, número a número: {'sí' if iguales else 'NO'}: {'OK' if n3 else 'MAL'}")
    ok &= n3

    print("SELFTEST OK (3 pruebas)" if ok else "SELFTEST FALLA")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    C.comprobar_entrada()
    if args.selftest:
        return selftest()
    red = K.entrenada()
    fondo, filas = medir(red)
    imprimir(fondo, filas, red)
    guardar_csv(filas)
    print(f"escrito {SALIDA_CSV}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
