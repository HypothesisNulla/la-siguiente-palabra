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

Orden 209 (23 nov 2026): al final de la salida imprime, tal cual está en el anexo A4 del libro, la tabla «Cuatro poquitos,
con el mismo peso de la red del colegio» (título, rótulos y cuatro filas). Las cifras salen de la medición; las frases de
«qué pasa» y «al final» están escritas aquí, y `comprobar_frases` (que el selftest y la propia salida ejecutan) comprueba
cada una contra los datos: si alguna no se cumple, el programa para y no imprime la tabla.

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
# Orden 209: la tabla del anexo A4.
TABLA_A4 = (0.5, 100.0, 500.0, 1000.0)      # los cuatro poquitos de la tabla
TITULO_A4 = "Cuatro poquitos, con el mismo peso de la red del colegio"
ROTULOS_A4 = ("poquito", "qué pasa", "al final", "error final")
DECIMALES_ERROR_A4 = 4          # el error final, a cuatro decimales...
UMBRAL_ERROR_A4 = UMBRAL_FONDO  # ...salvo si no llegó al fondo (error final por encima de este umbral): entonces, a dos (0,31)
PASOS_PRIMEROS_A4 = 5           # «sale a…, vuelve a…, salta a…»: tienen que estar en los 5 primeros pasos
ERROR_LLANO_A4 = 0.538          # el llano de la izquierda: el error ahí es de 0,538 o más
ERROR_MINIMO_NO_FONDO_A4 = 0.0001   # «casi en el fondo»: el error final del 0,5 está por encima de esto

# ==========================================================

import argparse
import csv
import platform
import sys
from datetime import date
from pathlib import Path

import numpy as np

import formato as FM
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
    print()
    print("--- LA TABLA DEL ANEXO A4 (Orden 209): el libro la copia tal cual ---")
    print()
    for linea in tabla_del_anexo(fondo, filas):
        print(linea)


# ---------------------------------------------------------------- la tabla del anexo A4 (Orden 209)

def miles(p):
    """1000 -> «1.000»; 0,5 -> «0,5»."""
    return K.miles(int(p)) if float(p).is_integer() else poq(p)


def comprobar_frases(filas, fondo):
    """Las frases de la tabla del A4, una a una, contra los datos. Devuelve la lista de las que NO se cumplen
    (vacía si todas). Fijadas en la Orden 209 antes de mirar el resultado."""
    por = {r["poquito"]: r for r in filas}
    mal = []
    r = por[0.5]                                   # baja despacio, sin pasarse; acaba en 13,12, casi en el fondo
    if r["cruces"] != 0 or f(r["camino"][-1], 2) != "13,12" or not r["error_final"] > ERROR_MINIMO_NO_FONDO_A4:
        mal.append("0,5: no cruza el fondo, acaba en 13,12 y el error final está por encima de 0,0001")
    r = por[100.0]                                 # dos pasos: a 8,42 y a 13,22; en el fondo
    if (f(r["camino"][1], 2), f(r["camino"][2], 2)) != ("8,42", "13,22") or r["acaba"] != "en el fondo":
        mal.append("100: sus dos primeros pasos caen en 8,42 y 13,22, y acaba en el fondo")
    r = por[500.0]                                 # sale a 26,08, vuelve a 6,56, salta a 66,25; cruza varias veces; en el fondo
    primeros = [f(v, 2) for v in r["camino"][1:PASOS_PRIMEROS_A4 + 1]]
    if (not all(x in primeros for x in ("26,08", "6,56", "66,25")) or r["cruces"] < 2 or r["acaba"] != "en el fondo"):
        mal.append("500: 26,08, 6,56 y 66,25 están en sus primeros cinco pasos, cruza el fondo más de una vez y acaba en el fondo")
    r = por[1000.0]                                # sale a 48,15, llano de la izquierda, un paso lo lanza a 130,9; allí, volviendo despacio
    c, e = r["camino"], r["errores"]
    izquierda = c < fondo
    llano = bool(np.any(izquierda) and np.nanmax(np.where(izquierda, e, -np.inf)) >= ERROR_LLANO_A4)
    lejos_y_volviendo = bool(abs(c[-1] - fondo) > UMBRAL_FONDO and abs(c[-1] - fondo) < abs(c[-2] - fondo))
    if f(c[1], 2) != "48,15" or not llano or f(np.nanmax(c), 1) != "130,9" or not lejos_y_volviendo:
        mal.append("1.000: su primer paso cae en 48,15, pasa por el llano de la izquierda (error 0,538 o más), su punto más alto "
                   "es 130,9 y al final está lejos del fondo y acercándose")
    return mal


def tabla_del_anexo(fondo, filas):
    """Las líneas del bloque «::: tabla» del anexo A4, tal cual está en el libro. Las cifras salen de las filas; solo las
    frases están escritas aquí (y comprobadas por comprobar_frases)."""
    mal = comprobar_frases(filas, fondo)
    assert not mal, "Alguna frase de la tabla del A4 no se cumple con los datos: " + "; ".join(mal)
    por = {r["poquito"]: r for r in filas}

    def err(r):
        return f(r["error_final"], DECIMALES_ERROR_A4 if r["error_final"] < UMBRAL_ERROR_A4 else 2)
    a, b, c, d = (por[p] for p in TABLA_A4)
    cam = lambda r, i: f(r["camino"][i], 2)
    celdas = [
        (miles(a["poquito"]), "baja despacio, sin pasarse", f"en {cam(a, -1)}, casi en el fondo", err(a)),
        (miles(b["poquito"]), f"dos pasos: a {cam(b, 1)} y a {cam(b, 2)}", "en el fondo", err(b)),
        (miles(c["poquito"]), f"sale a {cam(c, 1)}, vuelve a {cam(c, 4)}, salta a {cam(c, 5)} y regresa cruzando varias veces el fondo",
         "en el fondo", err(c)),
        (miles(d["poquito"]), f"sale a {cam(d, 1)}, cae en el llano de la izquierda y un paso lo lanza a {f(float(np.nanmax(d['camino'])), 1)}",
         "allí, volviendo muy despacio", err(d)),
    ]
    return FM.tabla_editorial(TITULO_A4, ROTULOS_A4, celdas, "diid")


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

    # [3 bis] (Orden 209) las frases de la tabla del A4 contra los datos: con los datos de verdad se cumplen todas; con
    # unos retocados (el 500 sin cruces, el 100 con otro primer paso) se detecta que no se cumplen. Y la tabla sale
    # con su título, sus cuatro rótulos y cuatro filas.
    fondo_k, filas_k = medir(red)
    mal = comprobar_frases(filas_k, fondo_k[0])
    retocadas = [dict(r) for r in filas_k]
    for r in retocadas:
        if r["poquito"] == 500.0:
            r["cruces"] = 0
        if r["poquito"] == 100.0:
            r["camino"] = r["camino"].copy(); r["camino"][1] = 9.0
    mal_retocadas = comprobar_frases(retocadas, fondo_k[0])
    bloque = tabla_del_anexo(fondo_k[0], filas_k)
    forma = (bloque[0] == "::: tabla" and bloque[1] == TITULO_A4 and bloque[-1] == ":::"
             and sum(1 for l in bloque if l.startswith("|")) == 2 + len(TABLA_A4))
    n3b = bool(not mal and len(mal_retocadas) == 2 and forma)
    print(f"[3 bis] frases de la tabla del A4: incumplidas con los datos de verdad: {len(mal)}; con unos retocados: "
          f"{len(mal_retocadas)} (se esperan 2); el bloque tiene la forma de la tabla: {'sí' if forma else 'NO'}: {'OK' if n3b else 'MAL'}")
    for m in mal:
        print("    NO SE CUMPLE:", m)
    ok &= n3b

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
