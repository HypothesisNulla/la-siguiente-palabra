#!/usr/bin/env python3
"""
El golf en los dos valles (Orden 201, 22 nov 2026): qué pasa con el poquito en el terreno de juguete de
`dos_valles.py` (un valle hondo, uno poco hondo, dos números).

Importa de `dos_valles.py` el terreno (`altura`), la pendiente, las constantes (VALLES, CUENCO, SALIDAS,
PASO, LIMITE) y `bajar` (solo para comprobar el invariante). No copia nada.

La regla de paso es la misma de siempre: nuevo = viejo - poquito x pendiente. Pero aquí NO se usa la
parada de `bajar` (que para en cuanto un paso no baja): dar un paso que sube es justo lo que se quiere
ver. Se dan PASOS_FIJOS pasos, o se para si algún número del peso se va más allá de PARADA_LEJOS (se
apunta en qué paso). La salida es la de la bola 2 de `dos_valles.py` (la que acaba en el valle hondo).

La rejilla de poquitos se fijó antes de medir: MULTIPLOS veces 1/c, donde c es la curvatura del fondo
del valle hondo, calculada con diferencias finitas (la mayor de las dos curvaturas de los dos números,
que es la que decide cuándo el paso se descontrola).

Cómo se decide cada cosa (fijado antes de ver ningún número):
  - «cruces del fondo»: cuántas veces el peso pasa al otro lado del fondo del valle hondo entre un paso
    y el siguiente, es decir, cuántas veces el vector que va del fondo al peso se da la vuelta
    (el producto escalar de dos posiciones seguidas, medidas desde el fondo, es negativo), sin contar los
    pasos a menos de EPS_CRUCE del fondo (ver el comentario de EPS_CRUCE);
  - «dónde acaba»: «lejos» si el recorrido se paró por irse más allá de PARADA_LEJOS o el último punto
    está fuera del cuadrado que se dibuja (de -LIMITE a LIMITE en los dos números); si no, «asentado»
    si en los últimos ULTIMOS_PASOS pasos ningún paso mide más de ASENTADO, y entonces en el valle hondo
    o en el poco hondo según cuál está más cerca (como `en_que_valle`); y si no, «yendo y viniendo sin
    asentarse».
  - «sale del cuadrado»: algún punto del recorrido con un número de valor absoluto mayor que LIMITE;
    se apunta hasta dónde llega (el mayor valor absoluto de los dos números).

Las tres bolas de la figura. La regla de la Orden 201 (a: el primero que entra sin cruzar el fondo; b: el primero
que cruza el fondo al menos dos veces y se asienta en el valle hondo; c: el primero que sale del cuadrado) se
sustituyó en la Orden 202 (22 nov 2026), porque (c) no existía y (a) y (b) salían como una sola línea. La medición
no cambió (la tabla es la de la 201, número a número). Regla vigente, sobre la tabla y en el orden de la rejilla:
  (a) «entra»: el primer poquito que acaba en el valle hondo sin cruzar el fondo;
  (b) «se pasa y vuelve»: el último poquito que todavía acaba asentado en el valle hondo;
  (c) «no se asienta nunca»: el primero que acaba «yendo y viniendo sin asentarse».
Si alguna no existe en la rejilla, no se amplía la rejilla: se dice.

Mediciones independientes de la máquina (sin GPU, sin tiempos).

Uso:
    python golf_en_dos_valles.py --selftest
    python golf_en_dos_valles.py > ../datos/salidas/golf_en_dos_valles.txt
"""

# ======================= CONSTANTES =======================

MULTIPLOS = (0.1, 0.5, 0.9, 1.2, 1.5, 1.9, 2.1, 3.0, 5.0)   # los poquitos, en veces 1/c (Orden 201; no se toca)
PASOS_FIJOS = 2_000              # pasos que se dan
PARADA_LEJOS = 50.0              # se para si algún número del peso pasa de aquí (en valor absoluto)
H_DIFERENCIAS = 1e-3             # el salto de las diferencias finitas para la curvatura
ASENTADO = 1e-6                  # «asentado»: en los últimos pasos ninguno mide más que esto...
ULTIMOS_PASOS = 100              # ...y estos son los últimos pasos que se miran
ACABA_SIN_ASENTARSE = "yendo y viniendo sin asentarse"
# Cambio técnico, hecho tras la primera medición (Orden 201; regla 2 de CLAUDE.md): el contador de cruces del
# fondo contaba también el ruido de coma flotante. Cuando el peso ya está asentado en el fondo, la posición
# vale el fondo más 1e-16 y el producto escalar de dos posiciones seguidas cambia de signo al azar. Con el
# contador sin este umbral salió (en el orden de la rejilla, de 0,1 a 5 veces 1/c): 0, 0, 5, 999, 1.015, 1.996,
# 1.996, 1.997 y 1.354 cruces, y la bola (b) caía en 0,9/c con 5 «cruces» que eran ruido. Ahora un cruce solo se
# cuenta si el peso está a más de EPS_CRUCE del fondo antes y después. La regla de elección de las bolas no
# cambia.
EPS_CRUCE = 1e-6
SALIDA_CSV = "../datos/salidas/golf_en_dos_valles_datos.csv"

# ==========================================================

import argparse
import csv
import math
import platform
import sys
from datetime import date
from pathlib import Path

import numpy as np

from dos_valles import CUENCO, LIMITE, PASO, SALIDAS, VALLES, altura, bajar, en_que_valle, pendiente
from formato import coma as coma_fija, miles

AQUI = Path(__file__).resolve().parent
SALIDA = SALIDAS[1]              # la bola 2: la que acaba en el valle hondo
NOMBRES = ("en el valle hondo", "en el valle poco hondo")


def f(x, n=2):
    if not np.isfinite(x):
        return str(x)
    return coma_fija(x, n)


def poq(x):
    return coma_fija(x, 4) if x < 1 else coma_fija(x, 3)


# ---------------------------------------------------------------- el terreno: el fondo y su curvatura

def fondo_del_valle_hondo(valles=VALLES, cuenco=CUENCO):
    """El punto más bajo del valle hondo: Newton con las diferencias finitas, desde el centro del valle."""
    x, y = valles[0][0], valles[0][1]
    for _ in range(50):
        gx, gy = pendiente(x, y, valles, cuenco)
        (a, b), (_, d) = hessiana(x, y, valles, cuenco)
        det = a * d - b * b
        dx, dy = (d * gx - b * gy) / det, (-b * gx + a * gy) / det
        x, y = x - dx, y - dy
        if math.hypot(dx, dy) < 1e-12:
            break
    return x, y


def hessiana(x, y, valles=VALLES, cuenco=CUENCO, h=H_DIFERENCIAS):
    """((hxx, hxy), (hxy, hyy)) con diferencias finitas de la altura."""
    A = lambda a, b: altura(a, b, valles, cuenco)
    hxx = (A(x + h, y) - 2 * A(x, y) + A(x - h, y)) / h ** 2
    hyy = (A(x, y + h) - 2 * A(x, y) + A(x, y - h)) / h ** 2
    hxy = (A(x + h, y + h) - A(x + h, y - h) - A(x - h, y + h) + A(x - h, y - h)) / (4 * h ** 2)
    return (hxx, hxy), (hxy, hyy)


def curvatura(valles=VALLES, cuenco=CUENCO):
    """(c, autovalores, fondo): la mayor curvatura del fondo del valle hondo, sus dos curvaturas y dónde está."""
    fx, fy = fondo_del_valle_hondo(valles, cuenco)
    H = np.array(hessiana(fx, fy, valles, cuenco))
    ev = np.sort(np.linalg.eigvalsh(H))
    return float(ev[-1]), (float(ev[0]), float(ev[-1])), (fx, fy)


# ---------------------------------------------------------------- el recorrido

def recorrer(salida, poquito, pasos=PASOS_FIJOS, valles=VALLES, cuenco=CUENCO, parada=PARADA_LEJOS):
    """[(x, y, altura)] en los pasos 0..: nuevo = viejo - poquito x pendiente. Se para si un número pasa de `parada`.
    Devuelve (camino, paro) donde paro es el paso en que se paró por irse lejos, o None."""
    x, y = salida
    camino = [(x, y, altura(x, y, valles, cuenco))]
    paro = None
    with np.errstate(all="ignore"):
        for n in range(1, pasos + 1):
            gx, gy = pendiente(x, y, valles, cuenco)
            x, y = x - poquito * gx, y - poquito * gy
            camino.append((x, y, altura(x, y, valles, cuenco)))
            if abs(x) > parada or abs(y) > parada:
                paro = n
                break
    return camino, paro


def cruces(camino, fondo, eps=EPS_CRUCE):
    """Veces que el peso pasa al otro lado del fondo entre dos pasos seguidos (producto escalar < 0), sin contar
    los pasos en que el peso está a menos de `eps` del fondo (ahí solo hay ruido de coma flotante)."""
    n = 0
    for (x1, y1, _), (x2, y2, _) in zip(camino[:-1], camino[1:]):
        d1, d2 = math.hypot(x1 - fondo[0], y1 - fondo[1]), math.hypot(x2 - fondo[0], y2 - fondo[1])
        if d1 <= eps or d2 <= eps:
            continue
        if (x1 - fondo[0]) * (x2 - fondo[0]) + (y1 - fondo[1]) * (y2 - fondo[1]) < 0:
            n += 1
    return n


def resumen(camino, paro, fondo):
    xs = np.array([c[0] for c in camino]); ys = np.array([c[1] for c in camino])
    pasos = np.hypot(np.diff(xs), np.diff(ys))
    fuera = bool(np.any((np.abs(xs) > LIMITE) | (np.abs(ys) > LIMITE)))
    mayor = float(max(np.max(np.abs(xs)), np.max(np.abs(ys))))
    ultimo_fuera = abs(xs[-1]) > LIMITE or abs(ys[-1]) > LIMITE
    if paro is not None or ultimo_fuera:
        acaba = "lejos"
    elif len(pasos) >= ULTIMOS_PASOS and np.max(pasos[-ULTIMOS_PASOS:]) <= ASENTADO:
        acaba = NOMBRES[en_que_valle(xs[-1], ys[-1])]
    else:
        acaba = ACABA_SIN_ASENTARSE
    return {"acaba": acaba, "paro": paro, "fuera": fuera, "mayor": mayor, "cruces": cruces(camino, fondo),
            "altura_final": camino[-1][2], "paso_largo": float(np.max(pasos)) if len(pasos) else 0.0,
            "pasos": len(camino) - 1}


def medir():
    c, ev, fondo = curvatura()
    filas = []
    for m in MULTIPLOS:
        p = m / c
        camino, paro = recorrer(SALIDA, p)
        filas.append({"mult": m, "poquito": p, "camino": camino, **resumen(camino, paro, fondo)})
    return c, ev, fondo, filas


def elegir(filas):
    """Las tres bolas de la figura, por la regla de la Orden 202 (ver la cabecera): {'a': fila|None, 'b': ..., 'c': ...}."""
    a = next((r for r in filas if r["acaba"] == NOMBRES[0] and r["cruces"] == 0), None)
    asentadas = [r for r in filas if r["acaba"] == NOMBRES[0]]
    b = asentadas[-1] if asentadas else None
    c = next((r for r in filas if r["acaba"] == ACABA_SIN_ASENTARSE), None)
    return {"a": a, "b": b, "c": c}


# ---------------------------------------------------------------- la salida

def imprimir(c, ev, fondo, filas):
    print("######## el golf en los dos valles (Orden 201) ########")
    print(f"fecha: {date.today().isoformat()}")
    print(f"máquina: {platform.system()} {platform.machine()}")
    print(f"python {platform.python_version()}, numpy {np.__version__}")
    print("GPU: no se usa (todo en numpy). Nada de lo que sigue")
    print("depende del ordenador: posiciones, alturas y recuentos.")
    print()
    print("El terreno es el de dos_valles.py (inventado: dos pesos).")
    print(f"Sale de ({f(SALIDA[0], 1)}; {f(SALIDA[1], 1)}), la salida de la bola 2.")
    print(f"Regla: nuevo = viejo - poquito x pendiente. {miles(PASOS_FIJOS)} pasos,")
    print(f"o se para si un número pasa de {f(PARADA_LEJOS, 0)} (en valor absoluto).")
    print(f"El cuadrado que se dibuja: de -{f(LIMITE, 0)} a {f(LIMITE, 0)} en cada número.")
    print()
    print(f"El fondo del valle hondo: ({f(fondo[0], 4)}; {f(fondo[1], 4)}).")
    print(f"Sus dos curvaturas (diferencias finitas, salto {f(H_DIFERENCIAS, 3)}):")
    print(f"  {f(ev[0], 4)} y {f(ev[1], 4)}. La mayor es c = {f(c, 4)}.")
    print(f"1/c = {f(1 / c, 4)}. (Para comparar: el PASO de")
    print(f"dos_valles.py es {f(PASO, 2)}, que es {f(PASO * c, 3)} veces 1/c.)")
    print()
    print("--- RESUMEN ---")
    print()
    print(f"{'x 1/c':>6} {'poquito':>8} {'cruces':>7} {'altura':>7} {'paso más':>9} {'sale del':>9}  dónde acaba")
    print(f"{'':>6} {'':>8} {'':>7} {'final':>7} {'largo':>9} {'cuadrado':>9}")
    for r in filas:
        sale = f"hasta {f(r['mayor'], 1)}" if r["fuera"] else "no"
        print(f"{f(r['mult'], 1):>6} {poq(r['poquito']):>8} {r['cruces']:>7} {f(r['altura_final'], 2):>7} "
              f"{f(r['paso_largo'], 2):>9} {sale:>9}  {r['acaba']}")
    print()
    print("Pasos dados, y dónde se paró por irse lejos:")
    for r in filas:
        extra = f"; se paró en el paso {r['paro']}" if r["paro"] is not None else ""
        print(f"  x {f(r['mult'], 1):>4}: {miles(r['pasos'])} pasos{extra}")
    print()
    print("--- LAS TRES BOLAS DE LA FIGURA (regla de la Orden 202) ---")
    print()
    el = elegir(filas)
    texto = {"a": "(a) «entra»: el primer poquito que acaba en el valle hondo, sin cruzar el fondo",
             "b": "(b) «se pasa y vuelve»: el último poquito que todavía acaba asentado en el valle hondo",
             "c": "(c) «no se asienta nunca»: el primero que acaba yendo y viniendo sin asentarse"}
    for k in "abc":
        r = el[k]
        print(texto[k] + ":")
        print("  " + (f"{f(r['mult'], 1)} veces 1/c (poquito {poq(r['poquito'])}); {r['acaba']}; {r['cruces']} cruces"
                      if r else "NO EXISTE en la rejilla (no se amplía)"))
    print()
    print("Las tres bolas: " + "; ".join(
        f"({k}) {f(el[k]['mult'], 1)} veces 1/c (poquito {poq(el[k]['poquito'])})" if el[k] else f"({k}) no existe" for k in "abc") + ".")
    print()


def guardar_csv(c, fondo, filas):
    with open(AQUI / SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["multiplo", "poquito", "paso", "x", "y", "altura"])
        for r in filas:
            for n, (x, y, h) in enumerate(r["camino"]):
                w.writerow([repr(r["mult"]), repr(r["poquito"]), n, repr(float(x)), repr(float(y)), repr(float(h))])


# ---------------------------------------------------------------- selftest

def selftest():
    print("=== SELFTEST de golf_en_dos_valles.py ===")
    ok = True

    # [1] NULO: con poquito 0 no se mueve
    cam, paro = recorrer(SALIDA, 0.0, pasos=50)
    n1 = bool(paro is None and all((x, y) == SALIDA for x, y, _ in cam))
    print(f"[1] nulo: con poquito 0 el peso se queda en ({f(cam[-1][0], 1)}; {f(cam[-1][1], 1)}) tras 50 pasos: {'OK' if n1 else 'MAL'}")
    ok &= n1

    # [2] SEÑAL IMPLANTADA: cuenco de curvatura conocida k en los dos números (sin valles, h = BASE + k/2 (x² + y²)):
    # cada paso multiplica la distancia al fondo (el origen) por (1 - poquito k); cruza el fondo cuando
    # poquito k > 1 y se aleja cuando > 2.
    k = 0.8
    cuenco = k / 2
    cruza, aleja = {}, {}
    pcs = (0.5, 0.9, 0.99, 1.01, 1.5, 1.99, 2.01, 3.0)
    for pc in pcs:
        cam, _ = recorrer((3.0, -2.0), pc / k, pasos=60, valles=(), cuenco=cuenco, parada=1e9)
        cruza[pc] = cruces(cam[:4], (0.0, 0.0))          # los 3 primeros pasos: con factor casi 0 la distancia baja de EPS_CRUCE enseguida
        d0 = math.hypot(cam[0][0], cam[0][1]); d1 = math.hypot(cam[-1][0], cam[-1][1])
        aleja[pc] = d1 > d0
    bien_c = all(cruza[q] == 0 for q in pcs if q < 1) and all(cruza[q] == 3 for q in pcs if q > 1)
    bien_a = all(not aleja[q] for q in pcs if q < 2) and all(aleja[q] for q in pcs if q > 2)
    n2 = bien_c and bien_a
    print("[2] señal implantada (cuenco, curvatura 0,8), poquito x curvatura = " + "; ".join(f(q, 2) for q in pcs) + ":")
    print("    cruces del fondo en los 3 primeros pasos: " + ", ".join(str(cruza[q]) for q in pcs))
    print("    ¿se aleja del fondo?: " + ", ".join("sí" if aleja[q] else "no" for q in pcs))
    print(f"    cruza desde poquito x curvatura > 1 y se aleja desde > 2, y no antes: {'OK' if n2 else 'MAL'}")
    ok &= n2
    # el ruido de coma flotante pegado al fondo no cuenta como cruce (el fallo de la primera medición)
    ruido = [(1e-12 * (-1) ** i, 2e-13 * (-1) ** (i + 1), 0.0) for i in range(100)]
    n2r = cruces(ruido, (0.0, 0.0)) == 0 and cruces(ruido, (0.0, 0.0), eps=0.0) == 99
    print(f"    cien posiciones que saltan de un lado a otro a 1e-12 del fondo: {cruces(ruido, (0.0, 0.0))} cruces con el umbral "
          f"(sin él, {cruces(ruido, (0.0, 0.0), eps=0.0)}): {'OK' if n2r else 'MAL'}")
    ok &= n2r
    # y la curvatura medida con diferencias finitas sale la del cuenco
    ev_cuenco = np.linalg.eigvalsh(np.array(hessiana(0.3, -0.2, valles=(), cuenco=cuenco)))
    n2b = bool(abs(ev_cuenco[-1] - k) < 1e-4 and abs(ev_cuenco[0] - k) < 1e-4)
    print(f"    curvatura de ese cuenco con diferencias finitas: {f(ev_cuenco[0], 5)} y {f(ev_cuenco[-1], 5)} (esperada {f(k, 1)}): {'OK' if n2b else 'MAL'}")
    ok &= n2b

    # [3] INVARIANTE: con el PASO de dos_valles.py reproduce el camino de la bola 2 hasta donde `bajar` se para
    ref = bajar(*SALIDA)
    cam, _ = recorrer(SALIDA, PASO, pasos=len(ref) - 1)
    dif = max(max(abs(a - b) for a, b in zip(r_, m_)) for r_, m_ in zip(ref, cam))
    n3 = bool(len(cam) == len(ref) and dif < 1e-12)
    print(f"[3] invariante: con el PASO {f(PASO, 2)} sale el camino de la bola 2 de dos_valles.py: {len(ref) - 1} pasos, "
          f"diferencia máxima {dif:.1e}: {'OK' if n3 else 'MAL'}")
    ok &= n3

    print("SELFTEST OK (3 pruebas)" if ok else "SELFTEST FALLA")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    c, ev, fondo, filas = medir()
    imprimir(c, ev, fondo, filas)
    guardar_csv(c, fondo, filas)
    print(f"escrito {SALIDA_CSV}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
