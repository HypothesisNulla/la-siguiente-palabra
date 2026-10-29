#!/usr/bin/env python3
"""
Capítulo 9, C — el mapa de la tarjeta (Orden 25.3, 19 oct 2026).

La Orden 24 midió, en la tarjeta (mps), con 1.024 posiciones, que la ventaja de la que mira todo a la
vez baja con el lote (`lotes_en_la_tarjeta.txt`: de 1,44x con lote 16 a 0,32x con lote 128, con el
modelo de 256 números). Falta el mapa entero: ¿a qué longitud deja de ganar, según el lote y el
modelo? Este programa mide, en la tarjeta, longitudes de 64, 128, 256, 512 y 1.024 posiciones, lotes de
16, 32, 64 y 128 textos y modelos de 256 y de 1.024 números por posición. Usa, importadas de
`lotes_en_la_tarjeta.py` (y por ella de `en_serie_o_a_la_vez.py`), la misma función de cronometrar con
su sincronización de la tarjeta, el mismo calentamiento y las dos mismas máquinas (la que lee en orden
y la que mira a la vez, con la atención rápida de PyTorch, `need_weights=False`).

Tablas: la ventaja por longitud (columnas) y lote (filas), una por modelo; y, para cada modelo y lote,
la longitud más corta a la que la que mira a la vez deja de ganar (su ventaja baja de 1,0x) o «gana
en todas». Las casillas que no quepan en memoria dicen «no cabe», como en `lotes_en_la_tarjeta.py`.

Invariante: las casillas de 1.024 posiciones tienen que quedar a menos de un 30 % (arriba o abajo) de
las de `datos/salidas/lotes_en_la_tarjeta.txt`.

=====================================================================================
PREDICCIÓN DE CLAUDE, escrita antes de medir (Orden 25, 19 oct 2026, copiada tal cual):

  Con 64 posiciones y lote 128, la que mira a la vez sigue ganando (0,7); la longitud a la que deja
  de ganar baja cuando sube el lote (0,75); con el modelo de 1.024, a cada lote deja de ganar a una
  longitud igual o mayor que con el de 256 (0,7).
=====================================================================================

Uso:
    python mapa_en_la_tarjeta.py --selftest
    python mapa_en_la_tarjeta.py > ../datos/salidas/mapa_en_la_tarjeta.txt
    python mapa_en_la_tarjeta.py --desde-guardado > ../datos/salidas/mapa_en_la_tarjeta.txt

Una sola medición en la tarjeta varía demasiado de una vez a otra (hasta un 50 % en una celda, Orden 98),
así que desde la Orden 99 la medición de verdad son BARRIDOS barridos completos (cada barrido, todas las
celdas, como siempre) y la ventaja de cada celda es la mediana de los BARRIDOS. «Deja de ganar con» se
decide con la mediana. Los BARRIDOS barridos, con sus tiempos, se guardan en
`datos/salidas/mapa_en_la_tarjeta.csv`.

Con `--desde-guardado` (Orden 98) no se mide nada: se lee ese csv (con las cabeceras de la salida:
fecha, máquina y versiones son las de la medición) y se escribe la salida entera. Sirve para cambiar
el texto de las notas sin que las cifras se muevan. La medición de verdad es la de sin modo.

Orden 105: la salida del libro sale de DOS repeticiones enteras, A y B (Orden 104), que `--desde-guardado` lee
juntas: `datos/salidas/mapa_en_la_tarjeta-A.csv` y `-B.csv`. Cada ventaja es la mediana de las diez mediciones
(cinco barridos de cada una). Y «deja de ganar con» usa una regla nueva: ganar es ir MÁS DE UN 2 % por delante
(MARGEN_EMPATE); a menos, es empate, que es lo que varía la misma medición al repetirla (la diferencia típica
entre A y B fue del 1,3 %). Esa regla se eligió DESPUÉS de ver el ruido (la propuesta de Carlos, aprobada el
29 de octubre): la nota de la tabla, tal como la dicta la Orden 105, dice que a menos del 2 % es empate, no que
se eligiera después. Con la regla, las ocho filas de la tabla salen iguales con A sola y con B sola.

La medición del 19 de octubre (una sola, con dos decimales) está en
`datos/salidas/mapa_en_la_tarjeta_2026-10-19.csv`; el programa ya no la lee.
"""

# ======================= CONSTANTES =======================

LONGITUDES = [64, 128, 256, 512, 1024]
LOTES = [16, 32, 64, 128]
ANCHOS = [256, 1024]                # números por posición
MARGEN_INVARIANTE = 0.30            # las casillas de 1.024 posiciones, a menos de un 30 % de las guardadas
MEDICIONES_TEST_NULO = 7            # Orden 101: el test nulo compara la mediana de estas mediciones con la mediana de otras tantas
REFERENCIA = "../datos/salidas/lotes_en_la_tarjeta.txt"
GUARDADO = "../datos/salidas/mapa_en_la_tarjeta.csv"   # donde escribe una medición nueva (luego se guarda como A o B)
GUARDADOS = ["../datos/salidas/mapa_en_la_tarjeta-A.csv",  # las dos repeticiones que lee --desde-guardado (Orden 105)
             "../datos/salidas/mapa_en_la_tarjeta-B.csv"]
MARGEN_EMPATE = 0.02                # ganar es ir MÁS de un 2 % por delante; a menos, es empate (Orden 105)
BARRIDOS = 5                        # barridos completos; la ventaja de cada celda es la mediana de los BARRIDOS
BARRIDOS_TIRADOS = 1                # barridos enteros que se miden antes y se tiran: calientan la tarjeta
# (Orden 103: cada modelo, el de 256 y el de 1.024, se mide en su propio proceso, con sus cuatro lotes.)
LONGITUDES_AL_REVES = True          # las longitudes de cada (modelo, lote), de la más larga a la más corta
# (Orden 100: con el orden inverso y un barrido que se tira, la dispersión mediana de las celdas baja de un
# 3,4 % a un 2,1 %, y la de las de 64 posiciones, de un 8,6 % a un 3,4 %; ver la Respuesta 100.)
# Orden 99: cifra que la nota escribe con letra; si cambia, falla aquí y no en la página.
EN_LETRA = {5: "cinco", 10: "diez"}
GANA_EN_TODAS = "gana en todas"
ANCHO_SALIDA = 62                   # líneas de no más de 62 caracteres, fuera de las tablas

# ==========================================================

import argparse
import csv
import os
import platform
import statistics
import subprocess
import time
import re
import sys
from datetime import date
from pathlib import Path

from formato import coma, comprobar_ancho, miles, pct, tabla_editorial

AQUI = Path(__file__).resolve().parent

# Orden 104: `torch` y los módulos que lo importan se cargan con cargar_torch(), no al importar este
# fichero. El proceso que LANZA los procesos de medición no puede haberlos importado mientras esperan:
# con `torch` importado en el padre, el hijo salía un 19 % más lento en la tarjeta (Respuesta 103).
torch = E = L = None


def cargar_torch():
    global torch, E, L
    import torch as _torch
    import en_serie_o_a_la_vez as _E
    import lotes_en_la_tarjeta as _L
    torch, E, L = _torch, _E, _L


def comprobar_entrada():
    L.comprobar_entrada()
    assert 1024 in LONGITUDES and LONGITUDES == sorted(LONGITUDES)
    assert LOTES == L.LOTES and ANCHOS == L.ANCHOS, "Los lotes y los modelos son los de lotes_en_la_tarjeta.py"


def ventaja(t_s, t_v):
    return None if t_s is None or t_v is None else t_s / t_v


def gana(v):
    """Orden 105: ganar es ir más de un MARGEN_EMPATE (2 %) por delante; a menos, o igual, es empate."""
    return v > 1.0 + MARGEN_EMPATE


def deja_de_ganar(ventajas):
    """La longitud más corta a la que la ventaja deja de pasar del 1,02x (es decir, a la que ya no gana: pierde
    o empata), o None si no hay ninguna. `ventajas`: {longitud: ventaja o None}; las casillas que no caben
    (None) no cuentan."""
    sin_ganar = [l for l in sorted(ventajas) if ventajas[l] is not None and not gana(ventajas[l])]
    return sin_ganar[0] if sin_ganar else None


def texto_umbral(ventajas):
    u = deja_de_ganar(ventajas)
    if u is not None:
        return miles(u)
    return GANA_EN_TODAS + (" las que caben" if None in ventajas.values() else "")


def referencia(ruta=AQUI / REFERENCIA):
    """{(ancho, lote): ventaja con 1.024 posiciones} de la tabla guardada de lotes_en_la_tarjeta.txt."""
    out = {}
    for l in Path(ruta).read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*([\d.]+)\s*\|\s*(\d+)\s*\|[^|]*\|[^|]*\|\s*([\d,]+)x\s*\|", l)
        if m:
            out[(int(m.group(1).replace(".", "")), int(m.group(2)))] = float(m.group(3).replace(",", "."))
    assert len(out) == len(ANCHOS) * len(LOTES), f"Se esperaban {len(ANCHOS) * len(LOTES)} casillas; hay {len(out)}"
    return out


def cerca(valor, ref, margen=MARGEN_INVARIANTE):
    return ref * (1 - margen) <= valor <= ref * (1 + margen)


def leer_guardado(rutas=None):
    """(cabecera, tiempos) de los csv de los barridos, uno o varios (por omisión, GUARDADOS: las repeticiones A y
    B, Orden 105). `tiempos`: {(barrido, ancho, lote, longitud): (en orden, a la vez)}, con None donde la casilla
    no cupo; los barridos del segundo csv siguen la numeración del primero (6 a 10 tras 1 a 5). La cabecera son
    las cinco primeras líneas de la salida de la medición, que el csv guarda tras «# salida: »; si los csv son
    de días distintos, la fecha lleva las dos, y lo demás (máquina, versiones, dispositivo) tiene que ser igual."""
    rutas = rutas or [AQUI / g for g in GUARDADOS]
    cabecera, tiempos = None, {}
    for ruta in rutas:
        cab, filas = [], []
        for l in Path(ruta).read_text(encoding="utf-8").splitlines():
            if l.startswith("# salida: "):
                cab.append(l[len("# salida: "):])
            elif not l.startswith("#"):
                filas.append(l)
        assert len(cab) == 5, f"Se esperaban cinco líneas de cabecera en {ruta}; hay {len(cab)}"
        if cabecera is None:
            cabecera = cab
        else:
            assert cab[:1] + cab[2:] == cabecera[:1] + cabecera[2:], (
                f"Se esperaban la misma máquina y las mismas versiones en todos los csv; {ruta} dice {cab} y "
                f"el primero {cabecera}")
            if cab[1] != cabecera[1]:
                cabecera[1] += " y " + cab[1][len("fecha: "):]
        desplazamiento = max((k[0] for k in tiempos), default=0)
        propios = {}
        for f in csv.DictReader(filas):
            t = (float(f["t_en_orden"]), float(f["t_a_la_vez"])) if f["t_en_orden"] else None
            propios[(int(f["barrido"]) + desplazamiento, int(f["ancho"]), int(f["lote"]), int(f["longitud"]))] = t
        barridos = sorted({k[0] for k in propios})
        esperadas = {(b, a, l, c) for b in barridos for a in ANCHOS for l in LOTES for c in LONGITUDES}
        assert set(propios) == esperadas, f"Se esperaban todas las casillas en cada barrido de {ruta}; hay {len(propios)}"
        tiempos.update(propios)
    return cabecera, tiempos


def escribir_guardado(cabecera, tiempos, ruta=AQUI / GUARDADO):
    """El csv de los barridos: los tiempos con seis decimales (de ellos sale cada ventaja, tanto al medir
    como al reimprimir, para que las dos salidas coincidan)."""
    with open(ruta, "w", encoding="utf-8", newline="") as fh:
        fh.write("# Los barridos de mapa_en_la_tarjeta.py (Orden 99): cada fila, una casilla de un barrido, con los\n"
                 "# segundos de la que lee en orden y de la que mira todo a la vez. La ventaja es lo primero entre\n"
                 "# lo segundo; la de cada celda de la salida, la mediana de los barridos. Las cabeceras de la salida\n"
                 "# van detrás de «# salida: ». Las lee mapa_en_la_tarjeta.py --desde-guardado.\n")
        for l in cabecera:
            fh.write("# salida: " + l + "\n")
        w = csv.writer(fh)
        w.writerow(["barrido", "ancho", "lote", "longitud", "t_en_orden", "t_a_la_vez", "ventaja"])
        for (b, a, l, c), t in sorted(tiempos.items()):
            w.writerow([b, a, l, c] + (["", "", ""] if t is None else
                                       [f"{t[0]:.6f}", f"{t[1]:.6f}", f"{ventaja(*t):.4f}"]))


def medianas(tiempos):
    """({(ancho, lote, longitud): mediana de las ventajas de los barridos, o None si alguno no cupo}, nº de barridos)."""
    barridos = sorted({k[0] for k in tiempos})
    v = {}
    for a in ANCHOS:
        for l in LOTES:
            for c in LONGITUDES:
                vs = [ventaja(*tiempos[(b, a, l, c)]) if tiempos[(b, a, l, c)] else None for b in barridos]
                v[(a, l, c)] = None if None in vs else statistics.median(vs)
    return v, len(barridos)


def medir_un_ancho(ancho):
    """Dentro de SU proceso: los barridos de un solo modelo (ancho), con sus cuatro lotes: BARRIDOS_TIRADOS
    que se tiran y BARRIDOS que cuentan. Escribe en la salida estándar una línea
    «barrido,lote,longitud,segundos en orden,segundos a la vez» por celda (vacíos, si no cupo)."""
    print(f"# torch {torch.__version__}")
    print(f"# dispositivo {E.DISPOSITIVO}")
    L.calentar_la_tarjeta()
    longitudes = LONGITUDES[::-1] if LONGITUDES_AL_REVES else LONGITUDES
    for barrido in range(1 - BARRIDOS_TIRADOS, BARRIDOS + 1):          # los de número 0 o menos se tiran
        for lote in LOTES:
            for longitud in longitudes:
                t = L.medir(ancho, lote, longitud)
                if barrido >= 1:
                    ok = t[0] is not None and t[1] is not None
                    print(f"{barrido},{lote},{longitud}," + (f"{t[0]:.6f},{t[1]:.6f}" if ok else ","))
        print(f"modelo de {ancho}, barrido {barrido} de {BARRIDOS}", file=sys.stderr)


def medir_en_procesos(inicio):
    """Los BARRIDOS barridos de todos los modelos, **cada modelo en su propio proceso** (Orden 103: después de
    las pasadas largas de un modelo grande, uno pequeño se queda más lento en ese mismo proceso; Respuesta 102).
    Orden 104: el que lanza no tiene `torch` importado (se comprueba aquí); solo recoge las líneas de los hijos.
    Devuelve ({(barrido, ancho, lote, longitud): (en orden, a la vez) o None}, {"torch": versión, "dispositivo": ...})."""
    assert "torch" not in sys.modules, "El proceso que lanza no puede haber importado torch (Orden 104)"
    tiempos, datos = {}, {}
    for ancho in ANCHOS:
        hijo = subprocess.run([sys.executable, os.path.abspath(__file__), "--medir-ancho", str(ancho)],
                              capture_output=True, text=True)
        assert hijo.returncode == 0, f"El proceso del modelo de {ancho} falló:\n{hijo.stderr}"
        for l in hijo.stdout.splitlines():
            if l.startswith("# torch ") or l.startswith("# dispositivo "):
                clave, valor = l[2:].split(" ", 1)
                assert datos.setdefault(clave, valor) == valor, f"Los procesos dan {clave} distintos"
        medidas = [l.split(",") for l in hijo.stdout.splitlines() if l.count(",") == 4]
        assert len(medidas) == BARRIDOS * len(LOTES) * len(LONGITUDES), (
            f"Se esperaban {BARRIDOS * len(LOTES) * len(LONGITUDES)} celdas del modelo de {ancho}; hay {len(medidas)}")
        for b, lote, longitud, t_s, t_v in medidas:
            tiempos[(int(b), ancho, int(lote), int(longitud))] = (float(t_s), float(t_v)) if t_s else None
        print(f"modelo de {ancho}: {time.time() - inicio:.0f} s desde el principio", file=sys.stderr)
    return tiempos, datos


def celda(v):
    return L.NO_CABE if v is None else coma(v, 2) + "x"


def informe(v, emitir, barridos):
    # Orden 99: la frase de las medianas, con el número de barridos del csv.
    mediana = f" Cada ventaja es la mediana de {EN_LETRA[barridos]} mediciones." if barridos > 1 else ""
    emitir(["--- LA VENTAJA, POR LONGITUD Y LOTE ---", ""])
    for ancho in ANCHOS:
        filas = [[str(lote)] + [celda(v[(ancho, lote, l)]) for l in LONGITUDES] for lote in LOTES]
        emitir(tabla_editorial(
            f"La ventaja, con el modelo de {miles(ancho)} números por posición",
            ["lote"] + [miles(l) for l in LONGITUDES], filas, "d" * (1 + len(LONGITUDES)),
            ["Columnas: posiciones del texto. Filas: textos a la vez.",
             "Ventaja: cuántas veces más rápida es la que mira todo a la vez; por debajo de 1,0x es "
             f"más lenta. «{L.NO_CABE}»: la tarjeta se quedó sin memoria.{mediana}"]))
        emitir([""])
    emitir(["--- DÓNDE DEJA DE GANAR ---", ""])
    filas = []
    for ancho in ANCHOS:
        for lote in LOTES:
            filas.append([miles(ancho), str(lote),
                          texto_umbral({l: v[(ancho, lote, l)] for l in LONGITUDES})])
    emitir(tabla_editorial(
        "La longitud más corta a la que deja de ganar",
        ["números por posición", "lote", "deja de ganar con"], filas, "dd" + "d",
        [# Orden 97: nota para el lector, en una sola frase larga.
         "En la tarjeta gráfica, para cada modelo y cada lote, es decir, cada número de textos a la "
         "vez: la longitud más corta, en posiciones, a la que la máquina que mira todo a la vez ya "
         f"no gana. Ganar es ir más de un {pct(MARGEN_EMPATE, 0)} por delante; a menos, es empate, que es "
         "más o menos lo que varía la misma medición al repetirla. "
         f"«{GANA_EN_TODAS.capitalize()}» quiere decir que gana en todas, de {miles(LONGITUDES[0])} a "
         f"{miles(LONGITUDES[-1])} posiciones.{mediana}"]))
    emitir([""])


def selftest():
    fallos = []
    comprobar_entrada()

    # 1. TEST NULO — la misma máquina contra sí misma no tiene ventaja (razón 1,0 con un 30 % de margen),
    #    y una tabla donde la que mira a la vez gana siempre no tiene longitud a la que deje de ganar.
    L.calentar_la_tarjeta()
    E.ANCHO, E.LOTE = 256, 16
    torch.manual_seed(E.SEMILLA)
    serie = E.EnSerie(256).to(E.DISPOSITIVO)
    # Orden 101: dos mediciones sueltas de 64 posiciones (las más inestables) fallaron una vez de cinco; se
    # comparan las medianas de MEDICIONES_TEST_NULO de cada lado. El margen del 30 % no se toca.
    a, b = [], []
    for _ in range(MEDICIONES_TEST_NULO):
        a.append(E.cronometrar(serie, 64))
        b.append(E.cronometrar(serie, 64))
    razon = statistics.median(a) / statistics.median(b)
    siempre = deja_de_ganar({64: 3.9, 128: 2.9, 256: 2.7, 512: 2.4, 1024: 1.2})
    print(f"[1] test nulo         la de en orden contra sí misma, medianas de {MEDICIONES_TEST_NULO}: {coma(razon, 2)}x (entre 0,70 y 1,30); "
          f"si gana siempre, deja de ganar con: {'ninguna' if siempre is None else siempre}")
    if not cerca(razon, 1.0) or siempre is not None:
        fallos.append("test nulo: la misma máquina contra sí misma, o una tabla que siempre gana, mal contadas")

    # 2. SEÑAL IMPLANTADA — con lote 16 y modelo de 256 la ventaja baja al crecer la longitud (3,9x con 64
    #    posiciones en la medición del portátil): con 64 posiciones tiene que ser mayor que con 1.024; y
    #    la función encuentra el umbral de una tabla fabricada.
    t64 = L.medir(256, 16, 64)
    t1024 = L.medir(256, 16, 1024)
    v64, v1024 = ventaja(*t64), ventaja(*t1024)
    fab = deja_de_ganar({64: 3.0, 128: 2.0, 256: 1.1, 512: 0.8, 1024: 0.5})
    # Orden 105: con la regla del 2 %, una ventaja de 1,01x es empate (no gana) y 1,03x sí gana.
    fab_empate = deja_de_ganar({64: 3.0, 128: 1.5, 256: 1.01, 512: 0.8, 1024: 0.5})
    fab_gana = deja_de_ganar({64: 3.0, 128: 1.5, 256: 1.03, 512: 0.8, 1024: 0.5})
    print(f"[2] señal implantada  ventaja con lote 16 y modelo de 256: {coma(v64, 2)}x con 64 posiciones y "
          f"{coma(v1024, 2)}x con 1.024; en una tabla fabricada deja de ganar con {fab}; con 1,01x "
          f"(empate) en la tercera, con {fab_empate}; con 1,03x, con {fab_gana}")
    if not v64 > v1024 * 1.2 or fab != 512 or fab_empate != 256 or fab_gana != 512:
        fallos.append("señal implantada: la ventaja no baja con la longitud, o el umbral no se encuentra, "
                      "o la regla del 2 % no separa 1,01x de 1,03x")

    # 3. INVARIANTE — la referencia (las ocho casillas de lotes_en_la_tarjeta.txt) se lee entera; las que no
    #    caben no cuentan al buscar el umbral, y se dice cuáles se saltan; y 1.024 está entre las longitudes.
    ref = referencia()
    con_hueco = deja_de_ganar({64: 2.0, 128: None, 256: 0.9})
    texto = texto_umbral({64: 2.0, 128: None, 256: 1.5})
    print(f"[3] invariante        casillas de referencia leídas: {len(ref)}; con una casilla que no cabe, "
          f"deja de ganar con {con_hueco}; y si gana en las demás: «{texto}»")
    if len(ref) != 8 or con_hueco != 256 or texto != GANA_EN_TODAS + " las que caben":
        fallos.append("invariante: la referencia, o el tratamiento de las casillas que no caben, no son los esperados")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--desde-guardado", action="store_true",
                    help="no mide: reimprime la salida con los barridos guardados en el csv")
    ap.add_argument("--medir-ancho", type=int, metavar="ANCHO",
                    help="(interno) los barridos de un solo modelo, en este proceso")
    args = ap.parse_args()
    if args.selftest or args.medir_ancho or args.desde_guardado:
        cargar_torch()              # estos modos no lanzan procesos: aquí `torch` se puede cargar ya
        comprobar_entrada()
    if args.selftest:
        return selftest()
    if args.medir_ancho:
        medir_un_ancho(args.medir_ancho)
        return 0

    def emitir(lineas):
        dentro = getattr(emitir, "dentro", False)
        for l in lineas:
            if l.startswith("::: "):
                dentro = True
            if not dentro:
                comprobar_ancho([l], ANCHO_SALIDA)
            if l == ":::":
                dentro = False
            print(l)
        emitir.dentro = dentro

    if args.desde_guardado:
        cabecera, tiempos = leer_guardado()
    else:
        # Orden 104: primero se mide, sin `torch` en este proceso; luego se carga, para el informe.
        tiempos, datos = medir_en_procesos(time.time())
        cargar_torch()
        comprobar_entrada()
        cabecera = ["######## el mapa de la tarjeta (Orden 25) ########",
                    f"fecha: {date.today().isoformat()}",
                    f"máquina: {platform.system()} {platform.machine()}, Apple M4 Max, 36 GB",
                    f"python {platform.python_version()}, torch {datos['torch']}",
                    f"se usa la tarjeta ({datos['dispositivo']}, con sincronización)."]
        # Los tiempos, a seis decimales, como van al csv: de ellos salen las ventajas de la salida.
        tiempos = {k: None if t is None else (round(t[0], 6), round(t[1], 6)) for k, t in tiempos.items()}
        escribir_guardado(cabecera, tiempos)
    v, barridos = medianas(tiempos)
    emitir(cabecera + ["Los tiempos solo valen en este portátil.", ""])
    informe(v, emitir, barridos)

    ref = referencia()
    filas, malas = [], []
    for (ancho, lote), r in sorted(ref.items()):
        mia = v[(ancho, lote, 1024)]
        ok = mia is not None and cerca(mia, r)
        filas.append([miles(ancho), str(lote), coma(r, 2) + "x", celda(mia), "sí" if ok else "NO"])
        if not ok:
            malas.append((ancho, lote, mia, r))
    emitir(tabla_editorial(
        "La comprobación contra lotes_en_la_tarjeta.txt",
        ["números por posición", "lote", "la guardada", "esta", "¿a menos de un 30 %?"], filas, "dddd" + "c",
        ["Las casillas de 1.024 posiciones. Si alguna no queda a menos de un 30 %, esta medición no vale."]))
    emitir([""])
    assert not malas, f"Casillas de 1.024 posiciones a más de un 30 % de las guardadas: {malas}"
    return 0


if __name__ == "__main__":
    sys.exit(main())
