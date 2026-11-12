#!/usr/bin/env python3
"""
Cuánto cuesta cambiar una palabra del contexto.

Un modelo guarda lo que ya ha calculado de cada token del contexto (la caché) y, si el texto
solo crece por el final, solo calcula lo nuevo. Pero cada token se calcula mirando todos los
anteriores: si se cambia un token del medio, todo lo que viene detrás deja de valer. Esto mide,
en este portátil y con la GPU (Metal):

  1. cuánto cuesta un turno que solo AÑADE texto frente a uno que CAMBIA un solo token del texto
     viejo, a distintas profundidades;
  2. qué pasaría si NO se recalculara: cuánto cambia la predicción de la siguiente palabra si
     se reaprovecha la caché vieja de lo que viene detrás del cambio.

Es una medición DEPENDIENTE DE LA MÁQUINA (lleva tiempos dentro): solo vale ejecutada en el
MacBook Pro M4 Max con 36 GB. La parte 2 (qué palabra sale) no depende de la máquina.

Uso (desde codigo/, con HF_HUB_OFFLINE=1 para no descargar nada):
    python editar_contexto.py --selftest
    python editar_contexto.py > ../datos/salidas/editar_contexto.txt
"""

# ======================= CONSTANTES =======================

MODELOS = [
    ("7B sin comprimir", "mlx-community/Qwen2.5-7B-Instruct-bf16"),
    ("32B comprimido",   "mlx-community/Qwen2.5-32B-Instruct-4bit"),
]
MODELO_SELFTEST = "mlx-community/Qwen2.5-7B-Instruct-4bit"
TEXTO = "../datos/entrada/quijote.txt"
INICIO = "En un lugar de la Mancha"   # el texto empieza en su primera aparición
VIEJOS = 7500        # tokens del texto viejo, ya calculados en la caché
NUEVOS = 500         # tokens que añade el turno
POSICIONES = [0.0, 0.25, 0.50, 0.75, 0.95]   # dónde se cambia un token, sobre VIEJOS
# Cambiado el 4 oct 2026. Antes: SUSTITUTO = " Sancho" y RESERVA = " molino". Los dos son DOS
# tokens en el tokenizador de Qwen2.5 (" San" + "cho", y otros dos), igual en el 7B y en el 32B:
# sustituir un token por dos habría desplazado todas las posiciones de después. La validación
# de «un solo token» (`un_solo_token`) es la que lo cazó, y se queda como estaba. Ahora son un
# solo token cada una.
SUSTITUTO = " casa"      # si el token original ya es ese, usa RESERVA
RESERVA = " mesa"
REPETICIONES = 5     # más una de calentamiento por condición, que no se cuenta
SEMILLA = 2026       # baraja el orden de las condiciones

# No venía en el encargo. Es el tamaño de trozo con el que mlx-lm prefila un texto largo
# (`prefill_step_size`, 2048 por omisión en la versión instalada): sin trocear, una sola pasada
# de 8.000 tokens construye matrices de atención enormes. Se usa el mismo valor para que lo
# medido sea lo que haría mlx-lm de verdad. No cambia cuántos tokens se calculan.
PASO_PREFILL = 2048

TOLERANCIA = 0.01    # del selftest: variación total por debajo de esto = misma predicción

SALIDA_CSV = "../datos/salidas/editar_contexto.csv"

# ==========================================================

import argparse
import csv
import datetime
import gc
import importlib.metadata
import platform
import random
import re
import statistics
import subprocess
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import (KVCache, can_trim_prompt_cache, make_prompt_cache,
                                 trim_prompt_cache)

from formato import ANCHO_CAJA, coma, comprobar_ancho, miles, pct

TOTAL = VIEJOS + NUEVOS


def caja(lineas):
    """Revienta si alguna línea no cabe en la caja del libro (regla 9)."""
    return comprobar_ancho(lineas, ANCHO_CAJA)

# Cuántos tokens se han pasado al modelo desde la última puesta a cero. Es lo que comprueba la
# tercera prueba del selftest: no se cuenta lo que el programa DICE que hace, sino lo que hace.
_PASADOS = [0]


# --------------------------- el texto y los tokens ---------------------------

def cargar_texto():
    with open(TEXTO, encoding="utf-8") as fh:
        texto = fh.read()
    assert INICIO in texto, f"Se esperaba «{INICIO}» en «{TEXTO}»; no aparece"
    return texto[texto.index(INICIO):]


def tokens_del_texto(tok, texto):
    ids = tok.encode(texto, add_special_tokens=False)
    assert len(ids) >= TOTAL, (
        f"Se esperaban al menos {TOTAL} tokens desde «{INICIO}»; hay {len(ids)}")
    return ids[:TOTAL]


def un_solo_token(tok, texto, nombre):
    ids = tok.encode(texto, add_special_tokens=False)
    assert len(ids) == 1, (
        f"Se esperaba que «{nombre}» ({texto!r}) fuera un solo token; salen {len(ids)}: {ids}")
    return ids[0]


def preparar(repo):
    """Carga el modelo y lo valida. Devuelve (modelo, tokenizador, tokens, sustituto, reserva)."""
    modelo, tok = load(repo)
    tokens = tokens_del_texto(tok, cargar_texto())
    sustituto = un_solo_token(tok, SUSTITUTO, "SUSTITUTO")
    reserva = un_solo_token(tok, RESERVA, "RESERVA")
    cache = make_prompt_cache(modelo)
    assert can_trim_prompt_cache(cache), (
        f"Se esperaba una caché recortable en «{repo}»; no lo es")
    assert all(isinstance(c, KVCache) for c in cache), (
        f"Se esperaba una caché de clase KVCache en «{repo}»; hay "
        f"{sorted({type(c).__name__ for c in cache})}")
    return modelo, tok, tokens, sustituto, reserva


def liberar():
    """Suelta lo que ya no tiene dueño. Quien llama hace antes su `del`: borrar una variable
    dentro de esta función no suelta la referencia de quien la pasó."""
    gc.collect()
    mx.clear_cache()


# --------------------------- calcular, con caché ---------------------------

def calcular(modelo, tokens, cache):
    """Pasa `tokens` por el modelo con su caché y devuelve los logits del último, ya calculados.

    Igual que el prefilado de mlx-lm: todos menos el último, de PASO_PREFILL en PASO_PREFILL y
    forzando solo el estado de la caché (los logits de esos no se calculan); el último token
    solo, con sus logits, que es lo que un turno necesita para decir la primera palabra. El
    `mx.eval` va DENTRO: MLX es perezoso y sin él una pasada no costaría nada."""
    n = len(tokens)
    assert n >= 1, "Se esperaba al menos un token que calcular; se encontró ninguno"
    i = 0
    while n - i > 1:
        paso = min(PASO_PREFILL, n - i - 1)
        modelo(mx.array(tokens[i:i + paso])[None], cache=cache)
        mx.eval([c.state for c in cache])
        _PASADOS[0] += paso
        i += paso
        mx.clear_cache()
    logits = modelo(mx.array(tokens[i:])[None], cache=cache)[0, -1, :].astype(mx.float32)
    mx.eval(logits, [c.state for c in cache])
    _PASADOS[0] += 1
    return logits


def construir_cache(modelo, tokens):
    """La caché de `tokens` y los logits del último. No se cronometra."""
    cache = make_prompt_cache(modelo)
    logits = calcular(modelo, tokens, cache)
    return cache, logits


def entradas(cache):
    return {c.offset for c in cache}


# --------------------------- las condiciones de la parte 1 ---------------------------

def nombre_condicion(cond):
    if cond[0] == "anadir":
        return "solo añadir"
    if cond[0] == "sin_cache":
        return "sin caché"
    return f"cambio al {pct(cond[1], 0, de_uno=True)}"


def posicion(p):
    return int(p * VIEJOS)


def sustituto_para(tokens, k, sustituto, reserva):
    return reserva if tokens[k] == sustituto else sustituto


def ejecutar_condicion(modelo, tokens, cond, sustituto, reserva, nuevo=None):
    """Una pasada de una condición. Devuelve un diccionario con el tiempo (segundos), los
    tokens que se pasaron al modelo, las entradas con que acaba la caché y los logits.

    Solo se cronometra calcular; construir la caché base y recortarla no cuenta."""
    if cond[0] == "anadir":
        cache, _ = construir_cache(modelo, tokens[:VIEJOS])
        pendientes = tokens[VIEJOS:]
        k = None
    elif cond[0] == "sin_cache":
        cache = make_prompt_cache(modelo)
        pendientes = tokens
        k = None
    else:
        k = posicion(cond[1])
        cache, _ = construir_cache(modelo, tokens)
        quitados = trim_prompt_cache(cache, TOTAL - k)
        assert quitados == TOTAL - k, (
            f"Se esperaba recortar {TOTAL - k} entradas de la caché; se recortaron {quitados}")
        cambiado = list(tokens)
        cambiado[k] = nuevo if nuevo is not None else sustituto_para(
            tokens, k, sustituto, reserva)
        pendientes = cambiado[k:]
    _PASADOS[0] = 0
    mx.synchronize()
    t0 = time.perf_counter()
    logits = calcular(modelo, pendientes, cache)
    segundos = time.perf_counter() - t0
    res = {"segundos": segundos, "pasados": _PASADOS[0], "entradas": entradas(cache),
           "logits": logits, "k": k}
    del cache
    liberar()
    return res


# --------------------------- la parte 2: la caché vieja ---------------------------

def caso_cache_vieja(modelo, tokens, k, nuevo):
    """Se parte de la caché de los TOTAL tokens originales; se calcula solo el token cambiado
    en la posición k y su entrada sustituye a la original en cada capa, dejando intactas las de
    k+1 en adelante; se quita el último token y se vuelve a calcular para tener la
    distribución. Devuelve los logits, los tokens pasados y las entradas finales de la caché."""
    cache, _ = construir_cache(modelo, tokens)
    quitados = trim_prompt_cache(cache, TOTAL - k)
    assert quitados == TOTAL - k, (
        f"Se esperaba recortar {TOTAL - k} entradas de la caché; se recortaron {quitados}")
    _PASADOS[0] = 0
    # La caché recortada a k entradas conserva, en sus buffers, las de k en adelante: calcular
    # el token cambiado escribe SOLO la entrada k (la posición sale del `offset`) y deja las
    # demás donde estaban. Luego se vuelve a exponer todo con `offset`, que es un atributo
    # llano de KVCache, y se quita el último token por la vía normal.
    calcular(modelo, [nuevo], cache)
    for c in cache:
        assert c.offset == k + 1, f"Se esperaba offset {k + 1}; se encontró {c.offset}"
        c.offset = TOTAL
    quitados = trim_prompt_cache(cache, 1)
    assert quitados == 1, f"Se esperaba quitar 1 entrada; se quitaron {quitados}"
    logits = calcular(modelo, [tokens[TOTAL - 1]], cache)
    res = {"logits": logits, "pasados": _PASADOS[0], "entradas": entradas(cache)}
    del cache
    liberar()
    return res


def distribucion(logits):
    p = np.array(mx.softmax(logits, axis=-1), dtype=np.float64)
    assert abs(p.sum() - 1.0) < 1e-3, f"Se esperaba que sumara 1; suma {p.sum():.6f}"
    return p


def variacion_total(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def mas_probable(p):
    i = int(np.argmax(p))
    return i, float(p[i])


# --------------------------- pruebas ---------------------------

def selftest():
    fallos = []
    modelo, tok, tokens, sustituto, reserva = preparar(MODELO_SELFTEST)
    orig = distribucion(construir_cache(modelo, tokens)[1])
    top_orig = mas_probable(orig)[0]

    # Por posición, una sola vez cada cosa; las tres pruebas se reparten lo que sale.
    nulo, implantada = [], []
    for p in POSICIONES:
        k = posicion(p)
        # nulo: «cambiar» el token k por sí mismo
        bien_n = distribucion(ejecutar_condicion(
            modelo, tokens, ("cambio", p), sustituto, reserva, nuevo=tokens[k])["logits"])
        vieja_n = distribucion(caso_cache_vieja(modelo, tokens, k, tokens[k])["logits"])
        nulo.append((p,
                     variacion_total(bien_n, orig), mas_probable(bien_n)[0] == top_orig,
                     variacion_total(vieja_n, orig), mas_probable(vieja_n)[0] == top_orig))
        # implantada: el cambio de verdad
        nuevo = sustituto_para(tokens, k, sustituto, reserva)
        bien = distribucion(ejecutar_condicion(
            modelo, tokens, ("cambio", p), sustituto, reserva)["logits"])
        cambiado = list(tokens)
        cambiado[k] = nuevo
        cero = distribucion(calcular(modelo, cambiado, make_prompt_cache(modelo)))
        vieja = distribucion(caso_cache_vieja(modelo, tokens, k, nuevo)["logits"])
        implantada.append((p, variacion_total(bien, cero),
                           mas_probable(bien)[0] == mas_probable(cero)[0],
                           variacion_total(vieja, bien)))
        mx.clear_cache()

    # 1. TEST NULO
    peor_bien = max(n[1] for n in nulo)
    peor_vieja = max(n[3] for n in nulo)
    misma = all(n[2] and n[4] for n in nulo)
    print(f"[1] test nulo         cambiar un token por sí mismo: variación total máxima "
          f"{coma(peor_bien, 4)} (bien calculada) y {coma(peor_vieja, 4)} (caché vieja); "
          f"misma palabra más probable en las {len(nulo)} posiciones: "
          f"{'sí' if misma else 'NO'}")
    if not misma or peor_bien >= TOLERANCIA or peor_vieja >= TOLERANCIA:
        fallos.append(f"test nulo: cambiar un token por sí mismo debería dejar la predicción "
                      f"igual (variación < {TOLERANCIA}); salió {nulo}")

    # 2. SEÑAL IMPLANTADA
    peor_cero = max(i[1] for i in implantada)
    misma_cero = all(i[2] for i in implantada)
    dif_vieja_0 = implantada[0][3]
    print(f"[2] señal implantada  recalcular desde k con la caché frente a calcular desde cero: "
          f"variación total máxima {coma(peor_cero, 4)}, misma palabra: "
          f"{'sí' if misma_cero else 'NO'}; en la posición 0 la caché vieja difiere de la "
          f"bien calculada en {coma(dif_vieja_0, 4)}")
    if not misma_cero or peor_cero >= TOLERANCIA:
        fallos.append(f"señal implantada: recalcular desde k no coincide con calcular desde "
                      f"cero: {implantada}")
    if dif_vieja_0 <= TOLERANCIA:
        fallos.append(f"señal implantada: en la posición 0 la caché vieja debería diferir de "
                      f"la bien calculada (> {TOLERANCIA}) y difiere {dif_vieja_0:.4f}: el "
                      f"montaje de la caché vieja no está guardando las entradas viejas")

    # 3. INVARIANTE DEL DOMINIO
    cuentas = []
    for cond, esperado in ([(("anadir",), NUEVOS), (("sin_cache",), TOTAL)]
                           + [(("cambio", p), TOTAL - posicion(p)) for p in POSICIONES]):
        r = ejecutar_condicion(modelo, tokens, cond, sustituto, reserva)
        cuentas.append((nombre_condicion(cond), esperado, r["pasados"], r["entradas"]))
    for p in POSICIONES:
        r = caso_cache_vieja(modelo, tokens, posicion(p), sustituto_para(
            tokens, posicion(p), sustituto, reserva))
        cuentas.append((f"caché vieja {pct(p, 0, de_uno=True)}", 2, r["pasados"],
                        r["entradas"]))
    mal = [c for c in cuentas if c[1] != c[2] or c[3] != {TOTAL}]
    print(f"[3] invariante        tokens pasados al modelo y entradas finales de la caché, "
          f"{len(cuentas)} casos: {'todos bien' if not mal else f'{len(mal)} MAL'} "
          f"(la caché acaba con {miles(TOTAL)} entradas en todos)")
    if mal:
        fallos.append(f"invariante: se esperaba (tokens pasados, entradas {TOTAL}) y salió "
                      f"{mal}")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


# --------------------------- la salida ---------------------------

def _sysctl(clave):
    return subprocess.run(["sysctl", "-n", clave], capture_output=True, text=True,
                          check=True).stdout.strip()


def swap_usado_gb():
    """Swap en uso ahora, en GB. Es una condición de la máquina que cambia lo que se mide: con
    la memoria justa, macOS pagina a disco y los tiempos dejan de ser los del cálculo."""
    salida = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True,
                            check=True).stdout
    m = re.search(r"used = ([\d.]+)M", salida)
    assert m, f"Se esperaba «used = N M» en vm.swapusage; se encontró {salida!r}"
    return float(m.group(1)) / 1024


def cabecera():
    memoria_gb = int(_sysctl("hw.memsize")) / 1073741824
    macos = subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True,
                           check=True).stdout.strip()
    # La fecha solo aquí, nunca dentro del cálculo.
    lineas = [
        f"Medido el {datetime.date.today()} en {platform.platform()}.",
        f"máquina: {_sysctl('machdep.cpu.brand_string')}, {coma(memoria_gb, 0)} GB de memoria",
        f"macOS {macos}; Python {platform.python_version()}",
        f"mlx {mx.__version__}; mlx-lm {importlib.metadata.version('mlx-lm')}",
        f"GPU (Metal): {'sí' if mx.metal.is_available() else 'NO'}; "
        f"dispositivo de MLX: {mx.default_device()}",
        f"swap en uso al empezar: {coma(swap_usado_gb(), 1)} GB",
        "",
        "constantes:",
        f"  texto: {TEXTO}, desde «{INICIO}»",
        f"  tokens viejos {miles(VIEJOS)}, nuevos {miles(NUEVOS)}, total {miles(TOTAL)}",
        "  posiciones del cambio: " + ", ".join(pct(p, 0, de_uno=True) for p in POSICIONES),
        f"  sustituto «{SUSTITUTO}», reserva «{RESERVA}»",
        f"  repeticiones {REPETICIONES} (y una de calentamiento), semilla {SEMILLA}",
        f"  prefilado en trozos de {miles(PASO_PREFILL)} tokens (el de mlx-lm)",
    ] + [f"  {etiqueta}: {repo}" for etiqueta, repo in MODELOS]
    for l in caja(lineas):
        print(l, flush=True)


def progreso(texto):
    print(texto, file=sys.stderr, flush=True)


def medir_parte1(modelo, tokens, sustituto, reserva, etiqueta, filas_csv):
    condiciones = [("anadir",), ("sin_cache",)] + [("cambio", p) for p in POSICIONES]
    random.Random(SEMILLA).shuffle(condiciones)
    resultados = {}
    for cond in condiciones:
        nombre = nombre_condicion(cond)
        tiempos, recalculados, k = [], None, None
        for rep in range(REPETICIONES + 1):          # la 0 es el calentamiento
            r = ejecutar_condicion(modelo, tokens, cond, sustituto, reserva)
            recalculados, k = r["pasados"], r["k"]
            filas_csv.append([etiqueta, nombre, "" if k is None else k, r["pasados"],
                              rep, f"{1000 * r['segundos']:.3f}"])
            if rep > 0:
                tiempos.append(1000 * r["segundos"])
            progreso(f"[{etiqueta}] {nombre}: repetición {rep} -> "
                     f"{1000 * r['segundos']:.0f} ms")
        resultados[cond] = (nombre, recalculados, tiempos)
    return resultados


def imprimir_parte1(resultados):
    base = statistics.median(resultados[("anadir",)][2])
    orden = [("anadir",)] + [("cambio", p) for p in POSICIONES] + [("sin_cache",)]
    ancho = 16
    cab = [
        f"{'condición':<{ancho}}{'tokens':>13}{'tiempo,':>9}{'mínimo,':>9}{'máximo,':>9}"
        f"{'veces el':>10}",
        f"{'':<{ancho}}{'recalculados':>13}{'mediana':>9}{'en ms':>9}{'en ms':>9}"
        f"{'coste de':>10}",
        f"{'':<{ancho}}{'':>13}{'en ms':>9}{'':>9}{'':>9}{'«añadir»':>10}",
        "-" * (ancho + 13 + 9 + 9 + 9 + 10),
    ]
    for l in caja(cab):
        print(l, flush=True)
    for cond in orden:
        nombre, recalculados, tiempos = resultados[cond]
        med = statistics.median(tiempos)
        fila = (f"{nombre:<{ancho}}{miles(recalculados):>13}{miles(round(med)):>9}"
                f"{miles(round(min(tiempos))):>9}{miles(round(max(tiempos))):>9}"
                f"{coma(med / base, 1):>10}")
        print(*caja([fila]), sep="", flush=True)
    print(flush=True)
    for l in caja([
            f"tiempo: mediana de {REPETICIONES} repeticiones, en milisegundos (ms).",
            "mínimo, máximo: la más rápida y la más lenta de esas repeticiones.",
            "tokens recalculados: los que el modelo vuelve a calcular.",
            "«solo añadir»: se añaden tokens al final; no se cambia ninguno.",
            "«cambio al N %»: se cambia un token al N % del texto viejo.",
            f"«sin caché»: calcular los {miles(TOTAL)} tokens desde cero.",
            "veces el coste de «añadir»: el tiempo entre el de «solo añadir»."]):
        print(l, flush=True)
    print(flush=True)


def medir_parte2(modelo, tok, tokens, sustituto, reserva, etiqueta):
    orig = distribucion(construir_cache(modelo, tokens)[1])
    liberar()
    filas = []
    for p in POSICIONES:
        k = posicion(p)
        nuevo = sustituto_para(tokens, k, sustituto, reserva)
        bien = distribucion(ejecutar_condicion(
            modelo, tokens, ("cambio", p), sustituto, reserva)["logits"])
        vieja = distribucion(caso_cache_vieja(modelo, tokens, k, nuevo)["logits"])
        filas.append((p, orig, bien, vieja))
        progreso(f"[{etiqueta}] parte 2, posición {pct(p, 0, de_uno=True)} hecha")
    return filas


MAX_PALABRA = 9   # lo que cabe en la columna: una palabra más larga sale cortada con «…»


def _palabra(tok, i):
    p = tok.decode([i]).replace(" ", "_").replace("\n", "\\n")
    return p if len(p) <= MAX_PALABRA else p[:MAX_PALABRA - 1] + "…"


def imprimir_parte2(tok, filas):
    celdas = []
    for p, orig, bien, vieja in filas:
        fila = []
        for d in (orig, bien, vieja):
            i, pr = mas_probable(d)
            fila.append((_palabra(tok, i), pr))
        celdas.append(fila)
    # Cada celda: la palabra a la izquierda y su probabilidad a la derecha, en un ancho que
    # cabe el rótulo de la columna («bien calculada», 14) y deja dos espacios hasta la siguiente.
    ancho_cel = max(14, max(len(c[0]) for fila in celdas for c in fila) + 8)
    assert 11 + 3 * ancho_cel + 4 <= ANCHO_CAJA, "la tabla 2 no cabe: bajar MAX_PALABRA"
    hueco = "  "
    ancho_p = 11
    cab = [
        f"{'posición':<{ancho_p}}palabra siguiente más probable, y su probabilidad",
        f"{'del cambio':<{ancho_p}}" + hueco.join(
            f"{t:<{ancho_cel}}" for t in ("original", "bien calculada", "caché vieja")).rstrip(),
        "-" * (ancho_p + 3 * ancho_cel + 2 * len(hueco)),
    ]
    for l in caja(cab):
        print(l, flush=True)
    for (p, *_), fila in zip(filas, celdas):
        texto = hueco.join(
            f"{pal:<{ancho_cel - 7}}{pct(pr, 1, de_uno=True):>7}" for pal, pr in fila)
        print(*caja([f"{pct(p, 0, de_uno=True):<{ancho_p}}{texto}"]), sep="", flush=True)
    print(flush=True)

    ancho_v = 21
    cab2 = [
        f"{'posición':<{ancho_p}}variación total de la predicción",
        f"{'del cambio':<{ancho_p}}{'bien calculada':>{ancho_v}}{hueco}{'caché vieja':>{ancho_v}}",
        f"{'':<{ancho_p}}{'frente a original':>{ancho_v}}{hueco}{'frente a bien calc.':>{ancho_v}}",
        "-" * (ancho_p + 2 * ancho_v + len(hueco)),
    ]
    for l in caja(cab2):
        print(l, flush=True)
    for p, orig, bien, vieja in filas:
        fila = (f"{pct(p, 0, de_uno=True):<{ancho_p}}"
                f"{coma(variacion_total(bien, orig), 4):>{ancho_v}}{hueco}"
                f"{coma(variacion_total(vieja, bien), 4):>{ancho_v}}")
        print(*caja([fila]), sep="", flush=True)
    print(flush=True)
    for l in caja([
            "«_»: marca un espacio delante de la palabra; «…», que se cortó.",
            "probabilidad: la que da el modelo a esa palabra, en %.",
            "variación total: la mitad de la suma de las diferencias entre",
            "dos reparticiones de probabilidad: 0 si son iguales, 1 si no",
            "tienen nada en común.",
            "«original»: el texto sin cambiar.",
            "«bien calculada»: texto cambiado, recalculado desde el cambio.",
            "«caché vieja»: solo se recalcula el token cambiado; lo que viene",
            "detrás se deja como estaba."]):
        print(l, flush=True)
    print(flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())

    cabecera()
    filas_csv = []
    for etiqueta, repo in MODELOS:
        modelo, tok, tokens, sustituto, reserva = preparar(repo)
        print(f"\n--- {etiqueta} ({repo}) ---\n", flush=True)
        resultados = medir_parte1(modelo, tokens, sustituto, reserva, etiqueta, filas_csv)
        print(f"PARTE 1: lo que cuesta cada turno ({etiqueta})\n", flush=True)
        imprimir_parte1(resultados)
        filas = medir_parte2(modelo, tok, tokens, sustituto, reserva, etiqueta)
        print(f"PARTE 2: qué pasaría si no se recalculara ({etiqueta})\n", flush=True)
        imprimir_parte2(tok, filas)
        pico = mx.get_peak_memory() / 1073741824
        print(f"memoria pico de la GPU con este modelo: {coma(pico, 1)} GB", flush=True)
        print(f"swap en uso al acabar con este modelo: {coma(swap_usado_gb(), 1)} GB\n",
              flush=True)
        del modelo
        liberar()
        mx.reset_peak_memory()

    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(
            [["modelo", "condicion", "k", "tokens_recalculados", "repeticion", "tiempo_ms"]]
            + filas_csv)
    print(f"Escrito {SALIDA_CSV}", flush=True)


if __name__ == "__main__":
    main()
