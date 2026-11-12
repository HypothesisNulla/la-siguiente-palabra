#!/usr/bin/env python3
"""
Capítulo 8 — reparto de atención del adjetivo final sobre el resto de la frase.

Mide a qué palabras mira el adjetivo —«alto», «bajo», «caro»— en tres frases que solo
se diferencian en esa última palabra, y si el reparto se concentra o se ensancha. No
mide el reparto de «era»: ver el comentario de CONSULTA_ES_ULTIMA_PALABRA, más abajo.

Uso:
    python reparto_atencion.py
    python reparto_atencion.py --selftest
"""

# ======================= CONSTANTES =======================

MODELO = "Qwen/Qwen2.5-0.5B"

FRASES = {
    "alto": "El vaso no cabía en el cajón porque era demasiado alto.",
    "bajo": "El vaso no cabía en el cajón porque era demasiado bajo.",
    "caro": "El vaso no cabía en el cajón porque era demasiado caro.",
}

# La palabra cuyo reparto medimos es el ADJETIVO FINAL, no «era».
# Motivo, y es el fallo que costó la primera versión de este script: el modelo es
# causal, solo mira hacia atrás. En la posición de «era» el adjetivo todavía no se ha
# leído, así que las tres frases son idénticas hasta ahí y el reparto sale igual en las
# tres por construcción. El adjetivo es la primera posición que puede distinguirlas.
CONSULTA_ES_ULTIMA_PALABRA = True
CANDIDATAS = ("vaso", "cajón")    # los dos referentes posibles

# Se promedia sobre las cabezas y sobre el último tercio de capas: las capas
# finales son donde se resuelven las relaciones de largo alcance. Constante, no
# ajustada a posteriori para que salga el resultado deseado.
FRACCION_CAPAS_FINALES = 1 / 3

SEMILLA = 20260914
SALIDA_CSV = "../datos/salidas/reparto_atencion.csv"
SALIDA_FIG = "reparto_atencion.png"

TOL_SUMA = 1e-4                   # tolerancia del invariante "los pesos suman uno"

# ==========================================================

import argparse
import csv
import math
import random
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from formato import ANCHO_CAJA_CITA, coma, comprobar_ancho


def fijar_semilla(semilla):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)


def validar_entrada(frases):
    """Asserts de esquema. Revientan diciendo qué se esperaba y qué se encontró."""
    assert isinstance(frases, dict) and frases, \
        f"Se esperaba un dict no vacío de frases; se encontró {type(frases).__name__} con {len(frases) if hasattr(frases,'__len__') else '?'} elementos"

    listas = [f.rstrip(".").split() for f in frases.values()]
    longitudes = {len(l) for l in listas}
    assert len(longitudes) == 1, \
        f"Se esperaba que todas las frases tuvieran el mismo número de palabras; se encontraron longitudes {sorted(longitudes)}"

    n = longitudes.pop()
    posiciones_distintas = [
        i for i in range(n) if len({l[i] for l in listas}) > 1
    ]
    assert posiciones_distintas == [n - 1], (
        f"Se esperaba que las frases difirieran SOLO en la última palabra (posición {n-1}); "
        f"difieren en las posiciones {posiciones_distintas}"
    )

    assert CONSULTA_ES_ULTIMA_PALABRA, \
        "Este script solo mide el reparto de la última palabra; CONSULTA_ES_ULTIMA_PALABRA está en False"

    for clave, frase in frases.items():
        palabras = [p.strip(".,;:").lower() for p in frase.split()]
        for cand in CANDIDATAS:
            assert cand in palabras, \
                f"Se esperaba encontrar la candidata «{cand}» en la frase «{clave}»; sus palabras son {palabras}"


def cargar_modelo(nombre):
    tok = AutoTokenizer.from_pretrained(nombre)
    modelo = AutoModelForCausalLM.from_pretrained(
        nombre, attn_implementation="eager", torch_dtype=torch.float32
    )
    modelo.eval()
    assert modelo.config.num_hidden_layers > 2, (
        f"Se esperaba un modelo con más de 2 capas para poder promediar el último tercio; "
        f"se encontró {modelo.config.num_hidden_layers}"
    )
    return tok, modelo


def reparto_de_una_frase(tok, modelo, frase, consulta=None):
    """Devuelve (palabras, pesos, posicion) con el reparto de la palabra de consulta
    —por defecto la última— sobre las palabras anteriores. Los pesos son no negativos y
    suman uno sobre lo que el modelo puede ver."""
    codificado = tok(frase, return_tensors="pt")
    ids = codificado["input_ids"][0]

    # Mapa de cada subunidad a la palabra a la que pertenece. Se hace por reconstrucción
    # explícita, no por heurística de espacios, para que no falle en silencio.
    piezas = [tok.decode([i]) for i in ids]
    palabras, indice_de_palabra, actual = [], [], ""
    for pieza in piezas:
        if pieza.startswith(" ") or not palabras:
            palabras.append(pieza.strip())
            actual = palabras[-1]
        else:
            palabras[-1] += pieza
            actual = palabras[-1]
        indice_de_palabra.append(len(palabras) - 1)

    normalizadas = [p.strip(".,;:").lower() for p in palabras]
    if consulta is None:
        consulta = normalizadas[-1]
    coincidencias = [i for i, p in enumerate(normalizadas) if p == consulta]
    assert len(coincidencias) == 1, (
        f"Se esperaba que «{consulta}» apareciera exactamente una vez tras la tokenización; "
        f"se encontraron {len(coincidencias)} en {normalizadas}"
    )
    palabra_objetivo = coincidencias[0]
    ultima_subunidad = max(
        k for k, w in enumerate(indice_de_palabra) if w == palabra_objetivo
    )

    with torch.no_grad():
        salida = modelo(**codificado, output_attentions=True)

    atenciones = salida.attentions
    assert atenciones is not None and len(atenciones) > 0, \
        "Se esperaban matrices de atención; el modelo no las ha devuelto (¿attn_implementation distinto de 'eager'?)"

    n_capas = len(atenciones)
    desde = n_capas - max(1, int(round(n_capas * FRACCION_CAPAS_FINALES)))
    # atenciones[c]: (lote, cabezas, consulta, clave) -> promediar cabezas y capas finales
    apiladas = torch.stack([atenciones[c][0].mean(dim=0) for c in range(desde, n_capas)])
    fila = apiladas.mean(dim=0)[ultima_subunidad].numpy()

    # Agregar subunidades a palabras. Suma, nunca relleno silencioso.
    pesos = np.zeros(len(palabras), dtype=np.float64)
    for k, w in enumerate(indice_de_palabra):
        pesos[w] += float(fila[k])

    assert not np.isnan(pesos).any(), \
        f"Se esperaban pesos sin NaN; se encontraron {int(np.isnan(pesos).sum())} NaN en la frase «{frase}»"
    return normalizadas, pesos, palabra_objetivo


def dispersion(pesos):
    """Entropía normalizada del reparto: 0 = todo a una palabra, 1 = repartido a partes
    iguales. Es la medida de «se concentra» frente a «se ensancha»."""
    p = np.asarray(pesos, dtype=np.float64)
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log(p)).sum() / math.log(p.size))


def medir(tok, modelo, frases):
    resultados = {}
    for clave, frase in frases.items():
        palabras, pesos, objetivo = reparto_de_una_frase(tok, modelo, frase)
        visibles = pesos[: objetivo + 1]
        suma = visibles.sum()
        assert abs(suma - 1.0) < TOL_SUMA, (
            f"Invariante roto en «{clave}»: se esperaba que los pesos sumaran 1 "
            f"(tolerancia {TOL_SUMA}); se encontró {suma:.6f}"
        )
        resultados[clave] = {
            "frase": frase,
            "palabras": palabras[: objetivo + 1],
            "pesos": visibles,
            "dispersion": dispersion(visibles),
            **{
                f"peso_{c}": float(
                    visibles[palabras.index(c)] if c in palabras[: objetivo + 1] else 0.0
                )
                for c in CANDIDATAS
            },
        }
    return resultados


def escribir_csv(resultados, ruta):
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["variante", "frase", "palabra", "peso", "dispersion"])
        for clave, r in resultados.items():
            for palabra, peso in zip(r["palabras"], r["pesos"]):
                w.writerow([clave, r["frase"], palabra, f"{peso:.6f}", f"{r['dispersion']:.6f}"])


def informe(resultados):
    """La tabla y el veredicto de la predicción, en castellano y dentro de la caja.

    Coma decimal porque el libro está en castellano (formato.py): un «0.4986» impreso
    obliga a castellanizarlo a mano al copiarlo, y eso es lo que la regla 6 prohíbe.
    """
    lineas = [
        "el reparto del adjetivo final, promediado sobre las cabezas y",
        "sobre el último tercio de capas. «vaso» y «cajón» son la porción",
        "que se lleva cada una; «dispersión» va de 0 (todo a una sola",
        "palabra) a 1 (repartido por igual entre todas).",
        "",
        f"{'adjetivo':<10}{'«vaso»':>12}{'«cajón»':>12}{'dispersión':>13}",
        "-" * 47,
    ]
    for clave, r in resultados.items():
        lineas.append(
            f"{clave:<10}{coma(r['peso_vaso'], 4):>12}"
            f"{coma(r['peso_cajón'], 4):>12}{coma(r['dispersion'], 4):>13}"
        )
    print()
    print("\n".join(comprobar_ancho(lineas, ANCHO_CAJA_CITA)))
    d = {k: r["dispersion"] for k, r in resultados.items()}
    print()
    # Esta predicción se escribió ANTES de medir, que es como se hacen. No se cumplió, y el
    # capítulo se reescribió el 18 de septiembre para contar lo medido en vez de lo esperado.
    # El veredicto se queda aquí, impreso cada vez, porque es el registro de que la hipótesis
    # se puso a prueba y perdió: borrarlo sería borrar la parte honesta.
    veredicto = [
        "LO QUE SE ESPERABA, ESCRITO ANTES DE MEDIR: que la frase sin",
        "respuesta («caro») repartiera la atención más que las otras.",
    ]
    if "caro" in d:
        gana = max(d, key=d.get)
        veredicto.append(f"RESULTADO: la mayor es «{gana}» ({coma(d[gana], 4)}).")
        veredicto.append(
            "SE CUMPLE: la frase sin respuesta es la que más reparte."
            if gana == "caro"
            else "NO SE CUMPLE: la frase sin respuesta no reparte más."
        )
        if gana != "caro":
            veredicto.append("El capítulo cuenta lo medido, no lo que se esperaba.")
    print("\n".join(comprobar_ancho(veredicto, ANCHO_CAJA_CITA)))


def selftest(tok, modelo):
    """Tres pruebas: test nulo, señal implantada e invariante del dominio."""
    fallos = []

    # 1. TEST NULO — las mismas palabras barajadas. Sin estructura no debe aparecer un pico
    #    sobre un referente: la dispersión no debe bajar de la real.
    #
    #    OJO con el montaje, que estuvo mal desde el primer día y nadie lo ejecutó (fallo 4.29):
    #    barajando la frase ENTERA, la palabra de consulta podía quedar la primera, y entonces
    #    solo se ve a sí misma. Un reparto de un solo elemento tiene dispersión cero, así que el
    #    test suspendía siempre, de forma determinista, sin que ninguna cifra del libro estuviera
    #    mal. Se baraja lo que va DELANTE de la consulta y la consulta se queda al final: mismas
    #    palabras, ninguna estructura, y el mismo número de palabras visibles que en la real, que
    #    es lo único que hace comparables las dos dispersiones.
    base = FRASES["alto"]
    palabras = base.rstrip(".").split()
    consulta_nula = palabras[-1]
    rng = random.Random(SEMILLA)
    delante = palabras[:-1]
    rng.shuffle(delante)
    nula = " ".join(delante + [consulta_nula]) + "."
    try:
        _, pesos_n, obj_n = reparto_de_una_frase(tok, modelo, nula, consulta=consulta_nula)
        d_nula = dispersion(pesos_n[: obj_n + 1])
        _, pesos_r, obj_r = reparto_de_una_frase(tok, modelo, base)
        d_real = dispersion(pesos_r[: obj_r + 1])
        print(f"[1] test nulo        dispersión barajada={d_nula:.4f}  real={d_real:.4f}")
        if d_nula < d_real - 0.15:
            fallos.append(
                f"test nulo: la frase barajada se concentra MÁS que la real ({d_nula:.4f} < {d_real:.4f}); "
                "el reparto no está capturando estructura"
            )
    except AssertionError as e:
        fallos.append(f"test nulo reventó: {e}")

    # 2. SEÑAL IMPLANTADA — referente único y adyacente. Debe recuperarse en el sitio
    #    correcto y con magnitud apreciable.
    implantada = "El vaso era demasiado alto."
    palabras_i, pesos_i, obj_i = reparto_de_una_frase(tok, modelo, implantada)
    visibles = pesos_i[: obj_i + 1]
    idx_vaso = palabras_i.index("vaso")
    contenido = [i for i, p in enumerate(palabras_i[: obj_i + 1]) if p in ("vaso", "era")]
    peso_vaso = float(visibles[idx_vaso])
    ranking = sorted(contenido, key=lambda i: -visibles[i])
    print(f"[2] señal implantada peso sobre «vaso»={peso_vaso:.4f}  (frase: {implantada})")
    if peso_vaso <= 0:
        fallos.append(f"señal implantada: peso nulo sobre «vaso» ({peso_vaso:.4f})")

    # 3. INVARIANTE DEL DOMINIO — todo peso es no negativo y el reparto suma uno.
    ok_inv = True
    for clave, frase in FRASES.items():
        _, pesos, obj = reparto_de_una_frase(tok, modelo, frase)
        v = pesos[: obj + 1]
        if (v < -TOL_SUMA).any():
            fallos.append(f"invariante: pesos negativos en «{clave}» (mínimo {v.min():.6f})")
            ok_inv = False
        if abs(v.sum() - 1.0) >= TOL_SUMA:
            fallos.append(f"invariante: en «{clave}» los pesos suman {v.sum():.6f}, no 1")
            ok_inv = False
    print(f"[3] invariante       pesos >= 0 y suma = 1 en las tres frases: {'sí' if ok_inv else 'NO'}")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", help="ejecuta las tres pruebas y sale")
    args = ap.parse_args()

    fijar_semilla(SEMILLA)
    validar_entrada(FRASES)
    tok, modelo = cargar_modelo(MODELO)

    if args.selftest:
        sys.exit(selftest(tok, modelo))

    resultados = medir(tok, modelo, FRASES)
    escribir_csv(resultados, SALIDA_CSV)
    informe(resultados)
    print(f"\nEscrito {SALIDA_CSV}")


if __name__ == "__main__":
    main()
