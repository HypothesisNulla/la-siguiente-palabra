#!/usr/bin/env python3
"""
Capítulo 8 — el reparto de atención mirada a mirada, sin promediar.

`reparto_atencion.py` promedia sobre las catorce miradas de cada capa antes de guardar
nada, así que la información de cada mirada por separado se pierde dentro del programa y
no llega a ningún fichero. Este programa no promedia: guarda las 24 x 14 = 336 filas de
cada frase, y con ellas contesta tres preguntas que el capítulo hace y que hasta hoy no
producía ningún guion:

  1. ¿Cuánto cambia el promedio según cuántas capas se metan en él? (Las cifras del
     capítulo son del último tercio de capas, no de las 336.)
  2. ¿Hay alguna mirada que se fije de verdad en «vaso», en vez de repartir?
  3. Al cambiar el adjetivo —«alto» por «bajo»—, ¿cambia alguna de opinión?

Lo que este programa NO dice, y conviene tenerlo escrito aquí para que nadie se lo
atribuya después: **no dice que ninguna mirada «siga al sujeto»**. Con un solo par de
frases, donde el sujeto es además el primer sustantivo y va en segunda posición, no hay
manera de distinguir «sigue al sujeto» de «sigue al primer sustantivo» o de «sigue a la
segunda palabra». Para eso harían falta frases donde esas tres cosas no coincidan, y
aquí no las hay.

Las frases, el modelo y la semilla se importan de `reparto_atencion.py` a propósito: si
cada programa se guardara su propia copia, al mes serían dos frases distintas midiendo
lo mismo y nadie lo notaría.

Uso:
    python reparto_una_a_una.py
    python reparto_una_a_una.py --selftest
"""

# ======================= CONSTANTES =======================

# Se toman de reparto_atencion.py para que no puedan divergir.
from reparto_atencion import (
    MODELO,
    FRASES,
    CANDIDATAS,
    FRACCION_CAPAS_FINALES,
    SEMILLA,
    cargar_modelo,
)

# Orden 97: cifras que la nota escribe con letra; si cambia el recuento, falla aquí y no en la página.
EN_LETRA = {4: "cuatro"}
PRIMERA_PALABRA = "el"            # la papelera: el primer trozo de la frase
SALIDA_CSV = "reparto_una_a_una.csv"

# Frases de control para saber a QUÉ se dedica la mirada que más se fija en «vaso».
# En la frase del vaso, el sujeto es además el primer sustantivo y va en segunda
# posición, así que esa frase sola no puede distinguir las tres cosas. Aquí hay frases
# donde no coinciden. El sujeto y el primer sustantivo van anotados a mano: son
# gramática, no medición, y cualquiera puede comprobarlos leyendo.
# Cada frase cabe en la caja del libro: ninguna pasa de 64 caracteres.
CONTROL_SUJETO = (
    ("El vaso no cabía en el cajón porque era demasiado alto.", "vaso", "vaso"),
    ("En el cajón el vaso no cabía porque era demasiado alto.", "vaso", "cajón"),
    ("El cajón no admitía el vaso porque era demasiado alto.", "cajón", "cajón"),
    ("La caja no cabía en el armario porque era demasiado alta.", "caja", "caja"),
    ("Con el cajón abierto el vaso no cabía porque era alto.", "vaso", "cajón"),
    ("Dentro del armario la caja no cabía porque era alta.", "caja", "armario"),
    ("Sobre la mesa el libro no cabía porque era demasiado ancho.", "libro", "mesa"),
)

TOL_SUMA = 1e-4                   # tolerancia del invariante «cada fila suma uno»

# Umbrales del selftest. Se fijan por razonamiento, ANTES de mirar el resultado, y el
# programa imprime el valor medido al lado para que se vea cuánto margen hay:
#  - test nulo: con las palabras barajadas no queda ningún referente que seguir, así que
#    ninguna mirada debería concentrarse sobre «vaso» como lo hace en la frase real. Se
#    exige que la barajada se quede por debajo del 90 % de la concentración real.
#  - señal implantada: con un referente único y pegado («El vaso era demasiado alto»),
#    alguna mirada tiene que encontrarlo. Se exige una que pase del 30 %.
#  - invariante: cada una de las 336 filas es un reparto, así que suma uno y no tiene
#    ninguna porción negativa.
FRACCION_MAXIMA_NULA = 0.90
MINIMO_IMPLANTADA = 0.30
BARAJADAS_DEL_NULO = 3            # tres barajados, no uno: uno solo es una anécdota

# ==========================================================

import argparse
import csv
import random
import sys

import numpy as np
import torch

from formato import coma, miles, muestra_editorial, pct, tabla_editorial


def fijar_semilla(semilla):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)


def reparto_por_mirada(tok, modelo, frase):
    """Devuelve (palabras, M) con M[capa, mirada, palabra] = la porción del reparto que
    esa mirada de esa capa le da a esa palabra, cuando la palabra que pregunta es la
    última de la frase (el adjetivo).

    Nada se promedia aquí. Las subunidades en que el troceador parte una palabra se SUMAN
    para volver a la palabra, que es lo que el lector ve; sumar es lo único que se puede
    hacer con porciones de un mismo pastel.
    """
    codificado = tok(frase, return_tensors="pt")
    ids = codificado["input_ids"][0]

    piezas = [tok.decode([i]) for i in ids]
    palabras, indice_de_palabra = [], []
    for pieza in piezas:
        if pieza.startswith(" ") or not palabras:
            palabras.append(pieza.strip())
        else:
            palabras[-1] += pieza
        indice_de_palabra.append(len(palabras) - 1)

    normalizadas = [p.strip(".,;:").lower() for p in palabras]
    objetivo = len(palabras) - 1
    ultima_subunidad = max(k for k, w in enumerate(indice_de_palabra) if w == objetivo)

    with torch.no_grad():
        salida = modelo(**codificado, output_attentions=True)

    atenciones = salida.attentions
    assert atenciones is not None and len(atenciones) > 0, \
        "Se esperaban matrices de atención; el modelo no las ha devuelto (¿attn_implementation distinto de 'eager'?)"

    n_capas = len(atenciones)
    n_miradas = atenciones[0].shape[1]
    M = np.zeros((n_capas, n_miradas, len(palabras)), dtype=np.float64)
    for c in range(n_capas):
        bloque = atenciones[c][0, :, ultima_subunidad, :].numpy()
        for h in range(n_miradas):
            for k, w in enumerate(indice_de_palabra):
                M[c, h, w] += float(bloque[h, k])

    assert not np.isnan(M).any(), \
        f"Se esperaba un reparto sin NaN; se encontraron {int(np.isnan(M).sum())} en «{frase}»"
    return normalizadas, M[:, :, : objetivo + 1]


def indice(palabras, cual):
    coincidencias = [i for i, p in enumerate(palabras) if p == cual]
    assert len(coincidencias) == 1, (
        f"Se esperaba que «{cual}» apareciera exactamente una vez tras el troceado; "
        f"se encontraron {len(coincidencias)} en {palabras}"
    )
    return coincidencias[0]


def medir(tok, modelo, frases):
    datos = {}
    for clave, frase in frases.items():
        palabras, M = reparto_por_mirada(tok, modelo, frase)
        sumas = M.sum(axis=2)
        assert abs(sumas.min() - 1.0) < TOL_SUMA and abs(sumas.max() - 1.0) < TOL_SUMA, (
            f"Invariante roto en «{clave}»: se esperaba que cada fila sumara 1 "
            f"(tolerancia {TOL_SUMA}); se encontró entre {sumas.min():.8f} y {sumas.max():.8f}"
        )
        assert (M >= -TOL_SUMA).all(), \
            f"Invariante roto en «{clave}»: hay porciones negativas (mínimo {M.min():.8f})"
        datos[clave] = (palabras, M)
    return datos


def vuelco(M1, M2):
    """Cuánto cambia un reparto entero al cambiar el adjetivo, mirada a mirada.

    0 = los dos repartos son idénticos; 1 = no tienen nada en común. Es la mitad de la
    suma de las diferencias, que es la fracción del pastel que ha cambiado de sitio.
    """
    return 0.5 * np.abs(M1 - M2).sum(axis=2)


# ------------------------- lo que se imprime -------------------------
# Todos los bloques pasan por comprobar_ancho() antes de salir. Todos: en el capítulo 7
# una tabla se libró de pasar por aquí y acabó ocupando una página entera, ilegible.


def bloque_cabecera(modelo, n_capas, n_miradas, rasgos, numeros_por_trozo):
    return tabla_editorial(
        "La forma de esta máquina", ["", "cuánto"],
        [["capas (rondas de mirar y mezclar)", str(n_capas)],
         ["miradas dentro de cada capa", str(n_miradas)],
         ["miradas en total", miles(n_capas * n_miradas)],
         ["rasgos que compara cada mirada", str(rasgos)],
         ["números con que se representa cada trozo", str(numeros_por_trozo)]], "id")


def bloque_promedios(datos, n_capas):
    """El promedio del reparto, con y sin el recorte a las capas finales.

    Las dos columnas existen porque el capítulo da como «promedio de todas las rondas»
    unas cifras que son del último tercio de capas, y no son las mismas.
    """
    desde = n_capas - max(1, int(round(n_capas * FRACCION_CAPAS_FINALES)))
    n_total = n_capas * next(iter(datos.values()))[1].shape[1]
    n_tercio = (n_capas - desde) * next(iter(datos.values()))[1].shape[1]
    filas = []
    for clave, (palabras, M) in datos.items():
        i_el = 0
        i_vaso = indice(palabras, "vaso")
        i_cajon = indice(palabras, "cajón")
        for etiqueta, trozo in (("todas", M), ("final", M[desde:])):
            p = trozo.mean(axis=(0, 1))
            filas.append([clave, etiqueta, pct(p[i_el], 1), pct(p[i_vaso], 1), pct(p[i_cajon], 1),
                          pct(p[-1], 1)])
    return tabla_editorial(
        "¿A quién le llega el reparto del adjetivo?",
        ["adjetivo", "miradas", "«El»", "«vaso»", "«cajón»", "el propio adjetivo"], filas, "iidddd",
        [f"Todas: promedio de las {n_total} miradas del modelo. Final: promedio de las {n_tercio} "
         f"miradas de las {n_capas - desde} últimas capas, que es el recorte con el que mide "
         "reparto_atencion.py.",
         "Cada número: la porción del pastel que se lleva esa palabra. «El» es la primera palabra "
         "de la frase: la papelera. El propio adjetivo: lo que la palabra que pregunta se queda "
         "para sí."])


def bloque_especializada(datos, cual):
    """La mirada que más se fija en una candidata, y las dos siguientes."""
    palabras_ref, M_ref = datos["alto"]
    i = indice(palabras_ref, cual)
    v = M_ref[:, :, i]
    orden = np.dstack(np.unravel_index(np.argsort(-v, axis=None), v.shape))[0][:3]
    capa, mirada = int(orden[0][0]), int(orden[0][1])
    cuantas = int((v > 0.5).sum())
    tres = tabla_editorial(
        f"Las tres miradas que más se fijan en «{cual}»", ["capa", "mirada", f"sobre «{cual}»"],
        [[str(int(c) + 1), str(int(h) + 1), pct(v[int(c), int(h)], 1)] for c, h in orden], "ccd",
        [f"Con el adjetivo «alto», de las {M_ref.shape[0] * M_ref.shape[1]} que tiene el modelo. "
         f"Miradas que pasan de la mitad sobre «{cual}»: {cuantas} de {v.size}."])
    misma = tabla_editorial(
        f"La misma mirada —capa {capa + 1}, mirada {mirada + 1}— en las tres frases",
        ["adjetivo", f"sobre «{cual}»"],
        [[clave, pct(M[capa, mirada, indice(palabras, cual)], 1)] for clave, (palabras, M) in datos.items()],
        "id")
    return tres + [""] + misma


def bloque_vuelco(datos):
    """Cuánto cambia cada mirada al cambiar «alto» por «bajo»."""
    pal_a, M_a = datos["alto"]
    pal_b, M_b = datos["bajo"]
    D = vuelco(M_a, M_b)
    c, h = np.unravel_index(D.argmax(), D.shape)
    dif = {}
    for cual in CANDIDATAS:
        dif[cual] = np.abs(M_a[:, :, indice(pal_a, cual)] - M_b[:, :, indice(pal_b, cual)]).max()
    return tabla_editorial(
        "¿Cambia alguna de opinión al cambiar el adjetivo?", ["", "cuánto"],
        [["vuelco mayor", f"{coma(D.max(), 3)} (capa {int(c) + 1}, mirada {int(h) + 1})"],
         ["vuelco medio", coma(D.mean(), 3)],
         ["vuelco mediano", coma(float(np.median(D)), 3)],
         ["miradas que mueven más de 0,10", f"{int((D > 0.10).sum())} de {D.size}"],
         ["el mayor cambio sobre «vaso», de todas", pct(dif["vaso"], 1)],
         ["el mayor cambio sobre «cajón», de todas", pct(dif["cajón"], 1)]], "id",
        ["Vuelco: la parte del pastel que una mirada mueve de sitio al cambiar «alto» por «bajo»: "
         f"0, no mueve nada; 1, lo mueve todo. Se mide sobre las {D.size} miradas del modelo."])


def bloque_sujeto(tok, modelo, datos, cual):
    """¿A qué se dedica de verdad la mirada que más se fija en «vaso»?

    Se localiza en la frase del vaso —que es donde el capítulo la señala— y se prueba en
    otras frases donde el sujeto, el primer sustantivo y la segunda palabra ya no son la
    misma cosa. Sin esto, «sigue al sujeto» no es una medición: es una lectura.
    """
    palabras_ref, M_ref = datos["alto"]
    v = M_ref[:, :, indice(palabras_ref, cual)]
    capa, mirada = (int(x) for x in np.unravel_index(v.argmax(), v.shape))
    frases = muestra_editorial(
        "Las siete frases", [f"{n}  {frase}" for n, (frase, _, _) in enumerate(CONTROL_SUJETO, 1)],
        [f"En {sum(s != p for _, s, p in CONTROL_SUJETO)} de ellas, el sujeto no es el primer "
         "sustantivo."])
    filas = []
    aciertos = distinguibles = 0
    for n, (frase, sujeto, primero) in enumerate(CONTROL_SUJETO, 1):
        palabras, M = reparto_por_mirada(tok, modelo, frase)
        fila = M[capa, mirada]
        elegida = palabras[int(fila.argmax())]
        if sujeto != primero:
            distinguibles += 1
            aciertos += int(elegida == sujeto)
        filas.append([str(n), sujeto, primero, elegida, "sí" if elegida == sujeto else "no"])
    tabla = tabla_editorial(
        "¿Sigue al sujeto, o al principio de la frase?",
        ["frase", "sujeto", "primer sustantivo", "se fija en", "¿es el sujeto?"], filas, "ciiic",
        [f"Es la mirada número {mirada + 1} de la capa {capa + 1}, la que más se fija en «{cual}» en "
         f"la frase del vaso. En las {EN_LETRA[distinguibles]} frases en que el sujeto no es el primer "
         + ("sustantivo, no se fija en él ni una vez." if aciertos == 0 else
            f"sustantivo, se fija en él {aciertos} {'vez' if aciertos == 1 else 'veces'}.")])
    return frases + [""] + tabla


def escribir_csv(datos, ruta):
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["adjetivo", "capa", "mirada", "palabra", "porcion"])
        for clave, (palabras, M) in datos.items():
            n_capas, n_miradas, _ = M.shape
            for c in range(n_capas):
                for h in range(n_miradas):
                    for j, palabra in enumerate(palabras[: M.shape[2]]):
                        w.writerow([clave, c + 1, h + 1, palabra, f"{M[c, h, j]:.6f}"])


def selftest(tok, modelo):
    """Tres pruebas: test nulo, señal implantada e invariante del dominio."""
    fallos = []

    # 1. TEST NULO — las mismas palabras barajadas, con el adjetivo todavía al final,
    #    que es lo único que hace comparable el montaje: si la consulta cambia de sitio,
    #    cambia también cuántas palabras puede ver, y entonces no se compara nada.
    base = FRASES["alto"]
    palabras_base = base.rstrip(".").split()
    _, M_real = reparto_por_mirada(tok, modelo, base)
    real = float(M_real[:, :, 1].max())     # «vaso» va en segunda posición en la real
    peor = 0.0
    rng = random.Random(SEMILLA)
    for _ in range(BARAJADAS_DEL_NULO):
        cuerpo = palabras_base[:-1]
        rng.shuffle(cuerpo)
        nula = " ".join(cuerpo + [palabras_base[-1]]) + "."
        pal_n, M_n = reparto_por_mirada(tok, modelo, nula)
        peor = max(peor, float(M_n[:, :, indice(pal_n, "vaso")].max()))
    print(f"[1] test nulo         concentración sobre «vaso»: real {real:.4f}, "
          f"barajada {peor:.4f} (tope {FRACCION_MAXIMA_NULA * real:.4f})")
    if peor >= FRACCION_MAXIMA_NULA * real:
        fallos.append(
            f"test nulo: con las palabras barajadas una mirada se concentra sobre «vaso» "
            f"casi igual que en la frase real ({peor:.4f} frente a {real:.4f}); "
            "lo que se está midiendo no depende de la frase"
        )

    # 2. SEÑAL IMPLANTADA — referente único y pegado. Alguna mirada tiene que encontrarlo.
    implantada = "El vaso era demasiado alto."
    pal_i, M_i = reparto_por_mirada(tok, modelo, implantada)
    v = M_i[:, :, indice(pal_i, "vaso")]
    c, h = np.unravel_index(v.argmax(), v.shape)
    print(f"[2] señal implantada  «{implantada}» -> capa {int(c)+1}, mirada {int(h)+1}: "
          f"{v.max():.4f} sobre «vaso» (mínimo {MINIMO_IMPLANTADA})")
    if v.max() < MINIMO_IMPLANTADA:
        fallos.append(
            f"señal implantada: la mirada que más se fija en «vaso» se queda en {v.max():.4f}, "
            f"por debajo de {MINIMO_IMPLANTADA}; el montaje no recupera un referente evidente"
        )

    # 3. INVARIANTE DEL DOMINIO — cada fila de cada mirada es un reparto: suma uno y no
    #    tiene porciones negativas.
    ok = True
    peor_suma = 0.0
    for clave, frase in FRASES.items():
        _, M = reparto_por_mirada(tok, modelo, frase)
        sumas = M.sum(axis=2)
        if max(abs(sumas.min() - 1.0), abs(sumas.max() - 1.0)) >= TOL_SUMA:
            fallos.append(f"invariante: en «{clave}» hay filas que suman {sumas.min():.8f}–{sumas.max():.8f}")
            ok = False
        if (M < -TOL_SUMA).any():
            fallos.append(f"invariante: en «{clave}» hay porciones negativas (mínimo {M.min():.8f})")
            ok = False
        peor_suma = max(peor_suma, abs(sumas.min() - 1.0), abs(sumas.max() - 1.0))
    print(f"[3] invariante        las 336 filas suman uno y no hay porciones negativas: "
          f"{'sí' if ok else 'NO'} (mayor desvío {peor_suma:.2e})")

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
    tok, modelo = cargar_modelo(MODELO)

    if args.selftest:
        sys.exit(selftest(tok, modelo))

    import datetime
    import platform

    datos = medir(tok, modelo, FRASES)
    palabras, M = datos["alto"]
    n_capas, n_miradas, _ = M.shape
    config = modelo.config

    print(f"máquina: {platform.machine()}, {platform.system()} {platform.release()}, procesador")
    print(f"modelo: {MODELO} en float32   fecha: {datetime.date.today().isoformat()}")
    print("la palabra que pregunta es el adjetivo final de cada frase")
    print()
    for bloque in (
        bloque_cabecera(
            modelo,
            n_capas,
            n_miradas,
            config.hidden_size // config.num_attention_heads,
            config.hidden_size,
        ),
        bloque_promedios(datos, n_capas),
        bloque_especializada(datos, "vaso"),
        bloque_sujeto(tok, modelo, datos, "vaso"),
        bloque_vuelco(datos),
    ):
        print("\n".join(bloque))
        print()

    escribir_csv(datos, SALIDA_CSV)
    print(f"Escrito {SALIDA_CSV}")


if __name__ == "__main__":
    main()
