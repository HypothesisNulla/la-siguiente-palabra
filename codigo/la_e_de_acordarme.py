#!/usr/bin/env python3
"""
Capítulo 8 — la mirada de una sola letra, hecha a mano (L24, «el lector no imagina»).

El capítulo 8 cuenta que cada letra hace una pregunta, que las de detrás tienen una etiqueta y un
contenido, que se puntúa cuánto encaja cada pregunta con cada etiqueta, que las puntuaciones se
convierten en un reparto y que la letra se lleva una mezcla de los contenidos. Este programa hace
esas cuentas UNA VEZ, enteras y con sus números, sobre la máquina del Quijote de
`mirada_a_mano.py`: la última letra de «quiero acordarme», la «e», la misma de la figura de ese
capítulo.

No entrena nada nuevo: vuelve a entrenar la máquina de `mirada_a_mano.py` con su misma semilla y
sus mismos pasos (medio minuto en el portátil) y el selftest comprueba que sale la misma: el mismo
acierto y el mismo reparto de la «e» que imprime `datos/salidas/mirada_a_mano.txt`. Después rehace
a mano, con numpy y en doble precisión, cada paso de la ida, y el selftest comprueba que da lo
mismo que la ida de la máquina.

Imprime, en este orden (el libro cita cada bloque por su título):
  1. la máquina, pieza a pieza (cuántos números tiene cada parte)
  2. tres fragmentos de prueba enteros y la frase de la figura: qué ve, qué viene, qué apuesta
  3. de la letra a la lista que entra: la misma «e» en dos sitios
  4. un número de la pregunta, hecho a mano (el comité del capítulo 3)
  5. la puntuación de una pareja, hecha a mano
  6. de las puntuaciones al reparto, letra a letra
  7. doble puntuación no es doble porción; y el reparto de la «e» sin el ajuste de escala
  8. «reparte entre», con repartos de ejemplo
  9. la mezcla, y lo que sigue adelante
 10. la apuesta
 11. sumas de muchos números que van y vienen (el ajuste de escala)

Uso:
    python la_e_de_acordarme.py --selftest
    python la_e_de_acordarme.py > ../datos/salidas/la_e_de_acordarme.txt
"""

# ======================= CONSTANTES =======================

SALIDA_MIRADA = "../datos/salidas/mirada_a_mano.txt"   # de aquí se comprueba que es la misma máquina
TRAMOS_DE_PRUEBA = 3          # cuántos tramos de prueba se enseñan enteros (los primeros, sin elegir)
MIRADOS = 4                   # números de cada lista que se enseñan en las cuentas a mano
MIRADOS_SITIO = 6             # números que se enseñan de la marca del sitio
MIRADOS_MEZCLA = 3            # números que se enseñan del contenido en la mezcla
SITIO_OTRA_E = 4              # la «e» de «quiero» está en el sitio 4 del tramo
LETRA_PAREJA = "m"            # la letra con la que se puntúa la pareja a mano (la más mirada)
APUESTAS = 5                  # cuántas letras se enseñan de la apuesta
EJEMPLOS_REPARTO = [          # repartos de ejemplo para «reparte entre», de cada cien
    ("100 a una sola", [100]),
    ("50 y 50", [50, 50]),
    ("90 y 10", [90, 10]),
    ("25, 25, 25 y 25", [25, 25, 25, 25]),
    ("por igual entre las 16", [100 / 16] * 16),
]
PAREJAS_PUNTUACION = [(1, 2), (2, 4), (10, 20)]   # doble puntuación no es doble porción
SUMANDOS = [4, 16, 32, 64]    # cuántos productos se suman en una puntuación
TIRADAS_SUMAS = 100_000       # sumas al azar por cada fila
SEMILLA_SUMAS = 20260927
# Orden 97: cifras que las notas escriben con letra; si cambia el parámetro, falla aquí y no en la página.
EN_LETRA = {16: "dieciséis", 6: "seis", 32: "treinta y dos"}
TOL_MANO = 1e-4               # diferencia máxima entre la cuenta a mano y la ida de la máquina

# ==========================================================

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

import mirada_a_mano as M
from formato import coma, leer_tablas, miles, tabla_editorial, trozo

AQUI = Path(__file__).resolve().parent
ANCHO = 64                    # ancho de línea: cabe dentro de una cita del libro


def letra(ch):
    return "_" if ch == " " else ch


def dec(x, d=2):
    """Un número con coma y sin «-0,00»."""
    s = coma(x, d)
    return s[1:] if s.startswith("-") and float(s.replace(",", ".")) == 0 else s


# ---------------------------------------------------------------- la máquina

def entrenar():
    texto, aprender, probar = M.cargar_texto()
    prueba = M.trozos(probar, np.random.default_rng(M.SEMILLA), M.EJEMPLOS_PRUEBA)
    p, med = M.entrenar(aprender, M.PASOS, [M.PASOS], prueba)
    return texto, prueba, p, med[M.PASOS]


def numpy_de(p):
    return {k: v.detach().double().cpu().numpy() for k, v in p.items()}


def ida_a_mano(P, ids):
    """La ida de la máquina para UN tramo, paso a paso y en doble precisión. Devuelve todo."""
    T = len(ids)
    ancho = P["letra"].shape[1]
    r = {}
    r["letra"] = P["letra"][ids]                       # la lista de cada letra
    r["sitio"] = P["pos"][:T]                          # la marca de su sitio
    r["entra"] = r["letra"] + r["sitio"]               # lo que entra
    r["pregunta"] = r["entra"] @ P["Wp"]
    r["etiqueta"] = r["entra"] @ P["We"]
    r["contenido"] = r["entra"] @ P["Wc"]
    r["divisor"] = math.sqrt(ancho)
    q = r["pregunta"][-1]
    r["suma"] = r["etiqueta"] @ q                      # pregunta de la última × etiqueta de cada una
    r["puntuacion"] = r["suma"] / r["divisor"]
    r["fuerza"] = np.exp(r["puntuacion"])
    r["reparto"] = r["fuerza"] / r["fuerza"].sum()
    r["mezcla"] = r["reparto"] @ r["contenido"]
    r["traido"] = r["mezcla"] @ P["Wo"]
    r["sigue"] = r["entra"][-1] + r["traido"]
    g = r["sigue"] + np.maximum(r["sigue"] @ P["W1"] + P["b1"], 0) @ P["W2"] + P["b2"]
    ap = g @ P["Ws"] + P["bs"]
    e = np.exp(ap - ap.max())
    r["apuesta"] = e / e.sum()
    return r


def reparte_entre(reparto):
    """A cuántas letras equivale un reparto: las que, repartiendo por igual, lo harían igual de
    concentrado (la misma cuenta que usa mirada_a_mano.py para su columna)."""
    a = np.asarray(reparto, dtype=float)
    a = a / a.sum()
    a = a[a > 0]
    return float(np.exp(-(a * np.log(a)).sum()))


def reparto_de(puntuaciones):
    f = np.exp(np.asarray(puntuaciones, dtype=float) - max(puntuaciones))
    return f / f.sum()


def leer_mirada(ruta):
    """De la salida de mirada_a_mano.py: el reparto final de la frase, de cada cien, y el acierto
    final (de sus tablas editoriales; L24, 9 de octubre)."""
    tablas = leer_tablas(Path(ruta).read_text(encoding="utf-8"))
    titulos = [t for t in tablas if t.startswith("El reparto de la última letra de «")]
    assert len(titulos) == 1, f"no está la tabla del reparto en {ruta}"
    final = [int(f[3]) for f in tablas[titulos[0]][1]]
    momentos = {f[0]: f[1] for f in tablas["Lo que acierta y cómo reparte la mirada, en cinco momentos"][1]}
    assert "20.000 pasos" in momentos, "no está la fila de 20.000 pasos del apartado 2"
    return final, float(momentos["20.000 pasos"].replace(" %", "").replace(",", "."))


# ---------------------------------------------------------------- selftest

def selftest(p, med, prueba):
    fallos = []
    P = numpy_de(p)
    tramo = M.FRASE[-M.CONTEXTO:]
    ids = np.array([M.INDICE[c] for c in tramo])

    # 1. TEST NULO — con la tabla de las preguntas a cero, todas las puntuaciones valen cero y el
    #    reparto tiene que salir por igual: 16 letras, 6,25 de cada cien cada una.
    P0 = dict(P, Wp=np.zeros_like(P["Wp"]))
    r0 = ida_a_mano(P0, ids)
    igual = np.allclose(r0["reparto"], 1 / M.CONTEXTO) and abs(reparte_entre(r0["reparto"]) - 16) < 1e-9
    print(f"[1] test nulo         sin pregunta, reparto por igual: {'sí' if igual else 'NO'} "
          f"(reparte entre {coma(reparte_entre(r0['reparto']))})")
    if not igual:
        fallos.append("test nulo: sin pregunta el reparto no sale por igual")

    # 2. SEÑAL — es la misma máquina del libro (mismo acierto y mismo reparto de la «e» que la salida
    #    de mirada_a_mano.py), y la cuenta a mano da lo mismo que la ida de la máquina.
    final_libro, acierto_libro = leer_mirada(AQUI / SALIDA_MIRADA)
    r = ida_a_mano(P, ids)
    mismo_reparto = [int(round(100 * v)) for v in r["reparto"]] == final_libro
    mismo_acierto = coma(100 * med["acierto"]) == coma(acierto_libro)
    x = torch.tensor([ids], device=M.DISPOSITIVO)
    ap, c = M.adelante(p, x)
    maquina = {"reparto": c["a"][0, -1], "traido": (c["traido"] @ p["Wo"])[0, -1],
               "apuesta": torch.softmax(ap[0, -1].double(), -1)}
    peor = max(float(np.abs(r[k] - v.detach().double().cpu().numpy()).max()) for k, v in maquina.items())
    print(f"[2] señal             misma máquina que el libro: reparto {'sí' if mismo_reparto else 'NO'}, "
          f"acierto {'sí' if mismo_acierto else 'NO'}; a mano contra la máquina: {peor:.1e}")
    if not (mismo_reparto and mismo_acierto):
        fallos.append("señal: el reentrenamiento no da la máquina de mirada_a_mano.txt")
    if peor > TOL_MANO:
        fallos.append(f"señal: la cuenta a mano se separa de la ida de la máquina ({peor:.1e})")

    # 3. INVARIANTE — un reparto suma cien; «reparte entre» da 1 con una sola, 2 con mitad y mitad
    #    y 4 con cuatro cuartos; multiplicar todas las puntuaciones por lo mismo mayor que 1 nunca
    #    abre el reparto; y la suma de n productos al azar se aleja de cero como la raíz de n.
    ok = abs(r["reparto"].sum() - 1) < 1e-12
    ok &= all(abs(reparte_entre(rep) - n) < 1e-9 for rep, n in (([100], 1), ([50, 50], 2), ([25] * 4, 4)))
    ok &= reparte_entre(reparto_de(3 * r["puntuacion"])) < reparte_entre(r["reparto"])
    dist = distancias(np.random.default_rng(SEMILLA_SUMAS))
    ok &= all(abs(d / math.sqrt(n) - dist[0] / 2) < 0.05 for n, d in zip(SUMANDOS, dist))
    print(f"[3] invariante        repartos que suman cien, «reparte entre» 1, 2 y 4, cerrar al "
          f"multiplicar y crecer como la raíz: {'sí' if ok else 'NO'}")
    if not ok:
        fallos.append("invariante: alguna de las cuatro comprobaciones no se cumple")
    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def dentro_de_uno(rng):
    """De cada cien números al azar de tamaño 1, cuántos caen entre menos 1 y 1."""
    return float((np.abs(rng.standard_normal(TIRADAS_SUMAS)) < 1).mean())


def distancias(rng):
    """Para cada número de sumandos, a qué distancia de cero queda de media la suma de productos
    de dos números al azar de tamaño 1 (como una pregunta y una etiqueta que empiezan así)."""
    out = []
    for n in SUMANDOS:
        a = rng.standard_normal((TIRADAS_SUMAS, n))
        b = rng.standard_normal((TIRADAS_SUMAS, n))
        out.append(float(np.abs((a * b).sum(1)).mean()))
    return out


# ---------------------------------------------------------------- lo que imprime

def informe(texto, prueba, p, med):
    """Cada apartado, en tablas editoriales (L24, 9 de octubre; regla 6 ter). Las cuentas no
    cambian."""
    P = numpy_de(p)
    tramo = M.FRASE[-M.CONTEXTO:]
    ids = np.array([M.INDICE[c] for c in tramo])
    r = ida_a_mano(P, ids)
    T = M.CONTEXTO
    L = ["", "####### capítulo 8: la mirada de la «e», hecha a mano #######",
         f"La máquina de mirada_a_mano.py, entrenada otra vez con su semilla ({M.SEMILLA}) y sus "
         f"{miles(M.PASOS)} pasos: acierta {coma(100 * med['acierto'])} %, como en el libro.",
         "Las cuentas se hacen con todos los decimales; se enseñan con dos.", ""]
    neg = lambda fila: [f"**{c}**" if c else "" for c in fila]
    num = lambda j: f"número: {j}"

    # 1. la máquina, pieza a pieza
    n = {k: v.numel() for k, v in p.items()}
    V, A = P["letra"].shape
    piezas = [
        (f"las listas de las {V} letras, {A} números cada una", n["letra"]),
        (f"las marcas de los {T} sitios, {A} números cada una", n["pos"]),
        ("la tabla de las preguntas", n["Wp"]),
        ("la tabla de las etiquetas", n["We"]),
        ("la tabla de los contenidos", n["Wc"]),
        ("la tabla que devuelve lo traído", n["Wo"]),
        ("los comités de después", n["W1"] + n["b1"] + n["W2"] + n["b2"]),
        (f"de la lista a los porcentajes de las {V} letras", n["Ws"] + n["bs"]),
    ]
    total = sum(v for _, v in piezas)
    de_mirar = n["Wp"] + n["We"] + n["Wc"] + n["Wo"]
    assert total == M.cuantos_numeros(p)
    L += ["--- 1. LA MÁQUINA, PIEZA A PIEZA ---", ""] + tabla_editorial(
        "La máquina, pieza a pieza", ["la pieza", "números"],
        [[nombre, miles(v)] for nombre, v in piezas] + [neg(["total", miles(total)])], "id",
        [f"Sin mirar atrás se quitan las cuatro tablas: {miles(de_mirar)} números. Quedan "
         f"{miles(total - de_mirar)}."]) + [""]

    # 2. fragmentos de prueba enteros
    i_frase = texto.find(M.FRASE)
    assert texto.count(M.FRASE) == 1 and i_frase >= 0
    casos = [("".join(M.LETRAS[i] for i in f[:T]), M.LETRAS[f[T]], str(j + 1))
             for j, f in enumerate(prueba[:TRAMOS_DE_PRUEBA])]
    casos.append((tramo, texto[i_frase + len(M.FRASE)], "el de la figura"))
    filas = []
    for ve, viene, nombre in casos:
        rr = ida_a_mano(P, np.array([M.INDICE[c] for c in ve]))
        orden = np.argsort(-rr["apuesta"])[:3]
        apuesta = ", ".join(f"{trozo(letra(M.LETRAS[k]))} {coma(100 * rr['apuesta'][k])} %" for k in orden)
        filas.append([nombre, trozo("".join(letra(ch) for ch in ve)), trozo(letra(viene)), apuesta,
                      "sí" if M.LETRAS[orden[0]] == viene else "no"])
    L += ["--- 2. FRAGMENTOS ---", ""] + tabla_editorial(
        "Fragmentos: lo que ve, lo que viene, lo más probable",
        ["fragmento", "lo que ve", "viene", "lo más probable", "¿acierta?"], filas, "ciiic",
        ["Fragmento: los tres primeros de prueba, sin elegir, y el de la figura.",
         "El guion bajo es un espacio. Acierta cuando la letra a la que da más porcentaje es la que "
         "viene de verdad."]) + [""]

    # 3. de la letra a la lista que entra
    k = MIRADOS_SITIO
    s1, s2 = SITIO_OTRA_E, T
    assert tramo[s1 - 1] == tramo[s2 - 1] == "e"
    ent1, ent2 = r["entra"][s1 - 1], r["entra"][s2 - 1]
    parecido = float(ent1 @ ent2 / np.linalg.norm(ent1) / np.linalg.norm(ent2))
    assert np.allclose(r["letra"][s1 - 1], r["letra"][s2 - 1])
    fila = lambda nombre, v: [nombre] + [dec(x) for x in v[:k]]
    L += ["--- 3. LA MISMA «e» EN DOS SITIOS ---", ""]
    for s, ent, notas in (
            (s1, ent1, [f"Los {k} primeros de sus {A} números. En «quiero acordarme» hay una «e» en "
                        f"el sitio {s1} y otra en el {s2}."]),
            (s2, ent2, [f"Los {k} primeros de sus {A} números.",
                        f"Con sus {EN_LETRA[A]} números, las dos listas que entran se parecen "
                        f"{coma(parecido, 2)}; dos listas idénticas darían 1."])):
        L += tabla_editorial(
            f"La «e» del sitio {s}: la lista que entra", [""] + [num(j + 1) for j in range(k)],
            [fila("la lista de la «e»", r["letra"][s2 - 1]),
             fila(f"más la marca del sitio {s}", r["sitio"][s - 1]),
             neg(fila("da: lo que entra", ent))], "i" + "d" * k, notas) + [""]

    # 4. un número de la pregunta, hecho a mano
    col = P["Wp"][:, 0]
    ent = r["entra"][-1]
    aporta = ent * col
    assert abs(aporta.sum() - r["pregunta"][-1][0]) < 1e-9
    L += ["--- 4. UN NÚMERO DE LA PREGUNTA, HECHO A MANO ---", ""] + tabla_editorial(
        "Un número de la pregunta de la «e», hecho a mano",
        ["número de la lista", "vale", "peso", "aporta"],
        [[f"número {j + 1}", dec(ent[j]), dec(col[j]), dec(aporta[j])] for j in range(MIRADOS)]
        + [[f"los otros {A - MIRADOS}", "", "", dec(aporta[MIRADOS:].sum())],
           neg(["total: el número 1 de la pregunta", "", "", dec(aporta.sum())])], "iddd",
        [f"La lista que entra en el sitio {T}, por la primera columna de la tabla de las "
         "preguntas: un comité como el del capítulo 3."]) + [""]
    L += tabla_editorial(
        f"La pregunta, la etiqueta y el contenido de la «e» del sitio {T}",
        [""] + [num(j + 1) for j in range(MIRADOS)],
        [[f"su {nombre}"] + [dec(v) for v in r[nombre][-1][:MIRADOS]]
         for nombre in ("pregunta", "etiqueta", "contenido")], "i" + "d" * MIRADOS,
        [f"Lo mismo con cada una de las {A} columnas da los {A} números de la pregunta; con la "
         "tabla de las etiquetas, los de la etiqueta; con la de los contenidos, los del contenido. "
         f"Los {MIRADOS} primeros de cada una."]) + [""]

    # 5. la puntuación de una pareja, hecha a mano
    j_par = max(i for i, ch in enumerate(tramo) if ch == LETRA_PAREJA)
    q, et = r["pregunta"][-1], r["etiqueta"][j_par]
    prod = q * et
    assert abs(prod.sum() - r["suma"][j_par]) < 1e-9
    L += ["--- 5. CUÁNTO ENCAJA UNA PREGUNTA CON UNA ETIQUETA ---", ""] + tabla_editorial(
        f"Cuánto encaja la pregunta de la «e» con la etiqueta de la «{LETRA_PAREJA}»",
        ["número", "pregunta de la «e»", f"etiqueta de la «{LETRA_PAREJA}»", "producto"],
        [[str(j + 1), dec(q[j]), dec(et[j]), dec(prod[j])] for j in range(MIRADOS)]
        + [[f"los otros {A - MIRADOS}", "", "", dec(prod[MIRADOS:].sum())],
           neg([f"suma de los {A} productos", "", "", dec(prod.sum())]),
           neg([f"entre {dec(r['divisor'])} (el ajuste de escala): la puntuación", "", "",
                dec(r["puntuacion"][j_par])])], "iddd",
        ["Número a número: se multiplican y se suma todo."]) + [""]

    # 6. de las puntuaciones al reparto
    L += ["--- 6. DE LAS PUNTUACIONES AL REPARTO ---", ""] + tabla_editorial(
        f"De las puntuaciones al reparto: la «e» y las {T} letras",
        ["sitio", "letra", "puntuación", "fuerza", "de cada cien"],
        [[str(i + 1), letra(ch), dec(r["puntuacion"][i], 3), dec(r["fuerza"][i]),
          f"{coma(100 * r['reparto'][i])} %"] for i, ch in enumerate(tramo)]
        + [neg(["suma", "", "", dec(r["fuerza"].sum()), f"{coma(100 * r['reparto'].sum())} %"])],
        "ccddd",
        ["El guion bajo es un espacio. Cada fila es la pregunta de la «e» contra la etiqueta de esa "
         "letra.",
         f"La fuerza vale 1 con puntuación 0; {coma(math.e, 2)} con 1; {coma(math.e ** 2, 2)} con 2; "
         f"{coma(math.e ** 3, 2)} con 3, y {coma(math.e ** -1, 2)} con menos 1. La última columna es "
         "esa fuerza entre la suma de todas, por cien."]) + [""]

    # 7. doble puntuación no es doble porción
    filas = []
    for a, b in PAREJAS_PUNTUACION:
        ra = reparto_de([a, b])
        filas.append([f"{a} y {b}", f"{coma(100 * ra[0])} y {coma(100 * ra[1])}"])
    sin = reparto_de(r["suma"])
    ult = list(range(T - 6, T))
    L += ["--- 7. DOBLE PUNTUACIÓN NO ES DOBLE PORCIÓN ---", ""] + tabla_editorial(
        "Doble puntuación no es doble porción",
        ["dos letras, con puntuaciones", "se llevan, de cada cien"], filas, "id") + [""]
    L += tabla_editorial(
        "El reparto de la «e», con el ajuste y sin él",
        [""] + [f"{i + 1} {letra(tramo[i])}" for i in ult] + ["reparte"],
        [["con el ajuste"] + [str(round(100 * r["reparto"][i])) for i in ult]
         + [coma(reparte_entre(r["reparto"]))],
         ["sin el ajuste"] + [str(round(100 * sin[i])) for i in ult] + [coma(reparte_entre(sin))]],
        "i" + "d" * len(ult) + "d",
        [f"Lo que se lleva cada una de las {EN_LETRA[len(ult)]} últimas letras, de cada cien; sin el "
         f"ajuste es sin dividir las puntuaciones entre {dec(r['divisor'])}. La última columna cuenta "
         "como «reparte entre»."]) + [""]

    # 8. reparte entre
    mayores = [i for i in range(T) if round(100 * r["reparto"][i]) > 0]
    L += ["--- 8. «REPARTE ENTRE» ---", ""] + tabla_editorial(
        "«Reparte entre», con repartos de ejemplo", ["el reparto, de cada cien", "reparte entre"],
        [[nombre, coma(reparte_entre(rep))] for nombre, rep in EJEMPLOS_REPARTO]
        + [["el de la «e»: " + ", ".join(str(round(100 * r["reparto"][i])) for i in mayores),
            coma(reparte_entre(r["reparto"]))]], "id",
        [f"Con 1, todo va a una sola letra; con {T}, las {EN_LETRA[T]} reciben lo mismo."]) + [""]

    # 9. la mezcla
    km = MIRADOS_MEZCLA
    resto = [i for i in range(T) if i not in mayores]
    otras = sum(r["reparto"][i] * r["contenido"][i] for i in resto)
    filas = [[f"{i + 1} {letra(tramo[i])}", f"{coma(100 * r['reparto'][i])} %"]
             + [dec(v) for v in r["contenido"][i][:km]]
             + [dec(v) for v in (r["reparto"][i] * r["contenido"][i])[:km]] for i in mayores]
    filas.append([f"otras {len(resto)}", f"{coma(100 * r['reparto'][resto].sum())} %"] + [""] * km
                 + [dec(v) for v in otras[:km]])
    filas.append(neg(["la mezcla (la suma)", ""] + [""] * km + [dec(v) for v in r["mezcla"][:km]]))
    L += ["--- 9. LA MEZCLA ---", ""] + tabla_editorial(
        "La mezcla: cada contenido, por su porción, y todo sumado",
        ["letra", "porción"] + [f"contenido: {j + 1}" for j in range(km)]
        + [f"por su porción: {j + 1}" for j in range(km)], filas, "id" + "d" * (2 * km),
        [f"Los {km} primeros de los {A} números de cada contenido."]) + [""]
    assert np.allclose(r["mezcla"], sum(r["reparto"][i] * r["contenido"][i] for i in range(T)))
    L += tabla_editorial(
        "Lo que sigue adelante desde la «e»", [""] + [num(j + 1) for j in range(km)],
        [[f"la lista que entró en el sitio {T}"] + [dec(v) for v in r["entra"][-1][:km]],
         ["más lo traído"] + [dec(v) for v in r["traido"][:km]],
         neg(["da lo que pasa a los comités"] + [dec(v) for v in r["sigue"][:km]])],
        "i" + "d" * km,
        ["Lo traído: la mezcla, pasada por la tabla que devuelve lo traído.",
         "La lista de la propia letra sigue adelante entera: la mirada le suma lo traído, no la "
         "sustituye. Sin mirar atrás, solo sigue la primera fila."]) + [""]

    # 10. la apuesta
    orden = np.argsort(-r["apuesta"])
    viene = texto[i_frase + len(M.FRASE)]
    kv = M.INDICE[viene]
    pv = int(np.where(orden == kv)[0][0]) + 1
    L += ["--- 10. LA LISTA DE LA «e» ---", ""] + tabla_editorial(
        "La lista de la «e»: qué letra viene detrás de «acordarme»",
        ["puesto", "letra", "de cada cien"],
        [[str(pu), f"«{letra(M.LETRAS[kk])}»", f"{coma(100 * r['apuesta'][kk])} %"]
         for pu, kk in enumerate(orden[:APUESTAS], 1)], "cid",
        [f"En el Quijote viene «{letra(viene)}», que está en el puesto {pv} con "
         f"{coma(100 * r['apuesta'][kv])} %."]) + [""]

    # 11. sumas de muchos números que van y vienen
    dist = distancias(np.random.default_rng(SEMILLA_SUMAS))
    L += ["--- 11. SUMAS DE MUCHOS NÚMEROS QUE VAN Y VIENEN ---", ""] + tabla_editorial(
        "Sumas de muchos números que van y vienen",
        ["se suman", "distancia a cero, de media", "ajuste entre", "tras el ajuste"],
        [[str(nn), dec(d), dec(math.sqrt(nn)), dec(d / math.sqrt(nn))] for nn, d in zip(SUMANDOS, dist)],
        "dddd",
        [f"Cada fila son {miles(TIRADAS_SUMAS)} sumas, y cada sumando, dos números al azar de tamaño 1 "
         "multiplicados; de tamaño 1 quiere decir que de cada cien, "
         f"{coma(100 * dentro_de_uno(np.random.default_rng(SEMILLA_SUMAS)), 0)} caen entre menos 1 y 1. "
         "El ajuste es el número que, multiplicado por sí mismo, da cuántos se suman (2 por 2 son 4; "
         "8 por 8, 64), y la última columna, la distancia a cero después de dividir entre él."])
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    texto, prueba, p, med = entrenar()
    print("--- selftest ---")
    codigo = selftest(p, med, prueba)
    if codigo or args.selftest:
        return codigo
    for l in informe(texto, prueba, p, med):
        print(l)
    return 0


if __name__ == "__main__":
    sys.exit(main())
