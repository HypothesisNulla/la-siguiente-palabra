#!/usr/bin/env python3
"""
Cuántas cuentas hay detrás de cambiar un trozo del contexto.

`editar_contexto.py` midió con el reloj, en un portátil, lo que cuesta un turno que solo añade
trozos al final frente a uno que cambia un trozo del texto viejo. Este programa no mide nada:
CUENTA las multiplicaciones que hace el modelo en cada uno de esos turnos, con las medidas
reales de los dos modelos (las lee de su fichero de configuración), y pone al lado lo que midió
el reloj. Sirve para dos cosas:

  1. ver qué parte del trabajo se va en las casillas de la cuadrícula (cada trozo comparado con
     él mismo y con todos los anteriores, y lo que trae de cada uno) y qué parte en el resto (las
     tablas que sacan pregunta, etiqueta y contenido, la que junta las miradas y la parte de
     mezclar), según lo largo que sea el texto;
  2. ver si el número de cuentas explica lo que midió el reloj.

Las cuentas no dependen de la máquina. Lo medido que se pone al lado sí (lo dice
`editar_contexto.txt`: MacBook Pro M4 Max, 36 GB, GPU).

Qué se cuenta y qué no:
  - se cuentan las multiplicaciones de las tablas de cada capa (pregunta, etiqueta, contenido,
    la que junta las miradas y las tres de la parte de mezclar) y las de las casillas (pregunta
    por etiqueta, y reparto por contenido);
  - se cuenta la tabla final que da la lista de probabilidades solo para el último trozo del
    turno, que es lo que hace mlx-lm al prefilar;
  - NO se cuentan las sumas sueltas, las normalizaciones, la regla de las fuerzas ni la marca
    de la posición: son unas pocas cuentas por número, no una por cada pareja de números;
  - en las casillas se cuentan solo las necesarias (cada trozo con él y los anteriores). Un
    programa real puede hacer más (calcular el cuadrado entero y tapar la mitad), y eso no se
    cuenta.

Uso (desde codigo/, con HF_HUB_OFFLINE=1 para no descargar nada):
    python lo_que_cuesta_cambiar.py --selftest
    python lo_que_cuesta_cambiar.py > ../datos/salidas/lo_que_cuesta_cambiar.txt
"""

# ======================= CONSTANTES =======================

# Los mismos dos que midió editar_contexto.py: (nombre en su csv, repositorio, rótulo en las
# tablas). El de 32.000M va comprimido (capítulo 13): eso cambia el reloj, no las cuentas.
MODELOS = [
    ("7B sin comprimir", "mlx-community/Qwen2.5-7B-Instruct-bf16",  "7.000M"),
    ("32B comprimido",   "mlx-community/Qwen2.5-32B-Instruct-4bit", "32.000M"),
]
LEYENDA_MODELOS = ["7.000M, 32.000M: millones de números de cada modelo. El de",
                   "32.000M va comprimido, con sus números redondeados para que",
                   "quepa en el portátil: eso cambia el reloj, no las cuentas."]
ROTULO = {csv_: libro for csv_, _, libro in MODELOS}
MEDIDO = "../datos/salidas/editar_contexto.csv"
CONDICIONES = ["solo añadir", "cambio al 0 %", "cambio al 25 %", "cambio al 50 %",
               "cambio al 75 %", "cambio al 95 %", "sin caché"]
REPETICIONES = 5     # las que cuentan, por condición, en editar_contexto.py
CALENTAMIENTO = 0    # en el csv, la repetición 0 es el calentamiento y no se cuenta
# Cómo se llama cada condición en el libro, que no usa la palabra «caché»: la llama «lo ya
# calculado» («lo guardado» ya es otra cosa en el capítulo 6).
EN_EL_LIBRO = {"sin caché": "todo desde cero"}
LARGOS = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000]
UMBRAL_NULO = 1e-4   # con un texto de un trozo, las casillas no llegan a esta fracción
TOLERANCIA_NULO = 0.01   # cambiar el último trozo viejo: a menos de esto de «solo añadir»
# Medidas del 7B escritas a mano SOLO para el selftest, que no debe necesitar internet ni la caché
# de Hugging Face. La ejecución normal las lee del config.json (copiadas de ahí el 5 oct 2026).
MEDIDAS_SELFTEST = {"hidden_size": 3584, "intermediate_size": 18944, "num_hidden_layers": 28,
                    "num_attention_heads": 28, "num_key_value_heads": 4, "vocab_size": 152064}

# ==========================================================

import argparse
import csv
import json
import statistics
import sys

from formato import coma, miles, pct, tabla_editorial


# --------------------------- las medidas de un modelo ---------------------------

CLAVES = {"hidden_size": "numeros", "intermediate_size": "mezclar",
          "num_hidden_layers": "capas", "num_attention_heads": "miradas",
          "num_key_value_heads": "miradas_kv", "vocab_size": "vocabulario"}


def medidas_de(config, origen):
    """De un config.json (ya leído) a las seis medidas que hacen falta, validadas."""
    faltan = [c for c in CLAVES if c not in config]
    assert not faltan, f"Se esperaban las claves {sorted(CLAVES)} en {origen}; faltan {faltan}"
    m = {CLAVES[c]: config[c] for c in CLAVES}
    for k, v in m.items():
        assert isinstance(v, int) and v > 0, (
            f"Se esperaba un entero positivo en «{k}» de {origen}; se encontró {v!r}")
    assert m["numeros"] % m["miradas"] == 0, (
        f"Se esperaba que {m['numeros']} números se repartieran a partes iguales entre "
        f"{m['miradas']} miradas en {origen}; no es así")
    assert m["miradas"] % m["miradas_kv"] == 0, (
        f"Se esperaba que {m['miradas']} miradas compartieran a partes iguales "
        f"{m['miradas_kv']} juegos de etiqueta y contenido en {origen}; no es así")
    return m


def leer_medidas(repo):
    from huggingface_hub import hf_hub_download
    ruta = hf_hub_download(repo, "config.json")
    with open(ruta, encoding="utf-8") as fh:
        return medidas_de(json.load(fh), repo)


# --------------------------- las cuentas ---------------------------

def tablas_por_trozo_y_capa(m):
    """Multiplicaciones de las tablas de UNA capa para UN trozo."""
    h = m["numeros"]
    kv = m["miradas_kv"] * (h // m["miradas"])   # números de etiqueta (y de contenido)
    return (h * h          # pregunta
            + h * kv       # etiqueta
            + h * kv       # contenido
            + h * h        # la tabla que junta las miradas
            + 3 * h * m["mezclar"])   # las tres de la parte de mezclar


def casillas_hasta(n):
    """Casillas de un texto de n trozos: cada trozo con él mismo y con todos los anteriores."""
    return n * (n + 1) // 2


def cuentas_turno(m, desde, hasta):
    """Multiplicaciones de calcular los trozos [desde, hasta), con lo anterior ya guardado.
    Devuelve (tablas, casillas, final)."""
    assert 0 <= desde < hasta, f"Se esperaba 0 <= desde < hasta; se encontró {desde}, {hasta}"
    tablas = (hasta - desde) * tablas_por_trozo_y_capa(m) * m["capas"]
    # en cada casilla: pregunta por etiqueta y reparto por contenido; en todas las miradas
    # juntas son 2 x números multiplicaciones por casilla y capa
    casillas = (casillas_hasta(hasta) - casillas_hasta(desde)) * 2 * m["numeros"] * m["capas"]
    final = m["numeros"] * m["vocabulario"]
    return tablas, casillas, final


def total(c):
    return sum(c)


def parte_casillas(m, n):
    """Fracción del trabajo que se va en las casillas, texto de n trozos desde cero, sin la
    tabla final (que es una sola vez por turno)."""
    tablas, casillas, _ = cuentas_turno(m, 0, n)
    return casillas / (tablas + casillas)


def largo_del_empate(m, tope=10**8):
    """El texto más corto, calculado desde cero, en el que las casillas ya pesan tanto como
    las tablas. Por bisección sobre la cuenta, no por la fórmula cerrada: el selftest compara
    las dos."""
    def ya(n):
        t, c, _ = cuentas_turno(m, 0, n)
        return c >= t
    assert ya(tope), f"Se esperaba el empate antes de {tope} trozos; no llega"
    bajo, alto = 1, tope
    while bajo < alto:
        medio = (bajo + alto) // 2
        if ya(medio):
            alto = medio
        else:
            bajo = medio + 1
    return bajo


# --------------------------- lo medido ---------------------------

def leer_medido(ruta=MEDIDO):
    """De editar_contexto.csv: para cada (modelo, condición), el primer trozo recalculado y la
    mediana de los tiempos. Valida lo que lee."""
    with open(ruta, encoding="utf-8", newline="") as fh:
        filas = list(csv.DictReader(fh))
    esperadas = ["modelo", "condicion", "k", "tokens_recalculados", "repeticion", "tiempo_ms"]
    assert filas and list(filas[0].keys()) == esperadas, (
        f"Se esperaban las columnas {esperadas} en {ruta}; se encontraron "
        f"{list(filas[0].keys()) if filas else 'ninguna fila'}")
    tiempos, recalc = {}, {}
    for f in filas:
        clave = (f["modelo"], f["condicion"])
        r = int(f["tokens_recalculados"])
        assert recalc.setdefault(clave, r) == r, (
            f"Se esperaba el mismo número de trozos recalculados en todas las filas de {clave}; "
            f"se encontraron {recalc[clave]} y {r}")
        t = float(f["tiempo_ms"])
        assert t > 0, f"Se esperaba un tiempo positivo en {clave}; se encontró {t}"
        rep = int(f["repeticion"])
        assert 0 <= rep <= REPETICIONES, (
            f"Se esperaba una repetición entre 0 y {REPETICIONES} en {clave}; se encontró {rep}")
        if rep == CALENTAMIENTO:
            continue
        tiempos.setdefault(clave, []).append(t)
    nombres = [n for n, _, _ in MODELOS]
    for n in nombres:
        for c in CONDICIONES:
            assert (n, c) in tiempos, f"Se esperaba la condición «{c}» de «{n}» en {ruta}; no está"
            assert len(tiempos[(n, c)]) == REPETICIONES, (
                f"Se esperaban {REPETICIONES} repeticiones de «{c}» en «{n}»; hay "
                f"{len(tiempos[(n, c)])}")
    total_trozos = {recalc[(n, "sin caché")] for n in nombres}
    assert len(total_trozos) == 1, f"Se esperaba el mismo texto en los dos modelos; {total_trozos}"
    (n_total,) = total_trozos
    medido = {}
    for (n, c), ts in tiempos.items():
        medido[(n, c)] = {"desde": n_total - recalc[(n, c)], "recalculados": recalc[(n, c)],
                          "mediana": statistics.median(ts)}
    # el «k» del csv tiene que cuadrar con los trozos recalculados (es el invariante de verdad:
    # que el texto tiene n_total trozos y se recalcula desde k hasta el final)
    for f in filas:
        if f["condicion"].startswith("cambio"):
            assert int(f["k"]) + int(f["tokens_recalculados"]) == n_total, (
                f"Se esperaba k + recalculados = {n_total} en {f}; no cuadra")
    return medido, n_total


# --------------------------- la salida ---------------------------

def imprimir(medidas, medido, n_total):
    """Las tablas editoriales (L24, 9 de octubre; regla 6 ter). Las cuentas no cambian."""
    leyenda = " ".join(LEYENDA_MODELOS)
    out = ["Cuentas, no tiempos: esta parte no depende de la máquina.",
           f"Lo medido con el reloj viene de {MEDIDO}",
           "(eso sí depende de la máquina: MacBook Pro M4 Max, 36 GB, GPU).", ""]
    a, b = (n for n, _, _ in MODELOS)
    ra, rb = ROTULO[a], ROTULO[b]
    filas = []
    for nombre, repo, _ in MODELOS:
        m = medidas[nombre]
        filas.append([ROTULO[nombre], repo, str(m["capas"]), miles(m["numeros"]),
                      miles(m["mezclar"]), str(m["miradas"]), str(m["miradas_kv"]),
                      miles(m["vocabulario"])])
    out += tabla_editorial(
        "Las medidas de los dos modelos, leídas de su config.json",
        ["modelo", "repositorio", "capas", "números por trozo", "mezclar", "miradas",
         "juegos de etiqueta y contenido", "vocabulario, en trozos"], filas, "iidddddd",
        [leyenda]) + [""]

    out += tabla_editorial(
        "Parte del trabajo que se va en las casillas",
        ["trozos", ra, rb],
        [[miles(n), pct(parte_casillas(medidas[a], n)), pct(parte_casillas(medidas[b], n))]
         for n in LARGOS], "ddd",
        ["Un texto entero, calculado desde cero; de cada cien multiplicaciones.",
         f"Las casillas pesan tanto como todo lo demás a partir de {miles(largo_del_empate(medidas[a]))} "
         f"trozos en el de {ra} y de {miles(largo_del_empate(medidas[b]))} en el de {rb}.",
         "Casillas: cada trozo comparado con él mismo y con todos los anteriores, y lo que trae de "
         "cada uno. Todo lo demás: las tablas que sacan pregunta, etiqueta y contenido, la que junta "
         "las miradas y la parte de mezclar.", leyenda]) + [""]

    base = {}
    for nombre in (a, b):
        d = medido[(nombre, "solo añadir")]
        base[nombre] = (total(cuentas_turno(medidas[nombre], d["desde"], n_total)), d["mediana"])
    filas = []
    for c in CONDICIONES:
        d = medido[(a, c)]
        assert d["recalculados"] == medido[(b, c)]["recalculados"], (
            f"Se esperaba lo mismo recalculado en los dos modelos en «{c}»; no es así")
        fila = [EN_EL_LIBRO.get(c, c), miles(d["recalculados"])]
        for nombre in (a, b):
            dn = medido[(nombre, c)]
            cu = total(cuentas_turno(medidas[nombre], dn["desde"], n_total)) / base[nombre][0]
            re = dn["mediana"] / base[nombre][1]
            fila += [coma(cu), coma(re)]
        filas.append(fila)
    out += tabla_editorial(
        "Cada turno, en veces el coste de «solo añadir»",
        ["condición", "trozos", f"{ra}: cuentas", f"{ra}: reloj", f"{rb}: cuentas", f"{rb}: reloj"],
        filas, "iddddd",
        [# Orden 97: nota para el lector. La mediana del reloj y el csv de donde sale se dicen arriba
         # (esta salida) y en editar_contexto.py; la leyenda de los modelos queda en las otras tablas.
         f"Un texto de {miles(n_total)} trozos. «Trozos» son los que hay que volver a calcular en ese "
         "turno; «cuentas» y «reloj», las multiplicaciones y el tiempo del turno, en veces los de "
         "«solo añadir». «Todo desde cero» es sin nada ya calculado.",
         f"El de {rb[:-1]} millones va comprimido, con sus números redondeados para que quepa en el "
         "portátil: eso cambia el reloj, no las cuentas."]) + [""]

    filas = []
    for nombre in (a, b):
        fila = [ROTULO[nombre]]
        for c in ("solo añadir", "cambio al 0 %"):
            t, k, f = cuentas_turno(medidas[nombre], medido[(nombre, c)]["desde"], n_total)
            fila.append(pct(k / (t + k + f)))
        filas.append(fila)
    out += tabla_editorial(
        "La parte de las casillas en cada turno", ["modelo", "solo añadir", "cambio al 0 %"],
        filas, "idd", ["De cada cien multiplicaciones del turno."])
    print("\n".join(out))


# --------------------------- selftest ---------------------------

def contar_a_mano(m, desde, hasta):
    """Las mismas cuentas, multiplicación a multiplicación, con bucles. Solo para modelos de
    juguete: es la comprobación independiente de las fórmulas de `cuentas_turno`."""
    h = m["numeros"]
    por_mirada = h // m["miradas"]
    kv = m["miradas_kv"] * por_mirada
    tablas = casillas = 0
    for _capa in range(m["capas"]):
        for p in range(desde, hasta):
            for filas, columnas in ((h, h), (h, kv), (h, kv), (h, h),
                                    (h, m["mezclar"]), (h, m["mezclar"]), (m["mezclar"], h)):
                for _f in range(filas):
                    for _c in range(columnas):
                        tablas += 1
            for _mirada in range(m["miradas"]):
                for _j in range(p + 1):              # él mismo y los anteriores
                    for _d in range(por_mirada):     # pregunta por etiqueta
                        casillas += 1
                    for _d in range(por_mirada):     # reparto por contenido
                        casillas += 1
    final = 0
    for _f in range(h):
        for _c in range(m["vocabulario"]):
            final += 1
    return tablas, casillas, final


def selftest():
    ok = True

    # [1] test nulo
    # a) cambiar el ÚLTIMO trozo del texto viejo obliga a recalcular un trozo más que «solo
    #    añadir», nada más: el turno tiene que costar lo mismo, a menos de TOLERANCIA_NULO;
    # b) con un texto de un solo trozo no hay cuadrícula que valga: las casillas no llegan a
    #    UMBRAL_NULO del trabajo.
    medido, n_total = leer_medido()
    m7 = medidas_de(MEDIDAS_SELFTEST, "MEDIDAS_SELFTEST")
    d = medido[(MODELOS[0][0], "solo añadir")]["desde"]
    r_a = total(cuentas_turno(m7, d - 1, n_total)) / total(cuentas_turno(m7, d, n_total))
    r_b = parte_casillas(m7, 1)
    p1 = abs(r_a - 1) < TOLERANCIA_NULO and r_b < UMBRAL_NULO
    ok &= p1
    print(f"[1] test nulo         cambiar el último trozo viejo: {coma(r_a, 3)} veces «solo "
          f"añadir»; casillas con un trozo: {r_b:.1e} del trabajo: {'bien' if p1 else 'MAL'}")

    # [2] señal implantada
    # a) las fórmulas frente a contar multiplicación a multiplicación, en un modelo de juguete;
    # b) un modelo de juguete hecho para que el empate caiga en 1.000 trozos exactos: por cada
    #    trozo y capa, las tablas hacen 2.002 multiplicaciones y las casillas 2 x 2 = 4 por
    #    casilla; desde cero, 4 x n(n+1)/2 >= 2.002 x n  <=>  n + 1 >= 1.001  <=>  n >= 1.000.
    juguete = medidas_de({"hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 2,
                          "num_attention_heads": 2, "num_key_value_heads": 1,
                          "vocab_size": 10}, "juguete")
    formulas = cuentas_turno(juguete, 3, 12)
    a_mano = contar_a_mano(juguete, 3, 12)
    empate_toy = medidas_de({"hidden_size": 2, "intermediate_size": 331, "num_hidden_layers": 3,
                             "num_attention_heads": 1, "num_key_value_heads": 1,
                             "vocab_size": 5}, "juguete del empate")
    assert tablas_por_trozo_y_capa(empate_toy) == 2002, "el juguete del empate está mal hecho"
    empate = largo_del_empate(empate_toy)
    p2 = formulas == a_mano and empate == 1000
    ok &= p2
    print(f"[2] señal implantada  fórmulas {formulas} frente a contadas a mano {a_mano}: "
          f"{'iguales' if formulas == a_mano else 'DISTINTAS'}; empate implantado en 1.000 "
          f"trozos, encontrado en {miles(empate)}: {'bien' if p2 else 'MAL'}")

    # [3] invariante: calcular [0, n) de una vez cuesta lo mismo que [0, k) y luego [k, n),
    # quitando la tabla final, que va una vez por turno (lo guardado no ahorra ni añade
    # cuentas: solo las reparte). Y el csv cuadra: k + recalculados = total (en leer_medido).
    fallos = []
    for k in (1, 100, 1875, 3750, 7499):
        t0, c0, f0 = cuentas_turno(m7, 0, n_total)
        t1, c1, _ = cuentas_turno(m7, 0, k)
        t2, c2, _ = cuentas_turno(m7, k, n_total)
        if (t0, c0) != (t1 + t2, c1 + c2):
            fallos.append(k)
    p3 = not fallos
    ok &= p3
    print(f"[3] invariante        partir el cálculo en dos no cambia las cuentas (5 cortes) y el "
          f"csv cuadra ({miles(n_total)} trozos): {'bien' if p3 else f'MAL en {fallos}'}")

    print()
    print("SELFTEST: las tres pruebas pasan." if ok else "SELFTEST: FALLA.")
    return ok


# --------------------------- principal ---------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    medidas = {nombre: leer_medidas(repo) for nombre, repo, _ in MODELOS}
    assert medidas[MODELOS[0][0]] == medidas_de(MEDIDAS_SELFTEST, "MEDIDAS_SELFTEST"), (
        f"Se esperaba que MEDIDAS_SELFTEST fueran las del config.json de {MODELOS[0][1]}; "
        f"el config.json dice {medidas[MODELOS[0][0]]}")
    medido, n_total = leer_medido()
    imprimir(medidas, medido, n_total)


if __name__ == "__main__":
    main()
