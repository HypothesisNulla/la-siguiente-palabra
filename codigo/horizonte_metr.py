#!/usr/bin/env python3
"""
Orden 210 (23 nov 2026): el horizonte temporal de METR — cuánto duran las tareas que estas máquinas terminan la mitad de las veces.

Lee, sin reescribir ningún número a mano, el fichero de resultados que METR publica de su medida «Time Horizon 1.1»
(`benchmark_results_1_1.yaml`, guardado en `datos/entrada/` con la fecha de la descarga, 23 nov 2026). De cada modelo saca la
fecha de salida y el horizonte del 50 %: la duración, medida en el tiempo que tardaría una persona experta, de las tareas
que el modelo termina la mitad de las veces (con su intervalo, y el horizonte del 80 %). Esos horizontes los calcula METR
con sus tareas; este programa no los calcula: los lee y mide lo que METR llama el tiempo de duplicación.

El tiempo de duplicación es el de un ajuste recto en escala logarítmica: log2(horizonte) = a + b × días, y duplicación = 1 / b.
METR no dice en su fichero con qué modelos hace ese ajuste; aquí se fijó, antes de ver ningún resultado propio, el que
describe su entrada «Time Horizon 1.1» (la frontera: los modelos que marcan el récord en su fecha, `is_sota`, y se
dejan fuera los de más de 16 horas, donde la propia METR dice que la medida no es fiable). Resultado: reproduce los dos
tiempos de duplicación que METR escribe en ese mismo fichero (187,778 días en todo el periodo; 128,744 desde 2023), y la
prueba 3 del selftest lo exige (a menos de medio día). Eso es lo que se compara; no se afinó nada para que saliera.

La entrada del blog de METR del 29 ene 2026 da 196 y 131 días. Salen de los modelos que había entonces (el fichero se
actualizó después, con los modelos hasta abril de 2026). Aquí se calcula también el ajuste con los modelos anteriores a esa
fecha, para ver que no es otra cosa.

Mediciones independientes de la máquina (numpy, sin GPU, sin tiempos): con los mismos datos dan lo mismo en cualquier ordenador.

Uso:
    python horizonte_metr.py --selftest
    python horizonte_metr.py > ../datos/salidas/horizonte_metr.txt
"""

# ======================= CONSTANTES =======================

ENTRADA = "../datos/entrada/metr-benchmark-results-1-1-2026-11-23.yaml"    # los resultados publicados por METR (TH1.1)
SALIDA_CSV = "../datos/salidas/horizonte_metr_datos.csv"
ORIGEN_DIAS = "2019-01-01"          # los días se cuentan desde aquí (solo cambia la ordenada, no la pendiente)
DESDE_2023 = "2023-01-01"           # el segundo ajuste: «desde 2023», como METR
LIMITE_MINUTOS = 16 * 60            # METR deja fuera del ajuste los puntos de más de 16 horas
FECHA_ENTRADA_DEL_BLOG = "2026-01-29"   # el blog «Time Horizon 1.1» de METR: los modelos que había hasta ese día
BLOG_TODO, BLOG_DESDE_2023 = 196.0, 131.0   # lo que dice esa entrada (196 días y 131 días)
TOPE_DIFERENCIA = 0.5               # días: cuánto se puede separar mi ajuste del que METR escribe en el fichero
REMUESTREOS, SEMILLA = 2000, 20261123   # el intervalo por remuestreo de modelos (solo informativo)
BARAJADAS = 1000                    # el test nulo: cuántas veces se barajan las fechas

# Los nombres con que se imprimen los modelos (a mano: son nombres, no números). Si METR añade un modelo, falla el selftest.
NOMBRES = {
    "gpt2": "GPT-2", "davinci_002": "GPT-3 (davinci)", "gpt_3_5_turbo_instruct": "GPT-3.5 Instruct",
    "gpt_4": "GPT-4 (mar 2023)", "gpt_4_1106_inspect": "GPT-4 (nov 2023)", "claude_3_opus_inspect": "Claude 3 Opus",
    "gpt_4_turbo_inspect": "GPT-4 Turbo", "gpt_4o_inspect": "GPT-4o",
    "claude_3_5_sonnet_20240620_inspect": "Claude 3.5 Sonnet (jun 2024)", "o1_preview": "o1-preview",
    "claude_3_5_sonnet_20241022_inspect": "Claude 3.5 Sonnet (oct 2024)", "o1_inspect": "o1",
    "claude_3_7_sonnet_inspect": "Claude 3.7 Sonnet", "o3_inspect": "o3", "claude_4_opus_inspect": "Claude 4 Opus",
    "claude_4_1_opus_inspect": "Claude 4.1 Opus", "gpt_5_2025_08_07_inspect": "GPT-5", "gemini_3_pro": "Gemini 3 Pro",
    "gpt_5_1_codex_max_inspect": "GPT-5.1-Codex-Max", "claude_opus_4_5_inspect": "Claude Opus 4.5", "gpt_5_2": "GPT-5.2",
    "claude_opus_4_6_inspect": "Claude Opus 4.6", "gpt_5_3_codex": "GPT-5.3-Codex", "gemini_3_1_pro": "Gemini 3.1 Pro",
    "gpt_5_4": "GPT-5.4", "claude_mythos_preview_early_inspect": "Claude Mythos Preview (inicial)",
}
MESES = ("ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic")

# ==========================================================

import argparse
import csv
import datetime as dt
import hashlib
import platform
import re
import sys
from pathlib import Path

import numpy as np

from formato import coma, comprobar_ancho, miles, tabla_editorial

AQUI = Path(__file__).resolve().parent


# ---------------------------------------------------------------- leer

def leer(ruta):
    """Lee el YAML de METR (sin PyYAML, que no está entre los requisitos): por sangría. Devuelve
    (tiempos_publicados, modelos): los dos tiempos de duplicación que METR escribe, y una lista de diccionarios
    (clave, fecha, p50, p50_bajo, p50_alto, p80, sota) en orden de fecha y, a igual fecha, de aparición en el fichero."""
    lineas = Path(ruta).read_text(encoding="utf-8").splitlines()
    publicados, modelos, actual, ruta_ = {}, [], None, []
    for l in lineas:
        if not l.strip() or l.lstrip().startswith("#"):
            continue
        sangria = len(l) - len(l.lstrip())
        nombre, _, valor = l.strip().partition(":")
        valor = valor.split("#")[0].strip()
        ruta_ = ruta_[:sangria // 2] + [nombre]
        if ruta_[:2] == ["doubling_time_in_days", "all_time_stitched"] and nombre == "point_estimate":
            publicados["todo"] = float(valor)
        elif ruta_[:2] == ["doubling_time_in_days", "from_2023_on"] and nombre == "point_estimate":
            publicados["desde_2023"] = float(valor)
        elif ruta_[:1] == ["results"] and len(ruta_) == 2:
            actual = {"clave": nombre}
            modelos.append(actual)
        elif actual is not None and ruta_[:1] == ["results"]:
            sub = ruta_[2:]
            if sub == ["release_date"]:
                actual["fecha"] = dt.date.fromisoformat(valor)
            elif sub == ["metrics", "is_sota"]:
                actual["sota"] = valor == "true"
            elif sub[:2] == ["metrics", "p50_horizon_length"] and len(sub) == 3:
                actual[{"estimate": "p50", "ci_low": "p50_bajo", "ci_high": "p50_alto"}[sub[2]]] = float(valor)
            elif sub[:2] == ["metrics", "p80_horizon_length"] and len(sub) == 3 and sub[2] == "estimate":
                actual["p80"] = float(valor)
    assert publicados.keys() == {"todo", "desde_2023"}, "no encuentro los dos tiempos de duplicación que escribe METR"
    for m in modelos:
        faltan = {"fecha", "sota", "p50", "p50_bajo", "p50_alto", "p80"} - m.keys()
        assert not faltan, f"al modelo {m['clave']} le falta {faltan}"
    modelos = sorted(modelos, key=lambda m: m["fecha"])       # estable: a igual fecha, el orden del fichero
    return publicados, modelos


def md5_de(ruta):
    return hashlib.md5(Path(ruta).read_bytes()).hexdigest()


# ---------------------------------------------------------------- las cuentas

def dias(fecha):
    return (fecha - dt.date.fromisoformat(ORIGEN_DIAS)).days


def ajustar(t, minutos):
    """Ajuste recto de log2(minutos) contra los días. Devuelve (duplicacion_en_dias, pendiente, ordenada)."""
    b, a = np.polyfit(np.asarray(t, float), np.log2(np.asarray(minutos, float)), 1)
    return 1.0 / b, b, a


def seleccion(modelos, solo_frontera=True, limite=LIMITE_MINUTOS, desde=None, hasta=None):
    sel = []
    for m in modelos:
        if solo_frontera and not m["sota"]:
            continue
        if limite is not None and m["p50"] > limite:
            continue
        if desde is not None and m["fecha"] < dt.date.fromisoformat(desde):
            continue
        if hasta is not None and m["fecha"] > dt.date.fromisoformat(hasta):
            continue
        sel.append(m)
    return sel


def duplicacion(sel):
    return ajustar([dias(m["fecha"]) for m in sel], [m["p50"] for m in sel])[0]


def intervalo(sel, semilla=SEMILLA, n=REMUESTREOS):
    """Intervalo del 95 % del tiempo de duplicación por remuestreo de modelos (con reemplazo), con semilla fija."""
    rng = np.random.default_rng(semilla)
    t = np.array([dias(m["fecha"]) for m in sel], float)
    y = np.log2([m["p50"] for m in sel])
    valores = []
    for _ in range(n):
        i = rng.integers(0, len(sel), len(sel))
        if len(set(t[i])) < 2:
            continue
        valores.append(1.0 / np.polyfit(t[i], y[i], 1)[0])
    return float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))


def duracion(minutos):
    """Una duración en palabras: «6 s», «20 min», «3 h 44 min», «17 h 25 min»."""
    if minutos < 1:
        return f"{round(minutos * 60)} s"
    if minutos < 60:
        return f"{round(minutos)} min"
    h, m = divmod(round(minutos), 60)
    return f"{h} h {m:02d} min" if m else f"{h} h"


def coma_miles(x, decimales=1):
    """1044,78 -> «1.044,8»: coma decimal y punto de miles."""
    entero, _, dec = f"{x:.{decimales}f}".partition(".")
    return miles(int(entero)) + ("," + dec if dec else "")


def fecha_corta(f):
    return f"{f.day} {MESES[f.month - 1]} {f.year}"


# ---------------------------------------------------------------- la salida

def resultados(modelos, publicados):
    """Todo lo que se imprime y se compara, calculado una vez."""
    r = {}
    r["frontera_todo"] = seleccion(modelos)
    r["frontera_2023"] = seleccion(modelos, desde=DESDE_2023)
    r["todos_todo"] = seleccion(modelos, solo_frontera=False, limite=None)
    r["todos_2023"] = seleccion(modelos, solo_frontera=False, limite=None, desde=DESDE_2023)
    r["blog_todo"] = seleccion(modelos, hasta=FECHA_ENTRADA_DEL_BLOG)
    r["blog_2023"] = seleccion(modelos, desde=DESDE_2023, hasta=FECHA_ENTRADA_DEL_BLOG)
    r["dup"] = {k: duplicacion(v) for k, v in r.items() if k != "dup"}
    r["ic_todo"] = intervalo(r["frontera_todo"])
    r["ic_2023"] = intervalo(r["frontera_2023"])
    r["publicados"] = publicados
    return r


def imprimir(modelos, publicados):
    r = resultados(modelos, publicados)
    dup = r["dup"]
    print("######## el horizonte temporal de METR (Orden 210) ########")
    print(f"fecha: {dt.date.today().isoformat()}")
    print(f"máquina: {platform.system()} {platform.machine()}")
    print(f"python {platform.python_version()}, numpy {np.__version__}")
    print("GPU: no se usa (todo en numpy). Nada de lo que sigue")
    print("depende del ordenador: son cuentas sobre un fichero.")
    print()
    print(f"Fichero de METR (TH1.1): {Path(ENTRADA).name}")
    print(f"md5 {md5_de(AQUI / ENTRADA)}; {len(modelos)} modelos, del {fecha_corta(modelos[0]['fecha'])} al {fecha_corta(modelos[-1]['fecha'])}.")
    print("El horizonte del 50 % es la duración, medida en lo que tardaría")
    print("una persona experta, de las tareas que el modelo termina la mitad")
    print("de las veces. Los calcula METR; aquí solo se leen.")
    print()
    filas = [(NOMBRES[m["clave"]], fecha_corta(m["fecha"]), duracion(m["p50"])) for m in modelos]
    lineas = tabla_editorial("Los modelos de METR: salida y horizonte del 50 %",
                             ("modelo", "salida", "horizonte"), filas, "iid")
    for l in comprobar_ancho(lineas):
        print(l)
    print()
    filas = [(NOMBRES[m["clave"]], miles(round(m["p50_bajo"])) if m["p50_bajo"] >= 10 else coma(m["p50_bajo"], 2),
              miles(round(m["p50_alto"])) if m["p50_alto"] >= 10 else coma(m["p50_alto"], 2), miles(round(m["p80"])) if m["p80"] >= 10 else coma(m["p80"], 2))
             for m in modelos]
    lineas = tabla_editorial("Los intervalos de METR, en minutos",
                             ("modelo", "50 %: de", "a", "80 %"), filas, "iddd",
                             notas=("El 80 %: las tareas que termina cuatro de cada cinco veces.",))
    for l in comprobar_ancho(lineas):
        print(l)
    print()
    print("--- EL TIEMPO DE DUPLICACIÓN ---")
    print()
    print("Ajuste recto de log2(horizonte) contra los días, con los modelos de la frontera")
    print(f"(los que marcan el récord en su fecha) y sin los de más de {LIMITE_MINUTOS // 60} horas.")
    print()
    pub_t, pub_2 = publicados["todo"], publicados["desde_2023"]
    ft, f2 = r["frontera_todo"], r["frontera_2023"]
    lineas = tabla_editorial("Tiempo de duplicación, en días",
                             ("periodo", "modelos", "aquí", "METR (fichero)"),
                             [("todo", str(len(ft)), coma(dup["frontera_todo"], 1), coma(pub_t, 1)),
                              ("desde 2023", str(len(f2)), coma(dup["frontera_2023"], 1), coma(pub_2, 1))], "idd" + "d")
    for l in comprobar_ancho(lineas):
        print(l)
    print()
    print(f"Diferencia con el fichero de METR: {coma(abs(dup['frontera_todo'] - pub_t), 3)} y "
          f"{coma(abs(dup['frontera_2023'] - pub_2), 3)} días (tope {coma(TOPE_DIFERENCIA, 1)}).")
    print(f"Intervalo del 95 % por remuestreo de modelos (semilla {SEMILLA}, {miles(REMUESTREOS)} remuestreos; solo informativo):")
    print(f"  todo: de {coma(r['ic_todo'][0], 0)} a {coma(r['ic_todo'][1], 0)} días; "
          f"desde 2023: de {coma(r['ic_2023'][0], 0)} a {coma(r['ic_2023'][1], 0)} días.")
    print()
    bt, b2 = r["blog_todo"], r["blog_2023"]
    lineas = tabla_editorial(f"Con los modelos hasta el {fecha_corta(dt.date.fromisoformat(FECHA_ENTRADA_DEL_BLOG))}",
                             ("periodo", "modelos", "aquí", "METR (blog)"),
                             [("todo", str(len(bt)), coma(dup["blog_todo"], 1), coma(BLOG_TODO, 0)),
                              ("desde 2023", str(len(b2)), coma(dup["blog_2023"], 1), coma(BLOG_DESDE_2023, 0))], "idd" + "d")
    for l in comprobar_ancho(lineas):
        print(l)
    print()
    print("Con todos los modelos, sin quedarse con la frontera ni quitar los de más de 16 horas:")
    print(f"  todo: {coma(dup['todos_todo'], 1)} días ({len(r['todos_todo'])} modelos); "
          f"desde 2023: {coma(dup['todos_2023'], 1)} días ({len(r['todos_2023'])} modelos).")
    print()
    print("--- LOS ÚLTIMOS ---")
    print()
    ultimo = modelos[-1]
    dentro = ft[-1]
    print(f"El último modelo del fichero: {NOMBRES[ultimo['clave']]}, {fecha_corta(ultimo['fecha'])}: horizonte del 50 % de")
    print(f"{coma_miles(ultimo['p50'], 1)} minutos ({duracion(ultimo['p50'])}; intervalo de {duracion(ultimo['p50_bajo'])} a "
          f"{duracion(ultimo['p50_alto'])}) y del 80 % de {duracion(ultimo['p80'])}.")
    if ultimo["p50"] > LIMITE_MINUTOS:
        print(f"Pasa de las {LIMITE_MINUTOS // 60} horas: METR lo deja fuera del ajuste.")
    print(f"El último dentro del ajuste: {NOMBRES[dentro['clave']]}, {fecha_corta(dentro['fecha'])}, {duracion(dentro['p50'])}.")
    primero = modelos[0]
    print(f"El primero: {NOMBRES[primero['clave']]}, {fecha_corta(primero['fecha'])}, {duracion(primero['p50'])}.")
    print(f"De uno a otro, el horizonte se multiplica por {miles(round(ultimo['p50'] / primero['p50']))}.")
    return r


def guardar_csv(modelos):
    with open(AQUI / SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["modelo", "nombre", "fecha", "p50_min", "p50_bajo_min", "p50_alto_min", "p80_min", "frontera", "en_el_ajuste"])
        for m in modelos:
            w.writerow([m["clave"], NOMBRES[m["clave"]], m["fecha"].isoformat(), repr(m["p50"]), repr(m["p50_bajo"]),
                        repr(m["p50_alto"]), repr(m["p80"]), int(m["sota"]), int(m["sota"] and m["p50"] <= LIMITE_MINUTOS)])


# ---------------------------------------------------------------- selftest

def lectura_independiente(ruta):
    """Otra forma de leer los p50: por expresiones regulares sobre el texto, sin sangrías. {clave: p50}."""
    texto = Path(ruta).read_text(encoding="utf-8")
    sal = {}
    for m in re.finditer(r"^  (\w[\w.\-]*):\n(.*?)(?=^  \w[\w.\-]*:\n|\Z)", texto.split("\nresults:\n", 1)[1], re.S | re.M):
        p = re.search(r"p50_horizon_length:\n(?:\s+ci_high: [\d.]+\n)?(?:\s+ci_low: [\d.]+\n)?\s+estimate: ([\d.]+)", m.group(2))
        assert p, f"no encuentro el p50 de {m.group(1)}"
        sal[m.group(1)] = float(p.group(1))
    return sal


def selftest():
    print("=== SELFTEST de horizonte_metr.py ===")
    ok = True
    publicados, modelos = leer(AQUI / ENTRADA)

    # [1] NULO — con las fechas barajadas entre los modelos, no hay tendencia: la pendiente media es casi cero y ninguna
    #     de las barajadas da una duplicación tan rápida como la de verdad.
    sel = seleccion(modelos, solo_frontera=False, limite=None)
    t = np.array([dias(m["fecha"]) for m in sel], float)
    y = np.log2([m["p50"] for m in sel])
    real = np.polyfit(t, y, 1)[0]
    rng = np.random.default_rng(SEMILLA)
    barajadas = np.array([np.polyfit(rng.permutation(t), y, 1)[0] for _ in range(BARAJADAS)])
    n1 = bool(abs(barajadas.mean()) < 0.1 * real and np.all(barajadas < real))
    print(f"[1] test nulo         con las fechas barajadas ({miles(BARAJADAS)} veces): pendiente media {barajadas.mean() * 365:+.2f} duplicaciones al año, "
          f"la de verdad {real * 365:+.2f}; barajadas tan rápidas como la de verdad: {int(np.sum(barajadas >= real))}: {'OK' if n1 else 'MAL'}")
    ok &= n1

    # [2] SEÑAL IMPLANTADA — unos datos inventados que se duplican cada 100 días dan 100 (también con ruido: tolerancia)
    t0 = dt.date(2020, 1, 1)
    inventados = [{"fecha": t0 + dt.timedelta(days=d), "p50": 2.0 ** (d / 100.0) * 3.0, "sota": True} for d in range(0, 1000, 37)]
    d_inv = duplicacion(inventados)
    rng = np.random.default_rng(1)
    con_ruido = [dict(m, p50=m["p50"] * float(np.exp(rng.normal(0, 0.1)))) for m in inventados]
    d_ruido = duplicacion(con_ruido)
    n2 = bool(abs(d_inv - 100.0) < 1e-6 and abs(d_ruido - 100.0) < 5.0)
    print(f"[2] señal implantada  datos inventados que se duplican cada 100 días: {coma(d_inv, 6)} días; con ruido del 10 %: {coma(d_ruido, 1)} días: {'OK' if n2 else 'MAL'}")
    ok &= n2

    # [3] INVARIANTE — (a) los horizontes leídos coinciden con los del fichero, leídos por otra vía, y cada modelo tiene
    #     nombre; (b) mi ajuste reproduce los dos tiempos de duplicación que METR escribe en ese fichero.
    otro = lectura_independiente(AQUI / ENTRADA)
    iguales = (set(otro) == {m["clave"] for m in modelos} and all(abs(otro[m["clave"]] - m["p50"]) < 1e-12 for m in modelos))
    con_nombre = all(m["clave"] in NOMBRES for m in modelos)
    d_todo, d_2023 = duplicacion(seleccion(modelos)), duplicacion(seleccion(modelos, desde=DESDE_2023))
    dif = max(abs(d_todo - publicados["todo"]), abs(d_2023 - publicados["desde_2023"]))
    n3 = bool(iguales and con_nombre and dif < TOPE_DIFERENCIA and len(modelos) == len(otro))
    print(f"[3] invariante        {len(modelos)} horizontes leídos, iguales a los del fichero por otra vía: {'sí' if iguales else 'NO'}; "
          f"todos con nombre: {'sí' if con_nombre else 'NO'}; mis duplicaciones {coma(d_todo, 3)} y {coma(d_2023, 3)} días frente a las de METR "
          f"{coma(publicados['todo'], 3)} y {coma(publicados['desde_2023'], 3)} (diferencia máxima {coma(dif, 3)}, tope {coma(TOPE_DIFERENCIA, 1)}): "
          f"{'OK' if n3 else 'MAL'}")
    ok &= n3
    print("SELFTEST OK (3 pruebas)" if ok else "SELFTEST FALLA")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    publicados, modelos = leer(AQUI / ENTRADA)
    assert all(m["clave"] in NOMBRES for m in modelos), "hay un modelo sin nombre en NOMBRES"
    r = imprimir(modelos, publicados)
    dif = max(abs(r["dup"]["frontera_todo"] - publicados["todo"]), abs(r["dup"]["frontera_2023"] - publicados["desde_2023"]))
    assert dif < TOPE_DIFERENCIA, f"mi duplicación se separa {dif:.3f} días de la que escribe METR: paro (Orden 210)"
    guardar_csv(modelos)
    print(f"escrito {SALIDA_CSV}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
