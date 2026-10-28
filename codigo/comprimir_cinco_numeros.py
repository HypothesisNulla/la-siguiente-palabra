#!/usr/bin/env python3
"""
Capítulo 13 y apéndice — qué quiere decir «comprimido», con los números delante (L24, E12 y E31).

El capítulo decía que la copia comprimida guarda sus números «con mucho menos detalle» y que «por
eso entra» en el portátil, sin enseñar un solo número antes y después ni un solo tamaño. Este
programa no ejecuta ningún modelo: abre los ficheros de números de las copias que usa
`comprimir.py` y enseña:

  1. CINCO NÚMEROS, ANTES Y DESPUÉS. Los cinco primeros números de una misma tabla de la máquina
     de siete mil millones, tal como salieron del entrenamiento y tal como quedan en la copia
     comprimida. Y cuántos valores distintos quedan en un grupo.
  2. LO QUE OCUPA CADA COPIA, frente a la memoria del portátil. Las tres copias que hay en el
     disco se miden por el tamaño de sus ficheros; la de treinta y dos mil millones sin comprimir no
     está descargada, y se calcula contando sus números en la copia comprimida y multiplicando por
     lo que ocupa cada uno sin comprimir (dos bytes, como en la copia de siete mil millones).

Uso (en el Mac: necesita MLX y los modelos descargados):
    python comprimir_cinco_numeros.py --selftest
    python comprimir_cinco_numeros.py > ../datos/salidas/comprimir_cinco_numeros.txt
"""

# ======================= CONSTANTES =======================

SIN_COMPRIMIR_7 = "mlx-community/Qwen2.5-7B-Instruct-bf16"
COMPRIMIDO_7 = "mlx-community/Qwen2.5-7B-Instruct-4bit"
COMPRIMIDO_32 = "mlx-community/Qwen2.5-32B-Instruct-4bit"
TABLA = "model.layers.0.self_attn.q_proj"   # una tabla cualquiera: la primera de la primera capa
CUANTOS = 5                                 # cuántos números se enseñan
BYTES_SIN_COMPRIMIR = 2                     # lo que ocupa cada número sin comprimir (bf16)
# Orden 97: cifras que las notas escriben con letra; si cambia el parámetro, falla aquí y no en la página.
EN_LETRA = {5: "cinco", 2: "dos"}
MEMORIA_PORTATIL_GB = 36                    # lo que dice la ficha: MacBook Pro M4 Max
# Pero la memoria no se cuenta en miles de millones de bytes: «36 GB» son 36 veces 1.024 por 1.024
# por 1.024 bytes, 38.654.705.664 (lo que da `sysctl -n hw.memsize` en este Mac). Compararla con el
# tamaño de las copias, que sí van en GB = 1e9, sin decirlo, era comparar 65,3 de una medida con 36
# de la otra (17 oct 2026).
MEMORIA_PORTATIL_BYTES = MEMORIA_PORTATIL_GB * 1024 ** 3
CHIP_DEL_LIBRO = "Apple M4 Max"             # el portátil del que habla el libro
GB = 1e9                                    # gigabytes como cuenta el fabricante el DISCO (no la memoria)

# ==========================================================

import argparse
import glob
import json
import os
import sys

from formato import coma, miles, tabla_editorial

# Desde el 8 oct 2026 las dos partes salen como tablas de libro (formato.py; REGLAS 6 ter).


def memoria_del_mac():
    """(chip, bytes de memoria) de este ordenador, si es un Mac; (None, None) si no."""
    import platform
    import subprocess
    if platform.system() != "Darwin":
        return None, None
    uno = lambda k: subprocess.run(["sysctl", "-n", k], capture_output=True, text=True,
                                   check=True).stdout.strip()
    return uno("machdep.cpu.brand_string"), int(uno("hw.memsize"))


def comprobar_memoria(medida=None):
    """En el Mac del libro (el chip de CHIP_DEL_LIBRO), la constante tiene que ser la memoria que
    dice el sistema; en cualquier otro ordenador no se compara con nada. `medida`, (chip, bytes),
    es para probarlo: si no se da, se pregunta al sistema."""
    chip, memoria = medida or memoria_del_mac()
    if chip == CHIP_DEL_LIBRO:
        assert memoria == MEMORIA_PORTATIL_BYTES, (
            f"Se esperaba la memoria del portátil del libro, {miles(MEMORIA_PORTATIL_BYTES)} bytes; "
            f"este {chip} dice {miles(memoria)}")
        return True
    return False


def linea_memoria():
    """La línea de la salida sobre la memoria: en GB de 1e9, como las copias, y con lo que son los
    «36 GB» de la ficha."""
    return (f"Memoria del portátil: {coma(MEMORIA_PORTATIL_BYTES / GB, 1)} GB, para todo: los "
            f"«{MEMORIA_PORTATIL_GB} GB» de su ficha son {MEMORIA_PORTATIL_GB} veces 1.024 por "
            "1.024 por 1.024 bytes.")


def carpeta(repo):
    """La carpeta del modelo en la caché de descargas (sin conectarse a nada)."""
    from huggingface_hub.constants import HF_HUB_CACHE
    donde = glob.glob(os.path.join(HF_HUB_CACHE, "models--" + repo.replace("/", "--"),
                                   "snapshots", "*"))
    assert len(donde) == 1, f"se esperaba una sola copia descargada de {repo}; hay {len(donde)}"
    return donde[0]


def ficheros(repo):
    fs = sorted(glob.glob(os.path.join(carpeta(repo), "*.safetensors")))
    assert fs, f"se esperaban ficheros de números en {repo}"
    return fs


def tamano(repo):
    return sum(os.path.getsize(os.path.realpath(f)) for f in ficheros(repo))


def config(repo):
    with open(os.path.join(carpeta(repo), "config.json"), encoding="utf-8") as fh:
        return json.load(fh)


def tablas(repo, nombres):
    """Las tablas pedidas, buscadas en los ficheros del modelo."""
    import mlx.core as mx
    out = {}
    for f in ficheros(repo):
        d = mx.load(f)
        for n in nombres:
            if n in d:
                out[n] = d[n]
    faltan = [n for n in nombres if n not in out]
    assert not faltan, f"no encuentro {faltan} en {repo}"
    return out


def originales(n=CUANTOS):
    import mlx.core as mx
    w = tablas(SIN_COMPRIMIR_7, [TABLA + ".weight"])[TABLA + ".weight"]
    return w, [float(x) for x in w[0, :n].astype(mx.float32).tolist()]


def comprimidos(repo=COMPRIMIDO_7, n=CUANTOS):
    import mlx.core as mx
    q = config(repo)["quantization"]
    t = tablas(repo, [TABLA + ".weight", TABLA + ".scales", TABLA + ".biases"])
    w = mx.dequantize(t[TABLA + ".weight"], t[TABLA + ".scales"], t[TABLA + ".biases"],
                      group_size=q["group_size"], bits=q["bits"])
    return w, [float(x) for x in w[0, :n].astype(mx.float32).tolist()], q


def cuantos_numeros(repo):
    """Cuántos números tiene el modelo. En una copia comprimida, cada casilla de 32 bits de una
    tabla comprimida lleva 32 / bits números; las escalas y los desplazamientos de cada grupo
    son la manera de guardarlos, no números del modelo, y no se cuentan."""
    import mlx.core as mx
    q = config(repo).get("quantization")
    total = 0
    for f in ficheros(repo):
        d = mx.load(f)
        for n, a in d.items():
            if n.endswith(".scales") or n.endswith(".biases"):
                continue
            base = n[: -len(".weight")] if n.endswith(".weight") else None
            if q and base and (base + ".scales") in d:
                total += a.size * (32 // q["bits"])
            else:
                total += a.size
    return total


def imprime(*bloques):
    for b in bloques:
        print("\n".join(b))
        print()


def informe():
    import mlx.core as mx
    w0, antes = originales()
    w1, despues, q = comprimidos()
    print("--- 1. CINCO NÚMEROS, ANTES Y DESPUÉS DE COMPRIMIR ---")
    print(f"máquina: {SIN_COMPRIMIR_7} y {COMPRIMIDO_7}")
    print(f"tabla: {TABLA}, fila 1, números 1 a {CUANTOS}\n")
    grupo0 = w0[0, :q["group_size"]].astype(mx.float32).tolist()
    grupo1 = w1[0, :q["group_size"]].astype(mx.float32).tolist()
    imprime(
        tabla_editorial(
            "Cinco números, antes y después de comprimir",
            ["número", "sin comprimir", "comprimido", "diferencia"],
            [[str(k), coma(a, 6), coma(b, 6), coma(b - a, 6)]
             for k, (a, b) in enumerate(zip(antes, despues), 1)], "cddd",
            [# Orden 97: nota para el lector. Que sea la primera fila de una tabla de la primera capa
             # (TABLA) y que la diferencia sea el comprimido menos el sin comprimir, está en el código.
             f"Los {EN_LETRA[CUANTOS]} primeros números de una de las tablas de la máquina de 7.000 "
             "millones adiestrada, tal como salieron del entrenamiento y en la copia comprimida. La "
             "diferencia es lo que se ha movido cada uno al comprimir."]),
        tabla_editorial(
            f"Valores distintos en el primer grupo de {q['group_size']} números de esa fila",
            ["sin comprimir", "comprimido"], [[str(len(set(grupo0))), str(len(set(grupo1)))]],
            "dd",
            [f"Comprimido, cada número toma uno de {2 ** q['bits']} valores posibles, los mismos "
             f"para los {q['group_size']} de su grupo; lo perdido no vuelve."]))

    print("--- 2. LO QUE OCUPA CADA COPIA ---\n")
    n32 = cuantos_numeros(COMPRIMIDO_32)
    n7 = cuantos_numeros(COMPRIMIDO_7)
    filas = [("7.000M sin comprimir", tamano(SIN_COMPRIMIR_7), "medido"),
             ("7.000M comprimido", tamano(COMPRIMIDO_7), "medido"),
             ("32.000M sin comprimir", n32 * BYTES_SIN_COMPRIMIR, "calculado"),
             ("32.000M comprimido", tamano(COMPRIMIDO_32), "medido")]
    imprime(tabla_editorial(
        "Lo que ocupa cada copia",
        ["copia", "ocupa", "cómo"],
        [[nombre, coma(b / GB, 1) + " GB", como] for nombre, b, como in filas], "idi",
        [# Orden 97: nota para el lector. Los números de cada copia (32.640.013.312 y 7.615.616.512) y
         # lo que son los «36 GB» de la ficha están en `cuantos_numeros` y en `linea_memoria`.
         "«Medido»: lo que ocupan sus ficheros en el disco. «Calculado»: "
         f"{EN_LETRA[BYTES_SIN_COMPRIMIR]} bytes por cada número, como la de 7.000M sin comprimir. "
         f"El portátil tiene {coma(MEMORIA_PORTATIL_BYTES / GB, 1)} GB de memoria para todo, y la "
         "grande sin comprimir no cabe."]))
    assert n32 * BYTES_SIN_COMPRIMIR > MEMORIA_PORTATIL_BYTES, (
        "la nota dice que la grande sin comprimir no cabe en la memoria del portátil")


def selftest():
    import mlx.core as mx
    fallos = []
    w0, antes = originales()
    w1, despues, q = comprimidos()
    g = q["group_size"]
    a = w0.astype(mx.float32)
    b = w1.astype(mx.float32)

    # 1. TEST NULO — que la copia comprimida se parezca a la original no lo pone el montaje:
    #    comparada con OTRA fila de la original, la diferencia media es mucho mayor que con la suya.
    buena = float(mx.mean(mx.abs(b[0] - a[0])).item())
    otra = float(mx.mean(mx.abs(b[0] - a[1])).item())
    print(f"[1] test nulo         diferencia media con su fila {buena:.2e}; con otra fila {otra:.2e}")
    if not otra > 5 * buena:
        fallos.append("test nulo: la copia se parece igual a cualquier fila; la comparación no dice nada")

    # 2. SEÑAL IMPLANTADA — la copia comprimida sí pierde detalle: sus números no son los
    #    originales, y en un grupo quedan como mucho 2^bits valores distintos.
    distintos = len(set(b[0, :g].tolist()))
    pierde = any(x != y for x, y in zip(antes, despues))
    print(f"[2] señal implantada  los números cambian: {'sí' if pierde else 'NO'}; "
          f"valores distintos en un grupo: {distintos} (como mucho {2 ** q['bits']})")
    if not pierde or distintos > 2 ** q["bits"]:
        fallos.append("señal: la copia comprimida no pierde el detalle que se dice")

    # 3. INVARIANTE DEL DOMINIO — las dos copias de siete mil millones son el mismo modelo: en toda
    #    la tabla, ningún número comprimido se aleja del original más de un escalón de su grupo
    #    (la distancia entre dos de sus valores posibles), y las dos tienen los mismos números.
    esc = tablas(COMPRIMIDO_7, [TABLA + ".scales"])[TABLA + ".scales"].astype(mx.float32)
    paso = mx.repeat(mx.abs(esc), g, axis=1)
    peor = float(mx.max(mx.abs(b - a) / paso).item())
    n_a, n_b = cuantos_numeros(SIN_COMPRIMIR_7), cuantos_numeros(COMPRIMIDO_7)
    print(f"[3] invariante        el mayor alejamiento, en escalones de su grupo: {peor:.2f}; "
          f"números en total {miles(n_a)} y {miles(n_b)}")
    if peor > 1.0 or n_a != n_b:
        fallos.append("invariante: la copia comprimida no es la original comprimida")
    comparada = comprobar_memoria()
    ajeno = comprobar_memoria(("Intel Core", 17_179_869_184))        # otro ordenador: no se compara
    try:
        comprobar_memoria((CHIP_DEL_LIBRO, 36 * 10 ** 9))             # el Mac del libro, con 36e9: salta
        salta = False
    except AssertionError:
        salta = True
    print(f"                      memoria del portátil {miles(MEMORIA_PORTATIL_BYTES)} bytes: "
          f"{'igual que hw.memsize' if comparada else 'no se compara (no es el Mac del libro)'}; "
          f"en otro ordenador no se compara: {'sí' if not ajeno else 'NO'}; con 36 × 10^9 en el Mac "
          f"del libro salta: {'sí' if salta else 'NO'}")
    if ajeno or not salta:
        fallos.append("invariante: la comprobación de la memoria no distingue el Mac del libro")
    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        sys.exit(selftest())
    import datetime
    import platform
    print(f"Medido el {datetime.date.today()} en {platform.platform()}.")
    informe()


if __name__ == "__main__":
    main()
