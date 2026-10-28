#!/usr/bin/env python3
"""
Capítulo 13 — el mando de la temperatura, hecho a mano (L24, E01 y E02).

El capítulo 13 contaba la temperatura con dos adjetivos («baja, el favorito se lleva casi todo;
alta, las diferencias se aplanan») y enseñaba después cuatro relatos, que son el resultado y no el
mecanismo. Este programa enseña el mecanismo con la urna que el lector ya tiene, la del capítulo 7:

  1. LA URNA A CUATRO TEMPERATURAS. «La capital de Francia es», con la máquina del capítulo 7 (la
     de quinientos millones, en crudo): de cada cien papeletas, cuántas llevan «la», «una», «un» y
     cuántas cualquier otro trozo, a temperatura 0; 1; 1,3 y 1,8 (las cuatro de los relatos).
     A temperatura 1 tiene que salir la urna del capítulo 7, y el selftest lo comprueba.

  2. LAS VEINTE TIRADAS DE CADA TEMPERATURA. Con la máquina de los relatos (la de siete mil
     millones, adiestrada, sin comprimir), el mismo encargo, las mismas veinte semillas que usa
     `romper_la_maquina.py`, y para cada temperatura: cuántas de las veinte tiradas se quedan hasta
     el final dentro de las letras del castellano, y cuántas acaban en bucle. La tirada 1 es la
     muestra que imprime `romper_la_maquina.py`, y se comprueba que sale igual.

Uso (en el Mac: la parte 2 necesita la tarjeta gráfica):
    python temperatura_a_mano.py --selftest
    python temperatura_a_mano.py > ../datos/salidas/temperatura_a_mano.txt
"""

# ======================= CONSTANTES =======================

TEMPERATURAS = (0.0, 1.0, 1.3, 1.8)      # las cuatro de los relatos del capítulo 13
CANDIDATOS = (" la", " una", " un")      # los tres primeros de la urna del capítulo 7
PAPELETAS = 100
# Orden 97: cifra que la nota escribe con letra; si cambia el parámetro, falla aquí y no en la página.
EN_LETRA = {20: "veinte"}
URNA_CAP7 = "../datos/maquina_entera.csv"   # la urna que imprime el capítulo 7
TOL_URNA = 0.0005                        # la urna a temperatura 1 es la del capítulo 7
SALIDA_CSV = "../datos/salidas/temperatura_a_mano.csv"
SALIDA_ROMPER = "../datos/salidas/romper_la_maquina.txt"   # donde está la muestra de la tirada 1

# ==========================================================

import argparse
import csv
import datetime
import platform
import re
import sys
import unicodedata
from pathlib import Path

from formato import barra, coma, miles, tabla_editorial

# Desde el 8 oct 2026 las tres tablas del libro (la urna, las puntuaciones y las veinte tiradas)
# salen como tablas de libro (formato.py, tabla_editorial; REGLAS 6 ter). Las cuentas no cambian.

AQUI = Path(__file__).resolve().parent


# ----------------------------- la cuenta del mando -----------------------------

def urna(logits, temperatura):
    """Las probabilidades a esa temperatura. A temperatura 0 no hay sorteo: todo para el
    favorito. Por encima, cada puntuación se divide por la temperatura antes de repartir."""
    import numpy as np
    x = np.asarray(logits, dtype=np.float64)
    if temperatura == 0.0:
        p = np.zeros_like(x)
        p[int(np.argmax(x))] = 1.0
        return p
    z = x / temperatura
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def en_papeletas(fracciones, total=PAPELETAS):
    """Reparte `total` papeletas enteras en proporción a las fracciones (método del mayor
    resto): así cada fila de la figura tiene exactamente cien, y cada grupo el número entero
    más cercano a su parte."""
    brutas = [f * total for f in fracciones]
    enteras = [int(b) for b in brutas]
    faltan = total - sum(enteras)
    orden = sorted(range(len(brutas)), key=lambda i: brutas[i] - enteras[i], reverse=True)
    for i in orden[:faltan]:
        enteras[i] += 1
    assert sum(enteras) == total
    return enteras


def dentro_del_castellano(texto):
    """¿Todas las letras del texto son del alfabeto latino? Las cifras, los signos y los
    espacios no cuentan: se mira solo lo que es letra."""
    return all(unicodedata.name(c, "").startswith("LATIN") for c in texto if c.isalpha())


# ----------------------------- parte 1: la urna -----------------------------

def logits_cap7():
    import torch
    from maquina_entera import DTYPE, FRASE, MODELO
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELO)
    modelo = AutoModelForCausalLM.from_pretrained(MODELO, dtype=getattr(torch, DTYPE))
    modelo.eval()
    ids = tok(FRASE, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        logits = modelo(ids).logits[0, -1].float().numpy()
    posiciones = []
    for c in CANDIDATOS:
        i = tok(c, add_special_tokens=False)["input_ids"]
        assert len(i) == 1, f"se esperaba que «{c}» fuera un solo trozo; son {len(i)}"
        posiciones.append(i[0])
    return FRASE, MODELO, logits, posiciones


def reparto(logits, posiciones, t):
    p = urna(logits, t)
    tres = [float(p[i]) for i in posiciones]
    return tres + [1.0 - sum(tres)]


def urna_del_cap7():
    """La urna que imprimió el capítulo 7, leída de su salida."""
    lee = {}
    with open(AQUI / URNA_CAP7, encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            if fila["medicion"] == "probabilidad":
                lee[fila["palabra"]] = float(fila["probabilidad"])
    assert all(c in lee for c in CANDIDATOS), f"se esperaban {CANDIDATOS} en {URNA_CAP7}"
    return [lee[c] for c in CANDIDATOS]


def parte_urna(filas):
    frase, modelo, logits, pos = logits_cap7()
    print(f"--- 1. EL SORTEO DEL CAPÍTULO 7 A CUATRO TEMPERATURAS ---")
    print(f"modelo: {modelo} (el del capítulo 7)")
    tabla = []
    otros = []
    for t in TEMPERATURAS:
        r = reparto(logits, pos, t)
        otros.append(r[3])
        e = en_papeletas(r)
        tabla.append([coma(t, 1)] + [coma(100 * x, 1) for x in r[:3]] + [coma(100 * r[3], 1),
                                                                         barra(r[3])])
        filas.append(["urna", f"{t:.1f}"] + [f"{x:.6f}" for x in r] + [str(x) for x in e])
    assert otros == sorted(otros) and otros[-1] > 0.5, (
        "la nota dice que al subir la temperatura «otros» se lo va quedando casi todo")
    print()
    print("\n".join(tabla_editorial(
        f"Las papeletas de «{frase}» a cuatro temperaturas",
        ["temperatura", "«la»", "«una»", "«un»", "otros", ""], tabla, "cddddi",
        [# Orden 97: nota para el lector; el modelo (Qwen/Qwen2.5-0.5B, el del capítulo 7) sale arriba.
         "Papeletas de cada cien para cada trozo. «Otros» son todos los demás trozos juntos, y la "
         "barra, su parte: al subir la temperatura se lo van quedando casi todo. A temperatura 0 no "
         "hay sorteo: gana siempre el favorito."])))
    print("\nlas mismas, redondeadas a papeletas enteras (para la figura):")
    for t in TEMPERATURAS:
        e = en_papeletas(reparto(logits, pos, t))
        print(f"  {coma(t, 1):<10}" + "".join(f"{x:>6}" for x in e))
    return logits, pos


# ----------------------------- parte 2: las veinte tiradas -----------------------------

def muestra_impresa(t):
    """La muestra de esa temperatura tal como la imprimió romper_la_maquina.py."""
    texto = (AQUI / SALIDA_ROMPER).read_text(encoding="utf-8")
    cab = f"temperatura {coma(t, 1)}"
    assert cab in texto, f"se esperaba «{cab}» en {SALIDA_ROMPER}"
    trozo = texto.split(cab, 1)[1].split("\n\n", 2)[0]
    lineas = [l.strip() for l in trozo.splitlines()[1:] if l.strip()]
    return " ".join(lineas)


def parte_tiradas(filas):
    import romper_la_maquina as R
    tok, modelo = R.cargar()
    texto = R.con_formato(tok, R.ENUNCIADO_A)
    print(f"\n--- 2. LAS VEINTE TIRADAS DE CADA TEMPERATURA ---")
    print(f"modelo: {R.MODELO}   dispositivo: {R.dispositivo()}")
    print(f"encargo: «{R.ENUNCIADO_A}»")
    print(f"trozos por tirada: {R.PASOS_A}; semillas: {R.SEMILLA} a "
          f"{R.SEMILLA + R.MUESTRAS_REPETICION - 1}, una por tirada")
    resumen = []
    for t in TEMPERATURAS:
        tiradas = [R.generar(tok, modelo, texto, R.PASOS_A, t, R.SEMILLA + k)
                   for k in range(R.MUESTRAS_REPETICION)]
        igual = " ".join(tiradas[0].split()) == " ".join(muestra_impresa(t).split())
        dentro = sum(dentro_del_castellano(x) for x in tiradas)
        bucle = sum(R.hay_bucle(x) for x in tiradas)
        distintas = len(set(tiradas))
        resumen.append((t, dentro, bucle, distintas, igual))
        filas.append(["tiradas", f"{t:.1f}", str(dentro), str(bucle), str(distintas),
                      "1" if igual else "0"])
        print(f"\n  temperatura {coma(t, 1)}: la tirada 1 es la muestra del capítulo: "
              f"{'sí' if igual else 'NO'}")
        for k, x in enumerate(tiradas, 1):
            marca = "dentro" if dentro_del_castellano(x) else "FUERA "
            print(f"    {k:>2} [{marca}] {x.replace(chr(10), ' ')[:70]}")
    tabla_de_tiradas(resumen, R.MUESTRAS_REPETICION)


def tabla_de_tiradas(resumen, n):
    """La tabla resumen de las tiradas. Se imprime al final de la parte 2 y, sin volver a
    ejecutar el modelo, con --solo-tabla, leyéndola del CSV (primera versión: el rótulo decía
    «se quedan en el castellano», que es más de lo que se mide; se cambió el 28 de septiembre)."""
    filas = [[coma(t, 1), f"{dentro} de {n}", f"{bucle} de {n}", str(distintas)]
             for t, dentro, bucle, distintas, _ in resumen]
    assert resumen[0][0] == 0.0 and resumen[0][3] == 1, "la nota dice que a temperatura 0 las tiradas son la misma"
    print()
    print("\n".join(tabla_editorial(
        f"Las {n} tiradas de cada temperatura",
        ["temperatura", "solo letras latinas", "acaban en bucle", "distintas entre sí"], filas,
        "cddd",
        [# Orden 97: nota para el lector; el modelo (7.000 millones, adiestrado y sin comprimir, el del
         # capítulo 12) sale arriba. «Las veinte son la misma» es lo que comprueba el assert.
         "«Solo letras latinas»: tiradas escritas de principio a fin con nuestro alfabeto, aunque se "
         "cuele alguna palabra de otro idioma que lo use, como el inglés. «Acaban en bucle»: repiten "
         "al final la misma frase tres veces. La última columna cuenta cuántas tiradas distintas hay: "
         f"a temperatura 0, como no hay sorteo, las {EN_LETRA[n]} son la misma."])))


def tabla_desde_csv():
    import ast   # MUESTRAS_REPETICION se lee sin importar el programa, que cargaría la tarjeta
    arbol = ast.parse((AQUI / "romper_la_maquina.py").read_text(encoding="utf-8"))
    n = next(ast.literal_eval(x.value) for x in arbol.body if isinstance(x, ast.Assign)
             and getattr(x.targets[0], "id", "") == "MUESTRAS_REPETICION")
    resumen = []
    with open(AQUI / SALIDA_CSV, encoding="utf-8") as fh:
        for f in csv.DictReader(fh):
            if f["parte"] == "tiradas":
                resumen.append((float(f["temperatura"]), int(f["a"]), int(f["b"]), int(f["c"]),
                                f["d"] == "1"))
    assert [t for t, *_ in resumen] == list(TEMPERATURAS), "faltan temperaturas en el CSV"
    tabla_de_tiradas(resumen, n)


# ----------------------------- selftest -----------------------------

def parte_puntuaciones():
    """Tercera vuelta (28 de septiembre): el mecanismo, no solo el resultado. La lista del final
    sale de unas puntuaciones igual que el reparto del capítulo 8 (cada punto más multiplica la
    fuerza por 2,72); la temperatura divide todas las puntuaciones antes de pasarlas a fuerzas.
    Se enseñan las de «la», «una» y la de un trozo del medio de la lista (el de la mediana), sin
    dividir y divididas, y cuántas veces la fuerza de «la» es la de ese trozo del medio."""
    import numpy as np
    frase, modelo, logits, pos = logits_cap7()
    medio = int(np.argsort(logits)[len(logits) // 2])
    print("--- 3. CÓMO SE REHACE LA LISTA: LAS PUNTUACIONES ---")
    print(f"modelo: {modelo} (el del capítulo 7)")
    filas = []
    for t in TEMPERATURAS[1:]:
        la, una, md = (float(logits[i]) / t for i in (pos[0], pos[1], medio))
        filas.append([coma(t, 1), coma(la, 2), coma(una, 2), coma(md, 2),
                      miles(round(float(np.exp(la - md))))])
    print()
    print("\n".join(tabla_editorial(
        f"Las puntuaciones de «{frase}», divididas entre la temperatura",
        ["temperatura", "puntuación: «la»", "puntuación: «una»", "puntuación: uno del medio",
         "«la», en veces el del medio"], filas, "cdddd",
        [# Orden 97: nota para el lector; la lista de {len(logits)} trozos y el modelo salen arriba.
         "«Uno del medio» es el trozo que queda a mitad de la lista, ordenados todos por puntuación. "
         "La última columna dice cuántas veces es más fuerte «la» que ése. A temperatura 0 no se "
         "divide nada: sale siempre el favorito."])))


def selftest():
    import numpy as np
    fallos = []

    # 1. TEST NULO — si todas las puntuaciones son iguales, la temperatura no puede inventar
    #    diferencias: a cualquier temperatura por encima de 0, cada candidato lleva lo mismo.
    plano = np.zeros(5)
    ok = all(np.allclose(urna(plano, t), 0.2) for t in (0.5, 1.0, 1.3, 1.8, 5.0))
    fuera = dentro_del_castellano("En la desolada costa, 1906: el Faro.")
    print(f"[1] test nulo         urna plana a cualquier temperatura: "
          f"{'sigue plana' if ok else 'NO sigue plana'}; texto castellano detectado como "
          f"{'dentro' if fuera else 'FUERA'}")
    if not ok:
        fallos.append("test nulo: la temperatura crea diferencias donde no las hay")
    if not fuera:
        fallos.append("test nulo: el detector saca del castellano un texto castellano")

    # 2. SEÑAL IMPLANTADA — con tres puntuaciones distintas, el favorito pierde papeletas a
    #    medida que sube la temperatura, y a temperatura 0 se las lleva todas. Y el detector
    #    ve la letra tailandesa y la china de la muestra a 1,3.
    x = np.array([2.0, 1.0, 0.0])
    fav = [urna(x, t)[0] for t in TEMPERATURAS]
    baja = fav[0] == 1.0 and all(a > b for a, b in zip(fav, fav[1:]))
    ve = (not dentro_del_castellano("En la biodiversa costa de Cantรวบรวม")
          and not dentro_del_castellano("costa 句子"))
    print(f"[2] señal implantada  el favorito a 0; 1; 1,3; 1,8: "
          + "; ".join(coma(100 * f, 1) for f in fav) + f" de cada cien; "
          f"otro alfabeto: {'detectado' if ve else 'NO detectado'}")
    if not baja:
        fallos.append("señal: el favorito no pierde papeletas al subir la temperatura")
    if not ve:
        fallos.append("señal: el detector no ve las letras de otro alfabeto")

    # 3. INVARIANTE DEL DOMINIO — a temperatura 1 la urna es la del capítulo 7, y cada fila
    #    reparte exactamente cien papeletas.
    _, _, logits, pos = logits_cap7()
    r = reparto(logits, pos, 1.0)
    cap7 = urna_del_cap7()
    igual = all(abs(a - b) < TOL_URNA for a, b in zip(r[:3], cap7))
    cien = all(sum(en_papeletas(reparto(logits, pos, t))) == PAPELETAS for t in TEMPERATURAS)
    print(f"[3] invariante        a temperatura 1, «la», «una», «un»: "
          + ", ".join(coma(100 * v, 2) for v in r[:3]) + " (capítulo 7: "
          + ", ".join(coma(100 * v, 2) for v in cap7) + f"); cien papeletas por fila: "
          f"{'sí' if cien else 'NO'}")
    if not igual:
        fallos.append("invariante: la urna a temperatura 1 no es la del capítulo 7")
    if not cien:
        fallos.append("invariante: alguna fila no reparte cien papeletas")

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
    ap.add_argument("--solo-urna", action="store_true", help="solo la parte 1 (sin tarjeta gráfica)")
    ap.add_argument("--puntuaciones", action="store_true",
                    help="el mecanismo: las puntuaciones divididas entre la temperatura (sin tarjeta)")
    ap.add_argument("--solo-tabla", action="store_true",
                    help="reimprime la tabla de las tiradas desde el CSV, sin ejecutar el modelo")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if args.solo_tabla:
        tabla_desde_csv()
        return
    if args.puntuaciones:
        parte_puntuaciones()
        return
    print(f"Medido el {datetime.date.today()} en {platform.platform()}.")
    filas = []
    parte_urna(filas)
    if not args.solo_urna:
        parte_tiradas(filas)
    with open(AQUI / SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(
            [["parte", "temperatura", "a", "b", "c", "d", "e", "f", "g", "h"]] + filas)
    print(f"\nEscrito {SALIDA_CSV}")


if __name__ == "__main__":
    main()
