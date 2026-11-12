#!/usr/bin/env python3
"""
Capítulo 10 — ¿qué pasa si se deja entrenando más? (L24, hallazgo D08)

El capítulo termina con una pregunta: «¿qué pasaría si la dejáramos entrenando mucho más
tiempo?», y la contesta con palabras: mejoraría cada vez menos y se quedaría lejos, porque
con 1.820.971 números y ocho millones de letras hay un techo. Este programa lo MIDE en vez de
afirmarlo: entrena exactamente el mismo modelo del capítulo (mismo guion, misma semilla, mismos
lotes) durante tres veces más pasos, y cada mil pasos mide lo mal que lo hace en dos textos:

  - texto visto: trozos de los mismos ocho millones de letras con los que entrena;
  - texto nuevo: trozos de los libros que vienen DESPUÉS de esos ocho millones, que la máquina
    no lee nunca.

La primera medida es la del capítulo (la curva de la figura). La segunda es la que dice si la
máquina sigue aprendiendo castellano o solo se está aprendiendo de memoria sus libros.

Se acota por PASOS, nunca por reloj (regla 10): con la semilla fija, el paso 50.000 de aquí es
el paso 50.000 del capítulo, y el programa lo comprueba contra la curva guardada.

Uso:
    python seguir_entrenando.py --selftest
    python seguir_entrenando.py
"""

# ======================= CONSTANTES =======================

PASOS_TOTAL = 150_000          # tres veces los 50.000 del capítulo
CADA = 1_000                   # cada cuántos pasos se mide en los dos textos
LOTES_MEDIDA = 32              # lotes fijos para cada medida (siempre los mismos)
LETRAS_NUEVAS = 500_000        # texto que viene después de los ocho millones, sin leer nunca
SEMILLA_MEDIDA = 7             # la de los lotes fijos de medida, distinta de la del entrenamiento
PASO_DEL_CAPITULO = 50_000     # el final del capítulo, para comprobar que es el mismo modelo
TOLERANCIA_CAPITULO = 0.03     # diferencia admitida con la curva guardada (tanto por uno)
VENTANA_COMPARA = 25           # puntos de la curva que se promedian para esa comprobación

PASOS_SELFTEST = 600
LETRAS_SELFTEST = 20_000       # texto diminuto: se aprende de memoria enseguida

SALIDA_CSV = "../datos/salidas/seguir_entrenando.csv"
CURVA_CAPITULO = "../datos/salidas/en_que_orden_aprende_curva.csv"

# ==========================================================

import argparse
import csv
import datetime
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import en_que_orden_aprende as cap10
from formato import coma, miles, pct


def lotes_fijos(datos, n=LOTES_MEDIDA, semilla=SEMILLA_MEDIDA):
    """Los mismos n lotes cada vez: así dos medidas solo se diferencian por el modelo."""
    rng = np.random.default_rng(semilla)
    return [cap10.lote(datos, rng) for _ in range(n)]


def medir(modelo, lotes, vocabulario):
    """Lo mal que lo hace (la misma medida que la curva del capítulo) y el acierto."""
    modelo.eval()
    total = aciertos = vistos = 0.0
    with torch.no_grad():
        for x, y in lotes:
            p = modelo(x)
            total += float(F.cross_entropy(p.reshape(-1, vocabulario), y.reshape(-1)))
            aciertos += int((p.argmax(-1) == y).sum())
            vistos += y.numel()
    modelo.train()
    return total / len(lotes), aciertos / vistos


def entrenar(datos, nuevos, vocabulario, pasos_totales, cada=CADA, lotes_medida=LOTES_MEDIDA):
    """El entrenamiento del capítulo, paso por paso igual (cap10.entrenar), sin las fotos.

    Devuelve la curva cruda (cada 20 pasos, como el CSV del capítulo) y las medidas en los
    dos textos (cada `cada` pasos)."""
    cap10.fijar_semilla()
    rng = np.random.default_rng(cap10.SEMILLA)
    modelo = cap10.Transformer(vocabulario).to(cap10.DISPOSITIVO)
    opt = torch.optim.AdamW(modelo.parameters(), lr=cap10.TASA)
    fijo_visto = lotes_fijos(datos, lotes_medida)
    fijo_nuevo = lotes_fijos(nuevos, lotes_medida)
    curva, medidas = [], []
    medidas.append((0,) + medir(modelo, fijo_visto, vocabulario) + medir(modelo, fijo_nuevo, vocabulario))
    modelo.train()
    t0 = time.time()
    for pasos in range(1, pasos_totales + 1):
        x, y = cap10.lote(datos, rng)
        opt.zero_grad()
        p = modelo(x)
        perdida = F.cross_entropy(p.reshape(-1, vocabulario), y.reshape(-1))
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        opt.step()
        if pasos % 20 == 0:
            curva.append((pasos, float(perdida.item())))
        if pasos % cada == 0:
            medidas.append((pasos,) + medir(modelo, fijo_visto, vocabulario)
                           + medir(modelo, fijo_nuevo, vocabulario))
            if pasos % (10 * cada) == 0:
                print(f"  paso {miles(pasos)}: visto {coma(medidas[-1][1], 3)}  "
                      f"nuevo {coma(medidas[-1][3], 3)}  ({coma((time.time() - t0) / 60, 1)} min "
                      "en esta máquina)", flush=True)
    return modelo, curva, medidas


def cargar():
    """Los ocho millones del capítulo y, detrás, el texto que no lee nunca.

    Se carga con el mismo cargador del capítulo y un poco más largo: los primeros ocho
    millones son, letra a letra, los del capítulo (se comprueba)."""
    largo = cap10.cargar_texto(cap10.MAX_CARACTERES + LETRAS_NUEVAS)
    del_capitulo = cap10.cargar_texto()
    assert largo[:cap10.MAX_CARACTERES] == del_capitulo, \
        "Se esperaba que los primeros ocho millones fueran el texto del capítulo"
    assert len(largo) == cap10.MAX_CARACTERES + LETRAS_NUEVAS, \
        f"Se esperaban {miles(cap10.MAX_CARACTERES + LETRAS_NUEVAS)} letras; hay {miles(len(largo))}"
    return del_capitulo, largo[cap10.MAX_CARACTERES:]


def a_tensores(texto, indice):
    faltan = set(texto) - set(indice)
    assert not faltan, f"Se esperaban solo símbolos del capítulo; sobran {sorted(faltan)}"
    return torch.tensor([indice[c] for c in texto], dtype=torch.long)


def media_curva(curva, paso, ventana=VENTANA_COMPARA):
    antes = [v for p, v in curva if p <= paso][-ventana:]
    assert len(antes) == ventana, f"Se esperaban {ventana} puntos hasta el paso {paso}"
    return sum(antes) / ventana


def selftest():
    fallos = []
    texto = cap10.cargar_texto(600_000)
    letras = sorted(set(texto))
    indice = {c: i for i, c in enumerate(letras)}

    # 1. TEST NULO — texto barajado letra a letra: no hay nada que aprender de un trozo para
    #    otro, así que en texto visto y en texto nuevo tiene que hacerlo igual de mal.
    barajado = list(texto[:400_000])
    np.random.default_rng(cap10.SEMILLA).shuffle(barajado)
    visto = a_tensores("".join(barajado[:300_000]), indice)
    nuevo = a_tensores("".join(barajado[300_000:]), indice)
    _, _, m = entrenar(visto, nuevo, len(letras), PASOS_SELFTEST, cada=PASOS_SELFTEST, lotes_medida=8)
    dif_nula = m[-1][3] - m[-1][1]
    print(f"[1] test nulo         texto barajado: visto {coma(m[-1][1], 3)}, nuevo {coma(m[-1][3], 3)}")
    if abs(dif_nula) > 0.05:
        fallos.append(f"test nulo: con texto barajado visto y nuevo difieren {coma(dif_nula, 3)}")

    # 2. SEÑAL IMPLANTADA — un texto diminuto se aprende de memoria: en él lo hace mucho mejor
    #    que en texto nuevo. Si la medida no viera esa diferencia, no serviría para ver el techo.
    visto = a_tensores(texto[:LETRAS_SELFTEST], indice)
    nuevo = a_tensores(texto[300_000:400_000], indice)
    _, _, m = entrenar(visto, nuevo, len(letras), PASOS_SELFTEST, cada=PASOS_SELFTEST, lotes_medida=8)
    dif = m[-1][3] - m[-1][1]
    print(f"[2] señal implantada  texto de {miles(LETRAS_SELFTEST)} letras: visto {coma(m[-1][1], 3)}, "
          f"nuevo {coma(m[-1][3], 3)}")
    if dif < 0.3:
        fallos.append(f"señal implantada: memorizando, visto y nuevo solo difieren {coma(dif, 3)}")

    # 3. INVARIANTE — medir dos veces el mismo modelo con los mismos lotes da lo mismo, y la
    #    medida de un modelo recién nacido está cerca del azar entre todos los símbolos.
    modelo = cap10.Transformer(len(letras)).to(cap10.DISPOSITIVO)
    lotes = lotes_fijos(a_tensores(texto[:200_000], indice), 4)
    a, b = medir(modelo, lotes, len(letras)), medir(modelo, lotes, len(letras))
    azar = float(np.log(len(letras)))
    print(f"[3] invariante        dos medidas iguales: {'sí' if a == b else 'NO'}; recién nacido "
          f"{coma(a[0], 3)} frente al azar {coma(azar, 3)}")
    if a != b:
        fallos.append("invariante: dos medidas del mismo modelo no coinciden")
    if abs(a[0] - azar) > 0.5:
        fallos.append(f"invariante: el modelo recién nacido no está cerca del azar ({coma(a[0], 3)})")
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
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())

    print(f"Medido el {datetime.date.today()} en {platform.platform()}; procesador usado: "
          f"{cap10.DISPOSITIVO}.")
    texto, texto_nuevo = cargar()
    letras = sorted(set(texto))
    indice = {c: i for i, c in enumerate(letras)}
    datos, nuevos = a_tensores(texto, indice), a_tensores(texto_nuevo, indice)
    print(f"texto visto: {miles(len(texto))} letras (las del capítulo); texto nuevo: "
          f"{miles(len(texto_nuevo))} letras de los libros que vienen detrás.\n")
    modelo, curva, medidas = entrenar(datos, nuevos, len(letras), PASOS_TOTAL)

    # El mismo modelo que el del capítulo: la curva cruda hasta el paso 50.000 tiene que
    # coincidir con la guardada (misma semilla, mismos lotes).
    guardada = cap10.leer_curva(CURVA_CAPITULO)
    a = media_curva(curva, PASO_DEL_CAPITULO)
    b = media_curva(guardada, PASO_DEL_CAPITULO)
    print(f"\ncomprobación: media de la curva cruda antes del paso {miles(PASO_DEL_CAPITULO)}: "
          f"aquí {coma(a, 4)}, en la del capítulo {coma(b, 4)}")
    assert abs(a - b) / b < TOLERANCIA_CAPITULO, \
        f"Se esperaba el mismo modelo que el del capítulo; la curva difiere {coma(abs(a - b) / b, 4)}"

    print("\n--- LO MAL QUE LO HACE, EN TEXTO VISTO Y EN TEXTO NUEVO ---")
    print(f"{'pasos':>9}{'visto':>10}{'nuevo':>10}{'acierto visto':>16}{'acierto nuevo':>16}")
    marcas = {0, 3_000, 10_000, 25_000, 50_000, 75_000, 100_000, 125_000, 150_000}
    for p, v, av, n, an in medidas:
        if p in marcas:
            print(f"{miles(p):>9}{coma(v, 3):>10}{coma(n, 3):>10}{pct(av, 0):>16}{pct(an, 0):>16}")
    mejor = min(medidas, key=lambda m: m[3])
    print(f"\nen texto nuevo, lo mejor que lo hace es en el paso {miles(mejor[0])}: {coma(mejor[3], 3)}")
    print(f"en el paso {miles(PASOS_TOTAL)}, en texto nuevo: {coma(medidas[-1][3], 3)}")
    print(f"\nlo que escribe tras {miles(PASOS_TOTAL)} pasos:")
    print(cap10.escribir(modelo, indice, letras))

    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(
            [["pasos", "visto", "nuevo", "acierto_visto", "acierto_nuevo"]]
            + [[p, f"{v:.4f}", f"{n:.4f}", f"{av:.4f}", f"{an:.4f}"] for p, v, av, n, an in medidas])
    print(f"\nEscrito {SALIDA_CSV}")


if __name__ == "__main__":
    main()
