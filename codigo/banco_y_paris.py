#!/usr/bin/env python3
"""
Capítulo 7 — dos lecturas de la máquina entera que el lector no tiene que hacer solo (L24).

Usa el mismo modelo, el mismo tipo de número y las mismas frases que `maquina_entera_detalle.py`
(importados de allí, no copiados) y añade lo que el capítulo necesita para leer dos de sus
bloques:

  1. PARÍS, TROZO A TROZO. Cada trozo de «París» se mira en la lista que da la máquina detrás
     del texto que tiene delante: «Par», detrás de «La capital de Francia es»; «ís», detrás de
     «La capital de Francia es Par». Y la cuenta de «entera», hecha en «de cada diez mil».
  2. LA ESCALA DEL PARECIDO DE «BANCO». La tabla del capítulo compara «banco» consigo mismo en
     frases distintas. Para saber si 0,74 es mucho o poco hace falta una referencia en esas
     mismas capas: el parecido de «banco» con otra palabra de su misma frase.

Uso:
    python banco_y_paris.py --selftest
    python banco_y_paris.py
"""

# ======================= CONSTANTES =======================

# La referencia: una palabra de B y otra de D que son un solo trozo y que no dicen nada del
# sentido de «banco». En la primera vuelta de L24 la de B era «pagar», que sí es pista del sentido
# del dinero; se cambió por «tanto» en la segunda (VERIFICACION-B, N8). Las palabras de las cuatro
# frases que sí son pista («parque», «hipoteca», «sentarme») son dos trozos o más.
REFERENCIAS = [("B", " tanto"), ("D", " hablar")]
CAPAS = [0, 4, 8, 16, 24]
POR_CADA = 10_000

# ==========================================================

import argparse
import datetime
import platform
import sys

import torch

import maquina_entera_detalle as M
from formato import coma, miles, pct, tabla_editorial, trozo

FRASES = {"A": M.BANCO_A, "B": M.BANCO_B, "C": M.BANCO_C, "D": M.BANCO_D}


def lista(modelo, ids):
    """La lista de probabilidades del trozo siguiente, como `maquina_entera_detalle.lista`, pero
    sumada en doble precisión: en un procesador ARM la suma en precisión simple de 151.936
    porcentajes se aparta de 1 en la cuarta cifra y aquel programa lo rechaza. La lista es la
    misma; solo cambia con cuántas cifras se hace la cuenta final."""
    with torch.no_grad():
        logits = modelo(torch.tensor([ids])).logits[0, -1]
    p = torch.softmax(logits.double(), dim=-1)
    assert abs(float(p.sum()) - 1.0) < 1e-9, f"la lista suma {float(p.sum())}"
    return p


def estados(modelo, ids):
    with torch.no_grad():
        s = modelo(torch.tensor([ids]), output_hidden_states=True)
    return [h[0].float() for h in s.hidden_states]      # capa -> (posición, números)


def posicion(tok, frase, palabra):
    ids = M.ids_de(tok, frase)
    t = M.ids_de(tok, palabra)
    assert len(t) == 1, f"{palabra!r} tenía que ser un solo trozo; es {len(t)}"
    assert ids.count(t[0]) == 1, f"{palabra!r} tenía que salir una sola vez en {frase!r}"
    return ids, ids.index(t[0])


def bloque_paris(tok, modelo):
    """L24 (9 de octubre): tablas editoriales (regla 6 ter). La cuenta no cambia."""
    frase = M.FRASES_PARIS[0]
    ids = M.ids_de(tok, frase)
    total, texto, valores, filas = 1.0, frase, [], []
    for i in M.ids_de(tok, M.PALABRA_BUSCADA):
        p = lista(modelo, ids)
        v, pu = float(p[i]), M.puesto(p, i)
        t = tok.decode([i])
        filas.append([f"«{texto}»", trozo(M.visible(t)), pct(v, 4 if v < 0.01 else 2),
                      f"{miles(pu)} de {miles(len(p))}"])
        valores.append(v)
        total *= v
        texto += t
        ids = ids + [i]
    trozos_ = tabla_editorial(
        "París, trozo a trozo", ["detrás de", "trozo", "probabilidad", "puesto"], filas, "iidd",
        ["Cada trozo se mira en la lista que da la máquina detrás del texto que tiene delante."])
    entera = tabla_editorial(
        f"«{M.PALABRA_BUSCADA.strip()}» entera", ["", "cuánto"],
        [[f"de cada {miles(POR_CADA)} veces, «Par» sale", coma(POR_CADA * valores[0], 2)],
         ["de esas, «ís» sigue, de cada 100", coma(100 * valores[1], 2)],
         [f"**«{M.PALABRA_BUSCADA.strip()}» entera, de cada {miles(POR_CADA)}**",
          f"**{coma(POR_CADA * total, 2)}**"]], "id",
        [f"El {pct(valores[1], 2)} del {pct(valores[0], 4)}, que da {pct(total, 4)}."])
    return trozos_ + [""] + entera, total


def bloque_banco(tok, modelo):
    """L24 (9 de octubre): tabla editorial (regla 6 ter). La cuenta no cambia."""
    banco = M.ids_de(tok, M.TROZO_BANCO)[0]
    datos = {}
    for f, w in REFERENCIAS:
        ids, pos = posicion(tok, FRASES[f], w)
        assert ids[-1] == banco
        e = estados(modelo, ids)
        datos[(f, w)] = [M.parecido(e[c][-1], e[c][pos]) for c in CAPAS]
    filas = [["entrada" if c == 0 else f"tras capa {c}"] + [coma(datos[r][k], 3) for r in REFERENCIAS]
             for k, c in enumerate(CAPAS)]
    return tabla_editorial(
        "«Banco» y otra palabra de su misma frase",
        [""] + [f"banco y {w.strip()} ({f})" for f, w in REFERENCIAS], filas, "i" + "d" * len(REFERENCIAS),
        ["Sirve de escala para leer la tabla anterior: es lo que se parece «banco» a otra palabra "
         "de su misma frase, en las mismas capas. Esa otra palabra no dice nada de su sentido. 1 es "
         "lo más parecido posible."]), datos


def selftest(tok, modelo):
    fallos = []
    # 1. TEST NULO — «banco» consigo mismo en la misma frase da 1 en todas las capas: la cuenta
    #    no se inventa diferencias.
    ids, _ = posicion(tok, M.BANCO_D, " hablar")
    e = estados(modelo, ids)
    peor = min(M.parecido(e[c][-1], e[c][-1]) for c in CAPAS)
    print(f"[1] test nulo         «banco» consigo mismo: parecido mínimo {coma(peor, 6)}")
    if abs(peor - 1) > M.TOL_IDENTICO:
        fallos.append("test nulo: una lista consigo misma no da 1")
    # 2. SEÑAL — la cuenta de «entera» da lo que imprimió maquina_entera_detalle.txt
    #    (0,0533 %), a la última cifra impresa.
    _, total = bloque_paris(tok, modelo)
    print(f"[2] señal             «París» entera: {pct(total, 4)} (maquina_entera_detalle.txt: 0,0533 %)")
    if abs(100 * total - 0.0533) > 0.00005:
        fallos.append("señal: la cuenta de París no da lo del otro programa")
    # 3. INVARIANTE — es el mismo modelo que el del capítulo: el primer paso de la frase del
    #    capítulo sale como en maquina_entera.txt, y a la entrada «banco» es la misma lista en A
    #    y en D.
    p = lista(modelo, M.ids_de(tok, M.FRASE))
    t = tok.decode([int(p.argmax())])
    v = float(p.max())
    print(f"[3] invariante        primer paso: «{t}» con {pct(v, 2)} (maquina_entera.txt: 17,46 %)")
    if t != M.PRIMER_PASO_ESPERADO[0] or abs(v - M.PRIMER_PASO_ESPERADO[1]) > M.TOL_PRIMER_PASO:
        fallos.append("invariante: no es el mismo modelo que el del capítulo")
    print()
    if fallos:
        for x in fallos:
            print("FALLA:", x)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    tok, modelo = M.cargar()
    M.validar_entrada(tok)
    codigo = selftest(tok, modelo)
    if codigo or args.selftest:
        sys.exit(codigo)
    print(f"\nmáquina: {platform.machine()}, {platform.system()} {platform.release()}, procesador")
    print(f"modelo: {M.MODELO} en {M.DTYPE}   fecha: {datetime.date.today().isoformat()}\n")
    print("--- 1. PARÍS, TROZO A TROZO ---\n")
    lin, _ = bloque_paris(tok, modelo)
    print("\n".join(lin) + "\n")
    print("--- 2. «BANCO» Y OTRA PALABRA DE SU MISMA FRASE ---\n")
    lin, _ = bloque_banco(tok, modelo)
    print("\n".join(lin))


if __name__ == "__main__":
    main()
