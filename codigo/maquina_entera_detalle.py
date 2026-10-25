#!/usr/bin/env python3
"""
Capítulo 7 — cuatro detalles de la máquina entera, pedidos por el análisis de claridad
(notas/CLARIDAD.md, M7-1 a M7-4, aprobadas por Carlos el 25 de septiembre de 2026).

No sustituye a maquina_entera.py ni toca su salida, que el libro ya cita. Usa el mismo modelo,
con el mismo tipo de número, y mide cuatro cosas que el capítulo necesita para explicarse:

  1. QUÉ SE PARTE Y QUÉ NO. Cuántos trozos son unas cuantas palabras corrientes, una rara, y
     la misma palabra en castellano y en inglés.
  2. DÓNDE ESTÁ PARÍS. «París» no es un trozo, son dos. Se calcula la probabilidad de la
     palabra entera: la del primer trozo por la del segundo detrás de él.
  3. LAS TRES PRIMERAS CANDIDATAS DE CADA PASO. Los ocho pasos del capítulo, pero enseñando
     también las dos opciones que perdieron.
  4. PARA QUÉ SIRVE MIRAR. El trozo «banco» en tres frases: dos en que es un asiento y una en
     que es donde se guarda el dinero. Cuánto se parece su lista de números entre frases, a la
     entrada y a la salida de cada capa. La medida de parecido es la misma del capítulo 5.

Uso:
    python maquina_entera_detalle.py
    python maquina_entera_detalle.py --selftest
"""

# ======================= CONSTANTES =======================

MODELO = "Qwen/Qwen2.5-0.5B"      # el mismo de maquina_entera.py
DTYPE = "float32"                 # los números del modelo
# L24 (9 de octubre): la lista de probabilidades se saca en float64, y todo se ejecuta en el Mac.
# Con torch 2.14 en el Mac (procesador ARM), la de float32 suma 1,000131 y no pasa la comprobación
# de lista(); en float64 suma 1. Contra la salida anterior (float32, x86, 25 de septiembre) cambian
# cuatro cifras en la cuarta decimal (97,8717 % → 97,8714 %; 15,5138 % → 15,5139 %; 96,1958 % →
# 96,1953 %; 14,9236 % → 14,9237 %) y una en la cuarta de «Paris» sin tilde; el resto, igual.
# (El comentario anterior, del 28 de septiembre, decía que esta salida se generaba en x86: lo puso
# el asistente sin que Carlos lo hubiera decidido.)
SEMILLA = 0                       # no hay nada al azar; se fija igual, por regla

FRASE = "La capital de Francia es"

# 1. Palabras elegidas a mano, sin medir su frecuencia (supuesto declarado en CLARIDAD.md).
#    Cada una lleva el espacio delante porque así aparece dentro de una frase.
PALABRAS_CORRIENTES = [" de", " casa", " agua", " perro", " gato", " caballo", " ventana"]
PALABRA_RARA = " esternocleidomastoideo"
PAREJAS_IDIOMA = [(" murciélago", " bat"), (" caballo", " horse"), (" gato", " cat"),
                  (" ventana", " window")]

# 2. La palabra entera que el lector espera, y en qué frases buscarla.
PALABRA_BUSCADA = " París"
VARIANTE_SIN_TILDE = " Paris"
FRASES_PARIS = ["La capital de Francia es", "La capital de Francia es la ciudad de"]

# 3. Los ocho pasos del capítulo, con sus tres primeras candidatas.
PASOS = 8
CANDIDATAS = 3

# 4. «Banco». Este modelo solo mira hacia ATRÁS: cada trozo ve los que tiene delante, nunca
#    los que vienen después. Por eso lo que decide el sentido va ANTES de «banco», y la frase
#    termina en él.
#    Cuatro frases, dos por dos: dos sentidos (asiento, dinero) por dos formas de frase. Con
#    solo tres frases —dos de asiento parecidas de forma y una de dinero distinta— no se podía
#    saber si lo que separaba era el sentido o la forma (visto el 25 de septiembre, antes de
#    publicar nada). Con cuatro se comparan las dos cosas: si mirar sirve para lo que dice el
#    capítulo, «mismo sentido, distinta forma» tiene que parecerse más que «misma forma,
#    distinto sentido».
TROZO_BANCO = " banco"
BANCO_A = "Cansado de caminar por el parque, me senté en el banco"    # asiento, forma 1
BANCO_B = "Cansado de pagar tanto de hipoteca, me quejé en el banco"  # dinero,  forma 1
BANCO_C = "Para descansar un rato, fui a sentarme en el banco"        # asiento, forma 2
BANCO_D = "Para pedir la hipoteca, fui a hablar con el banco"         # dinero,  forma 2
CAPAS_A_IMPRIMIR = [0, 1, 2, 4, 8, 12, 16, 20, 23, 24]

# selftest
PROMPT_FACIL = "uno, dos, tres, cuatro,"
CONTINUACION_FACIL = " cinco"
PROMPT_PALABRA_LARGA = "Madrid es la capital de España, y París es la capital de"
PALABRA_LARGA = " Francia"          # dos trozos: _Franc | ia
UMBRAL_SENAL = 0.30
PRIMER_PASO_ESPERADO = (" la", 0.1746)   # lo que imprimió maquina_entera.txt
TOL_PRIMER_PASO = 0.00005                # la mitad de la última cifra impresa
TOL_IDENTICO = 1e-6

# ==========================================================

from formato import coma, miles, pct, tabla_editorial, trozo, unidos

import argparse
import datetime
import platform
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def cargar():
    torch.manual_seed(SEMILLA)
    tok = AutoTokenizer.from_pretrained(MODELO)
    modelo = AutoModelForCausalLM.from_pretrained(MODELO, dtype=getattr(torch, DTYPE))
    modelo.eval()
    return tok, modelo


def ids_de(tok, texto):
    return tok(texto)["input_ids"]


def trozos(tok, texto):
    return [tok.decode([i]) for i in ids_de(tok, texto)]


def visible(t):
    """El trozo como lo escribe el libro: el espacio de delante, como guion bajo."""
    return t.replace(" ", "_")


def validar_entrada(tok):
    """Todo lo que el resto del programa da por hecho, comprobado antes de medir nada."""
    for p in PALABRAS_CORRIENTES + [PALABRA_RARA] + [x for par in PAREJAS_IDIOMA for x in par]:
        rehecha = "".join(trozos(tok, p))
        assert rehecha == p, (f"Se esperaba que los trozos de {p!r} volvieran a formar la "
                              f"palabra; se encontró {rehecha!r}")
    banco = ids_de(tok, TROZO_BANCO)
    assert len(banco) == 1, (f"Se esperaba que {TROZO_BANCO!r} fuera un solo trozo; "
                             f"se encontraron {len(banco)}: {trozos(tok, TROZO_BANCO)}")
    for f in (BANCO_A, BANCO_B, BANCO_C, BANCO_D):
        ids = ids_de(tok, f)
        assert ids[-1] == banco[0], (f"Se esperaba que la frase {f!r} terminara en el trozo "
                                     f"{TROZO_BANCO!r}; termina en {tok.decode([ids[-1]])!r}")
    assert len(ids_de(tok, PALABRA_BUSCADA)) >= 1


def lista(modelo, ids):
    """La lista de probabilidades del trozo siguiente, sobre todas las entradas."""
    with torch.no_grad():
        logits = modelo(torch.tensor([ids])).logits[0, -1]
    # L24 (9 de octubre): en float64; ver el comentario de DTYPE.
    p = torch.softmax(logits.double(), dim=-1)
    assert abs(float(p.sum()) - 1.0) < 1e-4, \
        f"Se esperaba que la lista sumara 1; suma {float(p.sum()):.6f}"
    return p


def puesto(p, i):
    """1 = la opción más probable."""
    return int((p > p[i]).sum()) + 1


def prob_palabra(tok, modelo, contexto, palabra):
    """Probabilidad de que detrás de `contexto` venga `palabra` entera: la del primer trozo,
    por la del segundo detrás del primero, y así. Devuelve también el desglose."""
    ids = ids_de(tok, contexto)
    total, desglose = 1.0, []
    for i in ids_de(tok, palabra):
        p = lista(modelo, ids)
        v = float(p[i])
        desglose.append((tok.decode([i]), v, puesto(p, i)))
        total *= v
        ids = ids + [i]
    return total, desglose


def estados_del_ultimo(modelo, ids):
    """La lista de números del ÚLTIMO trozo a la entrada (0) y a la salida de cada capa.
    Nota: en esta biblioteca, la última lista ya lleva aplicada la normalización final."""
    with torch.no_grad():
        salida = modelo(torch.tensor([ids]), output_hidden_states=True)
    return [h[0, -1].float() for h in salida.hidden_states]


def parecido(u, v):
    """La misma cuenta que el capítulo 5 (la_misma_direccion.coseno): 1 la misma dirección,
    0 ninguna relación."""
    return float(torch.dot(u, v) / (u.norm() * v.norm()))


# ------------------------------------------------------------------ las cuatro mediciones

def chips(piezas):
    return " ".join(trozo(visible(x)) for x in piezas)


def bloque_troceado(tok):
    partes = tabla_editorial(
        "Qué se parte y qué no", ["palabra", "trozos", "cómo se parte"],
        [[p.strip(), str(len(trozos(tok, p))), chips(trozos(tok, p))]
         for p in PALABRAS_CORRIENTES + [PALABRA_RARA]], "idi",
        ["El guion bajo es el espacio de delante, que la palabra lleva dentro de una frase."])
    idiomas = tabla_editorial(
        "La misma palabra, en castellano y en inglés",
        ["castellano", "trozos", "inglés", "trozos"],
        [[es.strip(), str(len(trozos(tok, es))), en.strip(), str(len(trozos(tok, en)))]
         for es, en in PAREJAS_IDIOMA], "idid",
        [f"Trozos distintos en el repertorio: {miles(len(tok))}."])
    return partes + [""] + idiomas


def bloque_paris(tok, modelo):
    out = []
    for f in FRASES_PARIS:
        total, desglose = prob_palabra(tok, modelo, f, PALABRA_BUSCADA)
        sin_tilde, d2 = prob_palabra(tok, modelo, f, VARIANTE_SIN_TILDE)
        n = len(lista(modelo, ids_de(tok, f)))
        filas = [[trozo(visible(t)), pct(v, 4), f"{miles(pu)} de {miles(n)}"] for t, v, pu in desglose]
        filas += [[f"**«{PALABRA_BUSCADA.strip()}» entera**", f"**{pct(total, 4)}**", ""],
                  [f"«{VARIANTE_SIN_TILDE.strip()}», sin tilde, un trozo", pct(sin_tilde, 4),
                   miles(d2[0][2])]]
        out += tabla_editorial(
            f"Dónde está París detrás de «{f}»", ["trozo", "probabilidad", "puesto"], filas, "idd",
            [f"«{PALABRA_BUSCADA.strip()}» son {len(ids_de(tok, PALABRA_BUSCADA))} trozos: "
             f"{unidos([visible(x) for x in trozos(tok, PALABRA_BUSCADA)])}. Cada uno, en la lista que da la máquina "
             "detrás del texto que tiene delante; «entera», los dos multiplicados."]) + [""]
    return out[:-1]


def bloque_candidatas(tok, modelo):
    ids = ids_de(tok, FRASE)
    elegidas, filas = [], []
    for paso in range(1, PASOS + 1):
        p = lista(modelo, ids)
        top = torch.topk(p, CANDIDATAS)
        filas.append([str(paso)] + [f"{trozo(visible(tok.decode([int(i)])))} {pct(float(v), 1)}"
                                    for v, i in zip(top.values, top.indices)])
        elegido = int(top.indices[0])
        elegidas.append((tok.decode([elegido]), float(top.values[0])))
        ids = ids + [elegido]
    tabla = tabla_editorial(
        "Las tres primeras candidatas de cada paso",
        ["paso"] + [f"{k}.ª opción" for k in range(1, CANDIDATAS + 1)], filas, "c" + "i" * CANDIDATAS,
        ["Se elige siempre la 1.ª; las otras dos son las que perdieron. Porcentajes con una cifra "
         "decimal: 0,0 % es menos de 0,05 %.",
         f"Texto al acabar: «{tok.decode(ids)}»."])
    return tabla, elegidas


def bloque_banco(tok, modelo):
    e = {k: estados_del_ultimo(modelo, ids_de(tok, f))
         for k, f in (("A", BANCO_A), ("B", BANCO_B), ("C", BANCO_C), ("D", BANCO_D))}
    parejas = [("A", "C"), ("B", "D"), ("A", "B"), ("C", "D")]
    frases = tabla_editorial(
        "Las cuatro frases de «banco»", ["", "la frase", "sentido", "forma de la frase"],
        [["A", f"«{BANCO_A}»", "asiento", "la de A"], ["B", f"«{BANCO_B}»", "dinero", "la de A"],
         ["C", f"«{BANCO_C}»", "asiento", "la de C"], ["D", f"«{BANCO_D}»", "dinero", "la de C"]],
        "ciii")
    filas = []
    tabla = []
    for c in CAPAS_A_IMPRIMIR:
        nombre = "entrada" if c == 0 else f"tras capa {c}"
        v = [parecido(e[a][c], e[b][c]) for a, b in parejas]
        filas.append((c, *v))
        tabla.append([nombre] + [coma(x, 3) for x in v])
    grupos = ["mismo sentido, distinta forma"] * 2 + ["misma forma, distinto sentido"] * 2
    parecidos = tabla_editorial(
        "Parecido de la lista de «banco» entre dos frases",
        [""] + [f"{g}: {a} con {b}" for g, (a, b) in zip(grupos, parejas)], tabla, "idddd",
        ["1 quiere decir idéntica; es la misma medida del capítulo 5."])
    return frases + [""] + parecidos, filas


# ------------------------------------------------------------------------------ selftest

def selftest(tok, modelo):
    fallos = []

    # 1. TEST NULO — «banco» en la MISMA frase dos veces: el parecido tiene que ser 1 en
    #    todas las capas. Y, como detector de que el nulo puede fallar, dos frases distintas
    #    tienen que dar menos de 1 en la última capa.
    e1 = estados_del_ultimo(modelo, ids_de(tok, BANCO_A))
    e2 = estados_del_ultimo(modelo, ids_de(tok, BANCO_A))
    peor = min(parecido(x, y) for x, y in zip(e1, e2))
    otra = estados_del_ultimo(modelo, ids_de(tok, BANCO_D))
    distinto = parecido(e1[-1], otra[-1])
    print(f"[1] test nulo         misma frase: parecido mínimo {coma(peor, 6)}; "
          f"frases distintas, última capa: {coma(distinto, 4)}")
    if abs(peor - 1.0) > TOL_IDENTICO:
        fallos.append(f"test nulo: la misma frase dos veces da un parecido de {coma(peor, 6)}, "
                      f"no de 1")
    if not distinto < 1.0 - TOL_IDENTICO:
        fallos.append("test nulo: dos frases distintas dan parecido 1; la medida no ve nada")

    # 2. SEÑAL IMPLANTADA — tras «uno, dos, tres, cuatro,» la primera candidata es «cinco»; y
    #    una palabra de dos trozos con respuesta evidente sale con probabilidad alta.
    p = lista(modelo, ids_de(tok, PROMPT_FACIL))
    primera = tok.decode([int(p.argmax())])
    larga, desglose = prob_palabra(tok, modelo, PROMPT_PALABRA_LARGA, PALABRA_LARGA)
    print(f"[2] señal implantada  «{PROMPT_FACIL}» -> «{primera}»; "
          f"«{PALABRA_LARGA.strip()}» ({len(desglose)} trozos) tras «…capital de»: "
          f"{pct(larga, 2)}")
    if primera != CONTINUACION_FACIL:
        fallos.append(f"señal implantada: se esperaba «{CONTINUACION_FACIL}» la primera; "
                      f"salió «{primera}»")
    if len(desglose) < 2:
        fallos.append(f"señal implantada: «{PALABRA_LARGA}» debía ser de dos trozos o más")
    if larga < UMBRAL_SENAL:
        fallos.append(f"señal implantada: se esperaba «{PALABRA_LARGA.strip()}» con al menos "
                      f"{pct(UMBRAL_SENAL, 0)}; salió {pct(larga, 2)}")

    # 3. INVARIANTE DEL DOMINIO — (a) el primer paso coincide con lo que ya imprimió
    #    maquina_entera.txt; (b) a la entrada, antes de ninguna capa, «banco» es la MISMA
    #    lista en cualquier frase: la posición no va sumada ahí en este modelo.
    _, elegidas = bloque_candidatas(tok, modelo)
    t, v = elegidas[0]
    ent = parecido(estados_del_ultimo(modelo, ids_de(tok, BANCO_A))[0],
                   estados_del_ultimo(modelo, ids_de(tok, BANCO_D))[0])
    print(f"[3] invariante        paso 1: «{t}» con {pct(v, 2)}; "
          f"«banco» a la entrada en A y D: {coma(ent, 6)}")
    if t != PRIMER_PASO_ESPERADO[0] or abs(v - PRIMER_PASO_ESPERADO[1]) > TOL_PRIMER_PASO:
        fallos.append(f"invariante: se esperaba «{PRIMER_PASO_ESPERADO[0]}» con "
                      f"{pct(PRIMER_PASO_ESPERADO[1], 2)} (maquina_entera.txt); "
                      f"salió «{t}» con {pct(v, 4)}")
    if abs(ent - 1.0) > TOL_IDENTICO:
        fallos.append(f"invariante: a la entrada «banco» debería ser la misma lista en "
                      f"cualquier frase; parecido {coma(ent, 6)}")

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

    tok, modelo = cargar()
    validar_entrada(tok)
    if args.selftest:
        sys.exit(selftest(tok, modelo))

    print(f"máquina: {platform.machine()}, {platform.system()} {platform.release()}, procesador")
    print(f"modelo: {MODELO} en {DTYPE}")
    print(f"torch {torch.__version__}   transformers {transformers.__version__}")
    print(f"fecha: {datetime.date.today().isoformat()}")
    print()
    print("--- 1. QUÉ SE PARTE Y QUÉ NO ---\n")
    print("\n".join(bloque_troceado(tok)))
    print("\n--- 2. DÓNDE ESTÁ PARÍS ---\n")
    print("\n".join(bloque_paris(tok, modelo)))
    print("\n--- 3. LAS TRES PRIMERAS CANDIDATAS DE CADA PASO ---\n")
    lin, _ = bloque_candidatas(tok, modelo)
    print("\n".join(lin))
    print("\n--- 4. PARA QUÉ SIRVE MIRAR: «BANCO» EN CUATRO FRASES ---\n")
    lin, _ = bloque_banco(tok, modelo)
    print("\n".join(lin))


if __name__ == "__main__":
    main()
