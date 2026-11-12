#!/usr/bin/env python3
"""
Capítulo 8 — la figura del reparto: a dónde va la atención del adjetivo final.

El capítulo enseña cuatro columnas de una tabla —«El», «vaso», «cajón» y el propio
adjetivo— y deja fuera las otras siete palabras de la frase. La figura las enseña
todas, en el orden en que van en la frase, y con eso se ven de un golpe las tres cosas
que el capítulo cuenta por separado en dos apartados:

  1. Que a los dos sustantivos entre los que había que elegir —«vaso» y «cajón»— les
     llega casi nada del reparto.
  2. Que más de la mitad se va a la primera palabra de la frase, «El», que es la
     papelera, y que la cuarta parte se la queda el adjetivo mirándose a sí mismo.
  3. Que las tres frases dan el mismo reparto: cambiar «alto» por «bajo» o por «caro»
     no mueve las barras.

De dónde salen los números. De `../datos/salidas/reparto_una_a_una.csv`, escrito por
`reparto_una_a_una.py`. Este programa **no carga ningún modelo**: lee el CSV y promedia.
Así la figura y las cifras del texto no se pueden separar nunca; si alguien vuelve a
medir, cambian las dos o no cambia ninguna.

Qué promedio se dibuja, que es la única decisión de esta figura y por eso va dicha en la
clave: el de **las 336 miradas del modelo entero**, no el del último tercio de capas. Es
el mismo que el capítulo llama «todas» en su tabla. Con el último tercio la primera
barra llegaría al 60,4 % y «vaso» se quedaría en el 1,2 %; el dibujo diría lo mismo y
sería menos honesto con la palabra «todas».

POR QUÉ LAS TRES FRASES VAN UNA DEBAJO DE OTRA DENTRO DE CADA PALABRA, y no en tres
cuadros separados, que es como estaba propuesta. Las imágenes de este libro entran a lo
ancho de la caja, 4,45 pulgadas, y `main.tex` le reserva a cada figura media caja de
alto: si una figura pasa de ahí, se sale de la página por abajo sin que ningún
verificador lo cante. Tres cuadros de once renglones rotulados son treinta y tres
renglones, y treinta y tres renglones que se lean no bajan de cuatro pulgadas y media de
alto. Con las once palabras rotuladas una sola vez y las tres frases dentro de cada
renglón, la figura mide tres pulgadas y media, entra en la media caja, y además enseña
mejor lo que más cuesta enseñar: que las tres frases dan lo mismo, porque las tres
barras de cada palabra salen del mismo largo y se ve sin comparar entre cuadros.

Uso:
    python figura_el_reparto.py
    python figura_el_reparto.py --selftest
"""

# ======================= CONSTANTES =======================

CSV = "../datos/salidas/reparto_una_a_una.csv"
SALIDA = "../figuras/el_reparto.png"
# 6,2 x 4,9 pulgadas. Lo que manda es la proporción: xelatex mete la imagen a lo ancho
# de la caja (4,45 pulgadas), así que en el papel sale de 4,45 x 3,52, y la media caja
# que le reserva `needspace` son 3,63. Si se toca el alto, hay que rehacer esta cuenta.
ANCHO_ALTO = (6.2, 4.9)
PUNTOS = 200                     # puntos por pulgada

# La figura se encoge a un 72 % al entrar en la página, así que estos cuerpos de letra
# están elegidos por lo que se lee EN EL PAPEL, no por lo que se ve en la pantalla:
# 11,5 en pantalla son 8,3 impresos.
CUERPO_TITULO = 13.0
CUERPO_PALABRA = 11.0
CUERPO_CIFRA = 9.4
CUERPO_EJE = 10.4
CUERPO_LLAMADA = 9.8
CUERPO_CLAVE = 9.0

NEGRO = "0.12"                   # las dos palabras entre las que había que elegir
GRIS = "0.62"                    # las otras nueve
GRIS_REJILLA = "0.88"

TOPE_EJE = 60                    # el eje va de 0 a 60 por ciento
MARCAS = [0, 10, 20, 30, 40, 50, 60]

TITULO = "A dónde mira el final de cada frase"
EJE_X = "porción del reparto, en por ciento"
ROTULO_ADJETIVO = "adjetivo\ny punto"  # la palabra 11 es distinta en cada frase; va pegada al punto, que es lo que pregunta
LLAMADA_PAPELERA = "la papelera"
LLAMADA_ADJETIVO = "el final de la frase,\nmirándose a sí mismo"

# La clave va debajo del cuadro y pegada a él, y dice las cuatro cosas que no se pueden
# adivinar mirando: qué separa el negro del gris, qué son las tres barras de cada
# palabra, que cada frase suma cien, y sobre cuántas miradas está hecho el promedio.
# El número de la última línea NO se escribe a mano: lo mide el programa.
CLAVE = (
    "En negro, las dos palabras entre las que había que elegir: «vaso» y «cajón»; en gris, las demás.\n"
    "Cada palabra lleva tres barras, una por frase: «alto» arriba, «bajo» en medio, «caro» abajo, y\n"
    "las once palabras de una frase suman el cien por cien. La cifra escrita es la de la frase del\n"
    "«alto»; las otras dos no se apartan de ella más de {mayor} puntos en ninguna palabra. Cada barra\n"
    "es el promedio de las 336 miradas del modelo entero, no el de las 112 de las 8 últimas capas."
)

# Umbrales del selftest, fijados por razonamiento antes de mirar el resultado. El
# programa imprime al lado el valor medido, para que se vea cuánto margen hay.
#  - test nulo: con un CSV inventado de reparto plano, las once barras tienen que salir
#    iguales (desvío máximo de una décima de punto) y la primera no puede destacar.
#    «Destacar» es llegar al doble de la barra mediana de su frase: es lo que hace
#    defendible la llamada de «la papelera», y con un reparto plano no puede pasar.
#  - señal implantada: con «vaso» llevándose el 90 %, la barra más larga del cuadro
#    tiene que ser la de «vaso», y tiene que salir en negro.
#  - invariante: cada frase suma cien, sin porciones negativas, y lleva tantas barras
#    como palabras trae el CSV para esa frase.
DESVIO_MAXIMO_NULO = 0.1         # en puntos porcentuales
FACTOR_DESTACA = 2.0
IMPLANTADA_VASO = 0.90
TOL_SUMA = 0.1                   # en puntos porcentuales

# ==========================================================

import argparse
import csv
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from formato import coma
from reparto_atencion import CANDIDATAS, FRASES


def leer_filas(ruta=CSV):
    with open(ruta, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def promedios(filas):
    """De las filas del CSV a lo que se dibuja: por frase, las palabras en el orden en
    que van y la porción media que se lleva cada una, en por ciento.

    El CSV trae una fila por (frase, capa, mirada, palabra) y aquí no se elige nada: se
    promedia sobre TODAS las miradas que traiga el fichero. Las palabras se toman en el
    orden de las filas, que es el orden de la frase; no se pueden ordenar por nombre
    porque «el» sale dos veces y son dos barras distintas.
    """
    miradas = OrderedDict()      # (frase, capa, mirada) -> [(palabra, porción), ...]
    for f in filas:
        clave = (f["adjetivo"], f["capa"], f["mirada"])
        miradas.setdefault(clave, []).append((f["palabra"], float(f["porcion"])))

    largos = {len(v) for v in miradas.values()}
    assert len(largos) == 1, (
        f"Se esperaba que todas las miradas trajeran el mismo número de palabras; "
        f"se encontró {sorted(largos)}")

    bandas, cuentas = OrderedDict(), {}
    for (frase, _, _), pares in miradas.items():
        if frase not in bandas:
            bandas[frase] = [[p, 0.0] for p, _ in pares]
            cuentas[frase] = 0
        acumulado = bandas[frase]
        for k, (palabra, porcion) in enumerate(pares):
            assert palabra == acumulado[k][0], (
                f"Se esperaba «{acumulado[k][0]}» en la posición {k + 1} de la frase "
                f"«{frase}»; se encontró «{palabra}»")
            acumulado[k][1] += porcion
        cuentas[frase] += 1

    salida = OrderedDict()
    for frase, acumulado in bandas.items():
        n = cuentas[frase]
        salida[frase] = {"palabras": [p for p, _ in acumulado],
                         "valores": [100.0 * s / n for _, s in acumulado],
                         "miradas": n}
    return salida


def color_de(palabra):
    """Negro para las dos palabras entre las que había que elegir, gris para el resto.
    Es la clave de la figura, y vive en una sola función para que el dibujo no pueda
    decir una cosa distinta de la que comprueba el selftest."""
    return NEGRO if palabra.strip(".,;:").lower() in CANDIDATAS else GRIS


def rotulos_de(bandas):
    """Cómo se escribe cada renglón. Las diez primeras palabras son las mismas en las
    tres frases, así que van rotuladas una sola vez; la undécima cambia de una frase a
    otra —es «alto», «bajo» o «caro»— y por eso se rotula por lo que es, «el adjetivo».
    La primera va con mayúscula porque es la primera de la frase y el capítulo la llama
    «El». Es ortografía, no dato: no toca ninguna porción."""
    listas = [d["palabras"] for d in bandas.values()]
    rotulos = []
    for k in range(len(listas[0])):
        columna = {l[k] for l in listas}
        palabra = listas[0][k] if len(columna) == 1 else ROTULO_ADJETIVO
        rotulos.append(palabra.capitalize() if k == 0 else palabra)
    return rotulos


def mayor_diferencia(bandas):
    """Cuánto se apartan entre sí las tres frases, palabra a palabra, en puntos. Es el
    número que la clave usa para justificar que la cifra se escriba una sola vez, así
    que se mide aquí y no se escribe a mano en ninguna parte."""
    listas = [d["valores"] for d in bandas.values()]
    return max(max(c) - min(c) for c in zip(*listas))


def destaca_la_primera(valores):
    """¿La primera barra sobresale de su frase? Es la afirmación que sostiene la llamada
    de «la papelera», y se mide: llegar al doble de la barra mediana."""
    ordenados = sorted(valores)
    n = len(ordenados)
    mediana = ordenados[n // 2] if n % 2 else (ordenados[n // 2 - 1] + ordenados[n // 2]) / 2
    return valores[0] >= FACTOR_DESTACA * mediana, mediana


def dibujar(bandas, salida=SALIDA):
    rotulos = rotulos_de(bandas)
    n_palabras = len(rotulos)
    n_frases = len(bandas)
    alto_barra = 0.72 / n_frases

    fig, ax = plt.subplots(figsize=ANCHO_ALTO)
    fig.subplots_adjust(left=0.185, right=0.985, top=0.885, bottom=0.330)

    for j, datos in enumerate(bandas.values()):
        # Dentro de cada palabra, la primera frase arriba y la última abajo.
        desplazamiento = (j - (n_frases - 1) / 2) * alto_barra
        ax.barh([k + desplazamiento for k in range(n_palabras)], datos["valores"],
                height=alto_barra * 0.68,
                color=[color_de(p) for p in datos["palabras"]],
                edgecolor="black", linewidth=0.4, zorder=3)

    # La cifra se escribe una sola vez por palabra, y va a la altura de la barra que
    # nombra —la de arriba, la de la frase del «alto»—, no en medio del grupo: en medio
    # se lee como si fuera de la barra del «bajo», que es otra.
    primera = next(iter(bandas.values()))["valores"]
    for k, v in enumerate(primera):
        ax.text(v + 1.0, k - alto_barra, coma(v, 1), va="center", ha="left",
                fontsize=CUERPO_CIFRA, color="0.2", zorder=4)

    ax.set_yticks(range(n_palabras))
    ax.set_yticklabels(rotulos, fontsize=CUERPO_PALABRA)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.invert_yaxis()
    ax.set_ylim(n_palabras - 0.45, -0.55)
    ax.set_xlim(0, TOPE_EJE)
    ax.set_xticks(MARCAS)
    ax.xaxis.set_tick_params(labelsize=CUERPO_EJE)
    ax.set_xlabel(EJE_X, fontsize=CUERPO_EJE, labelpad=5)
    ax.grid(axis="x", color=GRIS_REJILLA, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)

    # Las dos llamadas. Van con línea de guía porque las dos palabras que nombran son
    # las de las barras más largas y el sitio libre está lejos de su punta.
    ax.annotate(LLAMADA_PAPELERA, xy=(31.0, 0.26), xytext=(24.0, 1.05),
                fontsize=CUERPO_LLAMADA, color="0.15", ha="left", va="center", zorder=5,
                arrowprops=dict(arrowstyle="-", color="0.45", lw=0.8,
                                shrinkA=2.0, shrinkB=2.0))
    # La guía de esta llamada baja recta desde el texto hasta el LOMO de la primera de
    # las tres barras del adjetivo, no hasta su punta: en la punta está la cifra, y una
    # guía que pasa por encima de un número lo tacha. Es el mismo cuidado que costó la
    # figura del capítulo 4.
    ultima = n_palabras - 1
    ax.annotate(LLAMADA_ADJETIVO, xy=(18.0, ultima - alto_barra - 0.10),
                xytext=(13.0, ultima - 1.22),
                fontsize=CUERPO_LLAMADA, color="0.15", ha="left", va="center", zorder=5,
                arrowprops=dict(arrowstyle="-", color="0.45", lw=0.8,
                                shrinkA=2.0, shrinkB=1.0))

    fig.text(0.5, 0.975, TITULO, ha="center", va="top", fontsize=CUERPO_TITULO)
    fig.text(0.5, 0.028, CLAVE.format(mayor=coma(mayor_diferencia(bandas), 1)),
             ha="center", va="bottom", fontsize=CUERPO_CLAVE, color="0.3",
             linespacing=1.5)
    fig.savefig(salida, dpi=PUNTOS)
    plt.close(fig)


def informe(bandas):
    """Lo que el programa dice al terminar: los mismos números que dibuja, escritos para
    poder cotejarlos con la tabla del capítulo sin abrir la imagen."""
    lineas = [f"escrito {SALIDA}", f"leído {CSV}"]
    for frase, d in bandas.items():
        lineas.append(f"«…demasiado {frase}»: {d['miradas']} miradas promediadas, "
                      f"suman {coma(sum(d['valores']), 1)} %")
    lineas.append(f"mayor diferencia entre las tres frases, palabra a palabra: "
                  f"{coma(mayor_diferencia(bandas), 1)} puntos")
    rotulos = rotulos_de(bandas)
    ancho = max(len(p) for p in rotulos)
    cabecera = f"{'palabra':<{ancho}}" + "".join(f"{f'«{f}»':>10}" for f in bandas)
    lineas += ["", "porción media del reparto del adjetivo, en por ciento,",
               "sobre las 336 miradas del modelo:", "",
               cabecera, "-" * len(cabecera)]
    for k, palabra in enumerate(rotulos):
        fila = f"{palabra:<{ancho}}"
        for d in bandas.values():
            fila += f"{coma(d['valores'][k], 1):>10}"
        lineas.append(fila)
    return lineas


# ============================ SELFTEST ============================

def filas_inventadas(reparto):
    """Un CSV de mentira, con las palabras de la frase del capítulo y las porciones que
    se le pidan. Sirve para el test nulo y para la señal implantada: la figura tiene que
    dibujar lo que le den, y hay que poder comprobarlo sin volver a medir nada."""
    palabras = FRASES["alto"].rstrip(".").split()
    filas = []
    for capa in (1, 2):
        for mirada in (1, 2):
            for palabra, porcion in zip(palabras, reparto):
                filas.append({"adjetivo": "alto", "capa": str(capa),
                              "mirada": str(mirada), "palabra": palabra.lower(),
                              "porcion": f"{porcion:.6f}"})
    return filas


def selftest():
    fallos = []
    bandas = promedios(leer_filas())
    n = len(FRASES["alto"].rstrip(".").split())

    # [1] Test nulo: un CSV inventado donde las once palabras se reparten por igual. Si
    #     la figura tuviera cualquier adorno que alargara una barra por su cuenta, o si
    #     la llamada de «la papelera» estuviera puesta a mano en vez de medida, aquí
    #     saldría una barra más larga que las demás, o la primera destacaría igual.
    plano = promedios(filas_inventadas([1.0 / n] * n))["alto"]["valores"]
    desvio = max(plano) - min(plano)
    destaca_plano, _ = destaca_la_primera(plano)
    destaca_real, mediana_real = destaca_la_primera(bandas["alto"]["valores"])
    print(f"[1] test nulo         reparto plano: las {n} barras miden todas "
          f"{coma(min(plano), 2)} (desvío {coma(desvio, 2)}, tope "
          f"{coma(DESVIO_MAXIMO_NULO, 2)}) y la primera no destaca: "
          f"{'sí destaca' if destaca_plano else 'correcto'}.")
    print(f"                      medido de verdad, la primera sí destaca: "
          f"{coma(bandas['alto']['valores'][0], 1)} contra una mediana de "
          f"{coma(mediana_real, 1)}")
    if desvio > DESVIO_MAXIMO_NULO:
        fallos.append(f"test nulo: con un reparto plano las barras tendrían que salir "
                      f"iguales, y se diferencian en {coma(desvio, 2)} puntos")
    if destaca_plano:
        fallos.append("test nulo: con un reparto plano la primera barra destaca, así que "
                      "la llamada de «la papelera» no la estaría poniendo la medición")
    if not destaca_real:
        fallos.append(f"test nulo: sobre los datos de verdad la primera barra NO destaca "
                      f"({coma(bandas['alto']['valores'][0], 1)} contra una mediana de "
                      f"{coma(mediana_real, 1)}) y la figura la llama «la papelera»")

    # [2] Señal implantada: un CSV donde «vaso» se lleva el 90 %. La barra más larga del
    #     cuadro tiene que ser la de «vaso», y tiene que salir en negro, que es lo que
    #     promete la clave.
    palabras_frase = [p.lower() for p in FRASES["alto"].rstrip(".").split()]
    reparto = [(1.0 - IMPLANTADA_VASO) / (n - 1)] * n
    reparto[palabras_frase.index("vaso")] = IMPLANTADA_VASO
    implantada = promedios(filas_inventadas(reparto))["alto"]
    k_mayor = max(range(n), key=lambda i: implantada["valores"][i])
    palabra_mayor = implantada["palabras"][k_mayor]
    tinta = color_de(palabra_mayor)
    print(f"[2] señal implantada  metido un {coma(100 * IMPLANTADA_VASO, 0)} % sobre "
          f"«vaso»: la barra más larga es «{palabra_mayor}», con "
          f"{coma(implantada['valores'][k_mayor], 1)} %, y sale en "
          f"{'negro' if tinta == NEGRO else 'gris'}")
    if palabra_mayor != "vaso":
        fallos.append(f"señal implantada: se le mete el 90 % a «vaso» y la barra más larga "
                      f"es «{palabra_mayor}»")
    if tinta != NEGRO:
        fallos.append("señal implantada: «vaso» es una de las dos candidatas y la figura no "
                      "la pinta de negro, que es lo que dice su clave")

    # [3] Invariante del dominio: un reparto es un pastel entero. Cada frase tiene que
    #     sumar cien, no puede tener porciones negativas, y tiene que llevar tantas
    #     barras como palabras trae el CSV para esa frase.
    sumas = {f: sum(d["valores"]) for f, d in bandas.items()}
    peor = max(abs(s - 100.0) for s in sumas.values())
    minimo = min(min(d["valores"]) for d in bandas.values())
    barras = {f: len(d["palabras"]) for f, d in bandas.items()}
    print(f"[3] invariante        {len(bandas)} frases de {sorted(set(barras.values()))[0]} "
          f"barras, cada una promedio de {next(iter(bandas.values()))['miradas']} miradas: "
          f"suman cien (mayor desvío {coma(peor, 3)}) y la porción menor es "
          f"{coma(minimo, 2)} %")
    if peor > TOL_SUMA:
        fallos.append(f"invariante: alguna frase no suma cien; se desvía {coma(peor, 3)} puntos")
    if minimo < 0:
        fallos.append(f"invariante: hay una porción negativa ({coma(minimo, 2)} %)")
    for frase, cuantas in barras.items():
        esperadas = len(FRASES[frase].rstrip(".").split())
        if cuantas != esperadas:
            fallos.append(f"invariante: la frase «{frase}» tiene {cuantas} barras y el CSV "
                          f"trae {esperadas} palabras para ella")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        return selftest()
    bandas = promedios(leer_filas())
    dibujar(bandas)
    for linea in informe(bandas):
        print(linea)
    return 0


if __name__ == "__main__":
    sys.exit(main())
