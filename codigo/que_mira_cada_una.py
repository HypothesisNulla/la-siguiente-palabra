#!/usr/bin/env python3
"""
Capítulo de los conceptos — qué mira cada neurona de una capa.

El libro dice que una capa son varios comités a la vez, cada uno con una
pregunta distinta sobre lo mismo, y que nadie les reparte las preguntas. Esto lo
mide: entrena una red con una capa de en medio sobre dígitos manuscritos de verdad,
y luego mira, una por una, qué ha acabado mirando cada neurona de esa capa y cuánto
acierta ella sola.

La afirmación que sostiene el capítulo es que **ninguna de ellas es la respuesta**:
la red entera acierta más que cualquiera de sus neuronas de en medio por separado.
Eso lo comprueba el selftest.

Uso:
    python que_mira_cada_una.py
    python que_mira_cada_una.py --selftest
"""

# ======================= CONSTANTES =======================

# La pregunta es «¿este dígito es par?». Se elige a propósito porque una raya sola se
# queda corta con ella: aquí sale 91,9 %, y con una capa de en medio, 97,9 %, las dos
# como media de cinco entrenamientos desde cero.
# OJO con la justificación vieja, que estaba escrita aquí y era FALSA: decía que no hay
# ninguna forma que compartan el 0, el 2, el 4, el 6 y el 8 y que no tenga ninguno de
# los impares, y que esto «es el o exclusivo del capítulo anterior escrito a mano».
# `siete_segmentos.py` la pone a prueba en un reloj digital y la tumba: allí una sola
# raya separa par de impar, diez de diez. Y el o exclusivo es IMPOSIBLE para una raya,
# mientras que aquí una raya llega al 91,9 %: no es lo mismo imposible que difícil.
EN_MEDIO = 8                   # neuronas en la capa de en medio
LADO = 8                       # los dibujos son de ocho puntos por ocho
SEMILLA = 20260916
TASA = 0.5
PASOS = 20_000
FRACCION_PRUEBA = 0.3
UMBRAL = 0.5

ANCHURAS = [1, 2, 4, 8, 16]    # cuántas neuronas de en medio, para ver de dónde sale la mejora
REPETICIONES = 5               # arranques distintos por tamaño: una sola tirada oscila
# Los rótulos de las tablas. Son texto del libro: van aquí arriba, con nombre, y no
# escondidos dentro de un print a mitad del fichero.
TITULO_1 = "¿ES PAR ESTE DÍGITO ESCRITO A MANO?"
SUBTITULO_1A = "de cada 100 dígitos que nunca había visto, cuántos acierta"
SUBTITULO_1B = "media de {repeticiones} entrenamientos desde cero; entre paréntesis,"
SUBTITULO_1C = "el peor y el mejor de los {repeticiones}"
TITULO_2 = "LO QUE ACIERTA, ÉL SOLO, CADA UNO DE LOS OCHO COMITÉS DE LA PRIMERA CAPA"
TITULO_3 = "CUÁNTO SE PARECEN ENTRE SÍ ESOS OCHO COMITÉS"
SUBTITULO_3 = "(0 % serían ocho preguntas sin nada en común; 100 %, ocho copias)"
TITULO_4 = "¿Y SI EN LA PRIMERA CAPA HAY MÁS COMITÉS, O MENOS?"
SUBTITULO_4A = "media de {repeticiones} entrenamientos desde cero; entre paréntesis,"
SUBTITULO_4B = "el peor y el mejor de los {repeticiones}"

# L24 (28 de septiembre): tres bloques nuevos al final de la salida, sin tocar lo de arriba.
TITULO_5 = "LO QUE MEJORAN LAS DOS CAPAS, SEGÚN QUÉ DOS SE COMPAREN"
SUBTITULO_5 = "(en puntos: de cada 100 dígitos, cuántos más acierta)"
TITULO_6 = "CADA COMITÉ DE LA PRIMERA CAPA, LEÍDO DE LAS DOS MANERAS"
SUBTITULO_6 = "(de cada 100 dígitos que nunca había visto, cuántos acierta)"
TITULO_7 = "CUÁNTO SE PARECEN, CON LA CUENTA DE LOS PESOS DEL CUATRO"
SUBTITULO_7 = ["(de -1 a 1: 1, iguales salvo el tamaño de los números;",
               "0, nada que ver; -1, uno es el negativo del otro)"]

SALIDA_CSV = "que_mira_cada_una.csv"

# ==========================================================

import argparse
import csv
import sys

from formato import coma, pct, tabla_editorial
import numpy as np

from perceptron import cargar_digitos
from retropropagacion import Red


def datos(semilla=SEMILLA, permutar=False):
    Xs, t = cargar_digitos()
    ys = np.where(t % 2 == 0, 1.0, 0.0)
    rng = np.random.default_rng(semilla)
    if permutar:
        ys = rng.permutation(ys)
    idx = rng.permutation(len(Xs))
    corte = int(len(Xs) * (1 - FRACCION_PRUEBA))
    return Xs[idx[:corte]], ys[idx[:corte]], Xs[idx[corte:]], ys[idx[corte:]]


def comprobar(Xtr, ytr):
    assert Xtr.shape[1] == LADO * LADO, \
        f"Se esperaban {LADO*LADO} puntos por dibujo; se encontraron {Xtr.shape[1]}"
    assert set(np.unique(ytr)) <= {0.0, 1.0}, \
        f"Se esperaban respuestas de sí o no; se encontró {sorted(set(np.unique(ytr)))}"
    assert Xtr.min() >= 0.0 and Xtr.max() <= 1.0, \
        f"Se esperaban tonos entre 0 y 1; se encontró de {Xtr.min():.2f} a {Xtr.max():.2f}"


def entrenar_con_capa(semilla=SEMILLA, en_medio=EN_MEDIO, permutar=False):
    Xtr, ytr, Xte, yte = datos(semilla, permutar)
    comprobar(Xtr, ytr)
    red = Red([LADO * LADO, en_medio, 1], semilla=semilla).entrenar(Xtr, ytr, TASA, PASOS)
    act = red.adelante(Xte)
    entera = float((((act[-1].ravel() > UMBRAL) == (yte > UMBRAL)).mean()))
    return {"red": red, "Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte,
            "medio": act[1], "entera": entera}


def una_raya_sola(semilla=SEMILLA):
    """La misma pregunta con un solo comité: sin capa de en medio. Es el capítulo
    anterior aplicado a esto, y es la comparación que da sentido a la capa."""
    Xtr, ytr, Xte, yte = datos(semilla)
    red = Red([LADO * LADO, 1], semilla=semilla).entrenar(Xtr, ytr, TASA, PASOS)
    dice = red.adelante(Xte)[-1].ravel() > UMBRAL
    return float((dice == (yte > UMBRAL)).mean())


def acierto_de_cada_una(m):
    """Cuánto acierta cada neurona de en medio ELLA SOLA, mirando solo su respuesta.

    El sentido de cada neurona es arbitrario —nadie le dijo cuál es el sí—, así que se
    le concede el mejor de los dos sentidos. Es concederle ventaja a propósito: si aun
    así acierta menos que la red entera, la afirmación del capítulo aguanta."""
    aciertos = []
    for j in range(m["medio"].shape[1]):
        dice = m["medio"][:, j] > UMBRAL
        a = float((dice == (m["yte"] > UMBRAL)).mean())
        aciertos.append(max(a, 1.0 - a))
    return np.array(aciertos)


def parecido_maximo(red):
    """Lo más que se parecen dos de las neuronas de en medio entre sí, de 0 a 1.

    Si salieran todas iguales, la capa sería una sola neurona repetida ocho veces."""
    W = red.W[0]
    n = W / np.linalg.norm(W, axis=0, keepdims=True)
    s = np.abs(n.T @ n)
    np.fill_diagonal(s, 0.0)
    return float(s.max())


def barrido_de_anchura(semilla=SEMILLA):
    """La misma pregunta con capas de en medio de distinto tamaño. Sirve para ver que
    la mejora no viene de «tener una capa» sino de tener VARIAS neuronas en ella: con
    una sola no se gana nada respecto a no tener capa.

    Cada tamaño se entrena varias veces desde cero, porque una sola tirada oscila lo
    bastante como para dibujar una curva que no existe. Se da la media y el margen."""
    fila = []
    for n in ANCHURAS:
        a = [entrenar_con_capa(semilla + 1000 * k, n)["entera"] for k in range(REPETICIONES)]
        fila.append((n, float(np.mean(a)), float(np.min(a)), float(np.max(a))))
    return fila


def media_de_varias(funcion, semilla=SEMILLA):
    """La media de REPETICIONES entrenamientos desde cero, y el peor y el mejor.

    El titular del capítulo era una sola tirada, y resultó ser LA PEOR de las cinco que ya
    calculaba el barrido de más abajo: el margen de la fila de ocho empezaba justo en ese número.
    La mejora parecía de seis puntos por casualidad: comparaba una peor con otra peor. Un entrenamiento no da
    un número, da un margen, y las dos cifras que el capítulo compara tienen que medir lo mismo.

    Se usan las mismas semillas que barrido_de_anchura, para que las dos tablas del capítulo
    hablen de los mismos cinco entrenamientos y no de dos conjuntos parecidos."""
    a = [funcion(semilla + 1000 * k) for k in range(REPETICIONES)]
    return float(np.mean(a)), float(np.min(a)), float(np.max(a))


EN_LETRA = {5: "cinco", 8: "ocho"}   # el libro escribe «ocho comités» y «cinco veces», con letra


def frase(titulo):
    """«¿ES PAR ESTE DÍGITO…?» -> «¿Es par este dígito…?»: los títulos de las tablas de libro van
    en minúscula con la primera letra en mayúscula (el filtro los pone en versalitas)."""
    t = titulo.lower()
    k = next(i for i, c in enumerate(t) if c.isalpha())
    return t[:k] + t[k].upper() + t[k + 1:]


def nota_media(filas):
    """La nota de las dos tablas de medias («¿Es par…?» y «¿Y si en la primera capa…?»); `filas`
    es cómo se dicen las filas que comparten semillas: «las dos filas» o «todas las filas»."""
    cinco = EN_LETRA[REPETICIONES]
    return ("Cuántos dígitos acierta de cada 100 que nunca había visto. Cada "
            f"máquina se entrenó {cinco} veces desde cero, con {cinco} semillas distintas, las "
            f"mismas en {filas}.")


def imprimir(m, aciertos, parecido, sola, con_capa):
    """Las tablas que cita el capítulo 3. Desde el 9 oct 2026, tablas de libro (formato.py;
    REGLAS 6 ter): cada cifra en su columna, con su rótulo."""
    def media_peor_mejor(t):
        return [pct(t[0]), pct(t[1]), pct(t[2])]

    T = []
    T.append(tabla_editorial(
        frase(TITULO_1), ["", "media", "el peor", "el mejor"],
        [["una capa de un solo comité"] + media_peor_mejor(sola),
         [f"dos capas: {EN_LETRA.get(EN_MEDIO, EN_MEDIO)} comités en la primera"]
         + media_peor_mejor(con_capa)], "iddd",
        [nota_media("las dos filas")]))
    filas = [[f"el comité número {j}", pct(a)] for j, a in enumerate(aciertos, 1)]
    filas.append(["**el mejor de los ocho**", f"**{pct(aciertos.max())}**"])
    T.append(tabla_editorial(
        frase(TITULO_2), ["", "acierta"], filas, "id",
        ["De cada 100 dígitos que nunca había visto, cuántos acierta cada comité de la primera "
         "capa, él solo, con la lectura que mejor le va."]))
    T.append(tabla_editorial(
        frase(TITULO_3), ["", "parecido"], [["los dos que más se parecen", pct(parecido)]],
        "id", ["0 % serían ocho preguntas sin nada en común; 100 %, ocho copias."]))
    filas = [["ninguno: una capa de un solo comité"] + media_peor_mejor(sola)]
    for n, med, lo, hi in barrido_de_anchura():
        # «1 comité» y «2 comités»: el plural se dice bien o no se dice.
        filas.append([f"{n} comité" + ("s" if n > 1 else "") + " en la primera capa",
                      pct(med), pct(lo), pct(hi)])
    T.append(tabla_editorial(
        frase(TITULO_4), ["comités en la primera capa", "media", "el peor", "el mejor"],
        filas, "iddd", [nota_media("todas las filas")]))
    for t in T:
        print("\n".join(t))
        print()


def parecido_con_signo(red):
    """La cuenta de parecido del capítulo 3 (la de los pesos del cuatro contra la diferencia,
    `el_comite_por_dentro.parecido`): la correlación, de -1 a 1. Devuelve la pareja que más se
    parece sin mirar el signo, y su valor CON signo. Sin mirar el signo porque a cada comité se
    le deja elegir cuál de sus dos extremos es el «sí» (acierto_de_cada_una): dos comités que
    son uno el negativo del otro hacen la misma pregunta al revés."""
    W = red.W[0]
    C = np.corrcoef(W.T)
    np.fill_diagonal(C, 0.0)
    a, b = np.unravel_index(int(np.argmax(np.abs(C))), C.shape)
    a, b = sorted((int(a), int(b)))
    return a, b, float(C[a, b])


def imprimir_l24(m, sola, con_capa):
    """Los tres bloques del repaso L24 del capítulo 3 (hallazgos A01, A02 y A03), en tablas de
    libro desde el 9 oct 2026."""
    (sm, sp, sb), (cm, cp, cb) = sola, con_capa
    filas = [[rotulo, coma(100 * de), coma(100 * a), coma(100 * (a - de))]
             for rotulo, de, a in (("las dos medias de 5", sm, cm),
                                   ("la mejor con una capa y la peor con dos", sb, cp),
                                   ("la peor con una capa y la mejor con dos", sp, cb))]
    T = [tabla_editorial(
        frase(TITULO_5), ["qué dos se comparan", "una capa", "dos capas", "mejora"], filas,
        "iddd", ["Cada número es cuántos dígitos acierta de cada 100 que nunca había visto. La "
                 "columna «mejora» es cuántos más acierta con dos capas que con una, en puntos."])]
    filas = []
    for j in range(m["medio"].shape[1]):
        a = float(((m["medio"][:, j] > UMBRAL) == (m["yte"] > UMBRAL)).mean())
        filas.append([f"el comité {j + 1}", pct(a), pct(1 - a), pct(max(a, 1 - a))])
    T.append(tabla_editorial(
        frase(TITULO_6),
        ["", "si pasar de la mitad es: «par»", "si pasar de la mitad es: «impar»", "se le cuenta"],
        filas, "iddd",
        ["Cada número es cuántos dígitos acierta de cada 100 que nunca había visto. Como nadie ha "
         "fijado qué extremo es el «sí», cada comité se lee de las dos maneras, y la última "
         "columna le cuenta la que mejor le va."]))
    a, b, r = parecido_con_signo(m["red"])
    T.append(tabla_editorial(
        frase(TITULO_7),
        ["los dos que más se parecen", "parecido"],
        [[f"el comité {a + 1} y el comité {b + 1}", coma(r, 2)]], "id",
        ["El parecido va de −1 a 1: 1 quiere decir que los dos son iguales salvo el tamaño de los "
         "números; 0, que no tienen nada que ver; y −1, que uno es el otro en negativo. El "
         "negativo también cuenta como parecido, porque es la misma pregunta al revés."]))
    for t in T:
        print("\n".join(t))
        print()
    return a, b, r


def guardar(m, aciertos, parecido, sola, con_capa):
    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["medicion", "cual", "valor"])
        # Las dos cifras del titular son (media, peor, mejor) de REPETICIONES tiradas: se guardan
        # las tres, porque guardar solo la media vuelve a esconder que un entrenamiento es un
        # margen y no un número, que es justo lo que este capítulo acaba de aprender.
        for nombre, (media, peor, mejor) in (("una_raya_sola", sola),
                                             ("con_capa", con_capa)):
            w.writerow([nombre, "media", f"{media:.4f}"])
            w.writerow([nombre, "peor", f"{peor:.4f}"])
            w.writerow([nombre, "mejor", f"{mejor:.4f}"])
        w.writerow(["red_entera", "", f"{m['entera']:.4f}"])
        for j, a in enumerate(aciertos, 1):
            w.writerow(["una_sola", j, f"{a:.4f}"])
        w.writerow(["parecido_maximo", "", f"{parecido:.4f}"])


# ============================ SELFTEST ============================

def selftest():
    fallos = []
    m = entrenar_con_capa()
    aciertos = acierto_de_cada_una(m)

    # [1] Test nulo: con las etiquetas barajadas no hay nada que aprender y la red
    #     entera tiene que caerse al azar. Si no se cae, los mapas que dibujaríamos
    #     no significarían nada.
    nulo = entrenar_con_capa(permutar=True)
    print(f"[1] test nulo         etiquetas barajadas: la red acierta "
          f"{100 * nulo['entera']:.1f} % (azar = 50 %); de verdad: {100 * m['entera']:.1f} %")
    if nulo["entera"] > 0.75:
        fallos.append(f"test nulo: con etiquetas barajadas acierta {100*nulo['entera']:.1f} %, "
                      f"y no debería pasar del azar")

    # [2] Señal implantada: se añade un punto que delata la respuesta siempre. Alguna
    #     neurona de en medio tiene que acabar mirándolo más que a ningún otro punto.
    Xtr, ytr, Xte, yte = datos()
    chivato = np.zeros((len(Xtr), 1)); chivato[ytr > UMBRAL] = 1.0
    Xi = np.hstack([Xtr, chivato])
    red = Red([Xi.shape[1], EN_MEDIO, 1], semilla=SEMILLA).entrenar(Xi, ytr, TASA, PASOS)
    peso_chivato = np.abs(red.W[0][-1, :])
    mayor = np.abs(red.W[0]).max(axis=0)
    cuantas = int((peso_chivato >= mayor - 1e-12).sum())
    print(f"[2] señal implantada  un punto que delata la respuesta: "
          f"{cuantas} de {EN_MEDIO} neuronas lo miran más que a ningún otro")
    if cuantas == 0:
        fallos.append("señal implantada: puse un punto que delata la respuesta y ninguna "
                      "neurona de en medio acabó mirándolo por encima del resto")

    # [3] Invariante del dominio: la red entera tiene que acertar más que cualquiera de
    #     sus neuronas de en medio por separado. Ésta es la afirmación del capítulo: que
    #     ninguna de ellas es la respuesta, y que juntas sí.
    mejor = float(aciertos.max())
    sola = una_raya_sola()
    print(f"[3] invariante        la red entera {100 * m['entera']:.1f} %; la mejor de sus "
          f"neuronas sola {100 * mejor:.1f} %; sin capa {100 * sola:.1f} %")
    if m["entera"] <= mejor:
        fallos.append(f"invariante: la red entera acierta {100*m['entera']:.1f} % y una sola de "
                      f"sus neuronas {100*mejor:.1f} %; entonces la capa no está aportando nada")
    if m["entera"] <= sola:
        fallos.append(f"invariante: con capa {100*m['entera']:.1f} % y sin capa {100*sola:.1f} %; "
                      f"entonces esta tarea no necesita capa y el capítulo no se sostiene")

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
    a = p.parse_args()
    if a.selftest:
        return selftest()
    m = entrenar_con_capa()
    aciertos = acierto_de_cada_una(m)
    parecido = parecido_maximo(m["red"])
    sola = media_de_varias(una_raya_sola)
    con_capa = media_de_varias(lambda sem: entrenar_con_capa(sem, EN_MEDIO)["entera"])
    imprimir(m, aciertos, parecido, sola, con_capa)
    imprimir_l24(m, sola, con_capa)
    guardar(m, aciertos, parecido, sola, con_capa)
    return 0


if __name__ == "__main__":
    sys.exit(main())
