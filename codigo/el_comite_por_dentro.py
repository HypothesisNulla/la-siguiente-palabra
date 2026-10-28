#!/usr/bin/env python3
"""
Capítulo 3 — el comité real, por dentro, con todas sus cuentas (L16, T12; Orden 41, 23 oct 2026).

El capítulo abre por dentro el MISMO comité de la tabla «¿Es un cuatro?» del capítulo 2: un solo
comité que decide si un dibujo es un cuatro, con los otros nueve dígitos enfrente, y aprende con
1.257 dibujos y se evalúa con 540 que no vio. Este programa imprime todo lo que hace falta para
hacer sus cuentas con el dedo:

  1. el dibujo de un cuatro, punto a punto, con su tinta (de 0 a 16);
  2. los 64 casos (pesos) que aprendió el comité, uno por punto, y el listón;
  3. la cuenta de tres puntos, la de los que más suman y más restan, el total y el listón, y el
     total de un 1, que es el dígito que el comité más veces tomó por cuatro mientras aprendía;
  4. el cuatro medio, la media de los otros nueve dígitos y su diferencia, y cuánto se parecen los
     pesos a cada cosa;
  5. de dónde salen los pesos: las correcciones una a una, de qué dígito eran, y cómo cada peso y
     el listón son una suma de tinta.

El comité es el de `es_un_cuatro.py`, sección 1: el mismo reparto de los dibujos, la misma semilla,
el mismo tope de correcciones. El selftest comprueba que da las mismas correcciones y el mismo
acierto que imprime `datos/salidas/es_un_cuatro.txt`. (Hasta la Orden 40 el capítulo abría un comité
«¿un 4 o un 9?», el de `figura_una_neurona.py`; ese programa y su salida no han cambiado.)

LA TINTA, TAL COMO VIENE (Orden 38). El comité se entrena con la tinta tal como la da el conjunto de
dígitos, enteros de 0 a 16, sin dividir entre 16. Así la regla de Rosenblatt es literalmente la del
capítulo 2: si el comité no reconoce un cuatro, se suma a cada miembro la tinta que ve y el listón
baja un punto; si toma otro dígito por un cuatro, se resta y el listón sube un punto. Y cada peso es,
por construcción, la tinta de ese punto en los cuatros corregidos menos la de los otros dígitos
corregidos: el apartado 5 lo cuenta y el selftest lo comprueba en los 64 puntos.

Uso:
    python el_comite_por_dentro.py --selftest
    python el_comite_por_dentro.py > ../datos/salidas/el_comite_por_dentro.txt
"""

# ======================= CONSTANTES =======================

LADO = 8
PUNTOS_EN_TABLA = 5            # cuántos de los que más suman y de los que más restan se enseñan
EN_LETRA = {5: "cinco"}        # la nota de esa tabla lo dice con letra (Orden 50)
TINTA_ALTA = 8                 # «mucha tinta»: de 8 a 16, la mitad de arriba (apartado 5)
ES_UN_CUATRO = "../datos/salidas/es_un_cuatro.txt"    # donde el capítulo 2 imprime este comité

# Selftest
UMBRAL_NULO = 0.65             # acierto equilibrado máximo con las respuestas barajadas
PARECIDO_NULO_MAX = 0.3        # ni sus casos parecerse a la diferencia de verdad
PASOS_NULO = 20_000            # correcciones máximas del comité con las respuestas barajadas
PUNTO_IMPLANTADO = 27          # fila 4, columna 4

# ==========================================================

import argparse
import platform
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

from formato import ANCHO_CAJA, coma, comprobar_ancho, miles, pct, tabla_editorial
from es_un_cuatro import (DIGITO_DEL_CAPITULO, PASOS_MAX, acierto_equilibrado, es_este_digito,
                          reparto)
from perceptron import SEMILLA, TASA, acierto, cargar_digitos, entrenar

AQUI = Path(__file__).resolve().parent
DIGITO_SI = DIGITO_DEL_CAPITULO


def el_comite(permutar=False, pasos_max=PASOS_MAX):
    """El comité de `es_un_cuatro.py` (sección 1): ¿es un cuatro?, con los otros nueve dígitos
    enfrente. Mismo reparto, misma semilla, mismo tope de correcciones. Con permutar=True se le
    barajan las respuestas (el test nulo). Devuelve sus pesos y listón, y los dibujos de aprender
    y de evaluar con su dígito."""
    X, t = cargar_digitos(enteros=True)
    X = np.round(X).astype(int)
    y = np.where(t == DIGITO_SI, 1, -1)
    rng = np.random.default_rng(SEMILLA)
    tr, te = reparto(len(X), rng)
    y_aprender = rng.permutation(y[tr]) if permutar else y[tr]
    w, b, pasos = entrenar(X[tr].astype(float), y_aprender, pasos_max=pasos_max)
    return {"w": w, "b": b, "pasos": pasos, "X": X, "y": y, "t": t, "tr": tr, "te": te,
            "entrenamiento": (X[tr], y_aprender, t[tr]), "prueba": (X[te], y[te], t[te]),
            "acierto": acierto(X[te].astype(float), y[te], w, b)}


def lo_que_dice_es_un_cuatro(ruta=ES_UN_CUATRO):
    """Del capítulo 2 (`es_un_cuatro.txt`): (correcciones, acierto en tanto por ciento con una
    cifra decimal), tal como están impresos en la tabla «¿Es un cuatro?»."""
    texto = (AQUI / ruta).read_text(encoding="utf-8")
    c = re.search(r"\| correcciones hasta quedarse quieto \| ([\d.]+) \|", texto)
    a = re.search(r"\| acierta sobre los dibujos que no vio \| ([\d,]+) % \|", texto)
    assert c and a, f"se esperaban las filas de la tabla «¿Es un cuatro?» en {ruta}"
    return int(c.group(1).replace(".", "")), a.group(1)


def en_enteros(m):
    """Los casos (pesos), la tinta y el listón del comité, que salen enteros sin tocarlos: la
    tinta es entera y la regla suma o resta tinta entera y un punto de listón cada vez."""
    casos = m["w"]
    assert np.array_equal(casos, np.round(casos)), \
        "se esperaban pesos enteros; la regla de perceptron.py o la tinta han cambiado"
    tinta = m["X"]
    assert tinta.min() == 0 and tinta.max() <= 16, \
        "se esperaba tinta entera de 0 a 16; el conjunto de dígitos ha cambiado"
    liston = -m["b"]
    assert liston == round(liston), "se esperaba un listón entero"
    return np.round(casos).astype(int), tinta, int(round(liston))


def correcciones(m):
    """Las correcciones del entrenamiento, una a una: se repite el aprendizaje de
    `perceptron.entrenar` (la misma semilla, el mismo orden) apuntando cada vez que corrige. Devuelve
    la lista de (vuelta, dibujo, +1 si era un cuatro y -1 si era otro dígito) y cuántas vueltas dio,
    contando la última, en la que ya no hubo nada que corregir. Se comprueba que acaba en los mismos
    pesos y el mismo listón que el comité de verdad."""
    X, y, _ = m["entrenamiento"]
    rng = np.random.default_rng(SEMILLA)
    w, b = np.zeros(X.shape[1], dtype=int), 0
    orden = np.arange(len(X))
    apuntes = []
    for vuelta in range(1, PASOS_MAX + 1):
        rng.shuffle(orden)
        errores = 0
        for i in orden:
            if y[i] * (X[i] @ w + b) <= 0:
                assert TASA == 1.0, "se esperaba la tasa de la regla del capítulo 2 (1)"
                w += y[i] * X[i]
                b += y[i]
                apuntes.append((vuelta, int(i), int(y[i])))
                errores += 1
        if errores == 0:
            break
    assert errores == 0, "se esperaba que el comité dejara de corregirse"
    assert np.array_equal(w, m["w"]) and b == m["b"], \
        "las correcciones repetidas no dan los pesos y el listón del comité"
    return apuntes, vuelta


def ejemplos(m):
    """Un cuatro que el comité acierta y un 1 que rechaza bien, los primeros del conjunto de
    evaluar (los dibujos que no vio), para que la cuenta no haga trampa. El 1 es el dígito que el
    comité más veces tomó por cuatro mientras aprendía. Devuelve (índices en m["X"], el dígito)."""
    apuntes, _ = correcciones(m)
    _, _, t_aprender = m["entrenamiento"]
    X, y, _ = m["entrenamiento"]
    de_otro = Counter(int(t_aprender[i]) for _, i, s in apuntes if s == -1)
    rival = max(sorted(de_otro), key=lambda d: de_otro[d])         # con empate, el menor
    Xp, yp, tp = m["prueba"]
    total = Xp @ m["w"] + m["b"]
    i_si = next(i for i in range(len(yp)) if yp[i] == 1 and total[i] > 0)
    i_no = next(i for i in range(len(yp)) if tp[i] == rival and total[i] <= 0)
    return m["te"][i_si], m["te"][i_no], rival


def lugar(k):
    """El punto k (de 0 a 63), dicho como se busca en la cuadrícula: fila y columna, desde 1."""
    return f"fila {k // LADO + 1}, columna {k % LADO + 1}"


def cuadricula(valores, ancho=5):
    """Ocho filas de ocho números, con la fila y la columna numeradas."""
    lineas = ["      " + "".join(f"{c:>{ancho}}" for c in range(1, LADO + 1))]
    for f in range(LADO):
        fila = valores[f * LADO:(f + 1) * LADO]
        lineas.append(f"  {f + 1:>2}  " + "".join(f"{int(v):>{ancho}}" for v in fila))
    return lineas


def parecido(a, b):
    """Cuánto se parecen dos cuadrículas: la correlación, de −1 a 1 (1: iguales salvo escala)."""
    return float(np.corrcoef(a, b)[0, 1])


def mismo_lado(casos, diferencia):
    """En cuántos puntos, de los que tienen caso y diferencia distintos de cero, empujan los dos
    hacia el mismo lado."""
    vivos = (casos != 0) & (diferencia != 0)
    return int((np.sign(casos[vivos]) == np.sign(diferencia[vivos])).sum()), int(vivos.sum())


def medias(m):
    """La tinta media de los cuatros y la de los otros nueve dígitos, sobre los dibujos con los que
    aprendió el comité, y cuántos dibujos hay de cada."""
    X, y, _ = m["entrenamiento"]
    return X[y == 1].mean(0), X[y == -1].mean(0), int((y == 1).sum()), int((y == -1).sum())


def selftest():
    fallos = []
    m = el_comite()
    casos, tinta, liston = en_enteros(m)
    y_ap, t_ap = m["entrenamiento"][1], m["entrenamiento"][2]
    media_si, media_no, _, _ = medias(m)
    diferencia = media_si - media_no

    # 0. ES EL COMITÉ DEL CAPÍTULO 2 — las mismas correcciones y el mismo acierto que imprime
    #    `es_un_cuatro.txt`, y que el de `es_un_cuatro.es_este_digito` con su generador.
    corr_cap2, acierto_cap2 = lo_que_dice_es_un_cuatro()
    r = es_este_digito(m["X"].astype(float), m["t"], DIGITO_SI, np.random.default_rng(SEMILLA))
    mi_acierto = coma(100 * m["acierto"], 1)
    igual = (m["pasos"] == corr_cap2 == r["correcciones"] and mi_acierto == acierto_cap2
             and m["acierto"] == r["acierto_prueba"])
    print(f"[0] el del cap. 2     {m['pasos']} correcciones y acierta {mi_acierto} %; "
          f"es_un_cuatro.txt: {corr_cap2} y {acierto_cap2} %; como se espera: {'sí' if igual else 'NO'}")
    if not igual:
        fallos.append("no es el comité de es_un_cuatro.py: "
                      f"{m['pasos']} correcciones y {mi_acierto} % frente a {corr_cap2} y {acierto_cap2} %")

    # 1. TEST NULO — con las respuestas barajadas, el comité no aprende nada que se parezca a la
    #    diferencia entre un cuatro y los demás, ni acierta (equilibrado) más que el azar.
    nulo = el_comite(permutar=True, pasos_max=PASOS_NULO)
    Xp, yp, _ = m["prueba"]
    pred = np.where(Xp @ nulo["w"] + nulo["b"] > 0, 1, -1)
    eq, _, _ = acierto_equilibrado(yp, pred)
    p_nulo = parecido(nulo["w"], diferencia)
    print(f"[1] test nulo         respuestas barajadas: acierto equilibrado {coma(eq, 2)} (azar = 0,5); "
          f"parecido con la diferencia {coma(p_nulo, 2)}")
    if eq > UMBRAL_NULO or abs(p_nulo) > PARECIDO_NULO_MAX:
        fallos.append(f"test nulo: con respuestas barajadas acierta (equilibrado) {eq:.2f} y se "
                      f"parece {p_nulo:.2f} a la diferencia")

    # 2. SEÑAL IMPLANTADA — dos grupos de dibujos que solo se distinguen en un punto: el caso más
    #    grande tiene que caer justo ahí.
    rng = np.random.default_rng(20260926)
    A = rng.uniform(0, 0.3, size=(200, LADO * LADO))
    etiquetas = np.where(np.arange(200) < 100, 1, -1)
    A[etiquetas == 1, PUNTO_IMPLANTADO] += 0.7
    w, b, _ = entrenar(A, etiquetas)
    mayor = int(np.argmax(np.abs(w)))
    print(f"[2] señal implantada  la diferencia está en el punto {PUNTO_IMPLANTADO}; "
          f"el peso más grande, en el {mayor}")
    if mayor != PUNTO_IMPLANTADO:
        fallos.append(f"señal implantada: el peso más grande cae en {mayor}, no en {PUNTO_IMPLANTADO}")

    # 3. INVARIANTE — cada peso es la tinta de ese punto en los cuatros corregidos menos la de los
    #    otros dígitos corregidos, en los 64 puntos, y el listón es los otros corregidos menos los
    #    cuatros (las correcciones repetidas dan los mismos pesos que el comité); el comité decide
    #    como dice su suma en todos los dibujos; y la tabla de la cuenta suma exactamente el total.
    apuntes, vueltas = correcciones(m)
    Xtr = m["entrenamiento"][0]
    de_cuatros = [i for _, i, s in apuntes if s == 1]
    de_otros = [i for _, i, s in apuntes if s == -1]
    pesos_rehechos = Xtr[de_cuatros].sum(0) - Xtr[de_otros].sum(0)
    pesos_bien = int((pesos_rehechos == casos).sum())
    liston_rehecho = len(de_otros) - len(de_cuatros)
    total_ent = tinta @ casos
    decide_ent = total_ent > liston
    decide_dec = (m["X"] @ m["w"] + m["b"]) > 0
    i_si, _, _ = ejemplos(m)
    aporta = tinta[i_si] * casos
    orden = np.argsort(aporta)
    arriba, abajo = orden[::-1][:PUNTOS_EN_TABLA], orden[:PUNTOS_EN_TABLA]
    resto = np.setdiff1d(np.arange(LADO * LADO), np.concatenate([arriba, abajo]))
    cuadra = aporta[arriba].sum() + aporta[abajo].sum() + aporta[resto].sum() == total_ent[i_si]
    print(f"[3] invariante        peso = tinta de los cuatros corregidos menos la de los otros: "
          f"{pesos_bien} de {LADO * LADO} puntos; listón = otros menos cuatros corregidos: "
          f"{liston_rehecho} (el del comité, {liston})")
    print(f"                      decide como dice la suma: {int((decide_ent == decide_dec).sum())} "
          f"de {len(m['y'])} dibujos; la tabla suma el total: {cuadra}")
    if pesos_bien != LADO * LADO:
        fallos.append(f"invariante: solo {pesos_bien} de 64 pesos son la tinta de los cuatros "
                      "corregidos menos la de los otros")
    if liston_rehecho != liston:
        fallos.append(f"invariante: el listón sale {liston_rehecho} de las correcciones y es {liston}")
    if not (decide_ent == decide_dec).all():
        fallos.append("invariante: no decide lo que dice la suma de tinta por peso más el listón")
    if not cuadra:
        fallos.append("invariante: la tabla de la cuenta no suma el total")

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
    print("--- selftest ---")
    codigo = selftest()
    if codigo or args.selftest:
        sys.exit(codigo)

    m = el_comite()
    casos, tinta, liston = en_enteros(m)
    X_ap, y_ap, t_ap = m["entrenamiento"]
    Xp, yp, tp = m["prueba"]
    i_si, i_no, rival = ejemplos(m)
    total = tinta @ casos
    L = [
        "",
        "########## capítulo 3: el comité real, por dentro ##########",
        f"máquina: {platform.system()} {platform.machine()}. Medido el {date.today().isoformat()}.",
        f"La pregunta: ¿es un {DIGITO_SI}? Un solo comité, con los otros",
        "nueve dígitos enfrente. Es el de «¿Es un cuatro?» del capítulo 2.",
        f"Aprende con {miles(len(y_ap))} dibujos ({miles(int((y_ap == 1).sum()))} cuatros) y acierta "
        f"{pct(m['acierto'], 1)}",
        f"de los {miles(len(yp))} que no vio.",
        "La tinta va de 0 a 16, tal como viene en el conjunto de dígitos;",
        "los pesos y el listón salen enteros, sin multiplicar nada.",
        "",
        f"1. EL DIBUJO DE UN {DIGITO_SI}: LA TINTA DE CADA PUNTO (de 0 a 16)",
        "",
    ]
    L += cuadricula(tinta[i_si], 4)
    L += ["", "2. LO QUE APRENDIÓ: EL PESO DE CADA PUNTO", ""]
    L += cuadricula(casos, 6)
    L += ["", f"  el listón: {miles(liston)}. Dice «es un {DIGITO_SI}» si el total pasa de ahí.",
          "", f"3. LA CUENTA DEL {DIGITO_SI}: TINTA POR PESO, PUNTO A PUNTO"]
    # Desde el 9 oct 2026 lo que va al libro (la cuenta y el parecido) sale en tablas de libro
    # (formato.py; REGLAS 6 ter); las cuadrículas, que el libro enseña como figuras, siguen igual.
    for l in comprobar_ancho([l.rstrip() for l in L], ANCHO_CAJA):
        print(l)
    T = []
    aporta = tinta[i_si] * casos
    orden = np.argsort(aporta)
    mas, menos = orden[::-1][0], orden[0]
    sin_tinta = [k for k in np.argsort(-np.abs(casos)) if tinta[i_si][k] == 0][0]
    T.append(tabla_editorial(
        "Tres puntos, para ver cómo va", ["punto", "tinta", "peso", "aporta"],
        [[lugar(k), str(tinta[i_si][k]), miles(int(casos[k])), miles(int(aporta[k]))]
         for k in (mas, menos, sin_tinta)], "iddd",
        ["La última columna, «aporta», es la tinta del punto por su peso."]))
    arriba, abajo = orden[::-1][:PUNTOS_EN_TABLA], orden[:PUNTOS_EN_TABLA]
    resto = np.setdiff1d(np.arange(LADO * LADO), np.concatenate([arriba, abajo]))
    filas = [[lugar(k), str(tinta[i_si][k]), miles(int(casos[k])), miles(int(aporta[k]))]
             for k in list(arriba) + list(abajo[::-1])]
    filas.append([f"los otros {len(resto)} puntos", "", "", miles(int(aporta[resto].sum()))])
    filas.append([f"**total del {DIGITO_SI}**", "", "", f"**{miles(int(total[i_si]))}**"])
    T.append(tabla_editorial(
        f"La cuenta del {DIGITO_SI}, punto a punto", ["punto", "tinta", "peso", "aporta"], filas,
        "iddd",
        [f"Arriba, los {EN_LETRA[PUNTOS_EN_TABLA]} puntos que más suman; debajo, los "
         f"{EN_LETRA[PUNTOS_EN_TABLA]} que más restan; y en una sola fila, lo que suman juntos los "
         f"otros {len(resto)}. El listón es {miles(liston)}: si el total pasa de ahí, el comité "
         f"dice «es un {DIGITO_SI}»."]))
    T.append(tabla_editorial(
        f"Un {DIGITO_SI} y un {rival}, contra el listón",
        ["dibujo", "total", "listón", "¿pasa del listón?"],
        [[f"un {d}", miles(int(total[i])), miles(liston), "sí" if total[i] > liston else "no"]
         for d, i in ((DIGITO_SI, i_si), (rival, i_no))], "iddc",
        [f"La misma cuenta, tinta por peso en los {LADO * LADO} puntos, sumada. El {rival} es el "
         f"dígito que el comité más veces tomó por un {DIGITO_SI} mientras aprendía."]))
    for t in T:
        print()
        print("\n".join(t))

    media_si, media_no, n_si, n_no = medias(m)
    diferencia = media_si - media_no
    iguales, vivos = mismo_lado(casos, np.round(diferencia))
    L = ["", f"4. EL {DIGITO_SI} MEDIO, LA MEDIA DE LOS OTROS NUEVE Y SU DIFERENCIA",
         "(tinta media, redondeada)",
         "", f"  el {DIGITO_SI} medio ({miles(n_si)} dibujos)"]
    L += cuadricula(np.round(media_si), 4)
    L += ["", f"  la media de los otros nueve ({miles(n_no)} dibujos)"]
    L += cuadricula(np.round(media_no), 4)
    L += ["", f"  la diferencia: el {DIGITO_SI} medio menos la media de los otros nueve"]
    L += cuadricula(np.round(diferencia), 4)
    for l in comprobar_ancho([l.rstrip() for l in L], ANCHO_CAJA):
        print(l)
    print()
    print("\n".join(tabla_editorial(
        "Cuánto se parecen los pesos a cada cosa",
        ["los pesos, comparados con", "cuánto se parecen"],
        [[f"el {DIGITO_SI} medio", coma(parecido(casos, media_si), 2)],
         ["la diferencia", coma(parecido(casos, diferencia), 2)],
         ["puntos donde peso y diferencia empujan al mismo lado", f"{iguales} de {vivos}"]],
         "id",
        ["La columna «cuánto se parecen» va de −1 a 1: 1 quiere decir que las dos cuadrículas son "
         "iguales salvo el tamaño de los números, y 0, que no tienen nada que ver. La última fila "
         f"cuenta solo los {vivos} puntos en los que ni el peso ni la diferencia valen cero."])))
    de_donde_salen_los_pesos(m, casos, liston)


def de_donde_salen_los_pesos(m, casos, liston):
    """El apartado 5: cada peso y el listón, contados corrección a corrección."""
    apuntes, vueltas = correcciones(m)
    Xtr, ytr, ttr = m["entrenamiento"]
    de_cuatros = [i for _, i, s in apuntes if s == 1]
    de_otros = [i for _, i, s in apuntes if s == -1]
    n4, no = len(de_cuatros), len(de_otros)
    con_correccion = len({v for v, _, _ in apuntes})
    print()
    print("5. DE DÓNDE SALEN LOS PESOS")
    print()
    print("\n".join(tabla_editorial(
        "Las correcciones que hizo el comité al aprender",
        ["", "cuántas"],
        [["correcciones en total", miles(n4 + no)],
         ["de ellas, con un cuatro", miles(n4)],
         ["de ellas, con otro dígito", miles(no)],
         ["vueltas por todos los dibujos", str(vueltas)]], "id",
        ["Con un cuatro: el comité no lo reconocía y se sumó a cada peso la tinta del dibujo. Con "
         "otro dígito: lo tomó por un cuatro y se restó.",
         f"Vueltas: en las {con_correccion} primeras corrigió algo; en la última ya no tuvo que "
         "corregir nada y se quedó quieto."])))
    por_digito = Counter(int(ttr[i]) for i in de_otros)
    otros = [d for d in range(10) if d != DIGITO_SI]
    assert sum(por_digito.values()) == no
    print()
    print("\n".join(tabla_editorial(
        "De qué dígito eran las correcciones con otro dígito",
        ["dígito", "correcciones"],
        [[str(d), miles(por_digito[d])] for d in sorted(otros, key=lambda d: (-por_digito[d], d))]
        + [["**en total**", f"**{miles(no)}**"]], "id",
        ["Cada una es una vez que el comité tomó ese dígito por un cuatro y hubo que restarle su "
         "tinta."])))
    mas, menos = int(np.argmax(casos)), int(np.argmin(casos))
    filas = []
    for k in (mas, menos):
        suma4 = int(Xtr[de_cuatros][:, k].sum())
        sumaO = int(Xtr[de_otros][:, k].sum())
        assert suma4 - sumaO == casos[k], (k, suma4, sumaO, casos[k])
        filas.append([lugar(k), miles(suma4), miles(sumaO), miles(int(casos[k]))])
    print()
    print("\n".join(tabla_editorial(
        "De dónde sale el peso de los dos puntos que más cuentan",
        ["punto", "tinta sumada con los cuatros", "tinta restada con los otros dígitos", "peso"],
        filas, "iddd",
        [f"Tinta sumada con los cuatros: la suma de la tinta de ese punto en las {miles(n4)} "
         f"correcciones con un cuatro. Tinta restada con los otros dígitos: lo mismo en las "
         f"{miles(no)} correcciones con otro dígito. El peso es la primera menos la segunda.",
         "Son el punto de peso más positivo y el de peso más negativo."])))
    assert 0 - n4 + no == liston, (n4, no, liston)
    print()
    print("\n".join(tabla_editorial(
        "De dónde sale el listón",
        ["", "listón"],
        [["empieza en", "0"],
         [f"baja con los {miles(n4)} cuatros corregidos, un punto cada vez", miles(-n4)],
         [f"sube con los {miles(no)} otros dígitos corregidos, un punto cada vez", f"+{miles(no)}"],
         ["**acaba en**", f"**{miles(liston)}**"]], "id",
        [f"Dice «es un {DIGITO_SI}» si el total pasa del listón."])))
    Xp, yp, tp = m["prueba"]
    pred = np.where(Xp @ m["w"] + m["b"] > 0, 1, -1)
    mal = pred != yp
    fallos4 = int((mal & (yp == 1)).sum())
    fallos_de = Counter(int(d) for d in tp[mal & (yp == -1)])
    filas = [["dibujos que no vio", miles(len(yp))],
             ["los que acierta", miles(int((~mal).sum()))],
             ["los que falla, que son un cuatro", miles(fallos4)]]
    for d in sorted(fallos_de):
        filas.append([f"los que falla, que son un {d}", miles(fallos_de[d])])
    print()
    print("\n".join(tabla_editorial(
        "Lo que acierta de los dibujos que no vio",
        ["", "dibujos"], filas, "id",
        [f"Acierto: {pct(float((~mal).mean()), 1)}."])))
    cuatros, demas = Xtr[ytr == 1], Xtr[ytr == -1]
    filas = []
    for k in (mas, menos):
        a4 = int((cuatros[:, k] >= TINTA_ALTA).sum())
        aO = int((demas[:, k] >= TINTA_ALTA).sum())
        filas.append([lugar(k), f"{a4} de {miles(len(cuatros))} ({pct(a4 / len(cuatros), 0)})",
                      f"{miles(aO)} de {miles(len(demas))} ({pct(aO / len(demas), 0)})"])
    print()
    print("\n".join(tabla_editorial(
        f"Cuántos dibujos tienen tinta {TINTA_ALTA} o más en esos dos puntos",
        ["punto", "de los cuatros para aprender", "de los otros dígitos para aprender"],
        filas, "idd",
        [f"Los dibujos con los que aprendió: {miles(len(cuatros))} cuatros y {miles(len(demas))} "
         f"de otros dígitos. Tinta {TINTA_ALTA} o más: de la mitad de arriba de la escala, que va "
         "de 0 a 16."])))


if __name__ == "__main__":
    main()
