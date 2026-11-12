#!/usr/bin/env python3
"""
Capítulo 1 — la escalera, paso a paso.

La sección «La escalera» enseña lo que escribe la máquina de `ngrama.py` con el selector en cero,
una, dos, tres y cinco letras. El lector ve el resultado de doscientos sorteos por peldaño y no
los pasos que llevan a él (L11 y T27 de `notas/control/auditoria/PENDIENTES.md`). Este programa imprime
esos pasos, con las mismas muestras:

  1. cómo arranca cada muestra: las primeras letras no salen de un sorteo;
  2. un peldaño hecho a mano: los primeros sorteos de la muestra de dos letras, hoja a hoja;
  3. por qué al subir salen palabras: cuantas más letras mira la máquina, menos letras distintas
     caben en la urna de cada hoja;
  4. cuántas palabras del Quijote hay en cada muestra.

Las muestras NO se escriben aquí: se repiten los sorteos de `ngrama.py` con su misma semilla, en
su mismo orden, y el selftest revienta si no salen idénticas a las de `datos/salidas/ngrama.txt`.

Dos presentaciones (Órdenes 138 y 140, 8 nov 2026; regla 11 y notas de Carlos de las p. 15-16 y 21):

  - la de ahora (por defecto): imprime EXACTAMENTE las tablas del capítulo 1 y del anexo A1 reescritos (11.4 y
    11.5: el programa sirve al libro), con sus títulos, rótulos, filas y notas. Los rótulos de las columnas los
    escribe el programa y `verificar_muestras.py` no deja que el libro los cambie; el libro puede quitar columnas.
  - la de antes, con `--antes`: la que citaba el capítulo 1 viejo (copia en
    `notas/historico/antes-de-programas-cap01-reforma/traza_escalera.txt`). Mismos sorteos, mismas cifras.

Uso:
    python traza_escalera.py --selftest
    python traza_escalera.py > ../datos/salidas/traza_escalera.txt
    python traza_escalera.py --antes     # solo para reproducir la salida vieja
"""

# ======================= CONSTANTES =======================

MUESTRAS = "../datos/salidas/ngrama.txt"   # lo que el libro imprime, tal como salió

PELDANO_A_MANO = 2        # el selector del peldaño que se sigue sorteo a sorteo
SORTEOS_A_MANO = 6        # cuántos sorteos se enseñan
SORTEOS_PALABRAS = 4      # cuántos sorteos de la máquina de palabras (selector en dos) se enseñan
LETRAS_POR_URNA = 4       # cuántas letras de cada urna se enseñan una a una; el resto se agrupa

# Rótulos de las tablas del capítulo 1 y del anexo A1 (Orden 140.3: los pone el programa, el libro los copia).
# «Van detrás» y no «pueden venir»: no es condicional, en el Quijote detrás van tantas (notas de las p. 16 y 21).
ROTULOS_SORTEOS = ["lleva escrito", "mira la hoja de", "letras distintas que van detrás", "sale"]
ROTULOS_PALABRAS = ["mira", "veces en el Quijote", "palabras distintas detrás", "la que más sale", "sale"]
ROTULOS_QUIJO = ["selector", "mira", "letras distintas detrás", "la «t», de cada cien"]
ORDINALES = {3: "tercero", 4: "cuarto", 5: "quinto", 6: "sexto", 7: "séptimo", 8: "octavo"}

# El mismo momento de escribir, visto con el selector en cada posición: la máquina acaba de
# escribir «quijo» y va a sacar en un sorteo la letra siguiente. Con el selector en uno solo ve «o»; en dos,
# «jo»; en tres, «ijo»; en cinco, «quijo». Es la misma letra por salir, con más o menos pasado.
MOMENTO = "quijo"

# Selftest
TROZO_NULO = 400_000              # letras del corpus con las que se hace el test nulo
SELECTOR_NULA = 2
COCIENTE_NULO_MIN = 0.75          # barajado: la urna de dos letras casi tan llena como la de cero
COCIENTE_REAL_MAX = 0.65          # real: la urna de dos letras claramente más vacía
MARCA = "wñk"                     # señal implantada: una palabra que no existe (ni «wñ» en el libro)
REPETICIONES_MARCA = 300

# ==========================================================

import platform
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import ngrama as N
from formato import (ANCHO_CAJA, coma, comprobar_ancho, miles, muestra_editorial, partir, pct,
                     tabla_editorial)

AQUI = Path(__file__).resolve().parent
REFORMA = "--antes" not in sys.argv      # la presentación de ahora; con --antes, la vieja
DETRAS = "van detrás" if REFORMA else "pueden venir"    # el rótulo de la media, en las tablas de hojas


def rotulo(orden):
    """El rótulo con el que `ngrama.py` encabeza cada muestra de letras."""
    if orden == 0:
        return "--- LETRAS, azar puro ---"
    return f"--- LETRAS, {orden} letra{'s' if orden > 1 else ''} de contexto ---"


def selector(orden):
    return {0: "nada", 1: "1 letra"}.get(orden, f"{orden} letras")


def rotulo_palabras(orden):
    """El rótulo con el que `ngrama.py` encabeza cada muestra de palabras."""
    return ("--- PALABRAS, frecuencias sueltas ---" if orden == 1
            else f"--- PALABRAS, {orden} palabras de contexto ---")


def leer_muestras(ruta):
    """Las muestras que imprimió `ngrama.py`: las de letras con la clave del orden, y las de
    palabras con la clave ("palabras", orden)."""
    lineas = Path(ruta).read_text(encoding="utf-8").split("\n")
    muestras = {}
    claves = [(o, rotulo(o)) for o in N.ORDENES_LETRA]
    claves += [(("palabras", o), rotulo_palabras(o)) for o in N.ORDENES_PALABRA]
    for orden, r in claves:
        assert r in lineas, f"Se esperaba el rótulo «{r}» en {ruta}; no está"
        muestras[orden] = lineas[lineas.index(r) + 1]
    return muestras


def generar_con_traza(tabla, orden, largo, rng):
    """Lo mismo que `ngrama.generar`, llamando al sorteo en el mismo orden, pero apuntando cada
    paso: con qué contexto, qué había en la urna y qué salió. Si no gasta el azar exactamente
    igual, las muestras salen distintas y el selftest lo dice."""
    if orden == 0:
        u = tabla[()]
        salida = [N.elegir(u, rng) for _ in range(largo)]
        return salida, None, [((), u, s) for s in salida]
    contextos = list(tabla.keys())
    arranque = rng.choice(contextos)
    estado, salida, pasos = list(arranque), list(arranque), []
    for _ in range(largo - orden):
        clave = tuple(estado[-orden:])
        assert clave in tabla, (
            f"Se esperaba que todo contexto escrito estuviera en la libreta (lo que sigue a un "
            f"contexto del libro también está en el libro); no está «{''.join(clave)}»")
        siguiente = N.elegir(tabla[clave], rng)
        pasos.append((clave, tabla[clave], siguiente))
        salida.append(siguiente)
        estado.append(siguiente)
    return salida, arranque, pasos


def repetir_sorteos(secuencia):
    """Las muestras de letras y luego las de palabras, en el orden y con la semilla de
    `ngrama.py`: las de palabras gastan el azar que dejan las de letras, así que no se pueden
    repetir sin repetir antes aquellas."""
    rng = N.random.Random(N.SEMILLA)
    resultado = {}
    for orden in N.ORDENES_LETRA:
        tabla = N.construir(secuencia, orden)
        salida, arranque, pasos = generar_con_traza(tabla, orden, N.LARGO_MUESTRA_LETRAS, rng)
        resultado[orden] = (tabla, "".join(salida),
                            None if arranque is None else "".join(arranque), pasos)
    palabras = N.trocear(secuencia)
    for orden in N.ORDENES_PALABRA:
        tabla = N.construir(palabras, orden)
        salida, arranque, pasos = generar_con_traza(tabla, orden, N.LARGO_MUESTRA_PALABRAS, rng)
        resultado[("palabras", orden)] = (tabla, " ".join(salida), arranque, pasos)
    return resultado


def letras_por_urna(tabla):
    """De media, cada vez que la máquina hace un sorteo, cuántas letras distintas pueden venir en su hoja. Cada
    hoja pesa tantas veces como se usa: una hoja que sale mil veces cuenta mil."""
    usos = sum(sum(c.values()) for c in tabla.values())
    return sum(len(c) * sum(c.values()) for c in tabla.values()) / usos


def cuantas(n, singular, plural):
    """«1 papeleta», «24 papeletas»: el número con su nombre bien concordado."""
    return f"{miles(n)} {singular if n == 1 else plural}"


def letra(s):
    return "espacio" if s == " " else f"«{s}»"


def nombre_simbolo(s):
    """Cómo se escribe una letra en la columna de una tabla: el espacio, con su nombre."""
    return "espacio" if s == " " else s


def la_que_mas(contador, palabra=False):
    """La letra (o palabra) que más sale en una hoja, con cuántas veces de cada cien. Si
    empatan varias en cabeza, lo dice, porque nombrar una sola sería elegir por el lector."""
    total = sum(contador.values())
    orden = contador.most_common()
    arriba = [s for s, n in orden if n == orden[0][1]]
    parte = pct(orden[0][1] / total, 0)
    if len(arriba) > 1:
        return f"{len(arriba)} empatadas, {parte}"
    s = arriba[0]
    return f"{'«' + s + '»' if palabra else letra(s)}, {parte}"


def urna(contador, cuantas, unidad="letra"):
    """Las `cuantas` letras más frecuentes de una hoja, con su parte de cada cien, y el resto
    agrupado. Devuelve líneas."""
    total = sum(contador.values())
    orden = contador.most_common()
    nombre = letra if unidad == "letra" else (lambda s: f"«{s}»")
    partes = [f"{nombre(s)} {pct(n / total, 1)}" for s, n in orden[:cuantas]]
    resto = orden[cuantas:]
    if resto:
        n = sum(v for _, v in resto)
        otras = (f"otra {unidad}" if len(resto) == 1
                 else f"otras {miles(len(resto))} {unidad}s")
        partes.append(f"{otras}: {pct(n / total, 1)}")
    lineas, fila = [], "      "
    for p in partes:
        if len(fila) + len(p) + 3 > ANCHO_CAJA:
            lineas.append(fila.rstrip(" ·"))
            fila = "      "
        fila += p + " · "
    lineas.append(fila.rstrip(" ·"))
    return lineas


def palabras_reales(muestra, vocabulario):
    """Igual que `casillas_vacias.palabras_reales`: los trozos entre espacios, sin los signos
    pegados, y cuáles están en el Quijote."""
    trozos = [t.strip(N.SIGNOS) for t in muestra.split() if t.strip(N.SIGNOS)]
    return trozos, [t for t in trozos if t in vocabulario]


# ---- selftest ------------------------------------------------------------------------

def selftest(secuencia, muestras_libro, vocabulario):
    fallos = []

    # 1. TEST NULO — con las letras barajadas no hay castellano que estreche las urnas: mirar dos
    #    letras atrás deja la urna casi tan llena como no mirar nada. En el texto real, no.
    barajado = list(secuencia[:TROZO_NULO])
    random.Random(N.SEMILLA).shuffle(barajado)
    real = secuencia[:TROZO_NULO]
    c_nulo = letras_por_urna(N.construir(barajado, SELECTOR_NULA)) / letras_por_urna(N.construir(barajado, 0))
    c_real = letras_por_urna(N.construir(real, SELECTOR_NULA)) / letras_por_urna(N.construir(real, 0))
    print(f"[1] test nulo         letras posibles mirando {SELECTOR_NULA} / sin mirar: "
          f"barajado={coma(c_nulo, 2)}  real={coma(c_real, 2)}")
    if not (c_nulo > COCIENTE_NULO_MIN and c_real < COCIENTE_REAL_MAX):
        fallos.append(f"test nulo: se esperaba barajado > {COCIENTE_NULO_MIN} y real < "
                      f"{COCIENTE_REAL_MAX}; salió barajado={c_nulo:.3f}, real={c_real:.3f}")

    sin_grupo = N.palabra_de_ejemplo(secuencia, "qz", "e")            # «qz» no sale en el Quijote
    print(f"                      grupo «qz» (no aparece): palabra de ejemplo = {sin_grupo}"
          " (sin palabra, como tiene que ser)")
    if sin_grupo is not None:
        fallos.append(f"test nulo: un grupo que no aparece no debe dar palabra; dio {sin_grupo}")

    # 2. SEÑAL IMPLANTADA — una palabra inventada, repetida, deja una hoja con una sola letra,
    #    y exactamente tantas veces como se metió.
    implantado = secuencia[:200_000] + (" " + MARCA) * REPETICIONES_MARCA
    pagina = N.construir(implantado, 2)[tuple(MARCA[:2])]
    print(f"[2] señal implantada  hoja «{MARCA[:2]}»: {dict(pagina)}")
    if dict(pagina) != {MARCA[2]: REPETICIONES_MARCA}:
        fallos.append(f"señal implantada: se esperaba la hoja «{MARCA[:2]}» = "
                      f"{{'{MARCA[2]}': {REPETICIONES_MARCA}}}; salió {dict(pagina)}")

    # La palabra de ejemplo, con un texto fabricado: gana la más frecuente; con empate, la que
    # sale antes; un signo detrás del grupo no está dentro de ninguna palabra.
    fabricado = "extra extra texto texto sexto extrano texte exterior ext otro"
    esperado = [("xt", "r", "extra"), ("xt", "o", "texto"), ("xt", "e", "texte"),
                ("xt", " ", None), ("xt", "z", None)]
    obtenido = [(g, l, N.palabra_de_ejemplo(fabricado, g, l)) for g, l, _ in esperado]
    al_reves = N.palabra_de_ejemplo("exterior texte", "xt", "e")
    print(f"                      palabra de ejemplo en un texto fabricado: la más frecuente "
          f"(extra, texto), con empate la primera ({obtenido[2][2]}, y {al_reves} al revés)")
    if obtenido != esperado or al_reves != "exterior":
        fallos.append(f"señal implantada: palabra de ejemplo, se esperaba {esperado} y "
                      f"«exterior» al revés; salió {obtenido} y {al_reves}")

    # 3. INVARIANTES — los sorteos repetidos dan las muestras del libro, letra por letra; cada
    #    letra que sale estaba en su urna; y lo que se cuenta de las palabras cuadra.
    repetidas = repetir_sorteos(secuencia)
    todas = list(muestras_libro)
    distintas = [o for o in todas if repetidas[o][1].strip() != muestras_libro[o].strip()]
    fuera = sum(1 for o in todas for _, u, s in repetidas[o][3] if s not in u)
    print(f"[3] invariante        muestras repetidas iguales a las de ngrama.txt: "
          f"{len(todas) - len(distintas)} de {len(todas)}; "
          f"letras que salieron sin estar en su hoja: {fuera}")
    if distintas:
        fallos.append(f"invariante: las muestras de los selectores {distintas} no salen iguales "
                      f"que en {MUESTRAS}; este programa ya no repite los sorteos de ngrama.py")
    if fuera:
        fallos.append(f"invariante: {fuera} letras salieron sin estar en su hoja")
    clave0, u0, _ = repetidas[PELDANO_A_MANO][3][0]
    en_el_texto = set(re.findall(r"[^\W\d_]+", secuencia))
    ejemplos = {x: N.palabra_de_ejemplo(secuencia, "".join(clave0), x) for x in u0}
    malas = [x for x, e in ejemplos.items()
             if not (e and "".join(clave0) + x in e and e in en_el_texto)]
    print(f"                      las {len(u0)} palabras de ejemplo de «{''.join(clave0)}» "
          f"contienen el grupo y su letra y salen en el texto; las que no: {len(malas)}")
    if malas:
        fallos.append(f"invariante: palabras de ejemplo que no contienen el grupo y su letra, "
                      f"o no salen en el texto: {malas}")
    trozos, reales = palabras_reales(muestras_libro[1], vocabulario)
    print(f"[3] invariante        muestra de 1 letra: {len(reales)} palabras del Quijote en "
          f"{len(trozos)} trozos (casillas_vacias.txt dice 12 en 40)")
    if (len(reales), len(trozos)) != (12, 40):
        fallos.append(f"invariante: se esperaban 12 palabras en 40 trozos, como en "
                      f"casillas_vacias.txt; salieron {len(reales)} en {len(trozos)}")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1, repetidas
    print("SELFTEST: las tres pruebas pasan.")
    return 0, repetidas


# ---- la salida -------------------------------------------------------------------------

def main():
    secuencia = N.normalizar(N.cargar_corpus(str(AQUI / N.CORPUS)), N.ALFABETO)
    vocabulario = set(N.solo_palabras(N.trocear(secuencia)))
    muestras_libro = leer_muestras(AQUI / MUESTRAS)

    print("--- selftest ---")
    codigo, repetidas = selftest(secuencia, muestras_libro, vocabulario)
    if codigo or "--selftest" in sys.argv:
        sys.exit(codigo)

    L = [
        "",
        "########## capítulo 1: la escalera, paso a paso ##########",
        f"fecha: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
        f"máquina: {platform.system()} {platform.machine()}. La medición no",
        "depende de la máquina: repite los sorteos de ngrama.py con su",
        f"semilla ({N.SEMILLA}) y sale lo mismo en cualquier ordenador",
        "",
        "1. CÓMO ARRANCA CADA MUESTRA",
        "",
        "Las primeras letras no salen de un sorteo: la máquina coge al azar",
        "un contexto de la tabla, lo escribe, y detrás empiezan los sorteos.",
        "",
        f"  {'selector':<10}{'arranque':<12}principio de la muestra",
        f"  {'-------':<10}{'--------':<12}-----------------------",
    ]
    for orden in N.ORDENES_LETRA:
        _, texto, arranque, _ = repetidas[orden]
        a = "(ninguno)" if arranque is None else f"«{arranque}»"
        L.append(f"  {selector(orden):<10}{a:<12}{texto[:24]}…")

    # L24 (9 de octubre de 2026): del apartado 2 en adelante, tablas y muestras editoriales;
    # antes, renglones sangrados que el capítulo 1 copiaba. El apartado 1 no sale en el libro.
    E = []      # lo editorial va aparte: no se le mide el ancho, lo compone el libro
    tabla, texto, arranque, pasos = repetidas[PELDANO_A_MANO]
    clave, u, sale = pasos[0]
    total = sum(u.values())
    E += ["", f"2. EL PRIMER SORTEO DEL SELECTOR EN {PELDANO_A_MANO}, ENTERO", ""]
    E += tabla_editorial(
        f"Qué letra viene detrás de «{''.join(clave)}»",
        ["letra que sigue", "veces", "de cada cien veces", "por ejemplo"],
        [[nombre_simbolo(x), miles(n), pct(n / total, 2),
          N.palabra_de_ejemplo(secuencia, "".join(clave), x) or "—"]
         for x, n in u.most_common()], "cddi",
        [f"Las {miles(total)} veces que aparece «{''.join(clave)}» en el Quijote.",
         f"La máquina arranca con «{arranque}». Sale {letra(sale)}; escrito: «{arranque + sale}»."])

    if REFORMA:
        # Los SORTEOS_A_MANO sorteos, una fila cada uno, y una última fila, en negrita, con lo escrito al
        # final. «Lleva escrito» es lo escrito ANTES de cada sorteo, así que la letra del último sorteo no
        # aparece en esa columna de ninguna fila de sorteo: la fila final la enseña (notas de la p. 16).
        E += ["", f"3. LOS {SORTEOS_A_MANO} PRIMEROS SORTEOS Y LO ESCRITO AL FINAL", ""]
    else:
        E += ["", f"3. LOS {SORTEOS_A_MANO} PRIMEROS SORTEOS, UNO POR FILA", ""]
    filas, escrito = [], arranque
    for clave, u, sale in pasos[:SORTEOS_A_MANO]:
        filas.append([f"«{escrito}»", f"«{''.join(clave)}»", str(len(u)), la_que_mas(u),
                      letra(sale)])
        escrito += sale
    if REFORMA:
        assert SORTEOS_A_MANO in ORDINALES, f"Se esperaba un ordinal para {SORTEOS_A_MANO} sorteos; no lo hay"
        filas = [[f[0], f[1], f[2], f[4]] for f in filas] + [[f"**«{escrito}»**", "", "", ""]]
        E += tabla_editorial(
            f"Los {SORTEOS_A_MANO} primeros sorteos con el selector en {selector(PELDANO_A_MANO)}",
            ROTULOS_SORTEOS, filas, "iidi",
            [f"Cada fila es un sorteo. La última fila es lo que queda escrito después del "
             f"{ORDINALES[SORTEOS_A_MANO]}."])
    else:
        E += tabla_editorial(
            f"Los {SORTEOS_A_MANO} primeros sorteos con el selector en {selector(PELDANO_A_MANO)}",
            ["ha escrito", "mira", "pueden venir", "la que más sale", "sale"], filas, "iidii",
            ["Mira: las dos últimas letras escritas, que dicen a qué hoja va. Pueden venir: "
             "cuántas letras distintas hay en esa hoja. La que más sale: y cuántas veces de "
             "cada cien.", f"Escrito al final: «{escrito}»."])
    escrito_final = escrito

    tabla, texto, arranque, pasos = repetidas[("palabras", 2)]
    E += ["", f"3 BIS. LA MÁQUINA DE PALABRAS, SELECTOR EN 2: LOS {SORTEOS_PALABRAS} PRIMEROS "
              "SORTEOS", ""]
    filas, escrito = [], list(arranque)
    for clave, u, sale in pasos[:SORTEOS_PALABRAS]:
        filas.append([f"«{' '.join(clave)}»", miles(sum(u.values())), str(len(u)),
                      la_que_mas(u, palabra=True), f"«{sale}»"])
        escrito.append(sale)
    E += tabla_editorial(
        f"Los {SORTEOS_PALABRAS} primeros sorteos de la máquina de palabras, con el selector en 2",
        ROTULOS_PALABRAS if REFORMA else ["mira", "veces", "pueden venir", "la que más sale", "sale"],
        filas, "iddii",
        [f"Arranque: «{' '.join(arranque)}». Cada sorteo mira las dos últimas palabras"
         + ("." if REFORMA else "; «pueden venir» cuenta palabras distintas."),
         f"Escrito al final: «{' '.join(escrito)}»."])

    E += ["", "4. POR QUÉ AL SUBIR SALEN PALABRAS", ""]
    filas = []
    for orden in N.ORDENES_LETRA:
        t = repetidas[orden][0]
        clave = tuple(MOMENTO[len(MOMENTO) - orden:]) if orden else ()
        u = t[clave]
        filas.append([selector(orden), "nada" if orden == 0 else "«" + "".join(clave) + "»",
                      str(len(u)), coma(100 * u["t"] / sum(u.values()), 1)])
    E += tabla_editorial(
        f"La letra que {'va' if REFORMA else 'viene'} detrás de «{MOMENTO}», según el selector",
        ROTULOS_QUIJO if REFORMA else ["selector", "mira", "pueden venir", "la «t», de cada cien"], filas, "iidd",
        [f"La máquina ha escrito «{MOMENTO}» y tiene que sacar en un sorteo la letra siguiente. "
         "Lo que mira depende del selector."])
    E += [""] + tabla_editorial(
        f"Cuántas letras {DETRAS}, de media, según el selector",
        ["selector", "hojas", f"{DETRAS}, de media"],
        [[selector(o), miles(len(repetidas[o][0])), coma(letras_por_urna(repetidas[o][0]), 1)]
         for o in N.ORDENES_LETRA], "idd",
        [f"De media en todos los sorteos: cuántas letras distintas {'van detrás' if REFORMA else 'pueden venir'} en la hoja "
         "que le toca a la máquina."])

    E += ["", "5. PALABRAS DEL QUIJOTE EN CADA MUESTRA", ""]
    filas = []
    for orden in N.ORDENES_LETRA:
        trozos, reales = palabras_reales(muestras_libro[orden], vocabulario)
        filas.append([selector(orden), str(len(trozos)), str(len(reales)),
                      pct(len(reales) / len(trozos), 0)])
    _, reales = palabras_reales(muestras_libro[0], vocabulario)
    E += tabla_editorial(
        "Palabras del Quijote en cada muestra",
        ["selector", "grupos de letras", "palabras del Quijote", "de cada cien"], filas, "iddd",
        ["Grupos de letras: lo que hay entre espacio y espacio, sin los signos pegados.",
         "Las del selector en nada: " + " ".join(f"«{r}»" for r in reales) + "."])

    # 6. Las muestras, tal como las imprimió ngrama.py, partidas en renglones solo por espacios.
    E += ["", "6. LAS MUESTRAS, TAL CUAL", ""]
    titulos = [(o, "Lo que escribe sin mirar nada" if o == 0 else
                f"Lo que escribe con el selector en {selector(o)}") for o in N.ORDENES_LETRA]
    titulos += [(("palabras", o), "Lo que escribe palabra a palabra, sin mirar ninguna"
                 if o == 1 else f"Lo que escribe palabra a palabra, con el selector en {o} palabras")
                for o in N.ORDENES_PALABRA]
    for clave, titulo in titulos:
        m = muestras_libro[clave].rstrip()
        E += muestra_editorial(titulo, partir(m)) + [""]

    # Comprobaciones de la presentación (sobre lo que se va a imprimir, no sobre lo que se espera):
    texto_editorial = "\n".join(E)
    primero = repetidas[PELDANO_A_MANO][1]
    assert primero.strip().startswith(escrito_final.strip()) or escrito_final.strip() in primero, (
        f"Se esperaba que lo escrito al final de los sorteos a mano («{escrito_final}») saliera en la "
        f"muestra del selector en {PELDANO_A_MANO} que imprime ngrama.py; no sale")
    if REFORMA:
        assert "pueden venir" not in texto_editorial, (
            "Se esperaba ningún «pueden venir» en la presentación de ahora; queda alguno")
        assert f"| **«{escrito_final}»** |" in texto_editorial.split("3. LOS ")[1].split("3 BIS")[0], (
            f"Se esperaba la fila final en negrita con «{escrito_final}» en la tabla de los sorteos a mano")
    else:
        assert "van detrás" not in texto_editorial, (
            "Se esperaba la presentación de siempre sin «van detrás»; hay alguno")
    for l in comprobar_ancho([l.rstrip() for l in "\n".join(L).split("\n")], ANCHO_CAJA):
        print(l)
    for l in E:
        print(l)


if __name__ == "__main__":
    main()
