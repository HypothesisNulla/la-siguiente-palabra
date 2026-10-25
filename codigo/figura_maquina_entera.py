#!/usr/bin/env python3
"""
Capítulo 7 — la máquina entera, de un vistazo.

El capítulo explica la máquina en cinco pasos numerados y no tiene ninguna figura: es el
capítulo con menos prosa por bloque de todo el libro. Esto es el plano completo, con los
cinco pasos y, en cada uno, la cifra de verdad.

Ningún número está escrito aquí: todos se leen de `datos/salidas/maquina_entera.txt`, que es
lo que imprimió la máquina. Si esa salida cambia, la figura cambia; si no se puede leer, esto
revienta en vez de dibujar un número viejo.

Uso:
    python figura_maquina_entera.py --selftest
    python figura_maquina_entera.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/maquina_entera.txt"   # la medición
DESTINO = "../figuras/maquina_entera"                   # se escribe DESTINO_color.png y DESTINO_gris.png

# ==========================================================

import argparse, re, sys
from formato import leer_tablas
from infografia import Lienzo, COLOR, GRIS


def leer(ruta):
    """Saca de la salida de máquina lo que la figura necesita, de sus tablas editoriales (L24,
    9 de octubre). Si falta algo, revienta."""
    tablas = leer_tablas(open(ruta, encoding="utf-8").read())
    for t in ("El texto, partido en trozos", "El tamaño de la máquina", "Y se vuelve a empezar"):
        assert t in tablas, f"no encuentro la tabla «{t}» en la salida"
    assert any(t.startswith("Qué viene después de «") for t in tablas), "no encuentro la lista"
    d = {}
    frase, n, cortes = tablas["El texto, partido en trozos"][1][0]
    # Orden 72 (25 oct 2026): los trozos de la tabla van unidos con «|» en un solo código
    # (`La|_capital|…`), no cada uno en el suyo; antes se sacaban con re.findall de cada `…`.
    d["frase"], d["n_trozos"], d["trozos"] = frase, int(n), cortes.strip("`").split("|")
    tam = dict((f[0], f[1]) for f in tablas["El tamaño de la máquina"][1])
    for clave, rotulo in (("numeros", "números por trozo"), ("rondas", "rondas, una detrás de otra"),
                          ("miradas", "miradas a la vez dentro de cada ronda"),
                          ("total_miradas", "miradas en total"),
                          ("vocabulario", "trozos posibles en la salida"),
                          ("ajustables", "números ajustables en total")):
        assert rotulo in tam, f"no encuentro «{clave}» en la salida"
        d[clave] = tam[rotulo]
    lista = next(v for t, v in tablas.items() if t.startswith("Qué viene después de «"))
    d["probs"] = [(f[0].strip("`"), f[1].replace(" %", "")) for f in lista[1] if f[0] != "(el resto)"]
    assert len(d["probs"]) >= 5, "esperaba al menos cinco filas de probabilidad"
    d["pasos"] = [(f[0], f[1].strip("`"), f[2].replace(" %", "")) for f in tablas["Y se vuelve a empezar"][1]]
    assert len(d["pasos"]) >= 4, "esperaba al menos cuatro pasos de la cadena"
    return d


def dibujar(d, paleta, ruta):
    L = Lienzo("La máquina entera, de un vistazo",
               "Lo que le pasa a una frase desde que entra hasta que sale la palabra siguiente.\n"
               "Las cifras son las de la máquina que se mide en este capítulo.",
               # PENDIENTE (21 sep): esta figura no cabe en la página. La caja del libro mide
               # 7,25 pulgadas de alto y, con su pie, a la imagen le quedan 6,55. Sus cinco
               # paneles, apretados al mínimo que aguanta su contenido, piden unas 157 unidades
               # de lienzo y en 6,55 pulgadas caben 147. Con la regla nueva de infografia.py ya
               # no se dibuja: revienta con el aviso, que es lo que se quiere. Hay que decidir si
               # se parte en dos figuras o si se le quita un paso. Mientras tanto el libro no la
               # usa: no está incluida en ningún capítulo.
               paleta, alto=7.9)
    p = paleta

    # 1 ─ trozos
    x, y, w = L.panel(1, "El texto se parte en trozos", 22.0)
    L.texto(x, y, f"«{d['frase']}»", tam=8.2)
    ancho = (w - 2 * (len(d["trozos"]) - 1)) / len(d["trozos"])
    for i, tr in enumerate(d["trozos"]):
        L.ficha(x + i * (ancho + 2), y - 5.4, tr.replace("_", "␣"), ancho, mono=True, tam=6.2)
    L.texto(x, y - 10.4, f"{d['n_trozos']} trozos. No son palabras: son pedazos de palabra.",
            tam=7.2, color=p.suave)
    L.flecha()

    # 2 ─ números
    x, y, w = L.panel(2, "Cada trozo se vuelve una lista de números", 19.5)
    for i, (n, rot) in enumerate(((d["numeros"], "números\npor trozo"),
                                  (d["vocabulario"], "trozos posibles\nen la salida"),
                                  (d["ajustables"], "números ajustables\nen toda la máquina"))):
        cx = x + i * (w / 3)
        L.texto(cx + w / 6, y - 1.4, n, tam=12, negrita=True, ha="center", color=p.acento)
        L.texto(cx + w / 6, y - 6.2, rot, tam=6.6, ha="center", color=p.suave)
    L.flecha()

    # 3 ─ rondas
    x, y, w = L.panel(3, "Rondas de mirar y mezclar", 21.0)
    L.texto(x, y, f"{d['rondas']} rondas, una detrás de otra. En cada una, "
                  f"{d['miradas']} miradas a la vez.", tam=7.4)
    for i in range(int(d["rondas"])):
        L.ficha(x + i * (w / int(d["rondas"])), y - 5.0, "", w / int(d["rondas"]) - 0.6,
                alto=3.0, relleno=p.contra)
    L.texto(x, y - 9.4, f"{d['total_miradas']} miradas en total. Cada ronda recibe lo que "
                        f"dejó la anterior.", tam=7.2, color=p.suave)
    L.flecha()

    # 4 ─ probabilidades
    x, y, w = L.panel(4, "Sale una lista de probabilidades", 27.5)
    L.texto(x, y, "No elige una palabra: puntúa todas las que podría poner.", tam=7.2,
            color=p.suave)
    mayor = max(float(v.replace(",", ".")) for _, v in d["probs"])
    for i, (trozo, valor) in enumerate(d["probs"][:5]):
        yy = y - 4.2 - i * 2.9
        L.texto(x + 11, yy, trozo.replace("_", "␣"), tam=6.8, ha="right", mono=True)
        largo = (w - 26) * float(valor.replace(",", ".")) / mayor
        L.ficha(x + 13, yy, "", max(largo, 0.6), alto=2.1,
                relleno=p.acento if i == 0 else p.neutro)
        L.texto(x + 14 + largo, yy, f"{valor} %", tam=6.8, color=p.suave)
    L.flecha()

    # 5 ─ se vuelve a empezar
    x, y, w = L.panel(5, "Y se vuelve a empezar", 23.0)
    L.texto(x, y, "La palabra elegida se pega al final y todo el proceso se repite.", tam=7.2,
            color=p.suave)
    ancho = (w - 3 * 3) / 4
    for i, (n, trozo, valor) in enumerate(d["pasos"][:4]):
        cx = x + i * (ancho + 3)
        L.ficha(cx, y - 5.0, trozo.replace("_", "␣"), ancho, alto=3.4, mono=True, tam=6.4,
                relleno=p.fondo)
        L.texto(cx + ancho / 2, y - 8.4, f"{valor} %", tam=6.4, ha="center", color=p.suave)
        if i < 3:
            L.texto(cx + ancho + 1.5, y - 5.0, "›", tam=10, ha="center", color=p.acento)
    L.texto(x, y - 11.6, "Cada paso vuelve a mirar la frase entera, ya con lo recién escrito.",
            tam=7.2, color=p.suave)

    L.pie("Todas las cifras salen de datos/salidas/maquina_entera.txt, la salida del programa\n"
          "que mide esta máquina. La figura las lee de ahí: no hay ningún número escrito a mano.")
    L.guardar(ruta)
    return ruta


def selftest():
    fallos = []
    d = leer(SALIDA)
    # 1. TEST NULO — una salida sin los rótulos que la figura necesita tiene que reventar,
    #    no dibujar una figura vacía o con números de otro sitio.
    try:
        leer.__wrapped__ if False else None
        import io, tempfile, os
        tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
        tmp.write("aquí no hay nada que leer\n"); tmp.close()
        try:
            leer(tmp.name); revienta = False
        except AssertionError:
            revienta = True
        os.unlink(tmp.name)
    except Exception:
        revienta = False
    print(f"[1] test nulo         con una salida vacía de rótulos, "
          f"{'revienta' if revienta else 'NO revienta'}")
    if not revienta:
        fallos.append("test nulo: leer() no revienta con una salida sin rótulos")

    # 2. SEÑAL IMPLANTADA — los números que la figura pinta son los de la salida, carácter a
    #    carácter, y no una versión redondeada o reescrita.
    crudo = open(SALIDA, encoding="utf-8").read()
    literales = [d["numeros"], d["vocabulario"], d["ajustables"], d["total_miradas"]]
    fuera = [v for v in literales if v not in crudo]
    print(f"[2] señal implantada  {len(literales) - len(fuera)} de {len(literales)} cifras "
          f"aparecen literales en la salida")
    if fuera:
        fallos.append(f"señal implantada: estas cifras no están literales en la salida: {fuera}")

    # 3. INVARIANTE DEL DOMINIO — la frase se parte en tantos trozos como dice la salida, y
    #    las probabilidades de la lista bajan: si subieran, la figura estaría ordenando mal.
    baja = all(float(a[1].replace(",", ".")) >= float(b[1].replace(",", "."))
               for a, b in zip(d["probs"], d["probs"][1:]))
    print(f"[3] invariante        {len(d['trozos'])} trozos contra los {d['n_trozos']} que "
          f"declara la salida; la lista {'baja' if baja else 'NO baja'}")
    if len(d["trozos"]) != d["n_trozos"] or not baja:
        fallos.append("invariante: los trozos no cuadran o la lista no está ordenada")

    print()
    if fallos:
        for f in fallos: print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    d = leer(SALIDA)
    for nombre, paleta in (("color", COLOR), ("gris", GRIS)):
        print("Escrito", dibujar(d, paleta, f"{DESTINO}_{nombre}.png"))


if __name__ == "__main__":
    main()
