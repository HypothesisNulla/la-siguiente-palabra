#!/usr/bin/env python3
"""
Capítulo 10 — en qué orden aprende un transformer entrenado desde cero.

Entrena un transformer minúsculo, letra a letra, sobre libros en español, y guarda lo que
escribe en cuatro momentos del entrenamiento. Mide también lo mal que lo hace a lo largo del
tiempo y el acierto al principio y al final.

El capítulo tiene un presupuesto de números acotado (ver REGLAS.md del repositorio privado):
cuatro muestras de texto, una curva, dos porcentajes y una comparación de tamaño. Este guion
produce exactamente eso y nada más.

Uso:
    python en_que_orden_aprende.py
    python en_que_orden_aprende.py --selftest
    python en_que_orden_aprende.py --ficha > ../datos/salidas/en_que_orden_aprende_ficha.txt
"""

# ======================= CONSTANTES =======================

SEMILLA = 20260914
CORPUS = "../datos/corpus_es"
MAX_CARACTERES = 8_000_000

CONTEXTO = 128             # letras que puede mirar hacia atrás
ANCHO = 192                # números por letra
CAPAS = 4
CABEZAS = 4
LOTE = 64
TASA = 3e-4

# Cuántos pasos entrena el modelo. En PASOS, no en minutos, y por la razón de fondo: **las
# conclusiones del libro no pueden cambiar con la velocidad de la máquina**. Acotando por reloj,
# un ordenador lento llega a menos pasos, escribe peor y acierta menos, así que el «61 %» del
# final de este capítulo sería un número distinto en cada ordenador. Con los pasos fijos y la
# semilla fija, el modelo del paso 50.000 es el mismo en todas partes.
#
# Cincuenta mil, y no los 53.151 de la última tirada, porque ese número no lo eligió nadie: lo
# eligió el reloj de un portátil una tarde. Cincuenta mil es redondo y está por debajo de lo que
# dieron las dos tiradas hechas hasta hoy, así que ninguna máquina se queda corta.
#
# Los minutos siguen saliendo en la salida, pero como lo que son: una propiedad de la máquina
# que ejecuta, etiquetada «en esta máquina».
PASOS_TOTAL = 50_000
# Y lo mismo para el selftest, que también se acotaba por minutos: en el portátil del autor
# entrenaba unos dos mil pasos y en un contenedor lento unos doscientos, así que sus dos cifras
# —y por tanto la holgura con la que pasaba— dependían del ordenador. Ahora son pasos.
PASOS_SELFTEST = 400         # test nulo y señal implantada: bastan para que se vea la diferencia
PASOS_SELFTEST_MARCA = 800   # la cadena implantada necesita algo más para salir al escribir
# Cuándo pedirle que escriba: en PASOS, no en minutos. La diferencia no es cosmética.
#
# Este guion fotografiaba a los 0,5 / 2 / 8 / 25 minutos, y así lo publicó el repositorio. Pero
# las cuatro muestras que enseña el capítulo salieron de otra ejecución, con las fotos tomadas
# por pasos, y esa versión nunca se subió: el guion público **no podía reproducir** las cuatro
# muestras del libro, que es lo que el capítulo promete que puedes hacer. Fallo 4.36.
#
# Y por pasos es además lo correcto. Una foto «a los dos minutos» cae en un punto distinto del
# aprendizaje en cada máquina —en una tarjeta rápida ya ha dado veinte veces más pasos que en un
# procesador—, así que la escalera del capítulo no se vería igual en ningún otro ordenador. Con
# la semilla fija y las fotos por pasos, las tres primeras muestras salen IDÉNTICAS en cualquier
# máquina. Y la cuarta, la del final, también es reproducible desde que el entrenamiento se
# acota por pasos: antes era «hasta donde llegue en veinticinco minutos» y cambiaba con el
# ordenador.
INSTANTANEAS_PASOS = [30, 300, 3_000]
ARRANQUE = "el "
LARGO_MUESTRA = 200
TEMPERATURA = 0.8
EJEMPLOS_ACIERTO = 20_000

SALIDA_CURVA = "../datos/en_que_orden_aprende_curva.csv"
SALIDA_FIGURA = "../figuras/en_que_orden_aprende.png"
ANCHO_ALTO_FIGURA = (6.2, 3.0)    # pulgadas, para una página de 6 por 9
VENTANA_SUAVIZADO = 40            # puntos de la media que dibuja la línea negra
MINIMO_PUNTOS_CURVA = 100         # menos que esto y el CSV no es una medición de verdad
TITULO_FIGURA = "LO MAL QUE LO HACE, SEGÚN AVANZA EL ENTRENAMIENTO"
ETIQUETA_CRUDA = "la medición cruda, paso a paso"
ETIQUETA_SUAVE = "la misma, suavizada, para que se vea la forma"
# El recuadro ampliado. Sin él, el dibujo grande —con el eje anclado en cero, que es lo
# correcto para que se vea que la línea no llega al suelo— convierte el último 80 % de la
# curva en una raya horizontal a ojo, y el capítulo pide justo ahí una predicción sobre si
# la bajada sigue. El recuadro enseña ese tramo con el eje vertical SIN anclar en cero, y
# lo dice en su propia línea para que nadie lea la ampliación como si fuera el dibujo.
PASO_RECUADRO = 10_000            # desde qué paso se amplía el tramo final
ETIQUETA_RECUADRO = "el tramo final de cerca:\ndel paso {} al {}"
NOTA_RECUADRO = "aquí el cero no está en el dibujo, para que se vea que baja"
MODELO_GRANDE = 500_000_000_000   # orden de magnitud de un modelo grande de hoy

# ==========================================================

import argparse
import csv
import glob
import math
import os
import re
import sys
import time
import unicodedata

from formato import comprobar_ancho, miles, coma, pct, tabla_editorial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ALFABETO = "abcdefghijklmnñopqrstuvwxyzáéíóúü ,.;:¿?¡!\n"


def dispositivo():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DISPOSITIVO = dispositivo()


def fijar_semilla(s=SEMILLA):
    torch.manual_seed(s)
    np.random.seed(s)


def cargar_texto(maximo=MAX_CARACTERES):
    ficheros = sorted(glob.glob(os.path.join(CORPUS, "*.txt")))
    assert ficheros, f"Se esperaban libros en «{CORPUS}»; no se encontró ninguno"
    trozos, total = [], 0
    for ruta in ficheros:
        with open(ruta, encoding="utf-8", errors="replace") as fh:
            t = fh.read()
        if "*** START OF" in t:
            t = t.split("*** START OF", 1)[1].split("\n", 1)[-1]
        if "*** END OF" in t:
            t = t.split("*** END OF", 1)[0]
        t = unicodedata.normalize("NFC", t.lower())
        t = "".join(c if c in ALFABETO else " " for c in t)
        trozos.append(re.sub(r" {2,}", " ", t))
        total += len(trozos[-1])
        if total >= maximo:
            break
    texto = "".join(trozos)[:maximo]
    assert len(texto) >= 200_000, \
        f"Se esperaban al menos 200.000 caracteres; se encontraron {miles(len(texto))}"
    return texto


class Bloque(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(ANCHO)
        self.at = nn.MultiheadAttention(ANCHO, CABEZAS, batch_first=True)
        self.n2 = nn.LayerNorm(ANCHO)
        self.densa = nn.Sequential(nn.Linear(ANCHO, 4 * ANCHO), nn.GELU(),
                                   nn.Linear(4 * ANCHO, ANCHO))

    def forward(self, x, mascara):
        h = self.n1(x)
        a, _ = self.at(h, h, h, attn_mask=mascara, need_weights=False)
        x = x + a
        return x + self.densa(self.n2(x))


class Transformer(nn.Module):
    def __init__(self, vocabulario):
        super().__init__()
        self.emb = nn.Embedding(vocabulario, ANCHO)
        self.pos = nn.Embedding(CONTEXTO, ANCHO)
        self.bloques = nn.ModuleList([Bloque() for _ in range(CAPAS)])
        self.norma = nn.LayerNorm(ANCHO)
        self.salida = nn.Linear(ANCHO, vocabulario)

    def forward(self, idx):
        n = idx.shape[1]
        mascara = torch.triu(torch.ones(n, n, device=idx.device, dtype=torch.bool), 1)
        x = self.emb(idx) + self.pos(torch.arange(n, device=idx.device))
        for b in self.bloques:
            x = b(x, mascara)
        return self.salida(self.norma(x))


def lote(datos, rng):
    i = rng.integers(0, len(datos) - CONTEXTO - 1, size=LOTE)
    x = torch.stack([datos[j:j + CONTEXTO] for j in i]).to(DISPOSITIVO)
    y = torch.stack([datos[j + 1:j + CONTEXTO + 1] for j in i]).to(DISPOSITIVO)
    return x, y


def escribir(modelo, indice, letras, largo=LARGO_MUESTRA, arranque=ARRANQUE, semilla=SEMILLA):
    modelo.eval()
    g = torch.Generator(device="cpu").manual_seed(semilla)
    idx = torch.tensor([[indice[c] for c in arranque]], dtype=torch.long, device=DISPOSITIVO)
    salida = list(arranque)
    with torch.no_grad():
        for _ in range(largo):
            logits = modelo(idx[:, -CONTEXTO:])[0, -1]
            p = torch.softmax(logits.float() / TEMPERATURA, dim=-1).cpu()
            siguiente = int(torch.multinomial(p, 1, generator=g))
            salida.append(letras[siguiente])
            idx = torch.cat([idx, torch.tensor([[siguiente]], device=DISPOSITIVO)], dim=1)
    modelo.train()
    return "".join(salida).replace("\n", " ")


def acierto(modelo, datos, rng, n=EJEMPLOS_ACIERTO):
    modelo.eval()
    aciertos = vistos = 0
    with torch.no_grad():
        while vistos < n:
            x, y = lote(datos, rng)
            pred = modelo(x).argmax(-1)
            aciertos += int((pred == y).sum())
            vistos += y.numel()
    modelo.train()
    return aciertos / vistos


def entrenar(datos, vocabulario, indice, letras, pasos_totales, instantaneas):
    """Entrena `pasos_totales` pasos y fotografía en los de `instantaneas`, más uno al final.

    Devuelve las muestras como (paso, segundos transcurridos, texto): el paso es el dato
    reproducible y los segundos son de esta máquina, y por eso van etiquetados como tales."""
    fijar_semilla()
    rng = np.random.default_rng(SEMILLA)
    modelo = Transformer(vocabulario).to(DISPOSITIVO)
    opt = torch.optim.AdamW(modelo.parameters(), lr=TASA)
    curva, muestras = [], []
    pendientes = list(instantaneas)
    a_inicial = acierto(modelo, datos, np.random.default_rng(1))
    t0, pasos = time.time(), 0
    modelo.train()
    while True:
        if pendientes and pasos >= pendientes[0]:
            pendientes.pop(0)
            muestras.append((pasos, time.time() - t0, escribir(modelo, indice, letras)))
        if pasos >= pasos_totales:
            # La última foto, siempre: es la del final del entrenamiento.
            muestras.append((pasos, time.time() - t0, escribir(modelo, indice, letras)))
            break
        x, y = lote(datos, rng)
        opt.zero_grad()
        p = modelo(x)
        perdida = F.cross_entropy(p.reshape(-1, vocabulario), y.reshape(-1))
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        opt.step()
        pasos += 1
        if pasos % 20 == 0:
            curva.append((pasos, float(perdida.item())))
    a_final = acierto(modelo, datos, np.random.default_rng(1))
    return modelo, pasos, curva, muestras, a_inicial, a_final


def suavizar(v, ventana=VENTANA_SUAVIZADO):
    """La media de los últimos `ventana` valores. No es un dato nuevo: es el mismo dato
    visto de lejos, para que se vea la FORMA de la curva y no el temblor."""
    if len(v) < ventana:
        return list(v)
    fuera = []
    acumulado = 0.0
    for i, x in enumerate(v):
        acumulado += x
        if i >= ventana:
            acumulado -= v[i - ventana]
        fuera.append(acumulado / min(i + 1, ventana))
    return fuera


def leer_curva(ruta=SALIDA_CURVA):
    """La curva guardada, para poder redibujar la figura sin volver a entrenar.

    Entrenar esto son veinticinco minutos de GPU. Si la única manera de recuperar la
    figura es repetirlos, la figura acaba desincronizada del guion que dice haberla
    hecho —pasó: el CSV guardaba pasos y el guion dibujaba minutos— y nadie lo nota."""
    with open(ruta, newline="", encoding="utf-8") as fh:
        filas = list(csv.reader(fh))
    assert filas and filas[0] == ["pasos", "lo_mal_que_lo_hace"], \
        f"Se esperaba una cabecera ['pasos', 'lo_mal_que_lo_hace'] en {ruta}; se encontró {filas[:1]}"
    datos = [(int(a), float(b)) for a, b in filas[1:]]
    assert len(datos) >= MINIMO_PUNTOS_CURVA, \
        f"Se esperaban al menos {MINIMO_PUNTOS_CURVA} puntos en {ruta}; se encontraron {len(datos)}"
    return datos


def tramo_final(curva, desde=PASO_RECUADRO):
    """Los puntos del tramo que amplía el recuadro: del paso `desde` al último, ni uno más."""
    tramo = [c for c in curva if c[0] >= desde]
    assert len(tramo) >= 2, \
        f"Se esperaban al menos dos puntos desde el paso {desde}; se encontraron {len(tramo)}"
    assert tramo[-1] == curva[-1], "Se esperaba que el tramo llegara hasta el último punto"
    return tramo


def bajada_del_tramo(curva, desde=PASO_RECUADRO):
    """Cuánto baja el tramo del recuadro, del primer décimo al último, en tanto por uno.

    Es la cifra que el recuadro enseña dibujada. Se mide sobre el CSV y no sobre el
    dibujo: si lo que se ve en el recuadro fuera un efecto de estirar el eje y no una
    bajada de verdad, esto daría cero."""
    tramo = tramo_final(curva, desde)
    n = max(1, len(tramo) // 10)
    primero = sum(v for _, v in tramo[:n]) / n
    ultimo = sum(v for _, v in tramo[-n:]) / n
    assert primero > 0, "Se esperaba un primer décimo positivo"
    return (primero - ultimo) / primero


def figura(curva):
    """Lo mal que lo hace, paso a paso.

    Dos líneas: la medición cruda, que tiembla, y la misma suavizada, que enseña la
    forma. Las dos van explicadas DENTRO de la figura (regla 9): una figura tiene que
    poder entenderse sin el párrafo que la presenta, y aquí el gris y el negro son dos
    cosas distintas que nadie adivina."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(sin matplotlib: no se dibuja la figura)")
        return
    assert curva, "Se esperaba una curva con puntos; se encontró vacía"
    os.makedirs(os.path.dirname(SALIDA_FIGURA), exist_ok=True)
    t = [c[0] for c in curva]
    v = [c[1] for c in curva]
    fig, ax = plt.subplots(figsize=ANCHO_ALTO_FIGURA)
    ax.plot(t, v, linewidth=0.8, color="0.72", zorder=1, label=ETIQUETA_CRUDA)
    ax.plot(t, suavizar(v), linewidth=1.8, color="#222222", zorder=2,
            label=ETIQUETA_SUAVE)
    ax.set_title(TITULO_FIGURA, fontsize=10.5, pad=10)
    ax.set_xlabel("pasos de entrenamiento")
    # El eje de la izquierda no lleva números a propósito: el libro no usa notación, y lo
    # que importa aquí es la FORMA de la caída, no el valor. Lo que sí lleva es una
    # palabra en cada extremo, para que se sepa hacia dónde es peor.
    ax.set_ylabel("lo mal que lo hace")
    ax.set_yticks([])
    ax.text(-0.035, 0.99, "peor", transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color="0.35")
    ax.text(-0.035, 0.01, "mejor", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color="0.35")
    # El suelo del eje es el cero, y se ve que la línea no llega a él: el capítulo dice
    # justo eso, que baja y baja «sin llegar al suelo». Con el eje recortado por abajo esa
    # frase no se podría comprobar en el dibujo, y un eje sin números y además recortado
    # exagera la caída.
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", ".")))
    # La clave baja a la esquina de abajo a la izquierda porque el recuadro ocupa la de
    # arriba a la derecha; las dos palabras son las mismas.
    ax.legend(loc="lower left", frameon=False, fontsize=8.2, handlelength=2.2)

    # ---- el recuadro ampliado del tramo final ----
    tramo = tramo_final(curva)
    t2 = [c[0] for c in tramo]
    v2 = [c[1] for c in tramo]
    caja = ax.inset_axes([0.545, 0.57, 0.435, 0.26])
    caja.plot(t2, v2, linewidth=0.6, color="0.72", zorder=1)
    caja.plot(t2, suavizar(v2), linewidth=1.4, color="#222222", zorder=2)
    caja.set_xlim(t2[0], t2[-1])
    # Sin números en ninguno de los dos ejes: el vertical porque el dibujo grande tampoco
    # los lleva, y el horizontal porque el rótulo del recuadro ya dice de qué paso a qué
    # paso va, y una cifra suelta ahí abajo se pisaba con la línea del dibujo grande.
    caja.set_yticks([])
    caja.set_xticks([])
    for lado in caja.spines.values():
        lado.set_linewidth(0.6)
        lado.set_color("0.45")
    caja.set_title(ETIQUETA_RECUADRO.format(miles(t2[0]), miles(t2[-1])),
                   fontsize=7.4, pad=3.0, linespacing=1.25)
    caja.text(0.5, -0.09, NOTA_RECUADRO, transform=caja.transAxes, ha="center", va="top",
              fontsize=6.8, color="0.35")
    # Invariantes del recuadro: los puntos son exactamente los del tramo, el cero NO entra
    # en su eje vertical (que es para lo que está), y no se sale del dibujo grande.
    y0, _ = caja.get_ylim()
    assert y0 > 0, f"Se esperaba que el cero quedara fuera del recuadro; el eje empieza en {y0}"
    assert tramo == [c for c in curva if c[0] >= PASO_RECUADRO], \
        "Se esperaba que el recuadro llevara exactamente los puntos desde el paso del corte"
    assert ax.get_xlim()[0] <= t2[0] and t2[-1] <= ax.get_xlim()[1], \
        "Se esperaba que el tramo del recuadro cupiera dentro del eje del dibujo grande"
    bajada = bajada_del_tramo(curva)
    assert bajada > 0, \
        ("Se esperaba que el tramo del recuadro bajara; medido sobre el CSV baja "
         f"{bajada:.4f}. Con una curva plana esto da cero y el recuadro no se dibuja: "
         "lo que enseña tiene que ser el dato, no el estirón del eje.")

    fig.tight_layout()
    fig.savefig(SALIDA_FIGURA, dpi=200)
    plt.close(fig)
    print(f"Figura escrita en {SALIDA_FIGURA}")


def selftest():
    fallos = []
    texto = cargar_texto(600_000)
    letras = sorted(set(texto))
    indice = {c: i for i, c in enumerate(letras)}
    datos = torch.tensor([indice[c] for c in texto], dtype=torch.long)
    vocab = set(texto.split())

    # 1. TEST NULO — con el texto barajado letra a letra no hay estructura que aprender: el
    #    acierto tiene que quedarse muy por debajo del que se alcanza con texto real.
    barajado = list(texto[:400_000])
    np.random.default_rng(SEMILLA).shuffle(barajado)
    d_nulo = torch.tensor([indice[c] for c in barajado], dtype=torch.long)
    _, _, _, _, _, a_nulo = entrenar(d_nulo, len(letras), indice, letras, PASOS_SELFTEST, [])
    _, _, _, _, _, a_real = entrenar(datos, len(letras), indice, letras, PASOS_SELFTEST, [])
    print(f"[1] test nulo         acierto: texto real {coma(a_real, 3)}  texto barajado "
          f"{coma(a_nulo, 3)}")
    if a_nulo >= a_real:
        fallos.append(f"test nulo: el texto barajado alcanza {coma(a_nulo, 3)}, no menos que "
                      f"el real ({coma(a_real, 3)}); el montaje no distingue estructura")

    # 2. SEÑAL IMPLANTADA — una cadena rara repetida muchas veces debe acabar apareciendo.
    marca = "qxqxqx"
    implantado = texto[:300_000] + (" " + marca) * 3000
    letras2 = sorted(set(implantado))
    indice2 = {c: i for i, c in enumerate(letras2)}
    d2 = torch.tensor([indice2[c] for c in implantado], dtype=torch.long)
    m2, _, _, _, _, _ = entrenar(d2, len(letras2), indice2, letras2, PASOS_SELFTEST_MARCA, [])
    escrito = escribir(m2, indice2, letras2, 600, arranque="qxqx")
    print(f"[2] señal implantada  «{marca}» aparece {escrito.count(marca)} veces en 600 letras")
    if marca not in escrito:
        fallos.append(f"señal implantada: «{marca}» no aparece en lo que escribe")

    # 3. INVARIANTE DEL DOMINIO — la salida por letra son probabilidades: suman uno.
    modelo = Transformer(len(letras)).to(DISPOSITIVO)
    with torch.no_grad():
        p = torch.softmax(modelo(datos[:CONTEXTO].unsqueeze(0).to(DISPOSITIVO))[0].float(), -1)
    sumas = p.sum(-1)
    print(f"[3] invariante        {p.shape[1]} letras posibles; las probabilidades suman entre "
          f"{coma(float(sumas.min()), 6)} y {coma(float(sumas.max()), 6)}")
    if not torch.allclose(sumas, torch.ones_like(sumas), atol=1e-4):
        fallos.append("invariante: las probabilidades por letra no suman uno")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def ficha():
    """La ficha de la máquina del capítulo (Orden 79, 25 oct 2026): qué lee, cuántas piezas
    distintas, cuántos números lleva cada una, cuántas capas, cuántas miradas y cuántos pesos.
    Se cuenta del propio modelo, recién construido, SIN entrenar (el alfabeto sí se lee del
    corpus, que es rápido). No va al libro: es el respaldo de las cifras de la ficha del
    capítulo 10. Los pesos tienen que ser los que imprime la ejecución completa (--- TAMAÑO ---)."""
    letras = sorted(set(cargar_texto()))
    modelo = Transformer(len(letras))
    pesos = sum(p.numel() for p in modelo.parameters())
    miradas = {b.at.num_heads for b in modelo.bloques}
    assert len(miradas) == 1, f"Se esperaba el mismo número de miradas en todas las capas: {miradas}"
    filas = [
        ["qué lee", "letras"],
        ["piezas distintas", str(modelo.emb.num_embeddings)],
        ["números por pieza", str(modelo.emb.embedding_dim)],
        ["capas", str(len(modelo.bloques))],
        ["miradas por capa", str(miradas.pop())],
        ["letras de contexto", str(modelo.pos.num_embeddings)],
        ["pesos", miles(pesos)],
    ]
    print("--- LA FICHA DE LA MÁQUINA DEL CAPÍTULO 10 ---\n")
    print("\n".join(tabla_editorial(
        "La ficha de la máquina del capítulo 10", ["", "la máquina"], filas, "id",
        ["Contada del propio modelo, sin entrenar. Las piezas son los símbolos distintos del "
         "texto: las 42 letras y signos de ngrama.py y el salto de línea."])))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ficha", action="store_true",
                    help="imprime la ficha de la máquina (contada, sin entrenar) y sale")
    ap.add_argument("--pasos", type=int, default=PASOS_TOTAL,
                    help=f"pasos de entrenamiento (por omisión {PASOS_TOTAL})")
    ap.add_argument("--solo-figura", action="store_true",
                    help="redibuja la figura desde la curva ya guardada, sin entrenar")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if args.ficha:
        ficha()
        return
    if args.solo_figura:
        curva = leer_curva()
        print(f"{miles(len(curva))} puntos leídos de {SALIDA_CURVA} "
              f"(hasta el paso {miles(curva[-1][0])}). No se entrena nada.")
        figura(curva)
        return

    texto = cargar_texto()
    letras = sorted(set(texto))
    indice = {c: i for i, c in enumerate(letras)}
    datos = torch.tensor([indice[c] for c in texto], dtype=torch.long)
    print(f"Procesador usado: {DISPOSITIVO}")
    print(f"{miles(len(texto))} letras de libros en español, {len(letras)} símbolos "
          "distintos.\n")

    instantaneas = list(INSTANTANEAS_PASOS)
    modelo, pasos, curva, muestras, a_ini, a_fin = entrenar(
        datos, len(letras), indice, letras, args.pasos, instantaneas)

    total = sum(p.numel() for p in modelo.parameters())
    print("--- TAMAÑO ---")
    print(f"números ajustables: {miles(total)}")
    print(f"un modelo grande de hoy tiene del orden de {miles(MODELO_GRANDE)}, "
          f"unas {miles(round(MODELO_GRANDE / total))} veces más")
    print(f"pasos de entrenamiento: {miles(pasos)}\n")

    print("--- LO QUE ESCRIBE, EN CUATRO MOMENTOS ---")
    for paso, segundos, muestra in muestras:
        # El paso va delante porque es el dato: con la misma semilla, el paso 300 es el paso
        # 300 en cualquier ordenador. El reloj va detrás y dice «en esta máquina», porque es
        # lo único de esta línea que cambia según dónde se ejecute.
        reloj = (f"{segundos:.0f} s" if segundos < 300
                 else f"{coma(segundos / 60, 1)} min")
        print(f"\n[tras {miles(paso)} pasos — {reloj} en esta máquina]")
        print(muestra)

    print("\n--- ACIERTO AL ADIVINAR LA SIGUIENTE LETRA ---")
    print(f"al empezar: {pct(a_ini, 0)}")
    print(f"al acabar:  {pct(a_fin, 0)}")

    os.makedirs(os.path.dirname(SALIDA_CURVA), exist_ok=True)
    with open(SALIDA_CURVA, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["pasos", "lo_mal_que_lo_hace"]] +
                                 [[int(a), f"{b:.4f}"] for a, b in curva])
    print(f"\nCurva escrita en {SALIDA_CURVA}")
    figura(curva)


if __name__ == "__main__":
    main()
