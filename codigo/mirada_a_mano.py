#!/usr/bin/env python3
"""
Capítulo 8 — una mirada que nadie programa: un transformer diminuto, con la culpa repartida a mano.

El capítulo 8 cuenta que cada letra (allí, cada palabra) hace una pregunta, que las anteriores
tienen una etiqueta y un contenido, y que el reparto de la mirada «sale del entrenamiento: nadie se
lo dice». Este programa lo enseña en pequeño y sobre texto de verdad, el Quijote:

  - un transformer de una sola ronda y una sola mirada, que lee letras;
  - la ida (de las letras a la apuesta por la siguiente) y la vuelta (el reparto de la culpa entre
    todos los números) están ESCRITAS A MANO, fórmula a fórmula, en `adelante` y `atras`. No se usa
    el cálculo automático de PyTorch: solo sus tablas de números, que corren igual en el procesador
    y en la tarjeta gráfica. El cálculo automático se usa en el selftest, para comprobar que la
    culpa escrita a mano es la misma;
  - se mide cuánto acierta la letra siguiente, entre cuántas letras reparte la mirada y a dónde acaba
    mirando, en cuatro momentos del entrenamiento; se compara con no mirar atrás y con la máquina de
    contar del capítulo 1 sobre las mismas letras;
  - y un experimento aparte: el ajuste de escala de las puntuaciones que el capítulo 8 dice que hizo
    falta («sin ese ajuste el aprendizaje se atasca casi desde el principio»), con él y sin él.

Cómo se mueve cada número tras repartir la culpa: exactamente como en el capítulo 4, un poquito en
contra de su culpa y en proporción a ella (`ReglaDelCapitulo4`). Solo el experimento del ajuste de
escala usa el método de Adam (Kingma y Ba, 2014), también escrito a mano: con la regla simple hay que
buscar una tasa distinta para cada manera de empezar (una versión anterior de este programa necesitó
1,0, 0,3 y 4,0), y entonces no se sabe si la diferencia la pone el ajuste o la tasa. Adam da a cada
número su propio poquito y sirve igual para los cuatro casos.

Uso:
    python mirada_a_mano.py --selftest
    python mirada_a_mano.py --medir-velocidad        # procesador contra tarjeta, 300 pasos
    python mirada_a_mano.py > ../datos/salidas/mirada_a_mano.txt
"""

# ======================= CONSTANTES =======================

SEMILLA = 20260914
CORPUS = "../datos/quijote.txt"          # el mismo texto y la misma limpieza que ngrama.py
PARTE_APRENDER = 0.9                      # el 90 % primero para aprender, el 10 % último para probar

CONTEXTO = 16             # letras que puede mirar hacia atrás (la propia incluida)
ANCHO = 32                # números por letra
ANCHO_MEZCLA = 128        # números de la parte de mezclar (cuatro veces el ancho, como es costumbre)
LOTE = 64                 # trozos de texto por paso
TASA = 1.0                # el «poquito» del capítulo 4: cuánto se mueve cada número por unidad de
                          # culpa. Probado el 27 de septiembre: 0,3 y 1,0 dan lo mismo (44 %);
                          # con 3,0 los números se disparan y deja de aprender
TASA_ADAM = 3e-3          # el «poquito» de Adam, solo para el experimento del ajuste de escala
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
DESVIACION_INICIAL = 0.02 # los números empiezan al azar, pequeños (como GPT-2): el reparto nace plano

PASOS = 20_000
MOMENTOS = [0, 1_000, 5_000, 10_000, 20_000]  # cuándo se mide; el último tiene que ser PASOS.
# Con la regla del capítulo 4 la mirada se cierra más tarde que con Adam (a los 2.000 pasos aún
# reparte entre 15,7 letras), y hacen falta momentos intermedios para ver cuándo.
EJEMPLOS_PRUEBA = 20_000                  # trozos del 10 % final con los que se mide

SELECTORES = [0, 1, 2, 3, 5]
PASOS_REAPRENDER = 5_000   # apartado 7: con el reparto impuesto, cuánto se deja volver a aprender al resto
TASA_REAPRENDER = 0.1      # y con qué poquito. Medido el 27: con «todo a la de justo antes», 1,0 y 0,3
                           # disparan los números (se parte de una máquina que de golpe se equivoca
                           # mucho, y un poquito grande multiplica ese error); 0,1 y 0,03 dan lo mismo
                           # (38,4 y 38,1 %). «Por igual» aguanta hasta 1,0 y da 32,1 % con 1,0 y 0,3.                # la máquina de contar del capítulo 1, las mismas

# La frase en la que se enseña el reparto: la letra que tiene que adivinar es la que va detrás.
FRASE = "de la mancha, de cuyo nombre no quiero acordarme"
ARRANQUE_MUESTRA = "en un lugar de la mancha, de cuyo "
LARGO_MUESTRA = 300

# --- el experimento del ajuste de escala ---
ANCHO_ESCALA = 64                          # los 64 rasgos por mirada que cita el capítulo 8
PASOS_ESCALA = 3_000
MOMENTOS_ESCALA = [0, 100, 1_000, 3_000]
# Dos maneras de que empiecen los números: pequeños (0,02, como GPT-2) o del tamaño que dan por
# defecto las piezas de PyTorch (las letras con desviación 1, las tablas con 1/raíz del ancho). El
# argumento del artículo supone lo segundo; con lo primero, las puntuaciones nacen casi en cero.
INICIOS_ESCALA = {"pequeños (0,02)": "pequeno", "de tamaño 1": "unidad"}
OTRAS_SEMILLAS_ESCALA = [SEMILLA + 1, SEMILLA + 2]  # para ver si la diferencia es del azar

DISPOSITIVO = "cpu"       # se decide con --medir-velocidad; la salida dice cuál se usó

# --- selftest ---
PASOS_NULO = 3_000
PASOS_PERIODO = 3_000
PUNTOS_DIFERENCIAS = 24   # números que se comprueban moviéndolos un poquito, uno a uno
PASITO = 1e-6             # cuánto se mueven
TOLERANCIA_DIFERENCIAS = 1e-5  # el error de redondeo al restar dos errores casi iguales y dividir
                                # por un pasito de una millonésima ya es de este orden
PERIODO = 3               # texto fabricado en el que cada letra repite la de tres sitios atrás
TOLERANCIA_CULPA = 1e-6   # diferencia relativa máxima entre la culpa a mano y la automática


# ==========================================================

import argparse
import math
import platform
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch

from formato import coma, miles, muestra_editorial, tabla_editorial
from ngrama import ALFABETO, cargar_corpus, normalizar

AQUI = Path(__file__).resolve().parent
LETRAS = ALFABETO
V = len(LETRAS)
INDICE = {c: i for i, c in enumerate(LETRAS)}


# ---------------------------------------------------------------- el texto

def cargar_texto():
    texto = normalizar(cargar_corpus(AQUI / CORPUS), ALFABETO)
    ids = np.array([INDICE[c] for c in texto], dtype=np.int64)
    corte = int(len(ids) * PARTE_APRENDER)
    assert corte > 100 * CONTEXTO and len(ids) - corte > 100 * CONTEXTO, \
        f"se esperaba texto de sobra a los dos lados del corte; hay {corte} y {len(ids) - corte}"
    return texto, ids[:corte], ids[corte:]


def trozos(ids, rng, n):
    """n trozos al azar de CONTEXTO+1 letras: las CONTEXTO primeras entran, y cada una tiene que
    adivinar la de detrás."""
    inicio = rng.integers(0, len(ids) - CONTEXTO - 1, size=n)
    return np.stack([ids[i:i + CONTEXTO + 1] for i in inicio])


# ---------------------------------------------------------------- los números

def iniciar(ancho, mezcla, inicio, gen, dtype=torch.float32):
    """Todos los números ajustables, en un diccionario. `inicio` es «pequeno» o «unidad»."""
    def azar(filas, cols, desv):
        return (torch.randn(filas, cols, generator=gen, dtype=torch.float64) * desv).to(dtype)
    if inicio == "pequeno":
        d = dict(letra=DESVIACION_INICIAL, pos=DESVIACION_INICIAL, tabla=DESVIACION_INICIAL,
                 mezcla=DESVIACION_INICIAL)
    elif inicio == "unidad":
        d = dict(letra=1.0, pos=1.0, tabla=1 / math.sqrt(ancho), mezcla=1 / math.sqrt(ancho))
    else:
        raise AssertionError(f"se esperaba «pequeno» o «unidad»; llegó «{inicio}»")
    p = {
        "letra": azar(V, ancho, d["letra"]),          # la lista de números de cada letra
        "pos": azar(CONTEXTO, ancho, d["pos"]),       # y la de cada sitio
        "Wp": azar(ancho, ancho, d["tabla"]),         # la tabla que saca la pregunta
        "We": azar(ancho, ancho, d["tabla"]),         # la que saca la etiqueta
        "Wc": azar(ancho, ancho, d["tabla"]),         # la que saca el contenido
        "Wo": azar(ancho, ancho, d["tabla"]),         # la que devuelve lo traído a la letra
        "W1": azar(ancho, mezcla, d["mezcla"]),       # mezclar: ida
        "b1": torch.zeros(mezcla, dtype=dtype),
        "W2": azar(mezcla, ancho, d["mezcla"]),       # mezclar: vuelta
        "b2": torch.zeros(ancho, dtype=dtype),
        "Ws": azar(ancho, V, d["tabla"]),             # de los números a la apuesta por cada letra
        "bs": torch.zeros(V, dtype=dtype),
    }
    return p


def cuantos_numeros(p):
    return sum(t.numel() for t in p.values())


# ---------------------------------------------------------------- la ida, a mano

REPARTOS_IMPUESTOS = {
    "igual": "por igual entre las 16",
    "anterior": "todo a la de justo antes",
    "propia": "todo a sí misma",
}


def reparto_impuesto(cual, T, device, dtype):
    """Un reparto puesto a mano, para usarlo en lugar del aprendido (solo al medir, nunca al
    entrenar). Mira solo hacia atrás y cada fila suma 1, como el de verdad."""
    if cual == "igual":
        a = torch.tril(torch.ones(T, T, dtype=dtype, device=device))
        a = a / a.sum(-1, keepdim=True)
    elif cual == "anterior":
        a = torch.zeros(T, T, dtype=dtype, device=device)
        a[0, 0] = 1
        a[torch.arange(1, T), torch.arange(0, T - 1)] = 1
    elif cual == "propia":
        a = torch.eye(T, dtype=dtype, device=device)
    else:
        raise AssertionError(f"se esperaba uno de {sorted(REPARTOS_IMPUESTOS)}; llegó «{cual}»")
    assert torch.allclose(a.sum(-1), torch.ones(T, dtype=dtype, device=device))
    assert not torch.triu(a, 1).any(), "un reparto impuesto no puede mirar hacia delante"
    return a


def adelante(p, x, escalar=True, mirar=True, forzar=None):
    """x: (lote, CONTEXTO) índices de letras. Devuelve las apuestas (lote, CONTEXTO, V) y todo lo
    que hace falta para la vuelta."""
    B, T = x.shape
    ancho = p["letra"].shape[1]
    e = p["letra"][x] + p["pos"][:T]                          # cada letra, su lista de números
    if mirar:
        q = e @ p["Wp"]                                        # la pregunta de cada letra
        k = e @ p["We"]                                        # la etiqueta de cada letra
        v = e @ p["Wc"]                                        # el contenido de cada letra
        escala = 1 / math.sqrt(ancho) if escalar else 1.0
        s = (q @ k.transpose(1, 2)) * escala                   # cuánto encaja cada pregunta con cada etiqueta
        futuro = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        s = s.masked_fill(futuro, float("-inf"))               # no se puede mirar lo que viene
        a = torch.softmax(s, dim=-1)                           # el reparto: suma 1 en cada fila
        if forzar is not None:                                 # solo al medir (apartado 7)
            a = reparto_impuesto(forzar, T, x.device, a.dtype).expand(B, T, T)
        traido = a @ v                                         # la mezcla de contenidos
        h = e + traido @ p["Wo"]
    else:
        q = k = v = a = traido = None
        escala = 1.0
        h = e
    pre = h @ p["W1"] + p["b1"]                                # mezclar
    m = torch.relu(pre)
    g = h + m @ p["W2"] + p["b2"]
    apuestas = g @ p["Ws"] + p["bs"]
    cache = dict(x=x, e=e, q=q, k=k, v=v, a=a, traido=traido, h=h, pre=pre, m=m, g=g,
                 escala=escala, mirar=mirar, forzado=forzar is not None)
    return apuestas, cache


def perdida_y_culpa_de_salida(apuestas, y):
    """Lo mal que lo hace (entropía cruzada media) y su culpa sobre cada apuesta."""
    B, T, _ = apuestas.shape
    prob = torch.softmax(apuestas, dim=-1)
    n = B * T
    perdida = -torch.log(prob.reshape(n, V)[torch.arange(n, device=y.device), y.reshape(n)]).mean()
    d = prob.clone()
    d.reshape(n, V)[torch.arange(n, device=y.device), y.reshape(n)] -= 1
    return perdida, d / n


# ---------------------------------------------------------------- la vuelta, a mano

def atras(p, c, d_apuestas):
    """Reparte la culpa de las apuestas entre todos los números, en el orden contrario a la ida."""
    gr = {}
    t = lambda z: z.transpose(-1, -2)
    ancho = p["letra"].shape[1]
    plano = lambda z: z.reshape(-1, z.shape[-1])
    # apuestas = g Ws + bs
    gr["Ws"] = t(plano(c["g"])) @ plano(d_apuestas)
    gr["bs"] = plano(d_apuestas).sum(0)
    dg = d_apuestas @ t(p["Ws"])
    # g = h + relu(pre) W2 + b2
    gr["W2"] = t(plano(c["m"])) @ plano(dg)
    gr["b2"] = plano(dg).sum(0)
    dpre = (dg @ t(p["W2"])) * (c["pre"] > 0)
    gr["W1"] = t(plano(c["h"])) @ plano(dpre)
    gr["b1"] = plano(dpre).sum(0)
    dh = dg + dpre @ t(p["W1"])
    if c["mirar"]:
        # h = e + (a v) Wo
        gr["Wo"] = t(plano(c["traido"])) @ plano(dh)
        dtraido = dh @ t(p["Wo"])
        da = dtraido @ t(c["v"])                                # culpa de cada porción del reparto
        dv = t(c["a"]) @ dtraido
        ds = c["a"] * (da - (da * c["a"]).sum(-1, keepdim=True))  # a través del reparto (softmax)
        dq = (ds @ c["k"]) * c["escala"]
        dk = (t(ds) @ c["q"]) * c["escala"]
        if c["forzado"]:
            # con el reparto impuesto, la pregunta y la etiqueta no deciden nada: no tienen culpa, y
            # no se la pueden pasar a las letras. (Sin esto, una culpa inventada llegaba a las letras
            # y los números se disparaban: fallo del 27 de septiembre.)
            dq = torch.zeros_like(dq)
            dk = torch.zeros_like(dk)
        gr["Wp"] = t(plano(c["e"])) @ plano(dq)
        gr["We"] = t(plano(c["e"])) @ plano(dk)
        gr["Wc"] = t(plano(c["e"])) @ plano(dv)
        de = dh + dq @ t(p["Wp"]) + dk @ t(p["We"]) + dv @ t(p["Wc"])
    else:
        for n in ("Wp", "We", "Wc", "Wo"):
            gr[n] = torch.zeros_like(p[n])
        de = dh
    # e = letra[x] + pos
    T = c["x"].shape[1]
    gr["pos"] = torch.zeros_like(p["pos"])
    gr["pos"][:T] = de.sum(0)
    gr["letra"] = torch.zeros(V, ancho, dtype=de.dtype, device=de.device)
    gr["letra"].index_add_(0, c["x"].reshape(-1), plano(de))
    return gr


# ---------------------------------------------------------------- mover los números

class ReglaDelCapitulo4:
    """Mover cada número un poquito en contra de su culpa, en proporción a ella. Nada más."""

    def __init__(self, p, tasa=TASA):
        self.tasa = tasa

    def paso(self, p, gr):
        for k in p:
            p[k].sub_(self.tasa * gr[k])


class Adam:
    """Mover cada número un poquito en contra de su culpa, con un poquito propio para cada uno."""

    def __init__(self, p, tasa=TASA_ADAM):
        self.tasa, self.n = tasa, 0
        self.m = {k: torch.zeros_like(v) for k, v in p.items()}
        self.v = {k: torch.zeros_like(v) for k, v in p.items()}

    def paso(self, p, gr):
        self.n += 1
        for k in p:
            self.m[k].mul_(BETA1).add_(gr[k], alpha=1 - BETA1)
            self.v[k].mul_(BETA2).addcmul_(gr[k], gr[k], value=1 - BETA2)
            mh = self.m[k] / (1 - BETA1 ** self.n)
            vh = self.v[k] / (1 - BETA2 ** self.n)
            p[k].sub_(self.tasa * mh / (vh.sqrt() + EPS))


# ---------------------------------------------------------------- medir

def medir(p, prueba, escalar=True, mirar=True, forzar=None):
    """Acierto en la última letra de cada trozo de prueba (la que tiene las CONTEXTO letras delante),
    entre cuántas letras reparte su mirada, y a dónde mira más."""
    x, y = prueba[:, :CONTEXTO], prueba[:, CONTEXTO]
    aciertos, efectivas, destinos = 0, [], []
    for i in range(0, len(x), 2_000):
        xb = torch.from_numpy(x[i:i + 2_000]).to(DISPOSITIVO)
        ap, c = adelante(p, xb, escalar, mirar, forzar)
        assert torch.isfinite(ap).all(), "los números se han disparado: ya no son números (¿tasa grande?)"
        pred = ap[:, -1].argmax(-1).cpu().numpy()
        aciertos += int((pred == y[i:i + 2_000]).sum())
        if mirar:
            a = c["a"][:, -1].double().cpu().numpy()
            ent = -(a * np.log(np.clip(a, 1e-300, None))).sum(-1)
            efectivas.append(np.exp(ent))
            destinos.append(a.argmax(-1))
    r = {"acierto": aciertos / len(x)}
    if mirar:
        r["reparte_entre"] = float(np.concatenate(efectivas).mean())
        r["destino"] = np.concatenate(destinos)
    return r


ESPACIO = INDICE[" "]
CATEGORIAS = ["la propia letra", "la de justo antes", "un espacio",
              "la primera letra de la palabra", "otra"]


def clasificar(x, destino):
    """A qué letra mira más la última de cada trozo, en cinco categorías (CATEGORIAS), en este orden."""
    cuenta = dict.fromkeys(CATEGORIAS, 0)
    T = CONTEXTO
    for fila, j in zip(x, destino):
        d = T - 1 - j
        if d == 0:
            cuenta["la propia letra"] += 1
        elif d == 1:
            cuenta["la de justo antes"] += 1
        elif fila[j] == ESPACIO:
            cuenta["un espacio"] += 1
        else:
            espacios = [i for i in range(T - 1) if fila[i] == ESPACIO]
            primera = espacios[-1] + 1 if espacios else None
            cuenta["la primera letra de la palabra" if j == primera else "otra"] += 1
    return cuenta


def reparte_entre(a):
    """A cuántas letras equivale un reparto (la misma cuenta que la columna «reparte entre»)."""
    a = np.asarray(a, dtype=float)
    a = a[a > 0] / a.sum()
    return float(np.exp(-(a * np.log(a)).sum()))


def dentro_de(inicio):
    """De cada uno, cuántos números de las listas de las letras caen, al empezar, entre menos y más
    su tamaño (1 o 0,02)."""
    gen = torch.Generator().manual_seed(SEMILLA)
    p = iniciar(ANCHO_ESCALA, 4 * ANCHO_ESCALA, inicio, gen)
    tam = 1.0 if inicio == "unidad" else DESVIACION_INICIAL
    return float((p["letra"].abs() < tam).float().mean())


def contar_selector(aprender, prueba, k):
    """La máquina de contar del capítulo 1 con el selector en k: con las k letras de antes, la que
    más veces venía detrás en el texto de aprender. Si esa hoja está en blanco, cuenta como fallo y
    se dice cuántas veces pasó: no se rellena con nada."""
    from collections import Counter, defaultdict
    tabla = defaultdict(Counter)
    a = aprender.tolist()
    for i in range(k, len(a)):
        tabla[tuple(a[i - k:i])][a[i]] += 1
    mejor = {ctx: c.most_common(1)[0][0] for ctx, c in tabla.items()}
    guardados = sum(len(c) for c in tabla.values())   # cada (hoja, letra) con cuenta es un número
    aciertos = vacias = 0
    for fila in prueba:
        ctx = tuple(fila[CONTEXTO - k:CONTEXTO].tolist()) if k else ()
        if ctx not in mejor:
            vacias += 1
            continue
        aciertos += int(mejor[ctx] == fila[CONTEXTO])
    return aciertos / len(prueba), vacias, guardados


# ---------------------------------------------------------------- entrenar

def entrenar(aprender, pasos, momentos, prueba, ancho=ANCHO, escalar=True, mirar=True,
             inicio="pequeno", semilla=SEMILLA, al_medir=None,
             mover="capitulo4", congelar=(), desde=None, forzar=None, tasa=TASA):
    """Entrena desde cero, o `desde` unos números ya entrenados (se copian; los originales no se
    tocan). Con `forzar`, el reparto va impuesto también mientras aprende; entonces la pregunta y la
    etiqueta no deciden nada y tienen que ir en `congelar`."""
    if forzar is not None:
        assert {"Wp", "We"} <= set(congelar), "con el reparto impuesto, la pregunta y la etiqueta van congeladas"
    if desde is None:
        gen = torch.Generator().manual_seed(semilla)
        p = {k: v.to(DISPOSITIVO) for k, v in iniciar(ancho, 4 * ancho, inicio, gen).items()}
    else:
        p = {k: v.clone() for k, v in desde.items()}
    assert mover in ("capitulo4", "adam"), f"se esperaba «capitulo4» o «adam»; llegó «{mover}»"
    opt = ReglaDelCapitulo4(p, tasa) if mover == "capitulo4" else Adam(p)
    rng = np.random.default_rng(semilla)
    medidas = {}
    for paso in range(pasos + 1):
        if paso in momentos:
            medidas[paso] = medir(p, prueba, escalar, mirar, forzar)
            if al_medir:
                al_medir(paso, p)
        if paso == pasos:
            break
        lote = torch.from_numpy(trozos(aprender, rng, LOTE)).to(DISPOSITIVO)
        ap, c = adelante(p, lote[:, :-1], escalar, mirar, forzar)
        _, d = perdida_y_culpa_de_salida(ap, lote[:, 1:])
        gr = atras(p, c, d)
        for k in congelar:                 # estos números no se mueven nunca: se quedan al azar
            gr[k].zero_()
        opt.paso(p, gr)
    return p, medidas


def reparto_de_la_frase(p, frase=FRASE):
    """El reparto de la última letra de los CONTEXTO últimos de la frase, y la letra que apuesta."""
    tramo = frase[-CONTEXTO:]
    x = torch.tensor([[INDICE[ch] for ch in tramo]], device=DISPOSITIVO)
    ap, c = adelante(p, x)
    return tramo, c["a"][0, -1].double().cpu().numpy(), LETRAS[int(ap[0, -1].argmax())]


def escribir(p, rng, arranque=ARRANQUE_MUESTRA, largo=LARGO_MUESTRA):
    ids = [INDICE[ch] for ch in arranque]
    for _ in range(largo):
        x = torch.tensor([ids[-CONTEXTO:]], device=DISPOSITIVO)
        ap, _ = adelante(p, x)
        pr = torch.softmax(ap[0, -1].double(), -1).cpu().numpy()
        ids.append(int(rng.choice(V, p=pr / pr.sum())))
    return "".join(LETRAS[i] for i in ids)


# ---------------------------------------------------------------- selftest

def selftest():
    global DISPOSITIVO
    guardado, DISPOSITIVO = DISPOSITIVO, "cpu"
    fallos = []
    texto, aprender, probar = cargar_texto()
    rng = np.random.default_rng(SEMILLA)
    prueba = trozos(probar, rng, 5_000)

    # 1. TEST NULO — el mismo texto con las letras barajadas: ya no hay nada que mirar. El acierto
    #    no puede pasar de lo que da apostar siempre por la letra más frecuente (más un punto de
    #    margen), y con el texto de verdad, en los mismos pasos, tiene que pasar de largo.
    barajado = np.random.default_rng(SEMILLA).permutation(aprender)
    prueba_b = trozos(np.random.default_rng(SEMILLA + 1).permutation(probar), rng, 5_000)
    base = np.bincount(barajado, minlength=V).max() / len(barajado)
    _, mb = entrenar(barajado, PASOS_NULO, [PASOS_NULO], prueba_b)
    _, mr = entrenar(aprender, PASOS_NULO, [PASOS_NULO], prueba)
    ab, ar = mb[PASOS_NULO]["acierto"], mr[PASOS_NULO]["acierto"]
    print(f"[1] test nulo         barajado {coma(100 * ab)} % (la más frecuente da {coma(100 * base)} %); "
          f"texto de verdad {coma(100 * ar)} %")
    if ab > base + 0.01:
        fallos.append("test nulo: con las letras barajadas acierta más que apostar por la más frecuente")
    if ar < ab + 0.10:
        fallos.append("test nulo: con el texto de verdad no se separa del barajado")

    # 2. SEÑAL IMPLANTADA — (a) la culpa escrita a mano es la del cálculo automático de PyTorch, en
    #    todos los números, con y sin ajuste de escala; (b) en un texto fabricado donde cada letra
    #    repite la de PERIODO sitios atrás, la mirada tiene que ir a PERIODO-1 sitios atrás, o PERIODO más
    #    y acertar casi siempre.
    peor = 0.0
    # (y con el reparto impuesto del apartado 7, que tiene su propia vuelta: ahí la pregunta y la
    # etiqueta no tienen culpa, y el 27 de septiembre se la estaban pasando a las letras)
    for escalar, forzar in ((True, None), (False, None), (True, "igual"), (True, "anterior")):
        gen = torch.Generator().manual_seed(SEMILLA)
        p = iniciar(ANCHO, 4 * ANCHO, "unidad", gen, torch.float64)
        lote = torch.from_numpy(trozos(aprender, np.random.default_rng(1), 8))
        ap, c = adelante(p, lote[:, :-1], escalar, forzar=forzar)
        perd, d = perdida_y_culpa_de_salida(ap, lote[:, 1:])
        gr = atras(p, c, d)
        pa = {k: v.clone().requires_grad_(True) for k, v in p.items()}
        ap2, _ = adelante(pa, lote[:, :-1], escalar, forzar=forzar)
        perd2 = torch.nn.functional.cross_entropy(ap2.reshape(-1, V), lote[:, 1:].reshape(-1))
        perd2.backward()
        assert abs(float(perd) - float(perd2.detach())) < 1e-10, "la pérdida a mano no es la de PyTorch"
        for k in p:
            auto = pa[k].grad if pa[k].grad is not None else torch.zeros_like(gr[k])
            dif = (gr[k] - auto).abs().max() / (auto.abs().max() + 1e-12)
            peor = max(peor, float(dif))
    # y contra lo más elemental: mover un número un pasito arriba y abajo y ver cuánto cambia el error
    gen = torch.Generator().manual_seed(SEMILLA)
    p = iniciar(ANCHO, 4 * ANCHO, "unidad", gen, torch.float64)
    lote = torch.from_numpy(trozos(aprender, np.random.default_rng(5), 8))
    ap, c = adelante(p, lote[:, :-1])
    _, d = perdida_y_culpa_de_salida(ap, lote[:, 1:])
    gr = atras(p, c, d)
    elige = np.random.default_rng(6)
    peor_dif = 0.0
    for _ in range(PUNTOS_DIFERENCIAS):
        k = sorted(p)[elige.integers(len(p))]
        i = int(elige.integers(p[k].numel()))
        original = float(p[k].view(-1)[i])
        errores = []
        for signo in (1, -1):
            p[k].view(-1)[i] = original + signo * PASITO
            errores.append(float(perdida_y_culpa_de_salida(adelante(p, lote[:, :-1])[0], lote[:, 1:])[0]))
        p[k].view(-1)[i] = original
        a_ojo = (errores[0] - errores[1]) / (2 * PASITO)
        a_mano = float(gr[k].view(-1)[i])
        peor_dif = max(peor_dif, abs(a_ojo - a_mano) / max(abs(a_mano), 1e-3))
    # Cada 200 letras se eligen al azar PERIODO letras nuevas y se repiten: la única manera de acertar es
    # mirar PERIODO-1 sitios detrás de la última letra (o PERIODO más).
    trozo = 200
    base_f = np.random.default_rng(SEMILLA + 2).integers(0, V, size=(len(aprender) // 20 // trozo, PERIODO))
    fabricado = np.concatenate([np.tile(b, trozo // PERIODO + 1)[:trozo] for b in base_f])
    prueba_f = trozos(fabricado, np.random.default_rng(3), 2_000)
    _, mf = entrenar(fabricado, PASOS_PERIODO, [PASOS_PERIODO], prueba_f)
    dist = CONTEXTO - 1 - mf[PASOS_PERIODO]["destino"]
    # la letra que viene es la que había PERIODO sitios antes de ELLA, o sea PERIODO-1 antes de la
    # última que se ve (y cada PERIODO más atrás, que es la misma)
    multiplos = float(np.mean(dist % PERIODO == PERIODO - 1))
    af = mf[PASOS_PERIODO]["acierto"]
    print(f"[2] señal implantada  culpa a mano contra la automática: diferencia {peor:.1e}; "
          f"contra mover un pasito {PUNTOS_DIFERENCIAS} números: {peor_dif:.1e}; "
          f"texto que repite cada {PERIODO}: acierta {coma(100 * af)} %, mira a {PERIODO - 1}, "
          f"{2 * PERIODO - 1}, {3 * PERIODO - 1}… letras atrás el {coma(100 * multiplos)} %")
    if peor_dif > TOLERANCIA_DIFERENCIAS:
        fallos.append(f"señal: la culpa a mano no es lo que cambia el error al mover un número ({peor_dif:.1e})")
    if peor > TOLERANCIA_CULPA:
        fallos.append(f"señal: la culpa a mano se separa de la automática ({peor:.1e})")
    if af < 0.95 or multiplos < 0.90:
        fallos.append("señal: en el texto fabricado no encuentra la repetición")

    # 3. INVARIANTE — mirar solo hacia atrás: cambiar una letra no cambia nada de lo que va antes
    #    que ella; y cada reparto suma 1.
    gen = torch.Generator().manual_seed(SEMILLA)
    p = iniciar(ANCHO, 4 * ANCHO, "unidad", gen, torch.float64)
    x = torch.from_numpy(trozos(aprender, np.random.default_rng(4), 4)[:, :CONTEXTO])
    x2 = x.clone()
    x2[:, 10] = (x2[:, 10] + 7) % V
    a1, c1 = adelante(p, x)
    a2, _ = adelante(p, x2)
    antes = float((a1[:, :10] - a2[:, :10]).abs().max())
    despues = float((a1[:, 10:] - a2[:, 10:]).abs().max())
    suma = float((c1["a"].sum(-1) - 1).abs().max())
    print(f"[3] invariante        cambiar la letra 11 mueve lo de antes {antes:.1e} y lo de después "
          f"{despues:.1e}; los repartos suman 1 con error {suma:.1e}")
    if antes > 0 or despues == 0:
        fallos.append("invariante: la mirada ve el futuro, o cambiar una letra no cambia nada")
    if suma > 1e-12:
        fallos.append("invariante: un reparto no suma 1")

    DISPOSITIVO = guardado
    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


# ---------------------------------------------------------------- velocidad

def medir_velocidad():
    global DISPOSITIVO
    _, aprender, probar = cargar_texto()
    prueba = trozos(probar, np.random.default_rng(0), 2_000)
    for disp in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
        DISPOSITIVO = disp
        entrenar(aprender, 20, [], prueba)                     # calentar
        t0 = time.perf_counter()
        entrenar(aprender, 300, [], prueba)
        if disp == "mps":
            torch.mps.synchronize()
        print(f"{disp}: {coma(1000 * (time.perf_counter() - t0) / 300, 2)} ms por paso")


# ---------------------------------------------------------------- lo que imprime

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--medir-velocidad", action="store_true")
    args = ap.parse_args()
    if args.medir_velocidad:
        return medir_velocidad()
    print("--- selftest ---")
    codigo = selftest()
    if codigo or args.selftest:
        return codigo
    assert MOMENTOS[-1] == PASOS and MOMENTOS_ESCALA[-1] == PASOS_ESCALA
    for ch in FRASE + ARRANQUE_MUESTRA:
        assert ch in INDICE, f"«{ch}» no está en el alfabeto"

    texto, aprender, probar = cargar_texto()
    prueba = trozos(probar, np.random.default_rng(SEMILLA), EJEMPLOS_PRUEBA)
    # L24 (9 de octubre): cada apartado, en tablas y muestras editoriales (regla 6 ter). Las
    # cuentas, y el orden en que se hacen, no cambian.
    L = ["", "########## capítulo 8: una mirada que nadie programa ##########",
         f"máquina: {platform.system()} {platform.machine()}; calcula en: {DISPOSITIVO}. "
         f"Medido el {date.today().isoformat()}.",
         f"Semilla {SEMILLA}. Texto: el Quijote, {miles(len(texto))} letras:",
         f"{miles(len(aprender))} para aprender y {miles(len(probar))} para probar.", ""]
    ver = lambda ch: "_" if ch == " " else ch

    frases = {}
    t0 = time.perf_counter()
    p, med = entrenar(aprender, PASOS, MOMENTOS, prueba,
                      al_medir=lambda paso, p: frases.__setitem__(paso, reparto_de_la_frase(p)))
    minutos = (time.perf_counter() - t0) / 60
    _, med_sin = entrenar(aprender, PASOS, [PASOS], prueba, mirar=False)

    L += ["--- 1. EL TAMAÑO ---", ""] + tabla_editorial(
        "El tamaño de la máquina que mira", ["", "cuánto"],
        [["números ajustables", miles(cuantos_numeros(p))],
         ["rondas", "1"], ["miradas", "1"],
         ["letras de contexto", str(CONTEXTO)], ["números por letra", str(ANCHO)],
         ["pasos de entrenamiento", miles(PASOS)],
         ["lo que tarda en entrenar, en esta máquina", f"{coma(minutos)} min"],
         ["cada paso mueve cada número", f"{coma(TASA, 1)} veces su culpa, en contra"]], "id",
        ["Cada paso, con la regla del capítulo 4."]) + [""]

    assert len(MOMENTOS) == 5, "el título del apartado 2 dice «cinco momentos»"
    L += ["--- 2. LO QUE ACIERTA Y CÓMO REPARTE LA MIRADA ---", ""] + tabla_editorial(
        "Lo que acierta y cómo reparte la mirada, en cinco momentos",
        ["tras", "acierta la siguiente", "reparte entre"],
        [[f"{miles(paso)} pasos", f"{coma(100 * med[paso]['acierto'])} %",
          f"{coma(med[paso]['reparte_entre'])} letras"] for paso in MOMENTOS], "idd",
        [f"Medido con {miles(EJEMPLOS_PRUEBA)} fragmentos de prueba de {CONTEXTO} letras, que la "
         "máquina nunca ha visto. «Acierta la siguiente» dice, de cada cien, en cuántos la letra a la "
         "que da más porcentaje es la que viene de verdad; «reparte entre», a cuántas de las "
         f"{CONTEXTO} letras mira de media, de {CONTEXTO}, todas por igual, a 1, una sola."]) + [""]

    x = prueba[:, :CONTEXTO]
    c0, cf = clasificar(x, med[0]["destino"]), clasificar(x, med[PASOS]["destino"])
    L += ["--- 3. A DÓNDE MIRA MÁS LA ÚLTIMA LETRA ---", ""] + tabla_editorial(
        "A dónde mira más la última letra, al principio y al final",
        ["la letra a la que más mira", "al empezar", "al final"],
        [[cat, f"{coma(100 * c0[cat] / len(x))} %", f"{coma(100 * cf[cat] / len(x))} %"]
         for cat in CATEGORIAS], "idd",
        ["Cada columna cuenta, de cada cien fragmentos de prueba, en cuántos la letra más mirada es "
         "la de esa fila. Al empezar reparte casi por igual, y cuál sale la más mirada se decide por "
         "decimales: esa columna es casi azar."]) + [""]

    filas = []
    for k in SELECTORES:
        acc, vac, guard = contar_selector(aprender, prueba, k)
        filas.append([f"contar, selector en {k}", f"{coma(100 * acc)} %", miles(guard), miles(vac)])
    de_mirar = sum(p[n].numel() for n in ("Wp", "We", "Wc", "Wo"))
    filas += [["esta, sin mirar atrás", f"{coma(100 * med_sin[PASOS]['acierto'])} %",
               miles(cuantos_numeros(p) - de_mirar), ""],
              ["esta, mirando", f"{coma(100 * med[PASOS]['acierto'])} %",
               miles(cuantos_numeros(p)), ""]]
    L += ["--- 4. ¿SIRVE DE ALGO MIRAR? ---", ""] + tabla_editorial(
        "¿Sirve de algo mirar? Las mismas letras de prueba",
        ["la máquina", "acierta", "números", "vacías"], filas, "iddd",
        ["«Números» son los que guarda cada máquina; en la de contar, uno por cada hoja de su "
         "libreta y cada letra que vio detrás alguna vez. «Vacías» son los fragmentos de prueba que "
         "caen en una hoja en blanco: ahí la de contar no tiene nada que decir, y cuentan como "
         "fallo. Esta máquina no tiene libreta, y en sus filas esa columna va en blanco."]) + [""]

    tramo = frases[0][0]
    i_frase = texto.find(FRASE)
    assert texto.count(FRASE) == 1, "la frase tiene que salir una sola vez en el Quijote"
    viene = texto[i_frase + len(FRASE)]
    a0, af = frases[0][1], frases[PASOS][1]
    L += ["--- 5. EL REPARTO DE LA ÚLTIMA LETRA DE LA FRASE ---", ""] + tabla_editorial(
        f"El reparto de la última letra de «{FRASE[-CONTEXTO:]}»",
        ["sitio", "letra", "tras 0 pasos", f"tras {miles(PASOS)} pasos"],
        [[str(k + 1), ver(ch), str(round(100 * a0[k])), str(round(100 * af[k]))]
         for k, ch in enumerate(tramo)], "ccdd",
        ["De cada cien, cuánto de su mirada va a cada letra. El guion bajo es un espacio.",
         f"Lo más probable detrás: al empezar «{ver(frases[0][2])}», al final «{ver(frases[PASOS][2])}». "
         f"En el Quijote viene detrás: «{ver(viene)}».",
         f"Reparte entre: al empezar {coma(reparte_entre(a0))} letras; al final "
         f"{coma(reparte_entre(af))}."]) + [""]

    filas = []
    for nombre, ini in INICIOS_ESCALA.items():
        for escalar in (True, False):
            _, me = entrenar(aprender, PASOS_ESCALA, MOMENTOS_ESCALA, prueba[:5_000],
                             ancho=ANCHO_ESCALA, escalar=escalar, inicio=ini, mover="adam")
            filas.append([nombre, "sí" if escalar else "no"]
                         + [f"{coma(100 * me[s]['acierto'], 0)} %" for s in MOMENTOS_ESCALA]
                         + [coma(me[0]["reparte_entre"])])
    otras = []
    for sem in OTRAS_SEMILLAS_ESCALA:
        r = {}
        for escalar in (True, False):
            _, me = entrenar(aprender, PASOS_ESCALA, [PASOS_ESCALA], prueba[:5_000],
                             ancho=ANCHO_ESCALA, escalar=escalar, inicio="unidad", semilla=sem,
                             mover="adam")
            r[escalar] = me[PASOS_ESCALA]["acierto"]
        otras.append([str(sem), f"{coma(100 * r[True], 0)} %", f"{coma(100 * r[False], 0)} %"])
    dentro = {n: dentro_de(ini) for n, ini in INICIOS_ESCALA.items()}
    L += ["--- 6. EL AJUSTE DE ESCALA ---", ""] + tabla_editorial(
        "El ajuste de escala, con él y sin él",
        ["empiezan", "ajuste"] + [f"acierta, tras los pasos: {miles(s)}" for s in MOMENTOS_ESCALA]
        + ["reparte"], filas, "ic" + "d" * len(MOMENTOS_ESCALA) + "d",
        [f"La máquina del Quijote con {ANCHO_ESCALA} números por letra en lugar de {ANCHO}, y cada "
         "número moviéndose con su propio poquito. «Empiezan» dice de qué tamaño son al azar las "
         "listas de las letras antes de aprender: de tamaño 1, de cada cien números "
         f"{coma(100 * dentro['de tamaño 1'], 0)} caen entre menos 1 y 1; pequeños, entre menos 0,02 "
         "y 0,02. El acierto va de cada cien; «reparte», entre cuántas letras mira antes de aprender "
         "nada."]) + [""] + tabla_editorial(
        "El ajuste de escala, con otras semillas", ["semilla", "con ajuste", "sin ajuste"], otras,
        "cdd",
        [f"Números de tamaño 1; cuánto acierta tras {miles(PASOS_ESCALA)} pasos, de cada cien."]) + [""]

    # ---- 7. ¿mirar, o saber a dónde mirar?
    _, med_fija = entrenar(aprender, PASOS, [PASOS], prueba, congelar=("Wp", "We"))
    rotulos = ["la máquina", "acierta", "reparte entre"]
    L += ["--- 7. ¿MIRAR, O SABER A DÓNDE MIRAR? ---", ""] + tabla_editorial(
        "¿Mirar, o saber a dónde mirar?", rotulos,
        [["entrenada entera", f"{coma(100 * med[PASOS]['acierto'])} %",
          coma(med[PASOS]["reparte_entre"])],
         ["pregunta y etiqueta al azar, fijas", f"{coma(100 * med_fija[PASOS]['acierto'])} %",
          coma(med_fija[PASOS]["reparte_entre"])],
         ["sin mirar atrás", f"{coma(100 * med_sin[PASOS]['acierto'])} %", ""]], "idd",
        ["En la segunda fila se entrena todo menos las dos tablas que deciden a dónde mira, la de "
         "la pregunta y la de la etiqueta; el contenido sí aprende."]) + [""]
    filas = []
    for cual, nombre in REPARTOS_IMPUESTOS.items():
        mi = medir(p, prueba, forzar=cual)
        filas.append([nombre, f"{coma(100 * mi['acierto'])} %", coma(mi["reparte_entre"])])
    L += tabla_editorial(
        "La entrenada entera, con el reparto impuesto al usarla",
        ["el reparto impuesto", "acierta", "reparte entre"], filas, "idd",
        ["Impuesto: la máquina no cambia; solo se le dice a dónde mirar."]) + [""]
    _, mr = entrenar(aprender, PASOS_REAPRENDER, [PASOS_REAPRENDER], prueba, desde=p, tasa=TASA_REAPRENDER)
    mr_libre = mr
    filas = [["sin imponer nada (para comparar)", f"{coma(100 * mr[PASOS_REAPRENDER]['acierto'])} %",
              coma(mr[PASOS_REAPRENDER]["reparte_entre"])]]
    for cual in ("igual", "anterior"):
        _, mr = entrenar(aprender, PASOS_REAPRENDER, [PASOS_REAPRENDER], prueba, desde=p, forzar=cual,
                         congelar=("Wp", "We"), tasa=TASA_REAPRENDER)
        filas.append([REPARTOS_IMPUESTOS[cual], f"{coma(100 * mr[PASOS_REAPRENDER]['acierto'])} %",
                      coma(mr[PASOS_REAPRENDER]["reparte_entre"])])
    L += tabla_editorial(
        f"Con el reparto impuesto, y {miles(PASOS_REAPRENDER)} pasos para que el resto vuelva a aprender",
        rotulos, filas, "idd",
        ["Con el reparto impuesto, la pregunta y la etiqueta ya no deciden nada; el resto de los "
         "números vuelve a aprender, con el mismo poquito en las tres filas."]) + [""]
    _, m_igual = entrenar(aprender, PASOS_REAPRENDER, [PASOS_REAPRENDER], prueba, desde=p, tasa=TASA)
    L += tabla_editorial(
        f"La entrenada entera, {miles(PASOS_REAPRENDER)} pasos más sin imponer nada",
        ["cuánto se mueve", "acierta"],
        [[f"cada número {coma(TASA, 1)} veces su culpa (lo de siempre)",
          f"{coma(100 * m_igual[PASOS_REAPRENDER]['acierto'])} %"],
         [f"cada número {coma(TASA_REAPRENDER, 1)} veces su culpa",
          f"{coma(100 * mr_libre[PASOS_REAPRENDER]['acierto'])} %"]], "id",
        [f"Con los mismos fragmentos y en el mismo orden que sus {miles(PASOS_REAPRENDER)} "
         "primeros pasos."]) + [""]

    muestra = escribir(p, np.random.default_rng(SEMILLA))
    L += ["--- 8. LO QUE ESCRIBE AL FINAL ---", ""] + muestra_editorial(
        "Lo que escribe la máquina que mira, al final",
        [muestra[i:i + 60] for i in range(0, len(muestra), 60)],
        [f"Empieza por «{ARRANQUE_MUESTRA}»; lo demás lo escribe ella, letra a letra."])

    # si algo falla al imprimir, que la medición no se pierda: se deja entera en el disco antes
    Path(AQUI / "../datos/mirada_a_mano_bruto.txt").write_text("\n".join(L), encoding="utf-8")
    for l in L:
        print(l)
    return 0

if __name__ == "__main__":
    sys.exit(main())
