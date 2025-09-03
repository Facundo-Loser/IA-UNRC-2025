from deap import base, creator, tools, algorithms
import random
from collections import defaultdict, deque

# =========================
# Datos del problema
# =========================

# Capacidad de transmisión (Mbps)
CAPACIDAD_MBPS = [
#   0   1   2   3
   [0,  10, 40, 1 ],  # 0
   [10, 0,  4,  12],  # 1
   [40, 4,  0,  6 ],  # 2
   [1,  12, 6,  0 ]   # 3
]

# Costo de implementación
COSTO_IMPL = [
#    0  1   2  3
    [0, 20, 4, 12],   # 0
    [20, 0, 10, 6],   # 1
    [4, 10, 0, 16],   # 2
    [12, 6, 16, 0]    # 3
]

N = len(CAPACIDAD_MBPS)          # cantidad de nodos
# Construimos la lista de aristas (i<j) para grafo no dirigido
EDGES = [(i, j) for i in range(N) for j in range(i+1, N)]
M = len(EDGES)                   # cantidad de aristas posibles

# Restricción: como máximo K enlaces
K = 3  # probá con N-1, N, etc., según el caso


# =========================
# Utilidades de grafo
# =========================

def is_connected(active_edges):
    """Chequea conectividad usando BFS sobre las aristas activas (no dirigidas)."""
    if N == 0:
        return True
    if N == 1:
        return True

    if not active_edges:
        return False

    adj = defaultdict(list)
    for (u, v) in active_edges:
        adj[u].append(v)
        adj[v].append(u)

    # Empezamos desde un nodo presente en alguna arista, si no hay, no está conectado
    start = active_edges[0][0]
    visited = set([start])
    q = deque([start])

    while q:
        u = q.popleft()
        for w in adj[u]:
            if w not in visited:
                visited.add(w)
                q.append(w)

    # Puede haber nodos aislados (sin ninguna arista incidente)
    return len(visited) == N


def solution_edges(individual):
    """Devuelve la lista de aristas activas según el bitstring del individuo."""
    return [EDGES[i] for i, bit in enumerate(individual) if bit == 1]


def sums_for(individual):
    """Suma capacidad y costo de las aristas activas."""
    cap = 0.0
    cost = 0.0
    for i, bit in enumerate(individual):
        if bit == 1:
            u, v = EDGES[i]
            cap += CAPACIDAD_MBPS[u][v]
            cost += COSTO_IMPL[u][v]
    return cap, cost


# =========================
# DEAP: tipos y toolbox
# =========================

# Multi-objetivo: ( +capacidad , -costo )
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

# Inicialización: bitstring con bias a ~ N-1 aristas (árbol) para ayudar factibilidad
def init_individual():
    # prob de activar una arista; lo ajustamos alrededor de (N-1)/M
    p_on = max(0.1, min(0.9, (N - 1) / max(1, M)))
    return creator.Individual([1 if random.random() < p_on else 0 for _ in range(M)])

def init_population(n):
    return [init_individual() for _ in range(n)]

toolbox.register("individual", init_individual)
toolbox.register("population", init_population, n=80)  # tamaño de población

# Evaluación con penalizaciones suaves
def evaluate(individual):
    active = solution_edges(individual)
    cap, cost = sums_for(individual)

    # Penalizaciones
    penalty = 0.0
    # 1) Conectividad
    if not is_connected(active):
        # Penalizamos fuerte: reducimos capacidad y aumentamos costo
        penalty += 1000.0
    # 2) Límite K
    edges_count = len(active)
    if edges_count > K:
        # Penalizamos proporcional al exceso
        penalty += 100.0 * (edges_count - K)

    # Aplicamos penalización (suave): baja capacidad, sube costo
    cap_eff = cap - penalty
    cost_eff = cost + penalty

    return cap_eff, cost_eff

toolbox.register("evaluate", evaluate)

# Operadores genéticos estándar
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / M)
toolbox.register("select", tools.selNSGA2)


# =========================
# Bucle evolutivo
# =========================

def main():
    random.seed(42)

    pop = toolbox.population()     # población inicial
    hof = tools.ParetoFront()      # frente de Pareto
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: tuple(sum(f) / len(f) for f in zip(*fits)))
    stats.register("min", lambda fits: tuple(min(f) for f in zip(*fits)))
    stats.register("max", lambda fits: tuple(max(f) for f in zip(*fits)))

    # Evaluar población inicial y establecer orden de NSGA-II
    pop = tools.selNSGA2(pop, len(pop))
    # IMPORTANTÍSIMO: limpiar crowding distance para próxima gen
    pop = list(pop)

    # Evolución (mu+lambda) típica de NSGA-II
    MU = len(pop)
    LAMBDA = len(pop)
    CXPB = 0.7
    MUTPB = 0.2

    NGEN = 60

    # Inicial eval (DEAP requiere fitness antes de usar selNSGA2 adecuadamente)
    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind in invalid:
        ind.fitness.values = toolbox.evaluate(ind)

    for gen in range(1, NGEN + 1):
        # Variación
        offspring = algorithms.varOr(pop, toolbox, LAMBDA, CXPB, MUTPB)

        # Evaluar descendencia
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        # Selección ambiental NSGA-II
        pop = tools.selNSGA2(pop + offspring, MU)

        # Estadísticas
        record = stats.compile(pop)
        if gen % 10 == 0 or gen == 1:
            print(f"gen {gen:3d} | avg={tuple(round(x,2) for x in record['avg'])} "
                  f"min={tuple(round(x,2) for x in record['min'])} "
                  f"max={tuple(round(x,2) for x in record['max'])}")

    # Actualizamos hall of fame con el pop final
    hof.update(pop)

    print("\n--- Frente de Pareto (capacidad, costo) ---")
    for i, ind in enumerate(hof):
        cap, cost = evaluate(ind)  # valores efectivos (con penalización, deberían ser ya factibles)
        # Si querés ver factibilidad real:
        act = solution_edges(ind)
        connected = is_connected(act)
        print(f"{i+1:2d}. cap={cap:.2f}  cost={cost:.2f}  edges={len(act)}  connected={connected}  genes={ind}")

    # Mostrar soluciones factibles “reales” (sin penalización) ordenadas por costo
    factibles = []
    for ind in hof:
        act = solution_edges(ind)
        if is_connected(act) and len(act) <= K:
            cap, cost = sums_for(ind)  # sin penalización
            factibles.append((cap, cost, ind))

    if factibles:
        factibles.sort(key=lambda x: ( -x[0], x[1] ))  # más capacidad, menos costo
        print("\n--- Soluciones factibles del frente (sin penalización) ---")
        for i, (cap, cost, ind) in enumerate(factibles, 1):
            act = solution_edges(ind)
            print(f"{i:2d}. cap={cap:.2f}  cost={cost:.2f}  edges={len(act)}  enlaces={act}")
    else:
        print("\n(No se hallaron soluciones factibles en el frente; podés subir NGEN, POP o ajustar penalizaciones/K.)")

if __name__ == "__main__":
    main()
