from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_fish_data():
    return pd.read_csv(Path("Fish.csv"))

fish = load_fish_data()
print(fish)

# caract de los peces:
# Species (Especie)
# Weight  (Peso)
# Length1 (Longitud vertical cm)
# Length2 (Longitud diagonal cm)
# Length3 (Longitud transversal)
# Height  (Altura del pez)
# Width   (Ancho del pez cm)

# -> varaible objetivo: Weigth (peso del pez)

# histograma
fish.hist(bins=50, figsize=(12,8))
plt.show()

# muestreo aleatorio
