import rasterio
from pathlib import Path

file = rasterio.open(Path.cwd() / "GOTOVO2" / "S1A_3channel.tif")
r, g, b = file.read()
print(r)