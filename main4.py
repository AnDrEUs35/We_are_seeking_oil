import ee
import geemap
ee.Initialize(project="nice-hydra-465212-a1")

# Определим область интереса (например, точка в Москве)
aoi = ee.Geometry.Point([37.11651362719904, 44.942910730786004])
# Выберем коллекцию Landsat 8 SR за июнь 2020 года
collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
    .filterBounds(aoi) \
    .filterDate('2024-12-15', '2025-01-30') \
    .sort('CLOUD_COVER')
image = collection.first()  # кадр с минимальным облачным покрытием

# Рассчитаем NDVI: (B5 - B4) / (B5 + B4)
ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
# Посмотрим метаданные результата (не полное изображение!)
print(ndvi.getInfo()['properties'])
m = geemap.Map(center=(37.11651362719904, 44.942910730786004), zoom=8)
m.addLayer(ndvi, {'min': 0, 'max': 0.5, 'palette': ['blue','white','green']}, 'NDVI')
m