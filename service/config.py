from enum import Enum

class Cache(Enum):
    none = 1
    read = 2
    write = 3

class serviceConfig:
    cacheStrategy = Cache.write
    modelName = "SNIPER"
    modelVersion = 2
    modelStorePath = '/path/to/model/MaskRCNN/V?/'
    image_path = '/home/dingjian/Documents/vis_results/clint_region/Extra_Experiment_TIF/cropped_10_render.tif'
    groundTruthGeoJson = '/home/dingjian/Documents/Sniper/Davis_Monthan_AFB_20180814.tif.geojson'
    gtMappings = {
        'Buildings': 'Buildings',
        'Other': 'Buildings',
        'Hangar': 'Buildings',
        'Storage Tank': 'Buildings',
        'Residential': 'Buildings',

        'Cars': 'Vehicles',
        'Planes': 'Planes',
        'Fire Track': 'Vehicles',
        'Mil Bomber': 'Planes',
        'Mil Fighter': 'Planes',
        'Pickup Trucks': 'Vehicles',
        'Semi Trucks': 'Vehicles',
        'Vehicles': 'Vehicles'
    }
    modelMappings = {
        'Buildings': 'Buildings',
        'Planes': 'Planes',
        'Vehicles': 'Vehicles'
    }