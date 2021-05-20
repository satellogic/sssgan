from collections import OrderedDict 
import pandas as pd

#Based in https://github.com/gravitystorm/openstreetmap-carto/tree/master/style


class_dict=OrderedDict({
    "grass":set(["grass", "grass_2", "heath", "golf_course", "farmland", "farmland-line", "farmyard", "farmyard-line"]),
    "forest": set(["forest", "forest_2", "forest-text", "orchard", "scrub"]),
    "cemetery": set(["cemetery"]),
    "block" : set(["building-fill", "block_street", "church", "place_of_worship", "built-up-lowzoom"]),
    "residential": set(["residential", "residential-line", "land-color","land-color-2", "bare_ground"]),
    "allotments": set(["allotments"]),
    "commercial": set(["commercial", "commercial-line", "retail", "retail-line", "societal_amenities", "tourism", "pedestrian-fill", "transportation-area", "apron", "rest_area"]),
    "industrial": set(["railway","railway-line", "industrial", "industrial-line", "wastewater_plant", "wastewater_plant-line", "industrial_2"]),
    "parking": set(["garages", "parking"]),
    "construction": set(["construction", "construction_2", "built-up-z12","quarry", "water_works", "water_works-line", "sand", "beach"]),
    "sports": set(["pitch"]),
    #Road
    "motorway": set(["motorway-casing", "motorway-fill", "motorway-low-zoom", "motorway-low-zoom-casing", "motorway-shield", "trunk-casing", "trunk-fill", "trunk-low-zoom", "trunk-low-zoom-casing", "trunk-shield"]),
    "higway": set(["primary-casing", "primary-fill", "primary-low-zoom", "primary-low-zoom-casing", "primary-shield",
                    "secondary-casing", "secondary-fill", "secondary-low-zoom", "secondary-low-zoom-casing", "secondary-shield", "road-fill"
                    ]),
    "rail": set(["rail", "rail_2"]),
    "road": set(["default-casing", "tertiary-casing", "residential-casing", "tertiary-fill", "residential-fill", "living-street-fill"]),
    "footway": set(["footway-fill", "footway-fill-noaccess", "pedestrian-casing", "steps-fill-noaccess", "cycleway-fill", "cycleway-fill-noaccess", "bridleway-fill", "bridleway-fill-noaccess",
                    "track-fill", "track-fill-noaccess"]),
    "water": set(["water-color"]),

})

cat_df = pd.DataFrame([[k, v, i+1] for i,(k,v) in enumerate(class_dict.items())], columns=["category", "sub_cat", "code"])

subclass_to_class_dict = { e: k for k, s in class_dict.items() for e in s }

osm_render_class_color = OrderedDict({
    #Parks, woods, other green things
    "grass": "#cdebb0", 
    "grass_2":"#8cb873",
    "grass_3": "#dffce2",
    "forest": "#add19e" ,
    "forest_2": "#9dc29c",
    "scrub": "#c8d7ab",
    "forest-text": "#46673b"  ,
    "park": "#c8facc"         ,
    "allotments": "#c9e1bf"   ,
    "orchard": "#aedfa3" ,
    

    #base landuses
    "land-color": "#f2efe9",
    "land-color-2": "#f2eee9",
    "built-up-lowzoom": "#d0d0d0",
    "built-up-z12": "#dddddd",
    "residential": "#e0dfdf",      # Lch(89,0,0)
    "residential-line": "#b9b9b9", # Lch(75,0,0)
    "retail": "#ffd6d1",           # Lch(89,16,30)
    "retail-line": "#d99c95",      # Lch(70,25,30)
    "commercial": "#f2dad9",       # Lch(89,8.5,25)
    "commercial-line": "#d1b2b0",  # Lch(75,12,25)
    "industrial_2": "#e2cbde",
    "industrial": "#ebdbe8",       # Lch(89,9,330) (Also used for railway, wastewater_plant)
    "industrial-line": "#c6b3c3",  # Lch(75,11,330) (Also used for railway-line, wastewater_plant-line)
    "farmland": "#eef0d5",         # Lch(94,14,112)
    "farmland-line": "#c7c9ae",    # Lch(80,14,112)
    "farmyard": "#f5dcba",         # Lch(89,20,80)
    "farmyard-line": "#d1b48c",    # Lch(75,25,80)

    #Transport

    "transportation-area": "#e9e7e2",
    "apron": "#dadae0",
    "garages": "#dfddce",
    "parking": "#eeeeee",
    #"parking-outline": saturate(darken("parking, 40%), 20%),
    "rest_area": "#efc8c8", # also services

    #Other"
    "bare_ground": "#eee5dc",
    "campsite": "#def6c0", # also caravan_site, picnic_site
    "cemetery": "#aacbaf", # also grave_yard
    "construction": "#c7c7b4", # also brownfield
    "construction_2": "#b6b592",
    "heath": "#d6d99f",
    #"mud": rgba(203,177,154,0.3); # produces #e6dcd1 over "land
    "church": "#c4b6ab",
    #"place_of_worship_outline": darken("place_of_worship, 30%);
    #"leisure": lighten("park, 5%);
    #"power": darken("industrial, 5%);
    #"power-line": darken("industrial-line, 5%);
    "sand": "#f5e9c6",
    "societal_amenities": "#ffffe5",   # Lch(99,13,109)
    "tourism": "#660033",
    "quarry": "#c5c3c3",
    "military": "#f55",
    "beach": "#fff1ba",
    #Sports
    "pitch": "#aae0cb",# Lch(85,22,168) also track
    "golf_course": "#b5e3b5",

    #Building
    "building-fill": "#d9d0c9",  # Lch(84, 5, 68)
    "industrial_2": "#e2cbde",
    # @building-line: darken(@building-fill, 15%);  // Lch(70, 9, 66)
    # @building-low-zoom: darken(@building-fill, 4%);

    # @building-major-fill: darken(@building-fill, 10%);  // Lch(75, 8, 67)
    # @building-major-line: darken(@building-major-fill, 15%);  // Lch(61, 13, 65)
    # @building-major-z15: darken(@building-major-fill, 5%);  // Lch(70, 9, 66)
    # @building-major-z14: darken(@building-major-fill, 10%);  // Lch(66, 11, 65)

    # @entrance-permissive: darken(@building-line, 15%);
    # @entrance-normal: @building-line;

    #Adrressing
    # "addrres":"#666",
    # #Icons
    # "marina-text": "#576ddf", // also swimming_pool
    # #"wetland-text": darken("#4aa5fa", 25%), /* Also for mud */
    # "shop-icon": "#ac39ac",
    # "shop-text": "#939",
    # "transportation-icon": "#0092da",
    # "transportation-text": "#0066ff",
    # #"accommodation-icon": "transportation-icon,
    # #"accommodation-text": "transportation-text,
    # "airtransport": "#8461C4", #also ferry_terminal
    # "health-color": "#BF0000",
    # "amenity-brown": "#734a08",
    # "gastronomy-icon": "#C77400",
    # #"gastronomy-text": darken"gastronomy-icon, 5%),
    # #"memorials": "amenity-brown,
    # #"culture": "amenity-brown,
    # #"public-service": "amenity-brown,
    # "office": "#4863A0",
    # "man-made-icon": "#666666",
    # "advertising-grey": "man-made-icon,
    # "barrier-icon": "#3f3f3f",
    # "landform-color": "#d08f55",
    # #"leisure-green": darken"park, 60%),
    # "protected-area": "#008000",
    # "aboriginal": "#82643a",
    # "religious-icon": "#000000",

    #Powerline
    "power-line-color": "#888",
    "power-pole": "#928f8f",
    
    #Road
    "motorway-casing": "#dc2a67",
    "trunk-casing": "#c84e2f",
    "primary-casing": "#a06b00",
    "secondary-casing": "#707d05",
    "motorway-fill": "#e892a2",
    "trunk-fill": "#f9b29c",
    "primary-fill": "#fcd6a4",
    "secondary-fill": "#f7fabf",
    "motorway-low-zoom": "#e66e89",
    "trunk-low-zoom": "#f6967a",
    "primary-low-zoom": "#f4c37d",
    "secondary-low-zoom": "#e7ed9d",
    "motorway-low-zoom-casing": "#c24e6b",
    "trunk-low-zoom-casing": "#d1684a",
    "primary-low-zoom-casing": "#c78d2b",
    "secondary-low-zoom-casing": "#a4b329",
    "motorway-shield": "#620728",
    "trunk-shield": "#5f1c0c",
    "primary-shield": "#503000",
    "secondary-shield": "#364000",
    # Road
    "rail": "#707070",
    "rail_2": "#aaaaaa",
    "living-street-fill": "#ededed",
    "pedestrian-fill": "#dddde8",
    "raceway-fill": "#FFC0CB",
    "road-fill": "#ddd",
    "footway-fill": "#FA8072",
    "footway-fill-noaccess": "#bbbbbb",
    "cycleway-fill": "#0000FF",
    "cycleway-fill-noaccess": "#9999ff",
    "bridleway-fill": "#008000",
    "bridleway-fill-noaccess": "#aaddaa",
    "track-fill": "#996600",
    "track-fill-noaccess": "#e2c5bb",
    "aeroway-fill": "#bbc",
    "access-marking": "#eaeaea",
    "access-marking-living-street": "#cccccc",

    "default-casing": "#ffffff",
    "tertiary-casing": "#8f8f8f",
    "residential-casing": "#bbb",
    "pedestrian-casing": "#999",
    

    "tertiary-shield": "#3b3b3b",

    "minor-construction": "#aaa",

    "destination-marking": "#c2e0ff",
    "private-marking": "#efa9a9",
    "private-marking-for-red": "#c26363",

    "tunnel-casing": "#808080",
    "bridge-casing": "#000000",


    # @motorway-tunnel-fill: lighten(@motorway-fill, 10%);
    # @trunk-tunnel-fill: lighten(@trunk-fill, 10%);
    # @primary-tunnel-fill: lighten(@primary-fill, 10%);
    # @secondary-tunnel-fill: lighten(@secondary-fill, 5%);
    # @tertiary-tunnel-fill: lighten(@tertiary-fill, 5%);
    # @residential-tunnel-fill: darken(@residential-fill, 5%);
    # @living-street-tunnel-fill: lighten(@living-street-fill, 3%);

    #Station
    "station-color": "#7981b0",
    #@station-text: darken(saturate(@station-color, 15%), 10%);
    
    #Water
    "water-color": "#aad3df",
    "dam-line": "#444444",

})



osm_render_color_class = { v:k for k,v in osm_render_class_color.items()}

sub_cat_df = pd.DataFrame([[k, v, i+1] for i,(k,v) in enumerate(osm_render_class_color.items())], columns=["sub_category", "color", "code"])

area_rows = [ [area, idx] for idx, area in  enumerate(["vienna", "tyrol", "austin", "chicago"])]
area_df = pd.DataFrame(area_rows, columns=["area", "idx"])

#Kitsap exclude
kitsap = {
    "no": [1,21, 11, 12, 16, 17, 18, 2, 22, 23,28, 30, 31, 33, 4, 5, 7, 9],
    "modify":[10, 13, 14, 15, 24, 27, 29, 3, 32, 34, 35, 36, 6, 8],
    "yes": [20, 25, 24, 26]
}


