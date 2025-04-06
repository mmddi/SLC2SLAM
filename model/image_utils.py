
import numpy as np




nyu13_colour_code = (np.array([[0, 0, 0],
                       [0, 0, 1], # BED
                       [0.9137,0.3490,0.1882], #BOOKS
                       [0, 0.8549, 0], #CEILING
                       [0.5843,0,0.9412], #CHAIR
                       [0.8706,0.9451,0.0941], #FLOOR
                       [1.0000,0.8078,0.8078], #FURNITURE
                       [0,0.8784,0.8980], #OBJECTS
                       [0.4157,0.5333,0.8000], #PAINTING
                       [0.4588,0.1137,0.1608], #SOFA
                       [0.9412,0.1373,0.9216], #TABLE
                       [0,0.6549,0.6118], #TV
                       [0.9765,0.5451,0], #WALL
                       [0.8824,0.8980,0.7608]])*255).astype(np.uint8)


# color palette for nyu34 labels
nyu34_colour_code = np.array([
       (0, 0, 0),

       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair

       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
    #    (148, 103, 189),		# bookshelf

       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),       # blinds
       (247, 182, 210),		# desk
       (66, 188, 102),      # shelves

       (219, 219, 141),		# curtain
    #    (140, 57, 197),    # dresser
       (202, 185, 52),      # pillow
    #    (51, 176, 203),    # mirror
       (200, 54, 131),      # floor

       (92, 193, 61),       # clothes
       (78, 71, 183),       # ceiling
       (172, 114, 82),      # books
       (255, 127, 14), 		# refrigerator
       (91, 163, 138),      # tv

       (153, 98, 156),      # paper
       (140, 153, 101),     # towel
    #    (158, 218, 229),		# shower curtain
       (100, 125, 154),     # box
    #    (178, 127, 135),       # white board

    #    (120, 185, 128),       # person
       (146, 111, 194),     # night stand
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209),      # lamp

       (227, 119, 194),		# bathtub
       (213, 92, 176),      # bag
       (94, 106, 211),      # other struct
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)       # other prop
    ]).astype(np.uint8)



# color palette for nyu40 labels
nyu40_colour_code = np.array([
       (0, 0, 0),

       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair

       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf

       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),       # blinds
       (247, 182, 210),		# desk
       (66, 188, 102),      # shelves

       (219, 219, 141),		# curtain
       (140, 57, 197),    # dresser
       (202, 185, 52),      # pillow
       (51, 176, 203),    # mirror
       (200, 54, 131),      # floor

       (92, 193, 61),       # clothes
       (78, 71, 183),       # ceiling
       (172, 114, 82),      # books
       (255, 127, 14), 		# refrigerator
       (91, 163, 138),      # tv

       (153, 98, 156),      # paper
       (140, 153, 101),     # towel
       (158, 218, 229),		# shower curtain
       (100, 125, 154),     # box
       (178, 127, 135),       # white board

       (120, 185, 128),       # person
       (146, 111, 194),     # night stand
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209),      # lamp

       (227, 119, 194),		# bathtub
       (213, 92, 176),      # bag
       (94, 106, 211),      # other struct
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)       # other prop
    ]).astype(np.uint8)


if __name__ == "__main__":
        # nyu40_class_name_string = ["void",
        # "wall", "floor", "cabinet", "bed", "chair",
        # "sofa", "table", "door", "window", "book", 
        # "picture", "counter", "blinds", "desk", "shelves",
        # "curtain", "dresser", "pillow", "mirror", "floor",
        # "clothes", "ceiling", "books", "fridge", "tv",
        # "paper", "towel", "shower curtain", "box", "white board",
        # "person", "night stand", "toilet", "sink", "lamp",
        # "bath tub", "bag", "other struct", "other furntr", "other prop"] # NYUv2-40-class

        # legend_img_arr = plot_semantic_legend(np.arange(41), nyu40_class_name_string, 
        # colormap=nyu40_colour_code,
        # save_path="/home/shuaifeng/Documents/PhD_Research/SemanticSceneRepresentations/SSR",
        # img_name="nyu40_legned.png")


        nyu13_class_name_string = ["void",
                    "bed", "books", "ceiling", "chair", "floor",
                    "furniture", "objects", "painting/picture", "sofa", "table",
                    "TV", "wall", "window"] # NYUv2-13-class

        legend_img_arr = plot_semantic_legend(np.arange(14), nyu13_class_name_string, 
        colormap=nyu13_colour_code,
        save_path="/home/shuaifeng/Documents/PhD_Research/SemanticSceneRepresentations/SSR",
        img_name="nyu13_legned.png")