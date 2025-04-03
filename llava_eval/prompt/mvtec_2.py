from mvtec import MVTEC_PROMPT_V1_1_PROD as MVTEC_PRODUCTION_V0_PLUS, MVTEC_PROMPT_V1_PROD as MVTEC_CONTEXT_V1, MVTEC_PROMPT_V1_2_PROD as MVTEC_CONTEXT_V3
import copy
import json

def read_info(p):
    with open(p, 'r') as f:
        return json.load(f)
    
FASTMODE_QUESTION = """Are there any defects in this image?
A. Yes
B. No
Answer with the option's letter from the given choices directly."""

FASTMODE_CNAME_QUESTION = """Are there any defects on the {product_name} in this image?
A. Yes
B. No
Answer with the option's letter from the given choices directly."""

DETAILMODE_CNAME_QUESTION = """Are there any defects on the {product_name} in this image?
A. Yes
B. No"""

DETAILMODE_QUESTION = """Are there any defects in this image?
A. Yes
B. No"""

DETAILMODE2_QUESTION = """Are there any defects in this image? Give me more details about the product and defects."""

BASE_0SHOT_PROMPT="""Referencing the image that is shown below, please answer the question:
Image:
<image>
Question: Are there any defects in this image?
A. Yes
B. No
Answer with the option's letter from the given choices directly."""

BASE_0SHOT_PROMPT_CNAME="""Referencing the image that is shown below, please answer the question:
Image:
<image>
Question: Are there any defects on the {class_name} in this image?
A. Yes
B. No
Answer with the option's letter from the given choices directly."""

BASE_0SHOT_PROMPT_WITH_CONTEXT_AND_CNAME="""Referencing the image and production process that are shown below, please answer the question:
Image:
<image>
{context}
Question: Are there any defects on the {class_name} in this image?
A. Yes
B. No
Answer with the option's letter from the given choices directly."""


BASE_0SHOT_PROMPT_NOHINT_WITH_CONTEXT_AND_CNAME="""Referencing the image and production process that are shown below, please answer the question:
Image:
<image>
{context}
Question: Are there any defects on the {class_name} in this image?
A. Yes
B. No"""


BASE_0SHOT_PROMPT_WITH_PDESIGN_AND_CNAME="""Based on the image and the ideal appearance displayed below, please respond to the following question:
Image:
<image>
{context}
Question: Are there any defects on the {class_name} in this image?
A. Yes
B. No
Answer with the option's letter from the given choices directly."""

BASE_0SHOT_PROMPT_NOHINT_WITH_PDESIGN_AND_CNAME="""Based on the image and the ideal appearance displayed below, please respond to the following question:
Image:
<image>
{context}
Question: Are there any defects on the {class_name} in this image?
A. Yes
B. No"""

########################################

BASE_1SHOT_PROMPT="""Referencing the images that are shown below, please answer the question:
Image:
<image>
<image>
Question: Are there any defects in the first image?
A. Yes
B. No
Answer with the option's letter from the given choices directly."""

BASE_1SHOT_PROMPT_WITH_CONTEXT="""Referencing the images and production process that are shown below, please answer the question:
Image:
<image>
<image>
{context}
Question: Are there any defects in the first image?
A. Yes
B. No
Answer with the option's letter from the given choices directly."""

########################################

# As same as V4 in production.py
MVTEC_CNAME_MAPS_V1 = {
    'screw': 'screw',
    'pill': "pill",
    'capsule': "capsule",
    'carpet': "carpet",
    'grid': "grid",
    'tile': "marble tile",
    'wood': "wooden board",
    'zipper': "zipper",
    'cable': "cable",
    'toothbrush': "toothbrush",
    'transistor': "transistor",
    'metal_nut': "metal nut",
    'bottle': "glass bottle",
    'hazelnut': "hazelnut",
    'leather': "leather", 
}

########################################

MVTEC_CONTEXT_V3_FIXED = copy.copy(MVTEC_CONTEXT_V3)
MVTEC_CONTEXT_V3_FIXED['pill'] = """The following is the production process of the pills:
1. Ingredient Preparation: 
   - Active Ingredients: Cranberry extract or flavoring is prepared, possibly as a powder or liquid concentrate.
   - Binding Agents: Common binding agents like sucrose, dextrose, or sorbitol are measured out.
   - Coloring and Additives: Red specks are likely made from small particles of dried cranberry or an artificial coloring agent.
2. Mixing: 
   - The active ingredients, binding agents, and additives are thoroughly mixed in a large blender to create a blend. The mixture should have cranberry specks throughout the powder.
3. Granulation: 
   - The mixture undergoes wet or dry granulation to form granules, which help in better compression during tablet formation. This step ensures that the mixture has the correct flow properties and compressibility.
4. Drying: 
   - If wet granulation was used, the granules are dried to remove any moisture content, ensuring the mixture is ready for tablet compression.
5. Compression: 
   - The dried granules are fed into a tablet press machine where they are compressed into the lozenge shape. The machine uses a die to form the oval shape, and embosses the letters “FF” on the surface of each lozenge.
6. Cooling: 
   - After compression, the lozenges are allowed to cool to stabilize their structure and harden fully.\n"""

MVTEC_CONTEXT_V3_FIXED['cable'] = """The following is the production process of the cables:
1. Copper Wire Drawing: 
   - Start with large copper rods and draw them through a series of progressively smaller dies to create fine strands of copper wire.
2. Stranding:
   - Multiple fine copper wires are stranded together to form a single conductor for each wire.
3. Insulation Extrusion:
   - Extrude insulation material around each stranded copper conductor. The insulation material is heated and then extruded around the wire as it passes through a die.
4. Color Coding:
   - During the insulation process, the insulation is color-coded with different color to differentiate the wires within the cable.
5. Cable Assembly:
   - The three insulated wires are then twisted or laid together to form a single cable. This step ensures that the wires are held together in a compact and organized manner.
6. Outer Sheath Extrusion:
   - An outer sheath is extruded around the assembled wires to provide additional protection and integrity to the cable.
7. Cooling:
   - The extruded cable is passed through a cooling bath or air-cooling system to solidify the insulation and outer sheath materials.\n"""

MVTEC_CONTEXT_V3_FIXED['metal_nut'] = """The following is the production process of the metal nuts:
1. Material Selection:
   - Choose the appropriate metal, likely steel or stainless steel, for durability and strength.
2. Blanking:
   - Cut the raw material into blanks, which are flat pieces of metal in the general shape of the T-nut.
3. Forming:
   - Use a stamping machine to form the basic shape of the T-nut, including the central hole and the four prongs.
   - The stamping process will also create the initial profile of the prongs.
4. Threading:
   - Tap the central hole to create the internal threads necessary for the T-nut to function properly with bolts or screws.
5. Prong Shaping:
   - The prongs are then shaped further, potentially through a combination of stamping and bending, to ensure they are correctly angled and aligned in accordance with the T-nut's clock-wise rotation direction for optimal insertion into the material.
6. Heat Treatment:
   - The T-nut may undergo heat treatment to harden the material, increasing its strength and durability.
7. Surface Finishing:
   - Apply a surface treatment, such as galvanization or a coating, to enhance corrosion resistance and provide a uniform appearance.
8. Final Shaping and Trimming:
   - Any excess material is trimmed, and the prongs are given a final shape to ensure they will anchor securely when used.\n"""

########################################

MVTEC_PRODUCTION_V0_FINAL = {

    "cable" : (
        """The following is the ideal appearance of a defect-free cable:\nThis image shows a cross-section of a cable containing three insulated wires within a larger protective sheath, without any defects. The color of the top wire is closed to yellow color. The color of the bottom left wire is closed to blue color. The color of the bottom right wire is closed to brown color. Each one from three wires contains multiple strands of copper wires. All three wires are tightly encased in a white circular outer sheath, which provides additional protection and insulation. The arrangement of the wires within the sheath is such that they form a triangular pattern, with the yellow wire at the top and the blue and brown wires at the bottom. This kind of cable design is typically used for electrical power transmission or data communication, ensuring that each wire is well-insulated and protected from external interference.\n"""
    ), 
    "hazelnut": (
        """The following is the ideal appearance of a defect-free hazelnut:\nThis image shows a single hazelnut against a dark background. The hazelnut has a smooth, brown shell with a slight sheen, indicating its maturity and readiness for consumption. The top part of the hazelnut is covered by a lighter, rougher cap, which appears to be slightly worn, showing a natural, irregular pattern. The shell's texture transitions from smooth on the sides to a rougher surface near the cap, displaying subtle vertical striations along the sides.\n"""
    ),  
    "transistor": (
        """The following is the ideal appearance of a defect-free transistor:\nThe image shows a transistor mounted on a perforated board. The transistor has a black body with three metallic leads protruding from the bottom. On the front, there is a visible marking, which looks like a 'T'. The background is a perforated copper board, commonly used for prototyping electronic circuits. Each lead is bent downward and inserted into a separate hole in the board, making electrical contact with the perforated board.\n"""
    ),  
    'zipper': (
        """The following is the ideal appearance of a defect-free zipper:\nThe image shows a close-up view of a black zipper. The fabric part of the zipper is black and has a textured, woven appearance. The texture appears to be somewhat rough. In the meantime, the background is white. Thus there are sometimes tiny white spots on the fabric.The zipper teeth are also black and are made from plastic, which is common for lightweight and flexible zippers. The teeth are evenly spaced and have a continuous, interlocking pattern. The zipper appears to be in good condition with no visible signs of wear or damage. The teeth are aligned properly, indicating that the zipper would function smoothly.\n"""
    ),
    "screw": (
        """The following is the ideal appearance of a defect-free screw:\nThe image shows a metal screw. The screw has a flat, countersunk head with a conical shape. The shank is partially threaded, with threading starting from the head. The unthreaded portion of the shank near the head is smooth and cylindrical. The tip is sharp and pointed, which helps in penetrating the material easily. The screw is metallic, likely made of steel or a similar durable material.\n"""
    ), 

    "metal_nut":(
        """The following is the ideal appearance of a defect-free metal nut:\nThe image shows a metal nut with without any defects when viewed from above. The nut has a central circular hole, which is typical for a nut, allowing a bolt to pass through. Surrounding the central hole, the nut has a cross-like pattern with four equally spaced arms. Each arm extends outward from the center, creating a symmetrical design. The edges of the arms are rounded. The surface of the nut appears to be metallic, likely made of steel or another durable metal, suitable for industrial or mechanical applications. The texture of the metal shows some machining marks The color is metallic, likely silver or gray, common for such components. The arms of the nut are oriented in a way that they appear to rotate in a clockwise direction when viewed from above.\n"""
    ),
    
    "toothbrush" : (
        """The following is the ideal appearance of a defect-free toothbrush:\nThis image shows a toothbrush head from a top-down view. The toothbrush head is predominantly white with clusters of bristles arranged in a grid-like pattern. The bristles are organized into several rows, with each row containing multiple tufts of bristles. The bristles themselves are two-toned. In addition to white bristles, normal toothbrushes also have bristles in one of the following colors: red, yellow, or blue. This alternating color pattern appears in each tuft of bristles.\n"""
    ),
    "tile" : (
        """The following is the ideal appearance of a defect-free tile:\nThe surface of this tile features a speckled pattern. The background color appears to be a light gray or off-white, overlaid with numerous irregular black spots and splotches. The spots vary in size and shape, giving the surface a textured, almost organic look. The surface seems smooth, typical of ceramic tiles, and the pattern is random and natural-looking.\n"""
    ),
    "bottle" : (
        """The following is the ideal appearance of a defect-free bottle:\nThe image shows the top view of a glass bottle opening. The bottle's interior appears to be dark, indicating that it is either filled with a dark liquid or is empty and dark due to lighting or the bottle's color. The bottle's rim is smooth, with some minor reflections visible on the glass surface, suggesting that it is clean and has a glossy finish. There is a faint orange or brownish ring around the inner edge, possibly due to residue or reflections.\n"""
    ),
    "wood": (
        """The following is the ideal appearance of a defect-free wood:\nThe surface of the wood plank in the image features a natural grain pattern with a uniform, smooth texture. The grain lines are predominantly straight and parallel, creating a subtle yet consistent flow across the surface. The wood color is a warm, medium brown with slight variations in tone. The grain lines are slightly darker than the surrounding wood, enhancing the depth and character of the surface. The finish appears to be matte, with no signs of gloss or shine.\n"""
    ), 
    "capsule": (
        """The following is the ideal appearance of a defect-free capsule:\nThis image shows a capsule with two distinct parts, each of different colors. The left half of the capsule is black, and the right half is orange. The black part has the word "actavis" printed in white, while the orange part has "500" printed in white. Both ends of the capsule are rounded. The surface of the capsule is smooth and shiny, indicating it is likely made of a gelatin or similar ingestible material. The capsule is centrally positioned against a plain white background, which is free of any other markings or patterns.\n"""
    ), 
    "pill": (
        """The following is the ideal appearance of a defect-free pill:\nThe image shows an oval-shaped pill. The pill is white with red speckles scattered across its surface. There are two embossed letters "FF" on one side of the pill, centered in the middle. The edges of the pill are smooth, and the overall texture appears slightly mottled due to the presence of the red speckles. The background is black, which makes the pill stand out clearly.\n"""
    ), 
}

MVTEC_0SHOT2_CNAME = {
    k: BASE_0SHOT_PROMPT_CNAME.format(class_name=v) for k, v in MVTEC_CNAME_MAPS_V1.items()
}


# Origin V5 in production.py
MVTEC_0SHOT2_CONTEXT1 = {
    k: BASE_0SHOT_PROMPT_WITH_CONTEXT_AND_CNAME.format(context=MVTEC_PRODUCTION_V0_PLUS[k], class_name=MVTEC_CNAME_MAPS_V1[k]) if k in MVTEC_PRODUCTION_V0_PLUS else BASE_0SHOT_PROMPT
    for k in MVTEC_CNAME_MAPS_V1.keys()
}

# Origin 0shot_v1 in production.py
MVTEC_0SHOT2_CONTEXT2 = {
    k: BASE_0SHOT_PROMPT_WITH_CONTEXT_AND_CNAME.format(context=MVTEC_CONTEXT_V1[k], class_name=MVTEC_CNAME_MAPS_V1[k]) if k in MVTEC_CONTEXT_V1 else BASE_0SHOT_PROMPT
    for k in MVTEC_CNAME_MAPS_V1.keys()
}

MVTEC_0SHOT2_CONTEXT2_NOHINT = {
    k: BASE_0SHOT_PROMPT_NOHINT_WITH_CONTEXT_AND_CNAME.format(context=MVTEC_CONTEXT_V1[k], class_name=MVTEC_CNAME_MAPS_V1[k]) if k in MVTEC_CONTEXT_V1 else BASE_0SHOT_PROMPT
    for k in MVTEC_CNAME_MAPS_V1.keys()
}

# Origin 0shot_v2 in production.py
MVTEC_0SHOT2_CONTEXT3 = {
    k: BASE_0SHOT_PROMPT_WITH_CONTEXT_AND_CNAME.format(context=MVTEC_CONTEXT_V3_FIXED[k], class_name=MVTEC_CNAME_MAPS_V1[k]) if k in MVTEC_CONTEXT_V3 else BASE_0SHOT_PROMPT
    for k in MVTEC_CNAME_MAPS_V1.keys()
}

MVTEC_0SHOT2_CONTEXT3_NOHINT = {
    k: BASE_0SHOT_PROMPT_NOHINT_WITH_CONTEXT_AND_CNAME.format(context=MVTEC_CONTEXT_V3_FIXED[k], class_name=MVTEC_CNAME_MAPS_V1[k]) if k in MVTEC_CONTEXT_V3 else BASE_0SHOT_PROMPT
    for k in MVTEC_CNAME_MAPS_V1.keys()
}

# Origin v6 in production.py
MVTEC_0SHOT2_APPEARANCE = {
    k: BASE_0SHOT_PROMPT_WITH_PDESIGN_AND_CNAME.format(context=MVTEC_PRODUCTION_V0_FINAL[k], class_name=MVTEC_CNAME_MAPS_V1[k]) if k in MVTEC_PRODUCTION_V0_FINAL else BASE_0SHOT_PROMPT
    for k in MVTEC_CNAME_MAPS_V1.keys()
}

# Origin v6 in production.py
MVTEC_0SHOT2_APPEARANCE_NOCNAME = {
    k: BASE_0SHOT_PROMPT_WITH_PDESIGN_AND_CNAME.format(context=MVTEC_PRODUCTION_V0_FINAL[k], class_name='product') if k in MVTEC_PRODUCTION_V0_FINAL else BASE_0SHOT_PROMPT
    for k in MVTEC_CNAME_MAPS_V1.keys()
}

MVTEC_0SHOT2_APPEARANCE_NOHINT = {
    k: BASE_0SHOT_PROMPT_NOHINT_WITH_PDESIGN_AND_CNAME.format(context=MVTEC_PRODUCTION_V0_FINAL[k], class_name=MVTEC_CNAME_MAPS_V1[k]) if k in MVTEC_PRODUCTION_V0_FINAL else BASE_0SHOT_PROMPT
    for k in MVTEC_CNAME_MAPS_V1.keys()
}


# Origin 0shot_sv1 in production.py
MVTEC_0SHOT2_NULLCONTEXT = {
    k: BASE_0SHOT_PROMPT_WITH_CONTEXT_AND_CNAME.format(context='', class_name=MVTEC_CNAME_MAPS_V1[k]) if k in MVTEC_PRODUCTION_V0_FINAL else BASE_0SHOT_PROMPT
    for k in MVTEC_CNAME_MAPS_V1.keys()
}
