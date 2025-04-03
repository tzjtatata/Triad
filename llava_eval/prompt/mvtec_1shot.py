from .mvtec import MVTEC_PROMPT_V1_1_PROD, MVTEC_PROMPT_V1_PROD, MVTEC_PROMPT_V1_2_PROD

MVTEC_PRODUCTION_1SHOT_V0 = {
    "hazelnut": (
        "The second image shows an acceptable hazelnut. Compare with the acceptable hazelnut, find out whether there are defects on the hazelnut in the first image.\n"
        "A. Yes\n"
        "B. No\n"
        "Answer with the option's letter from the given choices directly."
    ),  # https://www.nutmachines.com/blog/how-is-hazelnut-processed.html#:~:text=From%20Harvest%20to%20Shelling%201%201.%20Harvest%20Hazelnuts%3A,6%206.%20Separation%20of%20shell%20and%20kernel%3A%20
    "transistor": (
        "The second image shows an acceptable transistor. Compare with the acceptable transistor, find out whether there are defects on the transistor in the first image.\n"
        "A. Yes\n"
        "B. No\n"
        "Answer with the option's letter from the given choices directly."
    ),  # https://www.wevolver.com/article/understanding-transistors-what-they-are-and-how-they-work
    'zipper': (
        "The second image shows an acceptable zipper. Compare with the acceptable zipper, find out whether there are defects on the zipper in the first image.\n"
        "A. Yes\n"
        "B. No\n"
        "Answer with the option's letter from the given choices directly."
        #" Please choose 'Yes' only when you are very confidence."
    ),
    "screw": (
        "The second image shows an acceptable screw. Compare with the acceptable screw, find out whether there are defects on the screw in the first image.\n"
        "A. Yes\n"
        "B. No\n"
        "Answer with the option's letter from the given choices directly."
    ), 
    "cable" : (
        "The second image shows an acceptable cable. Compare with the acceptable cable, find out whether there are defects on the cable in the first image.\n"
        "A. Yes\n"
        "B. No\n"
        "Answer with the option's letter from the given choices directly." 
    ), 

    "metal_nut" : (
        "The second image shows an acceptable metal nut. Compare with the acceptable metal nut, find out whether there are defects on the metal nut in the first image.\n"
        "A. Yes\n"
        "B. No\n"
        "Answer with the option's letter from the given choices directly." 
    ),
    "toothbrush" : (
        "The second image shows an acceptable toothbrush. Compare with the acceptable toothbrush, find out whether there are defects on the toothbrush in the first image.\n"
        "A. Yes\n"
        "B. No\n"
        "Answer with the option's letter from the given choices directly."
    ),
    "tile" : (
        "The second image shows an acceptable marble tile. Compare with the acceptable marble tile, find out whether there are defects on the marble tile in the first image.\n"
        "A. Yes\n"
        "B. No\n"
        "Answer with the option's letter from the given choices directly."
    ),
    "bottle" : (
        "The second image shows an acceptable glass bottle. Compare with the acceptable glass bottle, find out whether there are defects on the glass bottle in the first image.\n"
        "A. Yes\n"
        "B. No\n"
        "Answer with the option's letter from the given choices directly."
    ),
    "wood": (
        "The second image shows an acceptable wooden boards. Compare with the acceptable wooden boards, find out whether there are defects on the wooden boards in the first image.\n"
        "A. Yes\n"
        "B. No\n"
        "Answer with the option's letter from the given choices directly."
    ), 
    "capsule": (
        "The second image shows an acceptable capsule. Compare with the acceptable capsule, find out whether there are defects on the capsule in the first image.\n"
        "A. Yes\n"
        "B. No\n"
        "Answer with the option's letter from the given choices directly."
    ), 
    "pill": (
        "The second image shows an acceptable pill. Compare with the acceptable pill, find out whether there are defects on the pill in the first image.\n"
        "A. Yes\n"
        "B. No\n"
        "Answer with the option's letter from the given choices directly.\n"
    ), 
    "carpet": (
        "The second image shows an acceptable carpet. Compare with the acceptable carpet, find out whether there are defects on the carpet in the first image.\n"
        "A. Yes\n"
        "B. No\n"
        "Answer with the option's letter from the given choices directly."
    ), 
    "grid": (
        "The second image shows an acceptable grid. Compare with the acceptable grid, find out whether there are defects on the grid in the first image.\n"
        "A. Yes\n"
        "B. No\n"
        "Answer with the option's letter from the given choices directly."
    ), 
    "leather": (
        "The second image shows an acceptable leather. Compare with the acceptable leather, find out whether there are defects on the leather in the first image.\n"
        "A. Yes\n"
        "B. No\n"
        "Answer with the option's letter from the given choices directly."
    ), 
}


MVTEC_PRODUCTION_1SHOT_V1_1 = {
    k: MVTEC_PROMPT_V1_1_PROD[k] + MVTEC_PRODUCTION_1SHOT_V0[k] if k in MVTEC_PROMPT_V1_1_PROD else MVTEC_PRODUCTION_1SHOT_V0[k]
    for k in MVTEC_PRODUCTION_1SHOT_V0.keys()
}

MVTEC_PRODUCTION_1SHOT_V1 = {
    k: MVTEC_PROMPT_V1_PROD[k] + MVTEC_PRODUCTION_1SHOT_V0[k] if k in MVTEC_PROMPT_V1_PROD else MVTEC_PRODUCTION_1SHOT_V0[k]
    for k in MVTEC_PRODUCTION_1SHOT_V0.keys()
}

MVTEC_PRODUCTION_1SHOT_V1_2 = {
    k: MVTEC_PROMPT_V1_2_PROD[k] + MVTEC_PRODUCTION_1SHOT_V0[k] if k in MVTEC_PROMPT_V1_2_PROD else MVTEC_PRODUCTION_1SHOT_V0[k]
    for k in MVTEC_PRODUCTION_1SHOT_V0.keys()
}