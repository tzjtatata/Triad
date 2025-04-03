from prompt import *
import json

BASE_PROMPT="Are there any defects in this image?\nA. Yes\nB. No\nAnswer with the option's letter from the given choices directly."

SUPPORTED_QSET = {
    'mvtec': {
        'v0': MVTEC_PROMPT_V0, 
        'v1': MVTEC_PROMPT_V1_1, 
        'v2': MVTEC_PROMPT_V1, 
        'v3':MVTEC_PROMPT_V1_2,
    }, 
    'pcb_bank': {
        'v0': BASE_PROMPT,
        'prod_v1': OTHER_DATASETS["pcb_bank"]["prompt_v1"],
    },
    'WFDD': {
        'v0': BASE_PROMPT,
        'prod_v1': OTHER_DATASETS["WFDD"]["prompt_v1"],
    },
    'mvtec_1shot': {
        'v0': MVTEC_PRODUCTION_1SHOT_V0,
        'v1': MVTEC_PRODUCTION_1SHOT_V1_1,
        'v2':MVTEC_PRODUCTION_1SHOT_V1,
        'v3':MVTEC_PRODUCTION_1SHOT_V1_2,
    }
}
