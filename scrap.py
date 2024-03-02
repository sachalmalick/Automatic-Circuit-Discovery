import acdc.transprop.utils as tp
from acdc.ioi.utils import get_gpt2_small

model = get_gpt2_small()
prompts = tp.get_prompts(7, model)