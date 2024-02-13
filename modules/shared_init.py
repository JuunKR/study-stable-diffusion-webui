import os

import torch

from modules import shared
from modules.shared import cmd_opts


def initialize():
    """Initializes fields inside the shared module in a controlled manner.

    Should be called early because some other modules you can import mingt need these fields to be already set.
    """

    os.makedirs(cmd_opts.hypernetwork_dir, exist_ok=True)

    from modules import options, shared_options
    
    shared.options_templates = shared_options.options_templates
    
    # shared.opts:  <modules.options.Options object at 0x7f28b56c48b0>
    shared.opts = options.Options(shared_options.options_templates, shared_options.restricted_opts)

    # shared.restricted_opts {'outdir_txt2img_grids', 'outdir_init_images', 'directories_filename_pattern', 'outdir_save', 'outdir_img2img_samples', 'outdir_grids', 'outdir_extras_samples', 'outdir_samples', 'outdir_txt2img_samples', 'samples_filename_pattern'}
    shared.restricted_opts = shared_options.restricted_opts

    # 여기서 컨피그 로드 모델이름도 로드
    # /workspace/config.json
    if os.path.exists(shared.config_filename):
        shared.opts.load(shared.config_filename)

    from modules import devices
    devices.device, devices.device_interrogate, devices.device_gfpgan, devices.device_esrgan, devices.device_codeformer = \
        (devices.cpu if any(y in cmd_opts.use_cpu for y in [x, 'all']) else devices.get_optimal_device() for x in ['sd', 'interrogate', 'gfpgan', 'esrgan', 'codeformer'])

    # devices.dtype: torch.float16
    devices.dtype = torch.float32 if cmd_opts.no_half else torch.float16
    
    # devices.dtype_vae: torch.float16
    devices.dtype_vae = torch.float32 if cmd_opts.no_half or cmd_opts.no_half_vae else torch.float16
    
    # devices.device cuda
    shared.device = devices.device
    
    # devices.device cpu
    shared.weight_load_location = None if cmd_opts.lowram else "cpu"
    
    from modules import shared_state
    
    ### Style database not found: /workspace/styles.csv
    ### Style database not found: /workspace/styles.csv
    
    # shared.state: <modules.shared_state.State object at 0x7f2745fd3b80>
    shared.state = shared_state.State()

    from modules import styles
    # shared.prompt_styles: <modules.styles.StyleDatabase object at 0x7f273f162cb0>
    shared.prompt_styles = styles.StyleDatabase(shared.styles_filename)
    

    from modules import interrogate
    # shared.interrogator <modules.interrogate.InterrogateModels object at 0x7f273f163550>
    shared.interrogator = interrogate.InterrogateModels("interrogate")
    

    from modules import shared_total_tqdm
    # shared.total_tqdm  <modules.shared_total_tqdm.TotalTQDM object at 0x7f273f163550>
    shared.total_tqdm = shared_total_tqdm.TotalTQDM()
    

    from modules import memmon, devices
    # shared.mem_mon  <MemUsageMonitor(MemMon, initial daemon)>
    shared.mem_mon = memmon.MemUsageMonitor("MemMon", devices.device, shared.opts)
    
    shared.mem_mon.start()

