import importlib
import logging
import sys
import warnings
from threading import Thread

from modules.timer import startup_timer

def imports():
    logging.getLogger("torch.distributed.nn").setLevel(logging.ERROR)  # sshh...
    logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

    import torch  # noqa: F401
    startup_timer.record("import torch")
    import pytorch_lightning  # noqa: F401
    startup_timer.record("import torch")
    warnings.filterwarnings(action="ignore", category=DeprecationWarning, module="pytorch_lightning")
    warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision")
    
    import gradio  # noqa: F401
    startup_timer.record("import gradio")
    
    # ldm 
    from modules import paths, timer, import_hook, errors  # noqa: F401
    startup_timer.record("setup paths")

    import ldm.modules.encoders.modules  # noqa: F401
    startup_timer.record("import ldm")

    import sgm.modules.encoders.modules  # noqa: F401
    startup_timer.record("import sgm")

    from modules import shared_init
    
    # shared, devices, config 세팅
    shared_init.initialize()
    startup_timer.record("initialize shared")

    from modules import processing, gradio_extensons, ui  # noqa: F401
    startup_timer.record("other imports")


def check_versions():
    from modules.shared_cmd_options import cmd_opts

    if not cmd_opts.skip_version_check:
        from modules import errors
        errors.check_versions()


def initialize():
    from modules import initialize_util
    # 토치 버전 픽스
    initialize_util.fix_torch_version()
    # 이벤트 루프 policy 설정, 멀티
    initialize_util.fix_asyncio_event_loop_policy()
    # tls option 설정 x
    initialize_util.validate_tls_options()
    # 컨트롤 c 했을때 즉시 종료
    initialize_util.configure_sigint_handler()
    # 모델 세팅?
    initialize_util.configure_opts_onchange()

    from modules import modelloader
    # 모델 경로 이동?
    modelloader.cleanup_models()

    from modules import sd_models
    # 모델 다운로드
    sd_models.setup_model()
    startup_timer.record("setup SD model")

    from modules.shared_cmd_options import cmd_opts

    from modules import codeformer_model
    warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision.transforms.functional_tensor")
    
    #codeformer 세팅 - 얼굴복원 모델
    codeformer_model.setup_model(cmd_opts.codeformer_models_path)
    startup_timer.record("setup codeformer")

    from modules import gfpgan_model
    # gfpgan_model - 얼굴복원 모델
    gfpgan_model.setup_model(cmd_opts.gfpgan_models_path)
    startup_timer.record("setup gfpgan")

    # 업스케일러 로드, 스크립트 로드, 샘플러 세팅, 익스텐션 리스트 세팅, vae, textual_inversion, hijack optimizers, unet
    initialize_rest(reload_script_modules=False)


def initialize_rest(*, reload_script_modules=False):
    """
    Called both from initialize() and when reloading the webui.
    """
    from modules.shared_cmd_options import cmd_opts

    from modules import sd_samplers
    
    # set sampler
    sd_samplers.set_samplers()
    startup_timer.record("set samplers")

    from modules import extensions
    # extention 객체 및 리스트 세팅
    extensions.list_extensions()
    startup_timer.record("list extensions")

    from modules import initialize_util
    
    # 리턴당함 뭔지 모르겠음
    initialize_util.restore_config_state_file()
    startup_timer.record("restore config state file")

    from modules import shared, upscaler, scripts
    # 넘김
    if cmd_opts.ui_debug_mode:
        shared.sd_upscalers = upscaler.UpscalerLanczos().scalers
        scripts.load_scripts()
        return

    from modules import sd_models
    # 모델 리스트 생성 
    sd_models.list_models()
    startup_timer.record("list SD models")
    

    from modules import localization
    # localization ? 아마 번역?
    localization.list_localizations(cmd_opts.localizations_dir)
    startup_timer.record("list localizations")

    with startup_timer.subcategory("load scripts"):
        # 스크립트 로드
        """
        scripts_txt2img = ScriptRunner()
        scripts_img2img = ScriptRunner()
        scripts_postproc = scripts_postprocessing.ScriptPostprocessingRunner()
        """
        scripts.load_scripts()

    if reload_script_modules:
        print("여기들어오니?")
        for module in [module for name, module in sys.modules.items() if name.startswith("modules.ui")]:
            importlib.reload(module)
        startup_timer.record("reload script modules")

    from modules import modelloader
    # 업스케일 모델 로드
    modelloader.load_upscalers()
    startup_timer.record("load upscalers")

    from modules import sd_vae
    # sd_vae 리스트
    sd_vae.refresh_vae_list()
    startup_timer.record("refresh VAE")

    from modules import textual_inversion
    # textual_inversion
    textual_inversion.textual_inversion.list_textual_inversion_templates()
    startup_timer.record("refresh textual inversion templates")

    from modules import script_callbacks, sd_hijack_optimizations, sd_hijack
    # hijack optimizers
    script_callbacks.on_list_optimizers(sd_hijack_optimizations.list_optimizers)
    sd_hijack.list_optimizers()
    startup_timer.record("scripts list_optimizers")

    from modules import sd_unet
    # unet
    sd_unet.list_unets()
    startup_timer.record("scripts list_unets")

    def load_model():
        """
        Accesses shared.sd_model property to load model.
        After it's available, if it has been loaded before this access by some extension,
        its optimization may be None because the list of optimizaers has neet been filled
        by that time, so we apply optimization again.
        """

        shared.sd_model  # noqa: B018

        if sd_hijack.current_optimizer is None:
            sd_hijack.apply_optimizations()

        from modules import devices
        devices.first_time_calculation()
        
    if not shared.cmd_opts.skip_load_model_at_start:
        Thread(target=load_model).start()

    from modules import shared_items
    shared_items.reload_hypernetworks()
    startup_timer.record("reload hypernetworks")

    from modules import ui_extra_networks
    ui_extra_networks.initialize()
    ui_extra_networks.register_default_pages()

    from modules import extra_networks
    extra_networks.initialize()
    extra_networks.register_default_extra_networks()
    startup_timer.record("initialize extra networks")
