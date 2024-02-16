from contextlib import closing

import modules.scripts
from modules import processing
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.shared import opts
import modules.shared as shared
from modules.ui import plaintext_to_html
import gradio as gr


def txt2img(id_task: str, prompt: str, negative_prompt: str, prompt_styles, steps: int, sampler_name: str, n_iter: int, batch_size: int, cfg_scale: float, height: int, width: int, enable_hr: bool, denoising_strength: float, hr_scale: float, hr_upscaler: str, hr_second_pass_steps: int, hr_resize_x: int, hr_resize_y: int, hr_checkpoint_name: str, hr_sampler_name: str, hr_prompt: str, hr_negative_prompt, override_settings_texts, request: gr.Request, *args):
    override_settings = create_override_settings_dict(override_settings_texts)
    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=prompt_styles,
        negative_prompt=negative_prompt,
        sampler_name=sampler_name,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        enable_hr=enable_hr,
        denoising_strength=denoising_strength if enable_hr else None,
        hr_scale=hr_scale,
        hr_upscaler=hr_upscaler,
        hr_second_pass_steps=hr_second_pass_steps,
        hr_resize_x=hr_resize_x,
        hr_resize_y=hr_resize_y,
        hr_checkpoint_name=None if hr_checkpoint_name == 'Use same checkpoint' else hr_checkpoint_name,
        hr_sampler_name=None if hr_sampler_name == 'Use same sampler' else hr_sampler_name,
        hr_prompt=hr_prompt,
        hr_negative_prompt=hr_negative_prompt,
        override_settings=override_settings,
    )
    print("지금 세팅해요")
    p.scripts = modules.scripts.scripts_txt2img

    p.script_args = args
    # print("uiargs", p.script_args)

    """
args (0, False, '', 0.8, -1, False, -1, 0, 0, 0, False, False, {'ad_model': 'face_yolov8n.pt', 'ad_prompt': '', 'ad_negative_prompt': '', 'ad_confidence': 0.3, 'ad_mask_k_largest': 0, 'ad_mask_min_ratio': 0, 'ad_mask_max_ratio': 1, 'ad_x_offset': 0, 'ad_y_offset': 0, 'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 4, 'ad_denoising_strength': 0.4, 'ad_inpaint_only_masked': True, 'ad_inpaint_only_masked_padding': 32, 'ad_use_inpaint_width_height': False, 'ad_inpaint_width': 512, 'ad_inpaint_height': 512, 'ad_use_steps': False, 'ad_steps': 28, 'ad_use_cfg_scale': False, 'ad_cfg_scale': 7, 'ad_use_checkpoint': False, 'ad_checkpoint': 'Use same checkpoint', 'ad_use_vae': False, 'ad_vae': 'Use same VAE', 'ad_use_sampler': False, 'ad_sampler': 'DPM++ 2M Karras', 'ad_use_noise_multiplier': False, 'ad_noise_multiplier': 1, 'ad_use_clip_skip': False, 'ad_clip_skip': 1, 'ad_restore_face': False, 'ad_controlnet_model': 'None', 'ad_controlnet_module': 'None', 'ad_controlnet_weight': 1, 'ad_controlnet_guidance_start': 0, 'ad_controlnet_guidance_end': 1, 'is_api': ()}, {'ad_model': 'None', 'ad_prompt': '', 'ad_negative_prompt': '', 'ad_confidence': 0.3, 'ad_mask_k_largest': 0, 'ad_mask_min_ratio': 0, 'ad_mask_max_ratio': 1, 'ad_x_offset': 0, 'ad_y_offset': 0, 'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 4, 'ad_denoising_strength': 0.4, 'ad_inpaint_only_masked': True, 'ad_inpaint_only_masked_padding': 32, 'ad_use_inpaint_width_height': False, 'ad_inpaint_width': 512, 'ad_inpaint_height': 512, 'ad_use_steps': False, 'ad_steps': 28, 'ad_use_cfg_scale': False, 'ad_cfg_scale': 7, 'ad_use_checkpoint': False, 'ad_checkpoint': 'Use same checkpoint', 'ad_use_vae': False, 'ad_vae': 'Use same VAE', 'ad_use_sampler': False, 'ad_sampler': 'DPM++ 2M Karras', 'ad_use_noise_multiplier': False, 'ad_noise_multiplier': 1, 'ad_use_clip_skip': False, 'ad_clip_skip': 1, 'ad_restore_face': False, 'ad_controlnet_model': 'None', 'ad_controlnet_module': 'None', 'ad_controlnet_weight': 1, 'ad_controlnet_guidance_start': 0, 'ad_controlnet_guidance_end': 1, 'is_api': ()}, True, False, 1, False, False, False, 1.1, 1.5, 100, 0.7, False, False, True, False, False, 0, 'Gustavosta/MagicPrompt-Stable-Diffusion', '', UiControlNetUnit(enabled=False, module='none', model='None', weight=1, image=None, resize_mode='Crop and Resize', low_vram=False, processor_res=-1, threshold_a=-1, threshold_b=-1, guidance_start=0, guidance_end=1, pixel_perfect=False, control_mode='Balanced', inpaint_crop_input_image=False, hr_option='Both', save_detected_map=True, advanced_weighting=None), UiControlNetUnit(enabled=False, module='none', model='None', weight=1, image=None, resize_mode='Crop and Resize', low_vram=False, processor_res=-1, threshold_a=-1, threshold_b=-1, guidance_start=0, guidance_end=1, pixel_perfect=False, control_mode='Balanced', inpaint_crop_input_image=False, hr_option='Both', save_detected_map=True, advanced_weighting=None), UiControlNetUnit(enabled=False, module='none', model='None', weight=1, image=None, resize_mode='Crop and Resize', low_vram=False, processor_res=-1, threshold_a=-1, threshold_b=-1, guidance_start=0, guidance_end=1, pixel_perfect=False, control_mode='Balanced', inpaint_crop_input_image=False, hr_option='Both', save_detected_map=True, advanced_weighting=None), 'NONE:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\nALL:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1\nINS:1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0\nIND:1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0\nINALL:1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0\nMIDD:1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0\nOUTD:1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0\nOUTS:1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1\nOUTALL:1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1\nALL0.5:0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5', True, 0, 'values', '0,0.25,0.5,0.75,1', 'Block ID', 'IN05-OUT05', 'none', '', '0.5,1', 'BASE,IN00,IN01,IN02,IN03,IN04,IN05,IN06,IN07,IN08,IN09,IN10,IN11,M00,OUT00,OUT01,OUT02,OUT03,OUT04,OUT05,OUT06,OUT07,OUT08,OUT09,OUT10,OUT11', 1.0, 'black', '20', False, 'ATTNDEEPON:IN05-OUT05:attn:1\n\nATTNDEEPOFF:IN05-OUT05:attn:0\n\nPROJDEEPOFF:IN05-OUT05:proj:0\n\nXYZ:::1', False, False, False, False, 'Matrix', 'Columns', 'Mask', 'Prompt', '1,1', '0.2', False, False, False, 'Attention', [False], '0', '0', '0.4', None, '0', '0', False, False, False, 0, None, [], 0, False, [], [], False, 0, 1, False, False, 0, None, [], -2, False, [], False, 0, None, None, False, False, 'positive', 'comma', 0, False, False, 'start', '', 1, '', [], 0, '', [], 0, '', [], True, False, False, False, 0, False, None, None, False, None, None, False, None, None, False, 50, [], 30, '', 4, [], 1, '', '', '', '')
    """
    p.user = request.username

    if shared.opts.enable_console_prompts:
        print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)

    with closing(p):
        # selectable 정리
        processed = modules.scripts.scripts_txt2img.run(p, *args)

        if processed is None:
            processed = processing.process_images(p)

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return processed.images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")
