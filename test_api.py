prompt: (bird:1.0)
prompt_for_display: (bird:1.0)
negative_prompt: 
styles: []
seed: 1000
subseed: 2015311893
subseed_strength: 0.0
seed_resize_from_h: -1
seed_resize_from_w: -1
seed_enable_extras: True
sampler_name: DPM++ 2M SDE Karras
batch_size: 1
n_iter: 1
steps: 20
cfg_scale: 7.0
width: 512
height: 512
restore_faces: False
tiling: False
do_not_save_samples: False
do_not_save_grid: False
extra_generation_params: {}
overlay_images: None
eta: None
do_not_reload_embeddings: False
denoising_strength: None
ddim_discretize: None
s_min_uncond: 0.0
s_churn: 0.0
s_tmax: inf
s_tmin: 0.0
s_noise: 1.0
override_settings: {}
override_settings_restore_afterwards: True
sampler_index: None
refiner_checkpoint: None
refiner_switch_at: None
disable_extra_networks: False
comments: {}
enable_hr: False
firstphase_width: 0
firstphase_height: 0
hr_scale: 2.0
hr_upscaler: Latent
hr_second_pass_steps: 0
hr_resize_x: 0
hr_resize_y: 0
hr_checkpoint_name: None
hr_sampler_name: None
hr_prompt: 
hr_negative_prompt: 
sampler_noise_scheduler_override: None
# 5번째 다름
script_args_value: (0, False, '', 0.8, -1, False, -1, 0.0, 0, 0, False, False, 
{'ad_model': 'face_yolov8n.pt', 'ad_prompt': '', 'ad_negative_prompt': '', 'ad_confidence': 0.3, 'ad_mask_k_largest': 0, 'ad_mask_min_ratio': 0.0, 'ad_mask_max_ratio': 1.0, 'ad_x_offset': 0, 'ad_y_offset': 0, 'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 4, 'ad_denoising_strength': 0.4, 'ad_inpaint_only_masked': True, 'ad_inpaint_only_masked_padding': 32, 'ad_use_inpaint_width_height': False, 'ad_inpaint_width': 512, 'ad_inpaint_height': 512, 'ad_use_steps': False, 'ad_steps': 28, 'ad_use_cfg_scale': False, 'ad_cfg_scale': 7.0, 'ad_use_checkpoint': False, 'ad_checkpoint': 'Use same checkpoint', 'ad_use_vae': False, 'ad_vae': 'Use same VAE', 'ad_use_sampler': False, 'ad_sampler': 'DPM++ 2M Karras', 'ad_use_noise_multiplier': False, 'ad_noise_multiplier': 1.0, 'ad_use_clip_skip': False, 'ad_clip_skip': 1, 'ad_restore_face': False, 'ad_controlnet_model': 'None', 'ad_controlnet_module': 'None', 'ad_controlnet_weight': 1.0, 'ad_controlnet_guidance_start': 0.0, 'ad_controlnet_guidance_end': 1.0}, 

{'ad_model': 'None', 'ad_prompt': '', 'ad_negative_prompt': '', 'ad_confidence': 0.3, 'ad_mask_k_largest': 0, 'ad_mask_min_ratio': 0.0, 'ad_mask_max_ratio': 1.0, 'ad_x_offset': 0, 'ad_y_offset': 0, 'ad_dilate_erode': 4, 'ad_mask_merge_invert': 'None', 'ad_mask_blur': 4, 'ad_denoising_strength': 0.4, 'ad_inpaint_only_masked': True, 'ad_inpaint_only_masked_padding': 32, 'ad_use_inpaint_width_height': False, 'ad_inpaint_width': 512, 'ad_inpaint_height': 512, 'ad_use_steps': False, 'ad_steps': 28, 'ad_use_cfg_scale': False, 'ad_cfg_scale': 7.0, 'ad_use_checkpoint': False, 'ad_checkpoint': 'Use same checkpoint', 'ad_use_vae': False, 'ad_vae': 'Use same VAE', 'ad_use_sampler': False, 'ad_sampler': 'DPM++ 2M Karras', 'ad_use_noise_multiplier': False, 'ad_noise_multiplier': 1.0, 'ad_use_clip_skip': False, 'ad_clip_skip': 1, 'ad_restore_face': False, 'ad_controlnet_model': 'None', 'ad_controlnet_module': 'None', 'ad_controlnet_weight': 1.0, 'ad_controlnet_guidance_start': 0.0, 'ad_controlnet_guidance_end': 1.0}, 

True, False, 1, False, False, False, 1.1, 1.5, 100, 0.7, False, False, True, False, False, 0, 
'Gustavosta/MagicPrompt-Stable-Diffusion', '', 

# inpaint_crop_input_image
UiControlNetUnit(enabled=False, module='none', model='None', weight=1.0, image=None, resize_mode=<ResizeMode.INNER_FIT: 'Crop and Resize'>, low_vram=False, processor_res=-1, threshold_a=-1, threshold_b=-1, guidance_start=0.0, guidance_end=1.0, pixel_perfect=False, control_mode=<ControlMode.BALANCED: 'Balanced'>, inpaint_crop_input_image=True, hr_option=<HiResFixOption.BOTH: 'Both'>, save_detected_map=True, advanced_weighting=None), 

UiControlNetUnit(enabled=False, module='none', model='None', weight=1.0, image=None, resize_mode=<ResizeMode.INNER_FIT: 'Crop and Resize'>, low_vram=False, processor_res=-1, threshold_a=-1, threshold_b=-1, guidance_start=0.0, guidance_end=1.0, pixel_perfect=False, control_mode=<ControlMode.BALANCED: 'Balanced'>, inpaint_crop_input_image=True, hr_option=<HiResFixOption.BOTH: 'Both'>, save_detected_map=True, advanced_weighting=None), UiControlNetUnit(enabled=False, module='none', model='None', weight=1.0, image=None, resize_mode=<ResizeMode.INNER_FIT: 'Crop and Resize'>, low_vram=False, processor_res=-1, threshold_a=-1, threshold_b=-1, guidance_start=0.0, guidance_end=1.0, pixel_perfect=False, control_mode=<ControlMode.BALANCED: 'Balanced'>, inpaint_crop_input_image=True, hr_option=<HiResFixOption.BOTH: 'Both'>, save_detected_map=True, advanced_weighting=None), 

'NONE:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\nALL:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1\nINS:1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0\nIND:1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0\nINALL:1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0\nMIDD:1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0\nOUTD:1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0\nOUTS:1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1\nOUTALL:1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1\nALL0.5:0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5', True, 
# 다름
'Disable', 'values', '0,0.25,0.5,0.75,1', 'Block ID', 'IN05-OUT05', 'none', '', '0.5,1', 'BASE,IN00,IN01,IN02,IN03,IN04,IN05,IN06,IN07,IN08,IN09,IN10,IN11,M00,OUT00,OUT01,OUT02,OUT03,OUT04,OUT05,OUT06,OUT07,OUT08,OUT09,OUT10,OUT11', 1.0, 'black', '20', False, 'ATTNDEEPON:IN05-OUT05:attn:1\n\nATTNDEEPOFF:IN05-OUT05:attn:0\n\nPROJDEEPOFF:IN05-OUT05:proj:0\n\nXYZ:::1', False, False, False, False, 'Matrix', 'Columns', 'Mask', 'Prompt', '1,1', '0.2', False, False, False, 'Attention', [False], '0', '0', '0.4', None, '0', '0', False, False, False, '0', None, [], '0', False, [], [], False, '0', '2', False, False, '0', None, [], -2, False, [], False, '0', None, None, False, False, 'positive', 'comma', 0, False, False, 'start', '', 
# 다름
'Seed', '', None, 'Nothing', '', None, 'Nothing', '', None, 

True, False, False, False, 0, False, None, None, False, None, None, False, None, None, False, 50.0, [], 30, '', 4, [], 1, '', '', '', '')
refiner_checkpoint_info: None
cached_uc: [([''], 20, None, False, 1, <modules.sd_models.CheckpointInfo object at 0x7f419db79c30>, defaultdict(<class 'list'>, {}), 0, 0, 512, 512), [[ScheduledPromptConditioning(end_at_step=20, cond=tensor([[-0.3852,  0.0188, -0.0572,  ..., -0.4861, -0.3019,  0.0626],
        [-0.2964, -1.6715, -0.3428,  ...,  1.0424,  0.2908, -0.6744],
        [-0.2816, -1.7103, -0.2959,  ...,  1.0985,  0.1297, -0.5657],
        ...,
        [ 1.6024, -0.8372, -0.2422,  ...,  0.5231, -1.2136,  0.8820],
        [ 1.6097, -0.8291, -0.2428,  ...,  0.5444, -1.2206,  0.8794],
        [ 1.5998, -0.8147, -0.2004,  ...,  0.5235, -1.1818,  0.8787]],
       device='cuda:0'))]]]
cached_c: [(['(bird:1.0)'], 20, None, False, 1, <modules.sd_models.CheckpointInfo object at 0x7f419db79c30>, defaultdict(<class 'list'>, {}), 0, 0, 512, 512), <modules.prompt_parser.MulticondLearnedConditioning object at 0x7f40d4b0d480>]
cached_hr_uc: [None, None]
cached_hr_c: [None, None]
is_api: True
scripts_value: <modules.scripts.ScriptRunner object at 0x7f419db79e40>
scripts_setup_complete: True
sd_model_name: toonAme_version20
sd_model_hash: cacf2ede14
sd_vae_name: None
sd_vae_hash: None
all_prompts: ['(bird:1.0)']
all_negative_prompts: ['']
main_prompt: (bird:1.0)
main_negative_prompt: 
all_seeds: [1000]
all_subseeds: [2015311893]
_ad_disabled: True
controlnet_control_loras: []
iteration: 0
prompts: ['(bird:1.0)']
negative_prompts: ['']
seeds: [1000]
subseeds: [2015311893]
rng: <modules.rng.ImageRNG object at 0x7f40cfe71de0>
extra_network_data: defaultdict(<class 'list'>, {})
step_multiplier: 1
firstpass_steps: 20
uc: [[ScheduledPromptConditioning(end_at_step=20, cond=tensor([[-0.3852,  0.0188, -0.0572,  ..., -0.4861, -0.3019,  0.0626],
        [-0.2964, -1.6715, -0.3428,  ...,  1.0424,  0.2908, -0.6744],
        [-0.2816, -1.7103, -0.2959,  ...,  1.0985,  0.1297, -0.5657],
        ...,
        [ 1.6024, -0.8372, -0.2422,  ...,  0.5231, -1.2136,  0.8820],
        [ 1.6097, -0.8291, -0.2428,  ...,  0.5444, -1.2206,  0.8794],
        [ 1.5998, -0.8147, -0.2004,  ...,  0.5235, -1.1818,  0.8787]],
       device='cuda:0'))]]
c: <modules.prompt_parser.MulticondLearnedConditioning object at 0x7f40d4b0d480>
hr_uc: None
hr_c: None
sampler: <modules.sd_samplers_kdiffusion.KDiffusionSampler object at 0x7f40cfe97df0>
is_using_inpainting_conditioning: False
batch_index: 0
color_corrections: None
sd-lora-block POST