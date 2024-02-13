import json
import sys
from dataclasses import dataclass

import gradio as gr

from modules import errors
from modules.shared_cmd_options import cmd_opts



"""
옵션 섹션 종류
1) saving-images
2) saving-paths
3) saving-to-dirs
4) upscaling
5) face-restoration
6) system
7) API
8) training
9) sd
10) sdxl
11) vae
12) img2img
13) optimizations
14) compatibility
15) interrogate
16) extra_networks
17) ui_prompt_editing
18) ui_gallery
19) ui_alternatives
20) ui
21) infotext
22) sampler-params
23) postprocessing
24) Hidden options

"""
# 옵션에 대한 상세 설명 및 디폴트
class OptionInfo:
    def __init__(self, default=None, label="", component=None, component_args=None, onchange=None, section=None, refresh=None, comment_before='', comment_after='', infotext=None, restrict_api=False, category_id=None):
        self.default = default
        self.label = label
        self.component = component
        self.component_args = component_args
        self.onchange = onchange
        self.section = section
        self.category_id = category_id
        self.refresh = refresh
        self.do_not_save = False

        self.comment_before = comment_before
        """HTML text that will be added after label in UI"""

        self.comment_after = comment_after
        """HTML text that will be added before label in UI"""

        self.infotext = infotext

        self.restrict_api = restrict_api
        """If True, the setting will not be accessible via API"""

    def link(self, label, url):
        self.comment_before += f"[<a href='{url}' target='_blank'>{label}</a>]"
        return self

    def js(self, label, js_func):
        self.comment_before += f"[<a onclick='{js_func}(); return false'>{label}</a>]"
        return self

    def info(self, info):
        self.comment_after += f"<span class='info'>({info})</span>"
        return self

    def html(self, html):
        self.comment_after += html
        return self

    def needs_restart(self):
        self.comment_after += " <span class='info'>(requires restart)</span>"
        return self

    def needs_reload_ui(self):
        self.comment_after += " <span class='info'>(requires Reload UI)</span>"
        return self


class OptionHTML(OptionInfo):
    def __init__(self, text):
        super().__init__(str(text).strip(), label='', component=lambda **kwargs: gr.HTML(elem_classes="settings-info", **kwargs))

        self.do_not_save = True


def options_section(section_identifier, options_dict):
    for v in options_dict.values():
        if len(section_identifier) == 2:
            v.section = section_identifier
        elif len(section_identifier) == 3:
            v.section = section_identifier[0:2]
            v.category_id = section_identifier[2]

    return options_dict


options_builtin_fields = {"data_labels", "data", "restricted_opts", "typemap"}


class Options:
    typemap = {int: float}

    def __init__(self, data_labels: dict[str, OptionInfo], restricted_opts):
        self.data_labels = data_labels
        self.data = {k: v.default for k, v in self.data_labels.items() if not v.do_not_save}

        """
        self.data {'samples_save': True, 'samples_format': 'png', 'samples_filename_pattern': '', 'save_images_add_number': True, 'save_images_replace_action': 'Replace', 'grid_save': True, 'grid_format': 'png', 'grid_extended_filename': False, 'grid_only_if_multiple': True, 'grid_prevent_empty_spots': False, 'grid_zip_filename_pattern': '', 'n_rows': -1, 'font': '', 'grid_text_active_color': '#000000', 'grid_text_inactive_color': '#999999', 'grid_background_color': '#ffffff', 'save_images_before_face_restoration': False, 'save_images_before_highres_fix': False, 'save_images_before_color_correction': False, 'save_mask': False, 'save_mask_composite': False, 'jpeg_quality': 80, 'webp_lossless': False, 'export_for_4chan': True, 'img_downscale_threshold': 4.0, 'target_side_length': 4000, 'img_max_size_mp': 200, 'use_original_name_batch': True, 'use_upscaler_name_as_suffix': False, 'save_selected_only': True, 'save_init_img': False, 'temp_dir': '', 'clean_temp_dir_at_start': False, 'save_incomplete_images': False, 'notification_audio': True, 'notification_volume': 100, 'outdir_samples': '', 'outdir_txt2img_samples': 'outputs/txt2img-images', 'outdir_img2img_samples': 'outputs/img2img-images', 'outdir_extras_samples': 'outputs/extras-images', 'outdir_grids': '', 'outdir_txt2img_grids': 'outputs/txt2img-grids', 'outdir_img2img_grids': 'outputs/img2img-grids', 'outdir_save': 'log/images', 'outdir_init_images': 'outputs/init-images', 'save_to_dirs': True, 'grid_save_to_dirs': True, 'use_save_to_dirs_for_ui': False, 'directories_filename_pattern': '[date]', 'directories_max_prompt_words': 8, 'ESRGAN_tile': 192, 'ESRGAN_tile_overlap': 8, 'realesrgan_enabled_models': ['R-ESRGAN 4x+', 'R-ESRGAN 4x+ Anime6B'], 'upscaler_for_img2img': None, 'face_restoration': False, 'face_restoration_model': 'CodeFormer', 'code_former_weight': 0.5, 'face_restoration_unload': False, 'auto_launch_browser': 'Local', 'enable_console_prompts': False, 'show_warnings': False, 'show_gradio_deprecation_warnings': True, 'memmon_poll_rate': 8, 'samples_log_stdout': False, 'multiple_tqdm': True, 'print_hypernet_extra': False, 'list_hidden_files': True, 'disable_mmap_load_safetensors': False, 'hide_ldm_prints': True, 'dump_stacks_on_signal': False, 'api_enable_requests': True, 'api_forbid_local_requests': True, 'api_useragent': '', 'unload_models_when_training': False, 'pin_memory': False, 'save_optimizer_state': False, 'save_training_settings_to_txt': True, 'dataset_filename_word_regex': '', 'dataset_filename_join_string': ' ', 'training_image_repeats_per_epoch': 1, 'training_write_csv_every': 500, 'training_xattention_optimizations': False, 'training_enable_tensorboard': False, 'training_tensorboard_save_images': False, 'training_tensorboard_flush_every': 120, 'sd_model_checkpoint': None, 'sd_checkpoints_limit': 1, 'sd_checkpoints_keep_in_cpu': True, 'sd_checkpoint_cache': 0, 'sd_unet': 'Automatic', 'enable_quantization': False, 'enable_emphasis': True, 'enable_batch_seeds': True, 'comma_padding_backtrack': 20, 'CLIP_stop_at_last_layers': 1, 'upcast_attn': False, 'randn_source': 'GPU', 'tiling': False, 'hires_fix_refiner_pass': 'second pass', 'sdxl_crop_top': 0, 'sdxl_crop_left': 0, 'sdxl_refiner_low_aesthetic_score': 2.5, 'sdxl_refiner_high_aesthetic_score': 6.0, 'sd_vae_checkpoint_cache': 0, 'sd_vae': 'Automatic', 'sd_vae_overrides_per_model_preferences': True, 'auto_vae_precision': True, 'sd_vae_encode_method': 'Full', 'sd_vae_decode_method': 'Full', 'inpainting_mask_weight': 1.0, 'initial_noise_multiplier': 1.0, 'img2img_extra_noise': 0.0, 'img2img_color_correction': False, 'img2img_fix_steps': False, 'img2img_background_color': '#ffffff', 'img2img_editor_height': 720, 'img2img_sketch_default_brush_color': '#ffffff', 'img2img_inpaint_mask_brush_color': '#ffffff', 'img2img_inpaint_sketch_default_brush_color': '#ffffff', 'return_mask': False, 'return_mask_composite': False, 'img2img_batch_show_results_limit': 32, 'cross_attention_optimization': 'Automatic', 's_min_uncond': 0.0, 'token_merging_ratio': 0.0, 'token_merging_ratio_img2img': 0.0, 'token_merging_ratio_hr': 0.0, 'pad_cond_uncond': False, 'persistent_cond_cache': True, 'batch_cond_uncond': True, 'use_old_emphasis_implementation': False, 'use_old_karras_scheduler_sigmas': False, 'no_dpmpp_sde_batch_determinism': False, 'use_old_hires_fix_width_height': False, 'dont_fix_second_order_samplers_schedule': False, 'hires_fix_use_firstpass_conds': False, 'use_old_scheduling': False, 'interrogate_keep_models_in_memory': False, 'interrogate_return_ranks': False, 'interrogate_clip_num_beams': 1, 'interrogate_clip_min_length': 24, 'interrogate_clip_max_length': 48, 'interrogate_clip_dict_limit': 1500, 'interrogate_clip_skip_categories': [], 'interrogate_deepbooru_score_threshold': 0.5, 'deepbooru_sort_alpha': True, 'deepbooru_use_spaces': True, 'deepbooru_escape': True, 'deepbooru_filter_tags': '', 'extra_networks_show_hidden_directories': True, 'extra_networks_dir_button_function': False, 'extra_networks_hidden_models': 'When searched', 'extra_networks_default_multiplier': 1.0, 'extra_networks_card_width': 0, 'extra_networks_card_height': 0, 'extra_networks_card_text_scale': 1.0, 'extra_networks_card_show_desc': True, 'extra_networks_card_order_field': 'Path', 'extra_networks_card_order': 'Ascending', 'extra_networks_add_text_separator': ' ', 'ui_extra_networks_tab_reorder': '', 'textual_inversion_print_at_load': False, 'textual_inversion_add_hashes_to_infotext': True, 'sd_hypernetwork': 'None', 'keyedit_precision_attention': 0.1, 'keyedit_precision_extra': 0.05, 'keyedit_delimiters': '.,\\/!?%^*;:{}=`~() ', 'keyedit_delimiters_whitespace': ['Tab', 'Carriage Return', 'Line Feed'], 'keyedit_move': True, 'disable_token_counters': False, 'return_grid': True, 'do_not_show_images': False, 'js_modal_lightbox': True, 'js_modal_lightbox_initially_zoomed': True, 'js_modal_lightbox_gamepad': False, 'js_modal_lightbox_gamepad_repeat': 250, 'gallery_height': '', 'compact_prompt_box': False, 'samplers_in_dropdown': True, 'dimensions_and_batch_together': True, 'sd_checkpoint_dropdown_use_short': False, 'hires_fix_show_sampler': False, 'hires_fix_show_prompts': False, 'txt2img_settings_accordion': False, 'img2img_settings_accordion': False, 'localization': 'None', 'quicksettings_list': ['sd_model_checkpoint'], 'ui_tab_order': [], 'hidden_tabs': [], 'ui_reorder_list': [], 'gradio_theme': 'Default', 'gradio_themes_cache': True, 'show_progress_in_title': True, 'send_seed': True, 'send_size': True, 'enable_pnginfo': True, 'save_txt': False, 'add_model_name_to_info': True, 'add_model_hash_to_info': True, 'add_vae_name_to_info': True, 'add_vae_hash_to_info': True, 'add_user_name_to_info': False, 'add_version_to_infotext': True, 'disable_weights_auto_swap': True, 'infotext_skip_pasting': [], 'infotext_styles': 'Apply if any', 'show_progressbar': True, 'live_previews_enable': True, 'live_previews_image_format': 'png', 'show_progress_grid': True, 'show_progress_every_n_steps': 10, 'show_progress_type': 'Approx NN', 'live_preview_allow_lowvram_full': False, 'live_preview_content': 'Prompt', 'live_preview_refresh_period': 1000, 'live_preview_fast_interrupt': False, 'js_live_preview_in_modal_lightbox': False, 'hide_samplers': [], 'eta_ddim': 0.0, 'eta_ancestral': 1.0, 'ddim_discretize': 'uniform', 's_churn': 0.0, 's_tmin': 0.0, 's_tmax': 0.0, 's_noise': 1.0, 'k_sched_type': 'Automatic', 'sigma_min': 0.0, 'sigma_max': 0.0, 'rho': 0.0, 'eta_noise_seed_delta': 0, 'always_discard_next_to_last_sigma': False, 'sgm_noise_multiplier': False, 'uni_pc_variant': 'bh1', 'uni_pc_skip_type': 'time_uniform', 'uni_pc_order': 3, 'uni_pc_lower_order_final': True, 'postprocessing_enable_in_main_ui': [], 'postprocessing_operation_order': [], 'upscaling_max_images_in_cache': 5, 'postprocessing_existing_caption_action': 'Ignore', 'disabled_extensions': [], 'disable_all_extensions': 'none', 'restore_config_state_file': '', 'sd_checkpoint_hash': ''}
        """

        self.restricted_opts = restricted_opts

    def __setattr__(self, key, value):
        if key in options_builtin_fields:
            return super(Options, self).__setattr__(key, value)

        if self.data is not None:
            if key in self.data or key in self.data_labels:
                assert not cmd_opts.freeze_settings, "changing settings is disabled"

                info = self.data_labels.get(key, None)
                if info.do_not_save:
                    return

                comp_args = info.component_args if info else None
                if isinstance(comp_args, dict) and comp_args.get('visible', True) is False:
                    raise RuntimeError(f"not possible to set {key} because it is restricted")

                if cmd_opts.hide_ui_dir_config and key in self.restricted_opts:
                    raise RuntimeError(f"not possible to set {key} because it is restricted")

                self.data[key] = value
                return

        return super(Options, self).__setattr__(key, value)

    def __getattr__(self, item):
        if item in options_builtin_fields:
            return super(Options, self).__getattribute__(item)

        if self.data is not None:
            if item in self.data:
                return self.data[item]

        if item in self.data_labels:
            return self.data_labels[item].default

        return super(Options, self).__getattribute__(item)

    def set(self, key, value, is_api=False, run_callbacks=True):
        """sets an option and calls its onchange callback, returning True if the option changed and False otherwise"""

        oldval = self.data.get(key, None)
        if oldval == value:
            return False

        option = self.data_labels[key]
        if option.do_not_save:
            return False

        if is_api and option.restrict_api:
            return False

        try:
            setattr(self, key, value)
        except RuntimeError:
            return False

        if run_callbacks and option.onchange is not None:
            try:
                option.onchange()
            except Exception as e:
                errors.display(e, f"changing setting {key} to {value}")
                setattr(self, key, oldval)
                return False

        return True

    def get_default(self, key):
        """returns the default value for the key"""

        data_label = self.data_labels.get(key)
        if data_label is None:
            return None

        return data_label.default

    def save(self, filename):
        assert not cmd_opts.freeze_settings, "saving settings is disabled"

        with open(filename, "w", encoding="utf8") as file:
            json.dump(self.data, file, indent=4, ensure_ascii=False)

    def same_type(self, x, y):
        if x is None or y is None:
            return True

        type_x = self.typemap.get(type(x), type(x))
        type_y = self.typemap.get(type(y), type(y))

        return type_x == type_y

    def load(self, filename):
        with open(filename, "r", encoding="utf8") as file:
            self.data = json.load(file)

        # 1.6.0 VAE defaults
        if self.data.get('sd_vae_as_default') is not None and self.data.get('sd_vae_overrides_per_model_preferences') is None:
            self.data['sd_vae_overrides_per_model_preferences'] = not self.data.get('sd_vae_as_default')

        # 1.1.1 quicksettings list migration
        if self.data.get('quicksettings') is not None and self.data.get('quicksettings_list') is None:
            self.data['quicksettings_list'] = [i.strip() for i in self.data.get('quicksettings').split(',')]

        # 1.4.0 ui_reorder
        if isinstance(self.data.get('ui_reorder'), str) and self.data.get('ui_reorder') and "ui_reorder_list" not in self.data:
            self.data['ui_reorder_list'] = [i.strip() for i in self.data.get('ui_reorder').split(',')]

        bad_settings = 0
        for k, v in self.data.items():
            info = self.data_labels.get(k, None)
            if info is not None and not self.same_type(info.default, v):
                print(f"Warning: bad setting value: {k}: {v} ({type(v).__name__}; expected {type(info.default).__name__})", file=sys.stderr)
                bad_settings += 1

        if bad_settings > 0:
            print(f"The program is likely to not work with bad settings.\nSettings file: {filename}\nEither fix the file, or delete it and restart.", file=sys.stderr)

    def onchange(self, key, func, call=True):
        item = self.data_labels.get(key)
        item.onchange = func

        if call:
            func()

    def dumpjson(self):
        d = {k: self.data.get(k, v.default) for k, v in self.data_labels.items()}
        d["_comments_before"] = {k: v.comment_before for k, v in self.data_labels.items() if v.comment_before is not None}
        d["_comments_after"] = {k: v.comment_after for k, v in self.data_labels.items() if v.comment_after is not None}

        item_categories = {}
        for item in self.data_labels.values():
            category = categories.mapping.get(item.category_id)
            category = "Uncategorized" if category is None else category.label
            if category not in item_categories:
                item_categories[category] = item.section[1]

        # _categories is a list of pairs: [section, category]. Each section (a setting page) will get a special heading above it with the category as text.
        d["_categories"] = [[v, k] for k, v in item_categories.items()] + [["Defaults", "Other"]]

        return json.dumps(d)

    def add_option(self, key, info):
        self.data_labels[key] = info
        if key not in self.data and not info.do_not_save:
            self.data[key] = info.default

    def reorder(self):
        """Reorder settings so that:
            - all items related to section always go together
            - all sections belonging to a category go together
            - sections inside a category are ordered alphabetically
            - categories are ordered by creation order

        Category is a superset of sections: for category "postprocessing" there could be multiple sections: "face restoration", "upscaling".

        This function also changes items' category_id so that all items belonging to a section have the same category_id.
        """

        category_ids = {}
        section_categories = {}

        settings_items = self.data_labels.items()
        for _, item in settings_items:
            if item.section not in section_categories:
                section_categories[item.section] = item.category_id

        for _, item in settings_items:
            item.category_id = section_categories.get(item.section)

        for category_id in categories.mapping:
            if category_id not in category_ids:
                category_ids[category_id] = len(category_ids)

        def sort_key(x):
            item: OptionInfo = x[1]
            category_order = category_ids.get(item.category_id, len(category_ids))
            section_order = item.section[1]

            return category_order, section_order

        self.data_labels = dict(sorted(settings_items, key=sort_key))

    def cast_value(self, key, value):
        """casts an arbitrary to the same type as this setting's value with key
        Example: cast_value("eta_noise_seed_delta", "12") -> returns 12 (an int rather than str)
        """

        if value is None:
            return None

        default_value = self.data_labels[key].default
        if default_value is None:
            default_value = getattr(self, key, None)
        if default_value is None:
            return None

        expected_type = type(default_value)
        if expected_type == bool and value == "False":
            value = False
        else:
            value = expected_type(value)

        return value


@dataclass
class OptionsCategory:
    id: str
    label: str

class OptionsCategories:
    def __init__(self):
        self.mapping = {}

    def register_category(self, category_id, label):
        if category_id in self.mapping:
            return category_id

        self.mapping[category_id] = OptionsCategory(category_id, label)


categories = OptionsCategories()
