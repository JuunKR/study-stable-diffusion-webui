import os
import re
import sys
import inspect
from collections import namedtuple
from dataclasses import dataclass

import gradio as gr

from modules import shared, paths, script_callbacks, extensions, script_loading, scripts_postprocessing, errors, timer

AlwaysVisible = object()


class PostprocessImageArgs:
    def __init__(self, image):
        self.image = image


class PostprocessBatchListArgs:
    def __init__(self, images):
        self.images = images


@dataclass
class OnComponent:
    component: gr.blocks.Block


class Script:
    name = None
    """script's internal name derived from title"""

    section = None
    """name of UI section that the script's controls will be placed into"""

    filename = None
    args_from = None
    args_to = None
    alwayson = False

    is_txt2img = False
    is_img2img = False
    tabname = None

    group = None
    """A gr.Group component that has all script's UI inside it."""

    create_group = True
    """If False, for alwayson scripts, a group component will not be created."""

    infotext_fields = None
    """if set in ui(), this is a list of pairs of gradio component + text; the text will be used when
    parsing infotext to set the value for the component; see ui.py's txt2img_paste_fields for an example
    """

    paste_field_names = None
    """if set in ui(), this is a list of names of infotext fields; the fields will be sent through the
    various "Send to <X>" buttons when clicked
    """

    api_info = None
    """Generated value of type modules.api.models.ScriptInfo with information about the script for API"""

    on_before_component_elem_id = None
    """list of callbacks to be called before a component with an elem_id is created"""

    on_after_component_elem_id = None
    """list of callbacks to be called after a component with an elem_id is created"""

    setup_for_ui_only = False
    """If true, the script setup will only be run in Gradio UI, not in API"""

    def title(self):
        """this function should return the title of the script. This is what will be displayed in the dropdown menu."""

        raise NotImplementedError()

    def ui(self, is_img2img):
        """this function should create gradio UI elements. See https://gradio.app/docs/#components
        The return value should be an array of all components that are used in processing.
        Values of those returned components will be passed to run() and process() functions.
        """

        pass

    def show(self, is_img2img):
        """
        is_img2img is True if this function is called for the img2img interface, and Fasle otherwise

        This function should return:
         - False if the script should not be shown in UI at all
         - True if the script should be shown in UI if it's selected in the scripts dropdown
         - script.AlwaysVisible if the script should be shown in UI at all times
         """

        return True

    def run(self, p, *args):
        """
        This function is called if the script has been selected in the script dropdown.
        It must do all processing and return the Processed object with results, same as
        one returned by processing.process_images.

        Usually the processing is done by calling the processing.process_images function.

        args contains all values returned by components from ui()
        """

        pass

    def setup(self, p, *args):
        """For AlwaysVisible scripts, this function is called when the processing object is set up, before any processing starts.
        args contains all values returned by components from ui().
        """
        pass


    def before_process(self, p, *args):
        """
        This function is called very early during processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """

        pass

    def process(self, p, *args):
        """
        This function is called before processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """

        pass

    def before_process_batch(self, p, *args, **kwargs):
        """
        Called before extra networks are parsed from the prompt, so you can add
        new extra network keywords to the prompt with this callback.

        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
        """

        pass

    def after_extra_networks_activate(self, p, *args, **kwargs):
        """
        Called after extra networks activation, before conds calculation
        allow modification of the network after extra networks activation been applied
        won't be call if p.disable_extra_networks

        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
          - extra_network_data - list of ExtraNetworkParams for current stage
        """
        pass

    def process_batch(self, p, *args, **kwargs):
        """
        Same as process(), but called for every batch.

        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
        """

        pass

    def postprocess_batch(self, p, *args, **kwargs):
        """
        Same as process_batch(), but called for every batch after it has been generated.

        **kwargs will have same items as process_batch, and also:
          - batch_number - index of current batch, from 0 to number of batches-1
          - images - torch tensor with all generated images, with values ranging from 0 to 1;
        """

        pass

    def postprocess_batch_list(self, p, pp: PostprocessBatchListArgs, *args, **kwargs):
        """
        Same as postprocess_batch(), but receives batch images as a list of 3D tensors instead of a 4D tensor.
        This is useful when you want to update the entire batch instead of individual images.

        You can modify the postprocessing object (pp) to update the images in the batch, remove images, add images, etc.
        If the number of images is different from the batch size when returning,
        then the script has the responsibility to also update the following attributes in the processing object (p):
          - p.prompts
          - p.negative_prompts
          - p.seeds
          - p.subseeds

        **kwargs will have same items as process_batch, and also:
          - batch_number - index of current batch, from 0 to number of batches-1
        """

        pass

    def postprocess_image(self, p, pp: PostprocessImageArgs, *args):
        """
        Called for every image after it has been generated.
        """

        pass

    def postprocess(self, p, processed, *args):
        """
        This function is called after processing ends for AlwaysVisible scripts.
        args contains all values returned by components from ui()
        """

        pass

    def before_component(self, component, **kwargs):
        """
        Called before a component is created.
        Use elem_id/label fields of kwargs to figure out which component it is.
        This can be useful to inject your own components somewhere in the middle of vanilla UI.
        You can return created components in the ui() function to add them to the list of arguments for your processing functions
        """

        pass

    def after_component(self, component, **kwargs):
        """
        Called after a component is created. Same as above.
        """

        pass

    def on_before_component(self, callback, *, elem_id):
        """
        Calls callback before a component is created. The callback function is called with a single argument of type OnComponent.

        May be called in show() or ui() - but it may be too late in latter as some components may already be created.

        This function is an alternative to before_component in that it also cllows to run before a component is created, but
        it doesn't require to be called for every created component - just for the one you need.
        """
        if self.on_before_component_elem_id is None:
            self.on_before_component_elem_id = []

        self.on_before_component_elem_id.append((elem_id, callback))

    def on_after_component(self, callback, *, elem_id):
        """
        Calls callback after a component is created. The callback function is called with a single argument of type OnComponent.
        """
        if self.on_after_component_elem_id is None:
            self.on_after_component_elem_id = []

        self.on_after_component_elem_id.append((elem_id, callback))

    def describe(self):
        """unused"""
        return ""

    def elem_id(self, item_id):
        """helper function to generate id for a HTML element, constructs final id out of script name, tab and user-supplied item_id"""

        need_tabname = self.show(True) == self.show(False)
        tabkind = 'img2img' if self.is_img2img else 'txt2img'
        tabname = f"{tabkind}_" if need_tabname else ""
        title = re.sub(r'[^a-z_0-9]', '', re.sub(r'\s', '_', self.title().lower()))

        return f'script_{tabname}{title}_{item_id}'

    def before_hr(self, p, *args):
        """
        This function is called before hires fix start.
        """
        pass


class ScriptBuiltinUI(Script):
    setup_for_ui_only = True

    def elem_id(self, item_id):
        """helper function to generate id for a HTML element, constructs final id out of tab and user-supplied item_id"""

        need_tabname = self.show(True) == self.show(False)
        tabname = ('img2img' if self.is_img2img else 'txt2img') + "_" if need_tabname else ""

        return f'{tabname}{item_id}'


current_basedir = paths.script_path


def basedir():
    """returns the base directory for the current script. For scripts in the main scripts directory,
    this is the main directory (where webui.py resides), and for scripts in extensions directory
    (ie extensions/aesthetic/script/aesthetic.py), this is extension's directory (extensions/aesthetic)
    """
    return current_basedir


ScriptFile = namedtuple("ScriptFile", ["basedir", "filename", "path"])

scripts_data = []
postprocessing_scripts_data = []
ScriptClassData = namedtuple("ScriptClassData", ["script_class", "path", "basedir", "module"])

def topological_sort(dependencies):
    """Accepts a dictionary mapping name to its dependencies, returns a list of names ordered according to dependencies.
    Ignores errors relating to missing dependeencies or circular dependencies
    """

    visited = {}
    result = []

    def inner(name):
        visited[name] = True

        for dep in dependencies.get(name, []):
            if dep in dependencies and dep not in visited:
                inner(dep)

        result.append(name)

    for depname in dependencies:
        if depname not in visited:
            inner(depname)

    return result


@dataclass
class ScriptWithDependencies:
    script_canonical_name: str
    file: ScriptFile
    requires: list
    load_before: list
    load_after: list


def list_scripts(scriptdirname, extension, *, include_extensions=True):
    scripts = {}

    loaded_extensions = {ext.canonical_name: ext for ext in extensions.active()}
    loaded_extensions_scripts = {ext.canonical_name: [] for ext in extensions.active()}
    """
    loaded_extensions 
    {'ldsr': <modules.extensions.Extension object at 0x7f17cd3701c0>, 'lora': <modules.extensions.Extension object at 0x7f17cd371570>, 'scunet': <modules.extensions.Extension object at 0x7f17cd3716f0>, 'swinir': <modules.extensions.Extension object at 0x7f17cd371870>, 'canvas-zoom-and-pan': <modules.extensions.Extension object at 0x7f17cd3719f0>, 'extra-options-section': <modules.extensions.Extension object at 0x7f17cd371b70>, 'hypertile': <modules.extensions.Extension object at 0x7f17cd3430d0>, 'mobile': <modules.extensions.Extension object at 0x7f17cd371de0>, 'prompt-bracket-checker': <modules.extensions.Extension object at 0x7f17cd371f60>, 'adetailer': <modules.extensions.Extension object at 0x7f17cd3720e0>, 'openpose-editor': <modules.extensions.Extension object at 0x7f17cd372260>, 'sd-dynamic-prompts': <modules.extensions.Extension object at 0x7f17cd3723e0>, 'sd-webui-3d-open-pose-editor': <modules.extensions.Extension object at 0x7f17cd372560>, 'sd-webui-agent-scheduler': <modules.extensions.Extension object at 0x7f17cd3726e0>, 'sd-webui-ar': <modules.extensions.Extension object at 0x7f17cd372860>, 'sd-webui-controlnet': <modules.extensions.Extension object at 0x7f17cd3729e0>, 'sd-webui-lora-block-weight': <modules.extensions.Extension object at 0x7f17cd372b60>, 'sd-webui-openpose-editor': <modules.extensions.Extension object at 0x7f17cd372ce0>, 'sd-webui-regional-prompter': <modules.extensions.Extension object at 0x7f17cd372e60>, 'sd-webui-segment-anything': <modules.extensions.Extension object at 0x7f17cd372fe0>, 'stable-diffusion-webui-images-browser': <modules.extensions.Extension object at 0x7f17cd373160>, 'ultimate-upscale-for-automatic1111': <modules.extensions.Extension object at 0x7f17cd3732e0>}
    
    loaded_extensions_scripts 
    {'ldsr': [], 'lora': [], 'scunet': [], 'swinir': [], 'canvas-zoom-and-pan': [], 'extra-options-section': [], 'hypertile': [], 'mobile': [], 'prompt-bracket-checker': [], 'adetailer': [], 'openpose-editor': [], 'sd-dynamic-prompts': [], 'sd-webui-3d-open-pose-editor': [], 'sd-webui-agent-scheduler': [], 'sd-webui-ar': [], 'sd-webui-controlnet': [], 'sd-webui-lora-block-weight': [], 'sd-webui-openpose-editor': [], 'sd-webui-regional-prompter': [], 'sd-webui-segment-anything': [], 'stable-diffusion-webui-images-browser': [], 'ultimate-upscale-for-automatic1111': []}
    
    """
    # build script dependency map
    root_script_basedir = os.path.join(paths.script_path, scriptdirname)
    if os.path.exists(root_script_basedir):
        for filename in sorted(os.listdir(root_script_basedir)):
            if not os.path.isfile(os.path.join(root_script_basedir, filename)):
                continue

            if os.path.splitext(filename)[1].lower() != extension:
                continue

            script_file = ScriptFile(paths.script_path, filename, os.path.join(root_script_basedir, filename))
            scripts[filename] = ScriptWithDependencies(filename, script_file, [], [], [])

    if include_extensions:
        for ext in extensions.active():
            extension_scripts_list = ext.list_files(scriptdirname, extension)
            for extension_script in extension_scripts_list:
                if not os.path.isfile(extension_script.path):
                    continue

                script_canonical_name = ("builtin/" if ext.is_builtin else "") + ext.canonical_name + "/" + extension_script.filename
                relative_path = scriptdirname + "/" + extension_script.filename

                script = ScriptWithDependencies(
                    script_canonical_name=script_canonical_name,
                    file=extension_script,
                    requires=ext.metadata.get_script_requirements("Requires", relative_path, scriptdirname),
                    load_before=ext.metadata.get_script_requirements("Before", relative_path, scriptdirname),
                    load_after=ext.metadata.get_script_requirements("After", relative_path, scriptdirname),
                )

                scripts[script_canonical_name] = script
                loaded_extensions_scripts[ext.canonical_name].append(script)

    for script_canonical_name, script in scripts.items():
        # load before requires inverse dependency
        # in this case, append the script name into the load_after list of the specified script
        for load_before in script.load_before:
            # if this requires an individual script to be loaded before
            other_script = scripts.get(load_before)
            if other_script:
                other_script.load_after.append(script_canonical_name)

            # if this requires an extension
            other_extension_scripts = loaded_extensions_scripts.get(load_before)
            if other_extension_scripts:
                for other_script in other_extension_scripts:
                    other_script.load_after.append(script_canonical_name)

        # if After mentions an extension, remove it and instead add all of its scripts
        for load_after in list(script.load_after):
            if load_after not in scripts and load_after in loaded_extensions_scripts:
                script.load_after.remove(load_after)

                for other_script in loaded_extensions_scripts.get(load_after, []):
                    script.load_after.append(other_script.script_canonical_name)

    dependencies = {}

    for script_canonical_name, script in scripts.items():
        for required_script in script.requires:
            if required_script not in scripts and required_script not in loaded_extensions:
                errors.report(f'Script "{script_canonical_name}" requires "{required_script}" to be loaded, but it is not.', exc_info=False)

        dependencies[script_canonical_name] = script.load_after

    ordered_scripts = topological_sort(dependencies)
    scripts_list = [scripts[script_canonical_name].file for script_canonical_name in ordered_scripts]

    return scripts_list


def list_files_with_name(filename):
    res = []

    dirs = [paths.script_path] + [ext.path for ext in extensions.active()]

    for dirpath in dirs:
        if not os.path.isdir(dirpath):
            continue

        path = os.path.join(dirpath, filename)
        if os.path.isfile(path):
            res.append(path)

    return res


def load_scripts():
    global current_basedir
    
    scripts_data.clear()
    postprocessing_scripts_data.clear()
    script_callbacks.clear_callbacks()

    scripts_list = list_scripts("scripts", ".py") + list_scripts("modules/processing_scripts", ".py", include_extensions=False)
    syspath = sys.path

    def register_scripts_from_module(module):
        for script_class in module.__dict__.values():
            if not inspect.isclass(script_class):
                continue

            if issubclass(script_class, Script):
                scripts_data.append(ScriptClassData(script_class, scriptfile.path, scriptfile.basedir, module))
            elif issubclass(script_class, scripts_postprocessing.ScriptPostprocessing):
                postprocessing_scripts_data.append(ScriptClassData(script_class, scriptfile.path, scriptfile.basedir, module))

    # here the scripts_list is already ordered
    # processing_script is not considered though
    for scriptfile in scripts_list:
        try:
            if scriptfile.basedir != paths.script_path:
                sys.path = [scriptfile.basedir] + sys.path
            current_basedir = scriptfile.basedir

            script_module = script_loading.load_module(scriptfile.path)
            register_scripts_from_module(script_module)

        except Exception:
            errors.report(f"Error loading script: {scriptfile.filename}", exc_info=True)

        finally:
            sys.path = syspath
            current_basedir = paths.script_path
            timer.startup_timer.record(scriptfile.filename)

    global scripts_txt2img, scripts_img2img, scripts_postproc

    scripts_txt2img = ScriptRunner()
    scripts_img2img = ScriptRunner()
    scripts_postproc = scripts_postprocessing.ScriptPostprocessingRunner()


def wrap_call(func, filename, funcname, *args, default=None, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        errors.report(f"Error calling: {filename}/{funcname}", exc_info=True)

    return default


class ScriptRunner:
    def __init__(self):
        self.scripts = []
        self.selectable_scripts = []
        self.alwayson_scripts = []
        self.titles = []
        self.title_map = {}
        self.infotext_fields = []
        self.paste_field_names = []
        self.inputs = [None]

        self.on_before_component_elem_id = {}
        """dict of callbacks to be called before an element is created; key=elem_id, value=list of callbacks"""

        self.on_after_component_elem_id = {}
        """dict of callbacks to be called after an element is created; key=elem_id, value=list of callbacks"""

    def initialize_scripts(self, is_img2img):
        from modules import scripts_auto_postprocessing

        self.scripts.clear()
        self.alwayson_scripts.clear()
        self.selectable_scripts.clear()

        auto_processing_scripts = scripts_auto_postprocessing.create_auto_preprocessing_script_data()
        for script_data in auto_processing_scripts + scripts_data:
            script = script_data.script_class()
            script.filename = script_data.path
            script.is_txt2img = not is_img2img
            script.is_img2img = is_img2img
            script.tabname = "img2img" if is_img2img else "txt2img"

            visibility = script.show(script.is_img2img)
            """
            visibility script <prompt_matrix.py.Script object at 0x7f5b648a8d30>
            visibility script <prompts_from_file.py.Script object at 0x7f5b648a97e0>
            visibility script <xyz_grid.py.Script object at 0x7f5b648a98a0>
            alwayson_scripts <extra_options_section.py.ExtraOptionsSection object at 0x7f5b648a98d0>
            alwayson_scripts <hypertile_script.py.ScriptHypertile object at 0x7f5b648a9840>
            alwayson_scripts AfterDetailerScript(version=24.1.1)
            alwayson_scripts <main.py.Script object at 0x7f5b648a97b0>
            alwayson_scripts <sd_dynamic_prompts.dynamic_prompting.Script object at 0x7f5b648a8d00>
            alwayson_scripts <task_scheduler.py.Script object at 0x7f5b648a9750>
            alwayson_scripts <sd-webui-ar.py.AspectRatioScript object at 0x7f5b648a9450>
            alwayson_scripts <controlnet.py.Script object at 0x7f5b648a9480>
            visibility script <movie2movie.py.Script object at 0x7f5b648a9420>
            alwayson_scripts <lora_block_weight.py.Script object at 0x7f5b648aa350>
            alwayson_scripts <rp.py.Script object at 0x7f5b648aa380>
            visibility script <rps.py.Script object at 0x7f5b648aa3b0>
            alwayson_scripts <sam.py.Script object at 0x7f5b648aa3e0>
            alwayson_scripts <refiner.py.ScriptRefiner object at 0x7f5b648aa4a0>
            alwayson_scripts <seed.py.ScriptSeed object at 0x7f5b648aa470>
            sd-lora-block ui
            visibility script <img2imgalt.py.Script object at 0x7f5b63bd5180>
            visibility script <loopback.py.Script object at 0x7f5b63cb7a30>
            visibility script <outpainting_mk_2.py.Script object at 0x7f5b63bd5f60>
            visibility script <poor_mans_outpainting.py.Script object at 0x7f5b63bd51b0>
            visibility script <prompt_matrix.py.Script object at 0x7f5b63bd68c0>
            visibility script <prompts_from_file.py.Script object at 0x7f5b63bd68f0>
            visibility script <sd_upscale.py.Script object at 0x7f5b63bd7100>
            visibility script <xyz_grid.py.Script object at 0x7f5b63bec9a0>
            alwayson_scripts <extra_options_section.py.ExtraOptionsSection object at 0x7f5b63bec9d0>
            alwayson_scripts <hypertile_script.py.ScriptHypertile object at 0x7f5b63beca00>
            alwayson_scripts AfterDetailerScript(version=24.1.1)
            alwayson_scripts <main.py.Script object at 0x7f5b63beca60>
            alwayson_scripts <sd_dynamic_prompts.dynamic_prompting.Script object at 0x7f5b63beca90>
            alwayson_scripts <task_scheduler.py.Script object at 0x7f5b63becb50>
            alwayson_scripts <sd-webui-ar.py.AspectRatioScript object at 0x7f5b63bee530>
            alwayson_scripts <controlnet.py.Script object at 0x7f5b63bee4a0>
            visibility script <movie2movie.py.Script object at 0x7f5b63bee500>
            alwayson_scripts <lora_block_weight.py.Script object at 0x7f5b63becaf0>
            alwayson_scripts <rp.py.Script object at 0x7f5b63bee470>
            visibility script <rps.py.Script object at 0x7f5b63bed480>
            alwayson_scripts <sam.py.Script object at 0x7f5b63bee680>
            visibility script <ultimate-upscale.py.Script object at 0x7f5b63bee6b0>
            alwayson_scripts <refiner.py.ScriptRefiner object at 0x7f5b63bee710>
            alwayson_scripts <seed.py.ScriptSeed object at 0x7f5b63bee740>
            
            
            """ 
            if visibility == AlwaysVisible:
                # 스크립트는 다들어가있는거
                self.scripts.append(script)
                # alayson은 모르겠음
                self.alwayson_scripts.append(script)
                script.alwayson = True

            elif visibility:
                self.scripts.append(script)
                # xyz
                self.selectable_scripts.append(script)
        self.apply_on_before_component_callbacks()

    def apply_on_before_component_callbacks(self):
        for script in self.scripts:
            on_before = script.on_before_component_elem_id or []
            on_after = script.on_after_component_elem_id or []

            for elem_id, callback in on_before:
                if elem_id not in self.on_before_component_elem_id:
                    self.on_before_component_elem_id[elem_id] = []

                self.on_before_component_elem_id[elem_id].append((callback, script))

            for elem_id, callback in on_after:
                if elem_id not in self.on_after_component_elem_id:
                    self.on_after_component_elem_id[elem_id] = []

                self.on_after_component_elem_id[elem_id].append((callback, script))

            on_before.clear()
            on_after.clear()

    def create_script_ui(self, script):

        script.args_from = len(self.inputs)
        script.args_to = len(self.inputs)

        try:
            self.create_script_ui_inner(script)
        except Exception:
            errors.report(f"Error creating UI for {script.name}: ", exc_info=True)

    def create_script_ui_inner(self, script):
        import modules.api.models as api_models

        controls = wrap_call(script.ui, script.filename, "ui", script.is_img2img)

        if controls is None:
            return

        script.name = wrap_call(script.title, script.filename, "title", default=script.filename).lower()

        api_args = []

        for control in controls:
            control.custom_script_source = os.path.basename(script.filename)

            arg_info = api_models.ScriptArg(label=control.label or "")

            for field in ("value", "minimum", "maximum", "step"):
                v = getattr(control, field, None)
                if v is not None:
                    setattr(arg_info, field, v)

            choices = getattr(control, 'choices', None)  # as of gradio 3.41, some items in choices are strings, and some are tuples where the first elem is the string
            if choices is not None:
                arg_info.choices = [x[0] if isinstance(x, tuple) else x for x in choices]

            api_args.append(arg_info)

        script.api_info = api_models.ScriptInfo(
            name=script.name,
            is_img2img=script.is_img2img,
            is_alwayson=script.alwayson,
            args=api_args,
        )

        if script.infotext_fields is not None:
            self.infotext_fields += script.infotext_fields

        if script.paste_field_names is not None:
            self.paste_field_names += script.paste_field_names

        self.inputs += controls
        script.args_to = len(self.inputs)

    def setup_ui_for_section(self, section, scriptlist=None):
        if scriptlist is None:
            scriptlist = self.alwayson_scripts

        for script in scriptlist:
            if script.alwayson and script.section != section:
                continue

            if script.create_group:
                with gr.Group(visible=script.alwayson) as group:
                    self.create_script_ui(script)

                script.group = group
            else:
                self.create_script_ui(script)

    def prepare_ui(self):
        self.inputs = [None]

    def setup_ui(self):
        all_titles = [wrap_call(script.title, script.filename, "title") or script.filename for script in self.scripts]
        print("all_titles", all_titles)
        #  ['Prompt matrix', 'Prompts from file or textbox', 'X/Y/Z plot', 'Extra options', 'Hypertile', 'ADetailer', 'OpenPose Editor', 'Dynamic Prompts v2.17.1', 'Agent Scheduler', 'Aspect Ratio picker', 'ControlNet', 'controlnet m2m', 'LoRA Block Weight', 'Regional Prompter', 'Differential Regional Prompter', 'Segment Anything', 'Refiner', 'Seed']
        self.title_map = {title.lower(): script for title, script in zip(all_titles, self.scripts)}
        self.titles = [wrap_call(script.title, script.filename, "title") or f"{script.filename} [error]" for script in self.selectable_scripts]
        """
        self.title_map  {
            'prompt matrix': <prompt_matrix.py.Script object at 0x7f30bec94d30>, 
            'prompts from file or textbox': <prompts_from_file.py.Script object at 0x7f30bec957b0>, 
            'x/y/z plot': <xyz_grid.py.Script object at 0x7f30bec958d0>, 
            'extra options': <extra_options_section.py.ExtraOptionsSection object at 0x7f30bec95900>, 
            'hypertile': <hypertile_script.py.ScriptHypertile object at 0x7f30bec95870>, 
            'adetailer': AfterDetailerScript(version=24.1.1), 
            'openpose editor': <main.py.Script object at 0x7f30bec95780>, 
            'dynamic prompts v2.17.1': <sd_dynamic_prompts.dynamic_prompting.Script object at 0x7f30bec957e0>, 
            'agent scheduler': <task_scheduler.py.Script object at 0x7f30bed8d930>, 
            'aspect ratio picker': <sd-webui-ar.py.AspectRatioScript object at 0x7f30bed8d900>, 
            'controlnet': <controlnet.py.Script object at 0x7f30bed8dab0>,
            'controlnet m2m': <movie2movie.py.Script object at 0x7f30bed8dba0>, 
            'lora block weight': <lora_block_weight.py.Script object at 0x7f30bec950f0>, 
            'regional prompter': <rp.py.Script object at 0x7f30bec950c0>, 
            'differential regional prompter': <rps.py.Script object at 0x7f30bec95090>, 
            'segment anything': <sam.py.Script object at 0x7f30bec95060>, 
            'refiner': <refiner.py.ScriptRefiner object at 0x7f30bec94fa0>, 
            'seed': <seed.py.ScriptSeed object at 0x7f30bec94fd0>}
        
        self.titles  ['Prompt matrix', 'Prompts from file or textbox', 'X/Y/Z plot', 'controlnet m2m', 'Differential Regional Prompter']
        

        """

        self.setup_ui_for_section(None)

        dropdown = gr.Dropdown(label="Script", elem_id="script_list", choices=["None"] + self.titles, value="None", type="index")
        self.inputs[0] = dropdown

        self.setup_ui_for_section(None, self.selectable_scripts)

        def select_script(script_index):
            selected_script = self.selectable_scripts[script_index - 1] if script_index>0 else None

            return [gr.update(visible=selected_script == s) for s in self.selectable_scripts]

        def init_field(title):
            """called when an initial value is set from ui-config.json to show script's UI components"""

            if title == 'None':
                return

            script_index = self.titles.index(title)
            self.selectable_scripts[script_index].group.visible = True

        dropdown.init_field = init_field

        dropdown.change(
            fn=select_script,
            inputs=[dropdown],
            outputs=[script.group for script in self.selectable_scripts]
        )

        self.script_load_ctr = 0

        def onload_script_visibility(params):
            title = params.get('Script', None)
            if title:
                title_index = self.titles.index(title)
                visibility = title_index == self.script_load_ctr
                self.script_load_ctr = (self.script_load_ctr + 1) % len(self.titles)
                return gr.update(visible=visibility)
            else:
                return gr.update(visible=False)

        self.infotext_fields.append((dropdown, lambda x: gr.update(value=x.get('Script', 'None'))))
        self.infotext_fields.extend([(script.group, onload_script_visibility) for script in self.selectable_scripts])

        self.apply_on_before_component_callbacks()

        return self.inputs

    def run(self, p, *args):
        script_index = args[0]

        if script_index == 0:
            return None

        script = self.selectable_scripts[script_index-1]

        if script is None:
            return None

        script_args = args[script.args_from:script.args_to]
        processed = script.run(p, *script_args)

        shared.total_tqdm.clear()

        return processed

    def before_process(self, p):
        # [<extra_options_section.py.ExtraOptionsSection object at 0x7ff00a361b40>, <hypertile_script.py.ScriptHypertile object at 0x7ff00a361b70>, AfterDetailerScript(version=24.1.1), <main.py.Script object at 0x7ff00a3619f0>, <sd_dynamic_prompts.dynamic_prompting.Script object at 0x7ff00a361a50>, <task_scheduler.py.Script object at 0x7ff00a3619c0>, <sd-webui-ar.py.AspectRatioScript object at 0x7ff00a3616c0>, <sam.py.Script object at 0x7ff00a461c90>, <lora_block_weight.py.Script object at 0x7ff00a461d50>, <rp.py.Script object at 0x7ff00a461cf0>, <controlnet.py.Script object at 0x7ff00a3617e0>, <refiner.py.ScriptRefiner object at 0x7ff00a461ab0>, <seed.py.ScriptSeed object at 0x7ff00a461ba0>]
        '''
        10 10
        10 10
        10 14
        14 14
        14 32
        32 32
        32 32
        73 97
        35 53
        53 73
        32 35
        1 4
        4 10
        '''
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.before_process(p, *script_args)
                """
                script.before_process <bound method Script.before_process of <hypertile_script.py.ScriptHypertile object at 0x7f960113fac0>>
                script.before_process <bound method Script.before_process of AfterDetailerScript(version=24.1.1)>
                script.before_process <bound method Script.before_process of <main.py.Script object at 0x7f960113fa30>>
                script.before_process <bound method Script.before_process of <sd_dynamic_prompts.dynamic_prompting.Script object at 0x7f960113f9d0>>
                script.before_process <bound method Script.before_process of <task_scheduler.py.Script object at 0x7f960113f970>>
                script.before_process <bound method Script.before_process of <sd-webui-ar.py.AspectRatioScript object at 0x7f9601190700>>
                script.before_process <bound method Script.before_process of <sam.py.Script object at 0x7f960129c580>>
                script.before_process <bound method Script.before_process of <lora_block_weight.py.Script object at 0x7f960129c640>>
                script.before_process <bound method Script.before_process of <rp.py.Script object at 0x7f960129c5e0>>
                script.before_process <bound method Script.before_process of <controlnet.py.Script object at 0x7f9601190760>>
                script.before_process <bound method Script.before_process of <refiner.py.ScriptRefiner object at 0x7f960129c3a0>>
                script.before_process <bound method Script.before_process of <seed.py.ScriptSeed object at 0x7f960129c490>>
                """
            except Exception:
                errors.report(f"Error running before_process: {script.filename}", exc_info=True)

    def process(self, p):
        for script in self.alwayson_scripts:
            """
            <extra_options_section.py.ExtraOptionsSection object at 0x7efbf77719c0>
            <hypertile_script.py.ScriptHypertile object at 0x7efbf7771930>
            AfterDetailerScript(version=24.1.1)
            <main.py.Script object at 0x7efbf77718a0>
            <sd_dynamic_prompts.dynamic_prompting.Script object at 0x7efbf7770df0>
            <task_scheduler.py.Script object at 0x7efbf7771840>
            <sd-webui-ar.py.AspectRatioScript object at 0x7efbf7771540>
            <sam.py.Script object at 0x7efbf7871ab0>
            <lora_block_weight.py.Script object at 0x7efbf7871b70>
            """
            
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.process(p, *script_args)
            except Exception:
                errors.report(f"Error running process: {script.filename}", exc_info=True)

    def before_process_batch(self, p, **kwargs):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.before_process_batch(p, *script_args, **kwargs)
            except Exception:
                errors.report(f"Error running before_process_batch: {script.filename}", exc_info=True)

    def after_extra_networks_activate(self, p, **kwargs):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.after_extra_networks_activate(p, *script_args, **kwargs)
            except Exception:
                errors.report(f"Error running after_extra_networks_activate: {script.filename}", exc_info=True)

    def process_batch(self, p, **kwargs):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.process_batch(p, *script_args, **kwargs)
            except Exception:
                errors.report(f"Error running process_batch: {script.filename}", exc_info=True)

    def postprocess(self, p, processed):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.postprocess(p, processed, *script_args)
            except Exception:
                errors.report(f"Error running postprocess: {script.filename}", exc_info=True)

    def postprocess_batch(self, p, images, **kwargs):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.postprocess_batch(p, *script_args, images=images, **kwargs)
            except Exception:
                errors.report(f"Error running postprocess_batch: {script.filename}", exc_info=True)

    def postprocess_batch_list(self, p, pp: PostprocessBatchListArgs, **kwargs):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.postprocess_batch_list(p, pp, *script_args, **kwargs)
            except Exception:
                errors.report(f"Error running postprocess_batch_list: {script.filename}", exc_info=True)

    def postprocess_image(self, p, pp: PostprocessImageArgs):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.postprocess_image(p, pp, *script_args)
            except Exception:
                errors.report(f"Error running postprocess_image: {script.filename}", exc_info=True)

    def before_component(self, component, **kwargs):
        for callback, script in self.on_before_component_elem_id.get(kwargs.get("elem_id"), []):
            try:
                callback(OnComponent(component=component))
            except Exception:
                errors.report(f"Error running on_before_component: {script.filename}", exc_info=True)

        for script in self.scripts:
            try:
                script.before_component(component, **kwargs)
            except Exception:
                errors.report(f"Error running before_component: {script.filename}", exc_info=True)

    def after_component(self, component, **kwargs):
        for callback, script in self.on_after_component_elem_id.get(component.elem_id, []):
            try:
                callback(OnComponent(component=component))
            except Exception:
                errors.report(f"Error running on_after_component: {script.filename}", exc_info=True)

        for script in self.scripts:
            try:
                script.after_component(component, **kwargs)
            except Exception:
                errors.report(f"Error running after_component: {script.filename}", exc_info=True)

    def script(self, title):
        return self.title_map.get(title.lower())

    def reload_sources(self, cache):
        for si, script in list(enumerate(self.scripts)):
            args_from = script.args_from
            args_to = script.args_to
            filename = script.filename

            module = cache.get(filename, None)
            if module is None:
                module = script_loading.load_module(script.filename)
                cache[filename] = module

            for script_class in module.__dict__.values():
                if type(script_class) == type and issubclass(script_class, Script):
                    self.scripts[si] = script_class()
                    self.scripts[si].filename = filename
                    self.scripts[si].args_from = args_from
                    self.scripts[si].args_to = args_to

    def before_hr(self, p):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.before_hr(p, *script_args)
            except Exception:
                errors.report(f"Error running before_hr: {script.filename}", exc_info=True)

    def setup_scrips(self, p, *, is_ui=True):
        # print("p.script_args", p.script_args)
        
        for script in self.alwayson_scripts:
            if not is_ui and script.setup_for_ui_only:
                #이즈 유야이 펄스 <refiner.py.ScriptRefiner object at 0x7ff67b6ca1d0>
                #이즈 유야이 펄스 <seed.py.ScriptSeed object at 0x7ff67b6ca020>/
                print("@@@@@@@@@@@ not_ui")
                print("script", script)
                print("!!!", p.script_args[script.args_from:script.args_to])
                continue

            try:
                """
                여기가 오니!?!? <extra_options_section.py.ExtraOptionsSection object at 0x7f3c88685b10>
                argslice 10 10
                여기가 오니!?!? <hypertile_script.py.ScriptHypertile object at 0x7f3c88685a20>
                argslice 10 10
                여기가 오니!?!? AfterDetailerScript(version=24.1.1)
                argslice 10 14
                여기가 오니!?!? <main.py.Script object at 0x7f3c88685a50>
                argslice 14 14
                여기가 오니!?!? <sd_dynamic_prompts.dynamic_prompting.Script object at 0x7f3c88685960>
                argslice 14 32
                여기가 오니!?!? <task_scheduler.py.Script object at 0x7f3c88685900>
                argslice 32 32
                여기가 오니!?!? <sd-webui-ar.py.AspectRatioScript object at 0x7f3c88685810>
                argslice 32 32
                여기가 오니!?!? <sam.py.Script object at 0x7f3c88719e10>
                argslice 73 97
                여기가 오니!?!? <lora_block_weight.py.Script object at 0x7f3c88719ea0>
                argslice 35 53
                여기가 오니!?!? <rp.py.Script object at 0x7f3c8871ad40>
                argslice 53 73
                여기가 오니!?!? <controlnet.py.Script object at 0x7f3c88685750>
                argslice 32 35
                이즈 유야이 펄스 <refiner.py.ScriptRefiner object at 0x7f3c88685f30>
                이즈 유야이 펄스 <seed.py.ScriptSeed object at 0x7f3c88685690>
                """
                script_args = p.script_args[script.args_from:script.args_to]
                print("@@@@@@@@@@@ is_ui")
                print("script", script)
                print("!!!", p.script_args[script.args_from:script.args_to])
                script.setup(p, *script_args)
            except Exception:
                errors.report(f"Error running setup: {script.filename}", exc_info=True)


scripts_txt2img: ScriptRunner = None
scripts_img2img: ScriptRunner = None
scripts_postproc: scripts_postprocessing.ScriptPostprocessingRunner = None
scripts_current: ScriptRunner = None


def reload_script_body_only():
    cache = {}
    scripts_txt2img.reload_sources(cache)
    scripts_img2img.reload_sources(cache)


reload_scripts = load_scripts  # compatibility alias
