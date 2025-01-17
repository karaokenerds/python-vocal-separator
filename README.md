# Audio Separator 🎶

[![PyPI version](https://badge.fury.io/py/audio-separator.svg)](https://badge.fury.io/py/audio-separator)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/audio-separator.svg)](https://anaconda.org/conda-forge/audio-separator)
[![Docker pulls](https://img.shields.io/docker/pulls/beveradb/audio-separator.svg)](https://hub.docker.com/r/beveradb/audio-separator/tags)
[![codecov](https://codecov.io/gh/karaokenerds/python-audio-separator/graph/badge.svg?token=N7YK4ET5JP)](https://codecov.io/gh/karaokenerds/python-audio-separator)
[![Open In Colab](https://colab.research.google.com/drive/1gSlmSmna7f7fH6OjsiMEDLl-aJ9kGPkY?usp=sharing)
[![Open In Huggingface](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/theneos/audio-separator)

**Summary:** Easy to use audio stem separation from the command line or as a dependency in your own Python project, using the amazing MDX-Net, VR Arch, Demucs and MDXC models available in UVR by @Anjok07 & @aufr33.

Audio Separator is a Python package that allows you to separate an audio file into various stems, using models trained by @Anjok07 for use with [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui).

The simplest (and probably most used) use case for this package is to separate an audio file into two stems, Instrumental and Vocals, which can be very useful for producing karaoke videos! However, the models available in UVR can separate audio into many more stems, such as Drums, Bass, Piano, and Guitar, and perform other audio processing tasks, such as denoising or removing echo/reverb.

## Features

- Separate audio into multiple stems, e.g. instrumental and vocals.
- Supports all common audio formats (WAV, MP3, FLAC, M4A, etc.)
- Ability to inference using a pre-trained model in PTH or ONNX format.
- CLI support for easy use in scripts and batch processing.
- Python API for integration into other projects.

## Installation 🛠️

### 🐳 Docker

If you're able to use docker, you don't actually need to _install_ anything - there are [images published on Docker Hub](https://hub.docker.com/r/beveradb/audio-separator/tags) for GPU (CUDA) and CPU inferencing, for both `amd64` and `arm64` platforms.

You probably want to volume-mount a folder containing whatever file you want to separate, which can then also be used as the output folder.

For instance, if your current directory has the file `input.wav`, you could execute `audio-separator` as shown below (see [usage](#usage-) section for more details):

```sh
docker run -it -v `pwd`:/workdir beveradb/audio-separator input.wav
```

If you're using a machine with a GPU, you'll want to use the GPU specific image and pass in the GPU device to the container, like this:

```sh
docker run -it --gpus all -v `pwd`:/workdir beveradb/audio-separator:gpu input.wav
```

If the GPU isn't being detected, make sure your docker runtime environment is passing through the GPU correctly - there are [various guides](https://www.celantur.com/blog/run-cuda-in-docker-on-linux/) online to help with that.

### 🎮 Nvidia GPU with CUDA or 🧪 Google Colab

**Supported CUDA Versions:** 11.8 and 12.2

💬 If successfully configured, you should see this log message when running `audio-separator --env_info`:
 `ONNXruntime has CUDAExecutionProvider available, enabling acceleration`

Conda:
```sh
conda install pytorch=*=*cuda* onnxruntime=*=*cuda* audio-separator -c pytorch -c conda-forge
```

Pip:
```sh
pip install "audio-separator[gpu]"
```

Docker:
```sh
beveradb/audio-separator:gpu
```

###  Apple Silicon, macOS Sonoma+ with M1 or newer CPU (CoreML acceleration)

💬 If successfully configured, you should see this log message when running `audio-separator --env_info`:
 `ONNXruntime has CoreMLExecutionProvider available, enabling acceleration`

Pip:
```sh
pip install "audio-separator[cpu]"
```

### 🐢 No hardware acceleration, CPU only

Conda:
```sh
conda install audio-separator-c pytorch -c conda-forge
```

Pip:
```sh
pip install "audio-separator[cpu]"
```

Docker:
```sh
beveradb/audio-separator
```

### 🎥 FFmpeg dependency

💬 To test if `audio-separator` has been successfully configured to use FFmpeg, run `audio-separator --env_info`. The log will show `FFmpeg installed`.

If you installed `audio-separator` using `conda` or `docker`, FFmpeg should already be available in your environment.

You may need to separately install FFmpeg. It should be easy to install on most platforms, e.g.:

🐧 Debian/Ubuntu:
```sh
apt-get update; apt-get install -y ffmpeg
```

 macOS:
```sh
brew update; brew install ffmpeg
```

## GPU / CUDA specific installation steps with Pip

In theory, all you should need to do to get `audio-separator` working with a GPU is install it with the `[gpu]` extra as above.

However, sometimes getting both PyTorch and ONNX Runtime working with CUDA support can be a bit tricky so it may not work that easily.

You may need to reinstall both packages directly, allowing pip to calculate the right versions for your platform, for example:

- `pip uninstall torch onnxruntime`
- `pip cache purge`
- `pip install --force-reinstall torch torchvision torchaudio`
- `pip install --force-reinstall onnxruntime-gpu`

I generally recommend installing the latest version of PyTorch for your environment using the command recommended by the wizard here:
<https://pytorch.org/get-started/locally/>

### Multiple CUDA library versions may be needed

Depending on your CUDA version and environment, you may need to install specific version(s) of CUDA libraries for ONNX Runtime to use your GPU.

🧪 Google Colab, for example, now uses CUDA 12 by default, but ONNX Runtime still needs CUDA 11 libraries to work.

If you see the error `Failed to load library` or `cannot open shared object file` when you run `audio-separator`, this is likely the issue.

You can install the CUDA 11 libraries _alongside_ CUDA 12 like so:
```sh
apt update; apt install nvidia-cuda-toolkit
```

If you encounter the following messages when running on Google Colab or in another environment:
```
[E:onnxruntime:Default, provider_bridge_ort.cc:1862 TryGetProviderInfo_CUDA] /onnxruntime_src/onnxruntime/core/session/provider_bridge_ort.cc:1539 onnxruntime::Provider& onnxruntime::ProviderLibrary::Get() [ONNXRuntimeError] : 1 : FAIL : Failed to load library libonnxruntime_providers_cuda.so with error: libcudnn_adv.so.9: cannot open shared object file: No such file or directory

[W:onnxruntime:Default, onnxruntime_pybind_state.cc:993 CreateExecutionProviderInstance] Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 12.*. Please install all dependencies as mentioned in the GPU requirements page (https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements), make sure they're in the PATH, and that your GPU is supported.
```
You can resolve this by running the following command:
```sh
python -m pip install ort-nightly-gpu --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-12-nightly/pypi/simple/
```

> Note: if anyone knows how to make this cleaner so we can support both different platform-specific dependencies for hardware acceleration without a separate installation process for each, please let me know or raise a PR!

## Usage 🚀

### Command Line Interface (CLI)

You can use Audio Separator via the command line, for example:

```sh
audio-separator /path/to/your/input/audio.wav --model_filename UVR-MDX-NET-Inst_HQ_3.onnx
```

This command will download the specified model file, process the `audio.wav` input audio and generate two new files in the current directory, one containing vocals and one containing instrumental.

**Note:** You do not need to download any files yourself - audio-separator does that automatically for you!

To see a list of supported models, run `audio-separator --list_models`

Any file listed in the list models output can be specified (with file extension) with the model_filename parameter (e.g. `--model_filename UVR_MDXNET_KARA_2.onnx`) and it will be automatically downloaded to the `--model_file_dir` (default: `/tmp/audio-separator-models/`) folder on first usage.

### Listing and Filtering Available Models

You can view all available models using the `--list_models` (or `-l`) flag:

```sh
audio-separator --list_models
```

The output shows a table with the following columns:
- Model Filename: The filename to use with `--model_filename`
- Arch: The model architecture (MDX, MDXC, Demucs, etc.)
- Output Stems (SDR): The stems this model can separate, with Signal-to-Distortion Ratio scores where available
- Friendly Name: A human-readable name describing the model

#### Filtering Models

You can filter and sort the model list by stem type using `--list_filter`. For example, to find models that can separate drums:

```sh
audio-separator -l --list_filter=drums
```

Example output:
```
-----------------------------------------------------------------------------------------------------------------------------------
Model Filename        Arch    Output Stems (SDR)                                            Friendly Name
-----------------------------------------------------------------------------------------------------------------------------------
htdemucs_ft.yaml      Demucs  vocals (10.8), drums (10.1), bass (11.9), other               Demucs v4: htdemucs_ft
hdemucs_mmi.yaml      Demucs  vocals (10.3), drums (9.7), bass (12.0), other                Demucs v4: hdemucs_mmi
htdemucs.yaml         Demucs  vocals (10.0), drums (9.4), bass (11.3), other                Demucs v4: htdemucs
htdemucs_6s.yaml      Demucs  vocals (9.7), drums (8.5), bass (10.0), guitar, piano, other  Demucs v4: htdemucs_6s
```

#### Limiting Results

You can limit the number of results shown using `--list_limit`. This is useful for finding the best performing models for a particular stem. For example, to see the top 5 vocal separation models:

```sh
audio-separator -l --list_filter=vocals --list_limit=5
```

Example output:
```
--------------------------------------------------------------------------------------------------------------------------------------------------------------
Model Filename                             Arch  Output Stems (SDR)                   Friendly Name
--------------------------------------------------------------------------------------------------------------------------------------------------------------
model_bs_roformer_ep_317_sdr_12.9755.ckpt  MDXC  vocals* (12.9), instrumental (17.0)  Roformer Model: BS-Roformer-Viperx-1297
model_bs_roformer_ep_368_sdr_12.9628.ckpt  MDXC  vocals* (12.9), instrumental (17.0)  Roformer Model: BS-Roformer-Viperx-1296
vocals_mel_band_roformer.ckpt              MDXC  vocals* (12.6), other                Roformer Model: MelBand Roformer | Vocals by Kimberley Jensen
melband_roformer_big_beta4.ckpt            MDXC  vocals* (12.5), other                Roformer Model: MelBand Roformer Kim | Big Beta 4 FT by unwa
mel_band_roformer_kim_ft_unwa.ckpt         MDXC  vocals* (12.4), other                Roformer Model: MelBand Roformer Kim | FT by unwa
```

#### JSON Output

For programmatic use, you can output the model list in JSON format:

```sh
audio-separator -l --list_format=json
```

### Full command-line interface options

```sh
usage: audio-separator [-h] [-v] [-d] [-e] [-l] [--log_level LOG_LEVEL] [--list_filter LIST_FILTER] [--list_limit LIST_LIMIT] [--list_format {pretty,json}] [-m MODEL_FILENAME] [--output_format OUTPUT_FORMAT]
                       [--output_bitrate OUTPUT_BITRATE] [--output_dir OUTPUT_DIR] [--model_file_dir MODEL_FILE_DIR] [--download_model_only] [--invert_spect] [--normalization NORMALIZATION]
                       [--amplification AMPLIFICATION] [--single_stem SINGLE_STEM] [--sample_rate SAMPLE_RATE] [--use_soundfile] [--use_autocast] [--custom_output_names CUSTOM_OUTPUT_NAMES]
                       [--mdx_segment_size MDX_SEGMENT_SIZE] [--mdx_overlap MDX_OVERLAP] [--mdx_batch_size MDX_BATCH_SIZE] [--mdx_hop_length MDX_HOP_LENGTH] [--mdx_enable_denoise] [--vr_batch_size VR_BATCH_SIZE]
                       [--vr_window_size VR_WINDOW_SIZE] [--vr_aggression VR_AGGRESSION] [--vr_enable_tta] [--vr_high_end_process] [--vr_enable_post_process]
                       [--vr_post_process_threshold VR_POST_PROCESS_THRESHOLD] [--demucs_segment_size DEMUCS_SEGMENT_SIZE] [--demucs_shifts DEMUCS_SHIFTS] [--demucs_overlap DEMUCS_OVERLAP]
                       [--demucs_segments_enabled DEMUCS_SEGMENTS_ENABLED] [--mdxc_segment_size MDXC_SEGMENT_SIZE] [--mdxc_override_model_segment_size] [--mdxc_overlap MDXC_OVERLAP]
                       [--mdxc_batch_size MDXC_BATCH_SIZE] [--mdxc_pitch_shift MDXC_PITCH_SHIFT]
                       [audio_files ...]

Separate audio file into different stems.

positional arguments:
  audio_files                                            The audio file paths to separate, in any common format.

options:
  -h, --help                                             show this help message and exit

Info and Debugging:
  -v, --version                                          Show the program's version number and exit.
  -d, --debug                                            Enable debug logging, equivalent to --log_level=debug.
  -e, --env_info                                         Print environment information and exit.
  -l, --list_models                                      List all supported models and exit. Use --list_filter to filter/sort the list and --list_limit to show only top N results.
  --log_level LOG_LEVEL                                  Log level, e.g. info, debug, warning (default: info).
  --list_filter LIST_FILTER                              Filter and sort the model list by 'name', 'filename', or any stem e.g. vocals, instrumental, drums
  --list_limit LIST_LIMIT                                Limit the number of models shown
  --list_format {pretty,json}                            Format for listing models: 'pretty' for formatted output, 'json' for raw JSON dump

Separation I/O Params:
  -m MODEL_FILENAME, --model_filename MODEL_FILENAME     Model to use for separation (default: model_bs_roformer_ep_317_sdr_12.9755.yaml). Example: -m 2_HP-UVR.pth
  --output_format OUTPUT_FORMAT                          Output format for separated files, any common format (default: FLAC). Example: --output_format=MP3
  --output_bitrate OUTPUT_BITRATE                        Output bitrate for separated files, any ffmpeg-compatible bitrate (default: None). Example: --output_bitrate=320k
  --output_dir OUTPUT_DIR                                Directory to write output files (default: <current dir>). Example: --output_dir=/app/separated
  --model_file_dir MODEL_FILE_DIR                        Model files directory (default: /tmp/audio-separator-models/). Example: --model_file_dir=/app/models
  --download_model_only                                  Download a single model file only, without performing separation.

Common Separation Parameters:
  --invert_spect                                         Invert secondary stem using spectrogram (default: False). Example: --invert_spect
  --normalization NORMALIZATION                          Max peak amplitude to normalize input and output audio to (default: 0.9). Example: --normalization=0.7
  --amplification AMPLIFICATION                          Min peak amplitude to amplify input and output audio to (default: 0.0). Example: --amplification=0.4
  --single_stem SINGLE_STEM                              Output only single stem, e.g. Instrumental, Vocals, Drums, Bass, Guitar, Piano, Other. Example: --single_stem=Instrumental
  --sample_rate SAMPLE_RATE                              Modify the sample rate of the output audio (default: 44100). Example: --sample_rate=44100
  --use_soundfile                                        Use soundfile to write audio output (default: False). Example: --use_soundfile
  --use_autocast                                         Use PyTorch autocast for faster inference (default: False). Do not use for CPU inference. Example: --use_autocast
  --custom_output_names CUSTOM_OUTPUT_NAMES              Custom names for all output files in JSON format (default: None). Example: --custom_output_names='{"Vocals": "vocals_output", "Drums": "drums_output"}'

MDX Architecture Parameters:
  --mdx_segment_size MDX_SEGMENT_SIZE                    Larger consumes more resources, but may give better results (default: 256). Example: --mdx_segment_size=256
  --mdx_overlap MDX_OVERLAP                              Amount of overlap between prediction windows, 0.001-0.999. Higher is better but slower (default: 0.25). Example: --mdx_overlap=0.25
  --mdx_batch_size MDX_BATCH_SIZE                        Larger consumes more RAM but may process slightly faster (default: 1). Example: --mdx_batch_size=4
  --mdx_hop_length MDX_HOP_LENGTH                        Usually called stride in neural networks, only change if you know what you're doing (default: 1024). Example: --mdx_hop_length=1024
  --mdx_enable_denoise                                   Enable denoising during separation (default: False). Example: --mdx_enable_denoise

VR Architecture Parameters:
  --vr_batch_size VR_BATCH_SIZE                          Number of batches to process at a time. Higher = more RAM, slightly faster processing (default: 1). Example: --vr_batch_size=16
  --vr_window_size VR_WINDOW_SIZE                        Balance quality and speed. 1024 = fast but lower, 320 = slower but better quality. (default: 512). Example: --vr_window_size=320
  --vr_aggression VR_AGGRESSION                          Intensity of primary stem extraction, -100 - 100. Typically, 5 for vocals & instrumentals (default: 5). Example: --vr_aggression=2
  --vr_enable_tta                                        Enable Test-Time-Augmentation; slow but improves quality (default: False). Example: --vr_enable_tta
  --vr_high_end_process                                  Mirror the missing frequency range of the output (default: False). Example: --vr_high_end_process
  --vr_enable_post_process                               Identify leftover artifacts within vocal output; may improve separation for some songs (default: False). Example: --vr_enable_post_process
  --vr_post_process_threshold VR_POST_PROCESS_THRESHOLD  Threshold for post_process feature: 0.1-0.3 (default: 0.2). Example: --vr_post_process_threshold=0.1

Demucs Architecture Parameters:
  --demucs_segment_size DEMUCS_SEGMENT_SIZE              Size of segments into which the audio is split, 1-100. Higher = slower but better quality (default: Default). Example: --demucs_segment_size=256
  --demucs_shifts DEMUCS_SHIFTS                          Number of predictions with random shifts, higher = slower but better quality (default: 2). Example: --demucs_shifts=4
  --demucs_overlap DEMUCS_OVERLAP                        Overlap between prediction windows, 0.001-0.999. Higher = slower but better quality (default: 0.25). Example: --demucs_overlap=0.25
  --demucs_segments_enabled DEMUCS_SEGMENTS_ENABLED      Enable segment-wise processing (default: True). Example: --demucs_segments_enabled=False

MDXC Architecture Parameters:
  --mdxc_segment_size MDXC_SEGMENT_SIZE                  Larger consumes more resources, but may give better results (default: 256). Example: --mdxc_segment_size=256
  --mdxc_override_model_segment_size                     Override model default segment size instead of using the model default value. Example: --mdxc_override_model_segment_size
  --mdxc_overlap MDXC_OVERLAP                            Amount of overlap between prediction windows, 2-50. Higher is better but slower (default: 8). Example: --mdxc_overlap=8
  --mdxc_batch_size MDXC_BATCH_SIZE                      Larger consumes more RAM but may process slightly faster (default: 1). Example: --mdxc_batch_size=4
  --mdxc_pitch_shift MDXC_PITCH_SHIFT                    Shift audio pitch by a number of semitones while processing. May improve output for deep/high vocals. (default: 0). Example: --mdxc_pitch_shift=2
```

### As a Dependency in a Python Project

You can use Audio Separator in your own Python project. Here's a minimal example using the default two stem (Instrumental and Vocals) model:

```python
from audio_separator.separator import Separator

# Initialize the Separator class (with optional configuration properties, below)
separator = Separator()

# Load a machine learning model (if unspecified, defaults to 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt')
separator.load_model()

# Perform the separation on specific audio files without reloading the model
output_files = separator.separate('audio1.wav')

print(f"Separation complete! Output file(s): {' '.join(output_files)}")
```

#### Batch processing and processing with multiple models

You can process multiple files without reloading the model to save time and memory.

You only need to load a model when choosing or changing models. See example below:

```python
from audio_separator.separator import Separator

# Initialize the Separator with other configuration properties, below
separator = Separator()

# Load a model
separator.load_model(model_filename='UVR-MDX-NET-Inst_HQ_3.onnx')

# Separate multiple audio files without reloading the model
output_file_paths_1 = separator.separate('audio1.wav')
output_file_paths_2 = separator.separate('audio2.wav')
output_file_paths_3 = separator.separate('audio3.wav')

# Load a different model
separator.load_model(model_filename='UVR_MDXNET_KARA_2.onnx')

# Separate the same files with the new model
output_file_paths_4 = separator.separate('audio1.wav')
output_file_paths_5 = separator.separate('audio2.wav')
output_file_paths_6 = separator.separate('audio3.wav')
```

#### Renaming Stems

You can rename the output files by specifying the desired names. For example:
```python
output_names = {
    "Vocals": "vocals_output",
    "Instrumental": "instrumental_output",
}
output_files = separator.separate('audio1.wav', output_names)
```
In this case, the output file names will be: `vocals_output.wav` and `instrumental_output.wav`.

You can also rename specific stems:

- To rename the Vocals stem:
  ```python
  output_names = {
      "Vocals": "vocals_output",
  }
  output_files = separator.separate('audio1.wav', output_names)
  ```
  > The output files will be named: `vocals_output.wav` and `audio1_(Instrumental)_model_mel_band_roformer_ep_3005_sdr_11.wav`
- To rename the Instrumental stem:
  ```python
  output_names = {
      "Instrumental": "instrumental_output",
  }
  output_files = separator.separate('audio1.wav', output_names)
  ```
  > The output files will be named: `audio1_(Vocals)_model_mel_band_roformer_ep_3005_sdr_11.wav` and `instrumental_output.wav`
- List of stems for Demucs models:
  - htdemucs_6s.yaml
    ```python
    output_names = {
        "Vocals": "vocals_output",
        "Drums": "drums_output",
        "Bass": "bass_output",
        "Other": "other_output",
        "Guitar": "guitar_output",
        "Piano": "piano_output",
    }
    ```
  - Other Demucs models
    ```python
    output_names = {
        "Vocals": "vocals_output",
        "Drums": "drums_output",
        "Bass": "bass_output",
        "Other": "other_output",
    }
    ```

## Parameters for the Separator class

- **`log_level`:** (Optional) Logging level, e.g., INFO, DEBUG, WARNING. `Default: logging.INFO`
- **`log_formatter`:** (Optional) The log format. Default: None, which falls back to '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
- **`model_file_dir`:** (Optional) Directory to cache model files in. `Default: /tmp/audio-separator-models/`
- **`output_dir`:** (Optional) Directory where the separated files will be saved. If not specified, uses the current directory.
- **`output_format`:** (Optional) Format to encode output files, any common format (WAV, MP3, FLAC, M4A, etc.). `Default: WAV`
- **`normalization_threshold`:** (Optional) The amount by which the amplitude of the output audio will be multiplied. `Default: 0.9`
- **`amplification_threshold`:** (Optional) The minimum amplitude level at which the waveform will be amplified. If the peak amplitude of the audio is below this threshold, the waveform will be scaled up to meet it. `Default: 0.0`
- **`output_single_stem`:** (Optional) Output only a single stem, such as 'Instrumental' and 'Vocals'. `Default: None`
- **`invert_using_spec`:** (Optional) Flag to invert using spectrogram. `Default: False`
- **`sample_rate`:** (Optional) Set the sample rate of the output audio. `Default: 44100`
- **`use_soundfile`:** (Optional) Use soundfile for output writing, can solve OOM issues, especially on longer audio.
- **`use_autocast`:** (Optional) Flag to use PyTorch autocast for faster inference. Do not use for CPU inference. `Default: False`
- **`mdx_params`:** (Optional) MDX Architecture Specific Attributes & Defaults. `Default: {"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 1, "enable_denoise": False}`
- **`vr_params`:** (Optional) VR Architecture Specific Attributes & Defaults. `Default: {"batch_size": 1, "window_size": 512, "aggression": 5, "enable_tta": False, "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": False}`
- **`demucs_params`:** (Optional) Demucs Architecture Specific Attributes & Defaults. `Default: {"segment_size": "Default", "shifts": 2, "overlap": 0.25, "segments_enabled": True}`
- **`mdxc_params`:** (Optional) MDXC Architecture Specific Attributes & Defaults. `Default: {"segment_size": 256, "override_model_segment_size": False, "batch_size": 1, "overlap": 8, "pitch_shift": 0}`

## Requirements 📋

Python >= 3.10

Libraries: torch, onnx, onnxruntime, numpy, librosa, requests, six, tqdm, pydub

## Developing Locally

This project uses Poetry for dependency management and packaging. Follow these steps to setup a local development environment:

### Prerequisites

- Make sure you have Python 3.10 or newer installed on your machine.
- Install Conda (I recommend Miniforge: [Miniforge GitHub](https://github.com/conda-forge/miniforge)) to manage your Python virtual environments

### Clone the Repository

Clone the repository to your local machine:

```sh
git clone https://github.com/YOUR_USERNAME/audio-separator.git
cd audio-separator
```

Replace `YOUR_USERNAME` with your GitHub username if you've forked the repository, or use the main repository URL if you have the permissions.

### Create and activate the Conda Environment

To create and activate the conda environment, use the following commands:

```sh
conda env create
conda activate audio-separator-dev
```

### Install Dependencies

Once you're inside the conda env, run the following command to install the project dependencies:

```sh
poetry install
```

Install extra dependencies depending if you're running with GPU or CPU.
```sh
poetry install --extras "cpu"
```
or
```sh
poetry install --extras "gpu"
```

### Running the Command-Line Interface Locally

You can run the CLI command directly within the virtual environment. For example:

```sh
audio-separator path/to/your/audio-file.wav
```

### Deactivate the Virtual Environment

Once you are done with your development work, you can exit the virtual environment by simply typing:

```sh
conda deactivate
```

### Building the Package

To build the package for distribution, use the following command:

```sh
poetry build
```

This will generate the distribution packages in the dist directory - but for now only @beveradb will be able to publish to PyPI.


## How to Use in Colab

1. **Link Input**:

![step 1](https://github.com/user-attachments/assets/edb41e74-2082-43d8-9dde-30cc4eee3423)


   - **video_url**: This input is where you paste the URL of the audio or video you want to download. It can be from various platforms supported by yt-dlp. For a full list of supported websites, refer to [this link](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md).

   - Example: 
     ``` 
     https://www.youtube.com/watch?v=exampleID 
     ```

2. **Input Audio File for Separation**:

![2 and 3](https://github.com/user-attachments/assets/a040a17f-dad1-447a-afef-39fbbe59e556)


   - **input**: This is the file path of the audio you want to separate. After downloading the audio file, you will need to specify this path to continue with separation.

   - Example:
     ``` 
     /content/ytdl/your_downloaded_audio.wav 
     ```

3. **Output Directory**:
   - **output**: This is the path where the separated files will be saved. It defaults to `/content/output` but can be changed to another directory if desired.

   - Example:
     ``` 
     /content/custom_output 
     ```

## Contributing 🤝

Contributions are very much welcome! Please fork the repository and submit a pull request with your changes, and I'll try to review, merge and publish promptly!

- This project is 100% open-source and free for anyone to use and modify as they wish.
- If the maintenance workload for this repo somehow becomes too much for me I'll ask for volunteers to share maintainership of the repo, though I don't think that is very likely
- Development and support for the MDX-Net separation models is part of the main [UVR project](https://github.com/Anjok07/ultimatevocalremovergui), this repo is just a CLI/Python package wrapper to simplify running those models programmatically. So, if you want to try and improve the actual models, please get involved in the UVR project and look for guidance there!

## License 📄

This project is licensed under the MIT [License](LICENSE).

- **Please Note:** If you choose to integrate this project into some other project using the default model or any other model trained as part of the [UVR](https://github.com/Anjok07/ultimatevocalremovergui) project, please honor the MIT license by providing credit to UVR and its developers!

## Credits 🙏

- [Anjok07](https://github.com/Anjok07) - Author of [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui), which almost all of the code in this repo was copied from! Definitely deserving of credit for anything good from this project. Thank you!
- [DilanBoskan](https://github.com/DilanBoskan) - Your contributions at the start of this project were essential to the success of UVR. Thank you!
- [Kuielab & Woosung Choi](https://github.com/kuielab) - Developed the original MDX-Net AI code.
- [KimberleyJSN](https://github.com/KimberleyJensen) - Advised and aided the implementation of the training scripts for MDX-Net and Demucs. Thank you!
- [Hv](https://github.com/NaJeongMo/Colab-for-MDX_B) - Helped implement chunks into the MDX-Net AI code. Thank you!
- [zhzhongshi](https://github.com/zhzhongshi) - Helped add support for the MDXC models in `audio-separator`. Thank you!

## Contact 💌

For questions or feedback, please raise an issue or reach out to @beveradb ([Andrew Beveridge](mailto:andrew@beveridge.uk)) directly.

## Sponsors

<!-- sponsors --><!-- sponsors -->
