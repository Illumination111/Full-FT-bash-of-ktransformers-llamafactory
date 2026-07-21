# Conda environment dependency versions

Captured from `sapphire4` at `2026-07-21T06:44:26Z`. This is an installed-state inventory generated from `conda list --json`; it includes Conda packages and packages recorded from PyPI, including transitive dependencies.

## Environment summary

| Environment | Prefix | Packages | Python | pip |
| --- | --- | ---: | --- | --- |
| Kllama | `/mnt/data2/wbw/conda/envs/Kllama` | 204 | 3.12.13 | 26.1.2 |
| Deepspeed | `/mnt/data2/wbw/conda/envs/Deepspeed` | 198 | 3.12.13 | 26.1.2 |

## Key ML stack

| Package | Kllama | Deepspeed |
| --- | --- | --- |
| python | 3.12.13 | 3.12.13 |
| pip | 26.1.2 | 26.1.2 |
| torch | 2.9.1 | 2.9.1 |
| torchvision | 0.24.1+cu128 | 0.24.1+cu128 |
| torchaudio | 2.9.1+cu128 | 2.9.1+cu128 |
| triton | 3.5.1 | 3.5.1 |
| deepspeed | 0.19.2 | 0.19.2 |
| transformers | 5.6.0 | 5.6.0 |
| accelerate | 1.11.0 | 1.11.0 |
| accelerate-kt | 1.14.0.post1 | — |
| ktransformers | 0.6.3 | — |
| kt-kernel | 0.6.3.post1 | — |
| flash-attn | 2.8.3 | 2.8.3 |
| numpy | 2.5.1 | 2.5.1 |
| scipy | 1.18.0 | 1.18.0 |
| datasets | 4.0.0 | 4.0.0 |
| peft | 0.18.1 | 0.18.1 |
| trl | 0.24.0 | 0.24.0 |

## Kllama

Prefix: `/mnt/data2/wbw/conda/envs/Kllama`
Installed package records: **204**

| Package | Version | Build string | Build number | Channel | Platform |
| --- | --- | --- | ---: | --- | --- |
| _libgcc_mutex | 0.1 | main | 0 | pkgs/main | linux-64 |
| _openmp_mutex | 5.1 | 52_gnu | 52 | pkgs/main | linux-64 |
| accelerate | 1.11.0 | pypi_0 | 0 | pypi | pypi |
| accelerate-kt | 1.14.0.post1 | pypi_0 | 0 | pypi | pypi |
| aiofiles | 24.1.0 | pypi_0 | 0 | pypi | pypi |
| aiohappyeyeballs | 2.7.1 | pypi_0 | 0 | pypi | pypi |
| aiohttp | 3.14.1 | pypi_0 | 0 | pypi | pypi |
| aiosignal | 1.4.0 | pypi_0 | 0 | pypi | pypi |
| annotated-doc | 0.0.4 | pypi_0 | 0 | pypi | pypi |
| annotated-types | 0.7.0 | pypi_0 | 0 | pypi | pypi |
| antlr4-python3-runtime | 4.9.3 | pypi_0 | 0 | pypi | pypi |
| anyio | 4.14.1 | pypi_0 | 0 | pypi | pypi |
| attrs | 26.1.0 | pypi_0 | 0 | pypi | pypi |
| av | 16.0.0 | pypi_0 | 0 | pypi | pypi |
| binutils_impl_linux-64 | 2.44 | h78f17ca_3 | 3 | pkgs/main | linux-64 |
| brotli | 1.2.0 | pypi_0 | 0 | pypi | pypi |
| bzip2 | 1.0.8 | h5eee18b_6 | 6 | pkgs/main | linux-64 |
| ca-certificates | 2026.6.17 | hbd8a1cb_0 | 0 | conda-forge | noarch |
| certifi | 2026.6.17 | pypi_0 | 0 | pypi | pypi |
| charset-normalizer | 3.4.7 | pypi_0 | 0 | pypi | pypi |
| click | 8.4.2 | pypi_0 | 0 | pypi | pypi |
| contourpy | 1.3.3 | pypi_0 | 0 | pypi | pypi |
| cuda-bindings | 13.3.1 | pypi_0 | 0 | pypi | pypi |
| cuda-cudart | 11.8.89 | 0 | 0 | nvidia/label/cuda-11.8.0 | linux-64 |
| cuda-libraries | 11.8.0 | 0 | 0 | nvidia/label/cuda-11.8.0 | linux-64 |
| cuda-nvrtc | 11.8.89 | 0 | 0 | nvidia/label/cuda-11.8.0 | linux-64 |
| cuda-pathfinder | 1.5.6 | pypi_0 | 0 | pypi | pypi |
| cuda-runtime | 11.8.0 | 0 | 0 | nvidia/label/cuda-11.8.0 | linux-64 |
| cuda-toolkit | 13.0.2 | pypi_0 | 0 | pypi | pypi |
| cycler | 0.12.1 | pypi_0 | 0 | pypi | pypi |
| datasets | 4.0.0 | pypi_0 | 0 | pypi | pypi |
| deepspeed | 0.19.2 | pypi_0 | 0 | pypi | pypi |
| dill | 0.3.8 | pypi_0 | 0 | pypi | pypi |
| docstring-parser | 0.18.0 | pypi_0 | 0 | pypi | pypi |
| einops | 0.8.2 | pypi_0 | 0 | pypi | pypi |
| fastapi | 0.139.0 | pypi_0 | 0 | pypi | pypi |
| ffmpy | 1.0.0 | pypi_0 | 0 | pypi | pypi |
| filelock | 3.29.5 | pypi_0 | 0 | pypi | pypi |
| fire | 0.7.1 | pypi_0 | 0 | pypi | pypi |
| flash-attn | 2.8.3 | pypi_0 | 0 | pypi | pypi |
| fonttools | 4.63.0 | pypi_0 | 0 | pypi | pypi |
| frozenlist | 1.8.0 | pypi_0 | 0 | pypi | pypi |
| fsspec | 2025.3.0 | pypi_0 | 0 | pypi | pypi |
| gcc_impl_linux-64 | 15.2.0 | hcacfade_7 | 7 | conda-forge | linux-64 |
| gguf | 0.19.0 | pypi_0 | 0 | pypi | pypi |
| gradio | 5.50.0 | pypi_0 | 0 | pypi | pypi |
| gradio-client | 1.14.0 | pypi_0 | 0 | pypi | pypi |
| groovy | 0.1.2 | pypi_0 | 0 | pypi | pypi |
| h11 | 0.16.0 | pypi_0 | 0 | pypi | pypi |
| hf-transfer | 0.1.9 | pypi_0 | 0 | pypi | pypi |
| hf-xet | 1.5.1 | pypi_0 | 0 | pypi | pypi |
| hjson | 3.1.0 | pypi_0 | 0 | pypi | pypi |
| httpcore | 1.0.9 | pypi_0 | 0 | pypi | pypi |
| httpx | 0.28.1 | pypi_0 | 0 | pypi | pypi |
| huggingface-hub | 1.22.0 | pypi_0 | 0 | pypi | pypi |
| idna | 3.18 | pypi_0 | 0 | pypi | pypi |
| iniconfig | 2.3.0 | pypi_0 | 0 | pypi | pypi |
| jinja2 | 3.1.6 | pypi_0 | 0 | pypi | pypi |
| kernel-headers_linux-64 | 5.14.0 | he073ed8_3 | 3 | conda-forge | noarch |
| kiwisolver | 1.5.0 | pypi_0 | 0 | pypi | pypi |
| kt-kernel | 0.6.3.post1 | pypi_0 | 0 | pypi | pypi |
| ktransformers | 0.6.3 | pypi_0 | 0 | pypi | pypi |
| ld_impl_linux-64 | 2.44 | h9e0c5a2_3 | 3 | pkgs/main | linux-64 |
| libcublas | 11.11.3.6 | 0 | 0 | nvidia/label/cuda-11.8.0 | linux-64 |
| libcufft | 10.9.0.58 | 0 | 0 | nvidia/label/cuda-11.8.0 | linux-64 |
| libcufile | 1.4.0.31 | 0 | 0 | nvidia/label/cuda-11.8.0 | linux-64 |
| libcurand | 10.3.0.86 | 0 | 0 | nvidia/label/cuda-11.8.0 | linux-64 |
| libcusolver | 11.4.1.48 | 0 | 0 | nvidia/label/cuda-11.8.0 | linux-64 |
| libcusparse | 11.7.5.86 | 0 | 0 | nvidia/label/cuda-11.8.0 | linux-64 |
| libexpat | 2.8.2 | h7354ed3_1 | 1 | pkgs/main | linux-64 |
| libffi | 3.4.8 | h06d3fd0_3 | 3 | pkgs/main | linux-64 |
| libgcc | 15.2.0 | h69a1729_8 | 8 | pkgs/main | linux-64 |
| libgcc-devel_linux-64 | 15.2.0 | h73f6952_107 | 107 | conda-forge | noarch |
| libgcc-ng | 15.2.0 | h166f726_8 | 8 | pkgs/main | linux-64 |
| libgomp | 15.2.0 | h4751f2c_8 | 8 | pkgs/main | linux-64 |
| libnpp | 11.8.0.86 | 0 | 0 | nvidia/label/cuda-11.8.0 | linux-64 |
| libnvjpeg | 11.9.0.86 | 0 | 0 | nvidia/label/cuda-11.8.0 | linux-64 |
| libsanitizer | 15.2.0 | hb13aed2_7 | 7 | conda-forge | linux-64 |
| libstdcxx | 15.2.0 | h39759b7_8 | 8 | pkgs/main | linux-64 |
| libstdcxx-ng | 15.2.0 | hc03a8fd_8 | 8 | pkgs/main | linux-64 |
| libuuid | 1.41.5 | h5eee18b_0 | 0 | pkgs/main | linux-64 |
| libxcb | 1.17.0 | h9b100fa_0 | 0 | pkgs/main | linux-64 |
| libzlib | 1.3.2 | h47b2149_0 | 0 | pkgs/main | linux-64 |
| llamafactory | 0.9.6.dev0 | pypi_0 | 0 | pypi | pypi |
| markdown-it-py | 4.2.0 | pypi_0 | 0 | pypi | pypi |
| markupsafe | 3.0.3 | pypi_0 | 0 | pypi | pypi |
| matplotlib | 3.11.0 | pypi_0 | 0 | pypi | pypi |
| mdurl | 0.1.2 | pypi_0 | 0 | pypi | pypi |
| modelscope | 1.38.0 | pypi_0 | 0 | pypi | pypi |
| modelscope-hub | 0.1.6 | pypi_0 | 0 | pypi | pypi |
| mpmath | 1.3.0 | pypi_0 | 0 | pypi | pypi |
| msgpack | 1.2.1 | pypi_0 | 0 | pypi | pypi |
| multidict | 6.7.1 | pypi_0 | 0 | pypi | pypi |
| multiprocess | 0.70.16 | pypi_0 | 0 | pypi | pypi |
| ncurses | 6.5 | h7934f7d_0 | 0 | pkgs/main | linux-64 |
| networkx | 3.6.1 | pypi_0 | 0 | pypi | pypi |
| ninja | 1.13.0 | pypi_0 | 0 | pypi | pypi |
| numpy | 2.5.1 | pypi_0 | 0 | pypi | pypi |
| nvidia-cublas | 13.1.1.3 | pypi_0 | 0 | pypi | pypi |
| nvidia-cublas-cu12 | 12.8.4.1 | pypi_0 | 0 | pypi | pypi |
| nvidia-cuda-cupti | 13.0.85 | pypi_0 | 0 | pypi | pypi |
| nvidia-cuda-cupti-cu12 | 12.8.90 | pypi_0 | 0 | pypi | pypi |
| nvidia-cuda-nvrtc | 13.0.88 | pypi_0 | 0 | pypi | pypi |
| nvidia-cuda-nvrtc-cu12 | 12.8.93 | pypi_0 | 0 | pypi | pypi |
| nvidia-cuda-runtime | 13.0.96 | pypi_0 | 0 | pypi | pypi |
| nvidia-cuda-runtime-cu12 | 12.8.90 | pypi_0 | 0 | pypi | pypi |
| nvidia-cudnn-cu12 | 9.10.2.21 | pypi_0 | 0 | pypi | pypi |
| nvidia-cudnn-cu13 | 9.20.0.48 | pypi_0 | 0 | pypi | pypi |
| nvidia-cufft | 12.0.0.61 | pypi_0 | 0 | pypi | pypi |
| nvidia-cufft-cu12 | 11.3.3.83 | pypi_0 | 0 | pypi | pypi |
| nvidia-cufile | 1.15.1.6 | pypi_0 | 0 | pypi | pypi |
| nvidia-cufile-cu12 | 1.13.1.3 | pypi_0 | 0 | pypi | pypi |
| nvidia-curand | 10.4.0.35 | pypi_0 | 0 | pypi | pypi |
| nvidia-curand-cu12 | 10.3.9.90 | pypi_0 | 0 | pypi | pypi |
| nvidia-cusolver | 12.0.4.66 | pypi_0 | 0 | pypi | pypi |
| nvidia-cusolver-cu12 | 11.7.3.90 | pypi_0 | 0 | pypi | pypi |
| nvidia-cusparse | 12.6.3.3 | pypi_0 | 0 | pypi | pypi |
| nvidia-cusparse-cu12 | 12.5.8.93 | pypi_0 | 0 | pypi | pypi |
| nvidia-cusparselt-cu12 | 0.7.1 | pypi_0 | 0 | pypi | pypi |
| nvidia-cusparselt-cu13 | 0.8.1 | pypi_0 | 0 | pypi | pypi |
| nvidia-nccl-cu12 | 2.27.5 | pypi_0 | 0 | pypi | pypi |
| nvidia-nccl-cu13 | 2.29.7 | pypi_0 | 0 | pypi | pypi |
| nvidia-nvjitlink | 13.0.88 | pypi_0 | 0 | pypi | pypi |
| nvidia-nvjitlink-cu12 | 12.8.93 | pypi_0 | 0 | pypi | pypi |
| nvidia-nvshmem-cu12 | 3.3.20 | pypi_0 | 0 | pypi | pypi |
| nvidia-nvshmem-cu13 | 3.4.5 | pypi_0 | 0 | pypi | pypi |
| nvidia-nvtx | 13.0.85 | pypi_0 | 0 | pypi | pypi |
| nvidia-nvtx-cu12 | 12.8.90 | pypi_0 | 0 | pypi | pypi |
| omegaconf | 2.3.1 | pypi_0 | 0 | pypi | pypi |
| openssl | 3.6.3 | h35e630c_0 | 0 | conda-forge | linux-64 |
| orjson | 3.11.9 | pypi_0 | 0 | pypi | pypi |
| packaging | 26.0 | py312h06a4308_0 | 0 | pkgs/main | linux-64 |
| pandas | 2.3.3 | pypi_0 | 0 | pypi | pypi |
| peft | 0.18.1 | pypi_0 | 0 | pypi | pypi |
| pillow | 11.3.0 | pypi_0 | 0 | pypi | pypi |
| pip | 26.1.2 | pyhc872135_0 | 0 | pkgs/main | noarch |
| pluggy | 1.6.0 | pypi_0 | 0 | pypi | pypi |
| propcache | 0.5.2 | pypi_0 | 0 | pypi | pypi |
| protobuf | 7.35.1 | pypi_0 | 0 | pypi | pypi |
| psutil | 7.2.2 | pypi_0 | 0 | pypi | pypi |
| pthread-stubs | 0.3 | h0ce48e5_1 | 1 | pkgs/main | linux-64 |
| py-cpuinfo | 9.0.0 | pypi_0 | 0 | pypi | pypi |
| pyarrow | 24.0.0 | pypi_0 | 0 | pypi | pypi |
| pydantic | 2.12.3 | pypi_0 | 0 | pypi | pypi |
| pydantic-core | 2.41.4 | pypi_0 | 0 | pypi | pypi |
| pydub | 0.25.1 | pypi_0 | 0 | pypi | pypi |
| pygments | 2.20.0 | pypi_0 | 0 | pypi | pypi |
| pyparsing | 3.3.2 | pypi_0 | 0 | pypi | pypi |
| pytest | 9.1.1 | pypi_0 | 0 | pypi | pypi |
| python | 3.12.13 | h4d16e0c_1 | 1 | pkgs/main | linux-64 |
| python-dateutil | 2.9.0.post0 | pypi_0 | 0 | pypi | pypi |
| python-multipart | 0.0.32 | pypi_0 | 0 | pypi | pypi |
| pytz | 2026.2 | pypi_0 | 0 | pypi | pypi |
| pyyaml | 6.0.3 | pypi_0 | 0 | pypi | pypi |
| readline | 8.3 | hc2a1206_0 | 0 | pkgs/main | linux-64 |
| regex | 2026.6.28 | pypi_0 | 0 | pypi | pypi |
| requests | 2.34.2 | pypi_0 | 0 | pypi | pypi |
| rich | 15.0.0 | pypi_0 | 0 | pypi | pypi |
| ruff | 0.15.20 | pypi_0 | 0 | pypi | pypi |
| safehttpx | 0.1.7 | pypi_0 | 0 | pypi | pypi |
| safetensors | 0.8.0 | pypi_0 | 0 | pypi | pypi |
| scipy | 1.18.0 | pypi_0 | 0 | pypi | pypi |
| semantic-version | 2.10.0 | pypi_0 | 0 | pypi | pypi |
| sentencepiece | 0.2.1 | pypi_0 | 0 | pypi | pypi |
| setuptools | 81.0.0 | pypi_0 | 0 | pypi | pypi |
| shellingham | 1.5.4 | pypi_0 | 0 | pypi | pypi |
| shtab | 1.8.1 | pypi_0 | 0 | pypi | pypi |
| six | 1.17.0 | pypi_0 | 0 | pypi | pypi |
| sqlite | 3.53.2 | h795bf6d_0 | 0 | pkgs/main | linux-64 |
| sse-starlette | 3.4.5 | pypi_0 | 0 | pypi | pypi |
| starlette | 0.52.1 | pypi_0 | 0 | pypi | pypi |
| sympy | 1.14.0 | pypi_0 | 0 | pypi | pypi |
| sysroot_linux-64 | 2.34 | h087de78_3 | 3 | conda-forge | noarch |
| termcolor | 3.3.0 | pypi_0 | 0 | pypi | pypi |
| tiktoken | 0.13.0 | pypi_0 | 0 | pypi | pypi |
| tk | 8.6.15 | h54e0aa7_0 | 0 | pkgs/main | linux-64 |
| tokenizers | 0.22.2 | pypi_0 | 0 | pypi | pypi |
| tomlkit | 0.13.3 | pypi_0 | 0 | pypi | pypi |
| torch | 2.9.1 | pypi_0 | 0 | pypi | pypi |
| torchaudio | 2.9.1+cu128 | pypi_0 | 0 | pypi | pypi |
| torchdata | 0.11.0 | pypi_0 | 0 | pypi | pypi |
| torchvision | 0.24.1+cu128 | pypi_0 | 0 | pypi | pypi |
| tqdm | 4.68.3 | pypi_0 | 0 | pypi | pypi |
| transformers | 5.6.0 | pypi_0 | 0 | pypi | pypi |
| transformers-kt | 5.6.0.post1 | pypi_0 | 0 | pypi | pypi |
| triton | 3.5.1 | pypi_0 | 0 | pypi | pypi |
| trl | 0.24.0 | pypi_0 | 0 | pypi | pypi |
| typer | 0.26.8 | pypi_0 | 0 | pypi | pypi |
| typing-extensions | 4.16.0 | pypi_0 | 0 | pypi | pypi |
| typing-inspection | 0.4.2 | pypi_0 | 0 | pypi | pypi |
| tyro | 0.8.14 | pypi_0 | 0 | pypi | pypi |
| tzdata | 2026.2 | pypi_0 | 0 | pypi | pypi |
| urllib3 | 2.7.0 | pypi_0 | 0 | pypi | pypi |
| uvicorn | 0.50.1 | pypi_0 | 0 | pypi | pypi |
| websockets | 15.0.1 | pypi_0 | 0 | pypi | pypi |
| wheel | 0.47.0 | py312h06a4308_0 | 0 | pkgs/main | linux-64 |
| xorg-libx11 | 1.8.12 | h9b100fa_1 | 1 | pkgs/main | linux-64 |
| xorg-libxau | 1.0.12 | h9b100fa_0 | 0 | pkgs/main | linux-64 |
| xorg-libxdmcp | 1.1.5 | h9b100fa_0 | 0 | pkgs/main | linux-64 |
| xorg-xorgproto | 2024.1 | h47b2149_2 | 2 | pkgs/main | linux-64 |
| xxhash | 3.8.0 | pypi_0 | 0 | pypi | pypi |
| xz | 5.8.2 | h448239c_0 | 0 | pkgs/main | linux-64 |
| yarl | 1.24.2 | pypi_0 | 0 | pypi | pypi |
| zlib | 1.3.2 | h47b2149_0 | 0 | pkgs/main | linux-64 |

## Deepspeed

Prefix: `/mnt/data2/wbw/conda/envs/Deepspeed`
Installed package records: **198**

| Package | Version | Build string | Build number | Channel | Platform |
| --- | --- | --- | ---: | --- | --- |
| _libgcc_mutex | 0.1 | main | 0 | pkgs/main | linux-64 |
| _openmp_mutex | 5.1 | 52_gnu | 52 | pkgs/main | linux-64 |
| accelerate | 1.11.0 | pypi_0 | 0 | pypi | pypi |
| aiofiles | 24.1.0 | pypi_0 | 0 | pypi | pypi |
| aiohappyeyeballs | 2.7.1 | pypi_0 | 0 | pypi | pypi |
| aiohttp | 3.14.1 | pypi_0 | 0 | pypi | pypi |
| aiosignal | 1.4.0 | pypi_0 | 0 | pypi | pypi |
| annotated-doc | 0.0.4 | pypi_0 | 0 | pypi | pypi |
| annotated-types | 0.7.0 | pypi_0 | 0 | pypi | pypi |
| antlr4-python3-runtime | 4.9.3 | pypi_0 | 0 | pypi | pypi |
| anyio | 4.14.1 | pypi_0 | 0 | pypi | pypi |
| attrs | 26.1.0 | pypi_0 | 0 | pypi | pypi |
| av | 16.0.0 | pypi_0 | 0 | pypi | pypi |
| binutils_impl_linux-64 | 2.44 | h78f17ca_3 | 3 | pkgs/main | linux-64 |
| brotli | 1.2.0 | pypi_0 | 0 | pypi | pypi |
| bzip2 | 1.0.8 | h5eee18b_6 | 6 | pkgs/main | linux-64 |
| ca-certificates | 2026.6.17 | hbd8a1cb_0 | 0 | conda-forge | noarch |
| certifi | 2026.6.17 | pypi_0 | 0 | pypi | pypi |
| charset-normalizer | 3.4.7 | pypi_0 | 0 | pypi | pypi |
| click | 8.4.2 | pypi_0 | 0 | pypi | pypi |
| contourpy | 1.3.3 | pypi_0 | 0 | pypi | pypi |
| cuda-bindings | 13.3.1 | pypi_0 | 0 | pypi | pypi |
| cuda-cudart | 11.8.89 | 0 | 0 | nvidia/label/cuda-11.8.0 | linux-64 |
| cuda-libraries | 11.8.0 | 0 | 0 | nvidia/label/cuda-11.8.0 | linux-64 |
| cuda-nvrtc | 11.8.89 | 0 | 0 | nvidia/label/cuda-11.8.0 | linux-64 |
| cuda-pathfinder | 1.5.6 | pypi_0 | 0 | pypi | pypi |
| cuda-runtime | 11.8.0 | 0 | 0 | nvidia/label/cuda-11.8.0 | linux-64 |
| cuda-toolkit | 13.0.2 | pypi_0 | 0 | pypi | pypi |
| cycler | 0.12.1 | pypi_0 | 0 | pypi | pypi |
| datasets | 4.0.0 | pypi_0 | 0 | pypi | pypi |
| deepspeed | 0.19.2 | pypi_0 | 0 | pypi | pypi |
| dill | 0.3.8 | pypi_0 | 0 | pypi | pypi |
| docstring-parser | 0.18.0 | pypi_0 | 0 | pypi | pypi |
| einops | 0.8.2 | pypi_0 | 0 | pypi | pypi |
| fastapi | 0.139.0 | pypi_0 | 0 | pypi | pypi |
| ffmpy | 1.0.0 | pypi_0 | 0 | pypi | pypi |
| filelock | 3.29.5 | pypi_0 | 0 | pypi | pypi |
| fire | 0.7.1 | pypi_0 | 0 | pypi | pypi |
| flash-attn | 2.8.3 | pypi_0 | 0 | pypi | pypi |
| fonttools | 4.63.0 | pypi_0 | 0 | pypi | pypi |
| frozenlist | 1.8.0 | pypi_0 | 0 | pypi | pypi |
| fsspec | 2025.3.0 | pypi_0 | 0 | pypi | pypi |
| gcc_impl_linux-64 | 15.2.0 | hcacfade_7 | 7 | conda-forge | linux-64 |
| gguf | 0.19.0 | pypi_0 | 0 | pypi | pypi |
| gradio | 5.50.0 | pypi_0 | 0 | pypi | pypi |
| gradio-client | 1.14.0 | pypi_0 | 0 | pypi | pypi |
| groovy | 0.1.2 | pypi_0 | 0 | pypi | pypi |
| h11 | 0.16.0 | pypi_0 | 0 | pypi | pypi |
| hf-transfer | 0.1.9 | pypi_0 | 0 | pypi | pypi |
| hf-xet | 1.5.1 | pypi_0 | 0 | pypi | pypi |
| hjson | 3.1.0 | pypi_0 | 0 | pypi | pypi |
| httpcore | 1.0.9 | pypi_0 | 0 | pypi | pypi |
| httpx | 0.28.1 | pypi_0 | 0 | pypi | pypi |
| huggingface-hub | 1.22.0 | pypi_0 | 0 | pypi | pypi |
| idna | 3.18 | pypi_0 | 0 | pypi | pypi |
| jinja2 | 3.1.6 | pypi_0 | 0 | pypi | pypi |
| kernel-headers_linux-64 | 5.14.0 | he073ed8_3 | 3 | conda-forge | noarch |
| kiwisolver | 1.5.0 | pypi_0 | 0 | pypi | pypi |
| ld_impl_linux-64 | 2.44 | h9e0c5a2_3 | 3 | pkgs/main | linux-64 |
| libcublas | 11.11.3.6 | 0 | 0 | nvidia/label/cuda-11.8.0 | linux-64 |
| libcufft | 10.9.0.58 | 0 | 0 | nvidia/label/cuda-11.8.0 | linux-64 |
| libcufile | 1.4.0.31 | 0 | 0 | nvidia/label/cuda-11.8.0 | linux-64 |
| libcurand | 10.3.0.86 | 0 | 0 | nvidia/label/cuda-11.8.0 | linux-64 |
| libcusolver | 11.4.1.48 | 0 | 0 | nvidia/label/cuda-11.8.0 | linux-64 |
| libcusparse | 11.7.5.86 | 0 | 0 | nvidia/label/cuda-11.8.0 | linux-64 |
| libexpat | 2.8.2 | h7354ed3_1 | 1 | pkgs/main | linux-64 |
| libffi | 3.4.8 | h06d3fd0_3 | 3 | pkgs/main | linux-64 |
| libgcc | 15.2.0 | h69a1729_8 | 8 | pkgs/main | linux-64 |
| libgcc-devel_linux-64 | 15.2.0 | h73f6952_107 | 107 | conda-forge | noarch |
| libgcc-ng | 15.2.0 | h166f726_8 | 8 | pkgs/main | linux-64 |
| libgomp | 15.2.0 | h4751f2c_8 | 8 | pkgs/main | linux-64 |
| libnpp | 11.8.0.86 | 0 | 0 | nvidia/label/cuda-11.8.0 | linux-64 |
| libnvjpeg | 11.9.0.86 | 0 | 0 | nvidia/label/cuda-11.8.0 | linux-64 |
| libsanitizer | 15.2.0 | hb13aed2_7 | 7 | conda-forge | linux-64 |
| libstdcxx | 15.2.0 | h39759b7_8 | 8 | pkgs/main | linux-64 |
| libstdcxx-ng | 15.2.0 | hc03a8fd_8 | 8 | pkgs/main | linux-64 |
| libuuid | 1.41.5 | h5eee18b_0 | 0 | pkgs/main | linux-64 |
| libxcb | 1.17.0 | h9b100fa_0 | 0 | pkgs/main | linux-64 |
| libzlib | 1.3.2 | h47b2149_0 | 0 | pkgs/main | linux-64 |
| llamafactory | 0.9.6.dev0 | pypi_0 | 0 | pypi | pypi |
| markdown-it-py | 4.2.0 | pypi_0 | 0 | pypi | pypi |
| markupsafe | 3.0.3 | pypi_0 | 0 | pypi | pypi |
| matplotlib | 3.11.0 | pypi_0 | 0 | pypi | pypi |
| mdurl | 0.1.2 | pypi_0 | 0 | pypi | pypi |
| modelscope | 1.38.0 | pypi_0 | 0 | pypi | pypi |
| modelscope-hub | 0.1.6 | pypi_0 | 0 | pypi | pypi |
| mpmath | 1.3.0 | pypi_0 | 0 | pypi | pypi |
| msgpack | 1.2.1 | pypi_0 | 0 | pypi | pypi |
| multidict | 6.7.1 | pypi_0 | 0 | pypi | pypi |
| multiprocess | 0.70.16 | pypi_0 | 0 | pypi | pypi |
| ncurses | 6.5 | h7934f7d_0 | 0 | pkgs/main | linux-64 |
| networkx | 3.6.1 | pypi_0 | 0 | pypi | pypi |
| ninja | 1.13.0 | pypi_0 | 0 | pypi | pypi |
| numpy | 2.5.1 | pypi_0 | 0 | pypi | pypi |
| nvidia-cublas | 13.1.1.3 | pypi_0 | 0 | pypi | pypi |
| nvidia-cublas-cu12 | 12.8.4.1 | pypi_0 | 0 | pypi | pypi |
| nvidia-cuda-cupti | 13.0.85 | pypi_0 | 0 | pypi | pypi |
| nvidia-cuda-cupti-cu12 | 12.8.90 | pypi_0 | 0 | pypi | pypi |
| nvidia-cuda-nvrtc | 13.0.88 | pypi_0 | 0 | pypi | pypi |
| nvidia-cuda-nvrtc-cu12 | 12.8.93 | pypi_0 | 0 | pypi | pypi |
| nvidia-cuda-runtime | 13.0.96 | pypi_0 | 0 | pypi | pypi |
| nvidia-cuda-runtime-cu12 | 12.8.90 | pypi_0 | 0 | pypi | pypi |
| nvidia-cudnn-cu12 | 9.10.2.21 | pypi_0 | 0 | pypi | pypi |
| nvidia-cudnn-cu13 | 9.20.0.48 | pypi_0 | 0 | pypi | pypi |
| nvidia-cufft | 12.0.0.61 | pypi_0 | 0 | pypi | pypi |
| nvidia-cufft-cu12 | 11.3.3.83 | pypi_0 | 0 | pypi | pypi |
| nvidia-cufile | 1.15.1.6 | pypi_0 | 0 | pypi | pypi |
| nvidia-cufile-cu12 | 1.13.1.3 | pypi_0 | 0 | pypi | pypi |
| nvidia-curand | 10.4.0.35 | pypi_0 | 0 | pypi | pypi |
| nvidia-curand-cu12 | 10.3.9.90 | pypi_0 | 0 | pypi | pypi |
| nvidia-cusolver | 12.0.4.66 | pypi_0 | 0 | pypi | pypi |
| nvidia-cusolver-cu12 | 11.7.3.90 | pypi_0 | 0 | pypi | pypi |
| nvidia-cusparse | 12.6.3.3 | pypi_0 | 0 | pypi | pypi |
| nvidia-cusparse-cu12 | 12.5.8.93 | pypi_0 | 0 | pypi | pypi |
| nvidia-cusparselt-cu12 | 0.7.1 | pypi_0 | 0 | pypi | pypi |
| nvidia-cusparselt-cu13 | 0.8.1 | pypi_0 | 0 | pypi | pypi |
| nvidia-ml-py | 13.610.43 | pypi_0 | 0 | pypi | pypi |
| nvidia-nccl-cu12 | 2.27.5 | pypi_0 | 0 | pypi | pypi |
| nvidia-nccl-cu13 | 2.29.7 | pypi_0 | 0 | pypi | pypi |
| nvidia-nvjitlink | 13.0.88 | pypi_0 | 0 | pypi | pypi |
| nvidia-nvjitlink-cu12 | 12.8.93 | pypi_0 | 0 | pypi | pypi |
| nvidia-nvshmem-cu12 | 3.3.20 | pypi_0 | 0 | pypi | pypi |
| nvidia-nvshmem-cu13 | 3.4.5 | pypi_0 | 0 | pypi | pypi |
| nvidia-nvtx | 13.0.85 | pypi_0 | 0 | pypi | pypi |
| nvidia-nvtx-cu12 | 12.8.90 | pypi_0 | 0 | pypi | pypi |
| omegaconf | 2.3.1 | pypi_0 | 0 | pypi | pypi |
| openssl | 3.6.3 | h35e630c_0 | 0 | conda-forge | linux-64 |
| orjson | 3.11.9 | pypi_0 | 0 | pypi | pypi |
| packaging | 26.0 | py312h06a4308_0 | 0 | pkgs/main | linux-64 |
| pandas | 2.3.3 | pypi_0 | 0 | pypi | pypi |
| peft | 0.18.1 | pypi_0 | 0 | pypi | pypi |
| pillow | 11.3.0 | pypi_0 | 0 | pypi | pypi |
| pip | 26.1.2 | pyhc872135_0 | 0 | pkgs/main | noarch |
| propcache | 0.5.2 | pypi_0 | 0 | pypi | pypi |
| protobuf | 7.35.1 | pypi_0 | 0 | pypi | pypi |
| psutil | 7.2.2 | pypi_0 | 0 | pypi | pypi |
| pthread-stubs | 0.3 | h0ce48e5_1 | 1 | pkgs/main | linux-64 |
| py-cpuinfo | 9.0.0 | pypi_0 | 0 | pypi | pypi |
| pyarrow | 24.0.0 | pypi_0 | 0 | pypi | pypi |
| pydantic | 2.12.3 | pypi_0 | 0 | pypi | pypi |
| pydantic-core | 2.41.4 | pypi_0 | 0 | pypi | pypi |
| pydub | 0.25.1 | pypi_0 | 0 | pypi | pypi |
| pygments | 2.20.0 | pypi_0 | 0 | pypi | pypi |
| pyparsing | 3.3.2 | pypi_0 | 0 | pypi | pypi |
| python | 3.12.13 | h4d16e0c_1 | 1 | pkgs/main | linux-64 |
| python-dateutil | 2.9.0.post0 | pypi_0 | 0 | pypi | pypi |
| python-multipart | 0.0.32 | pypi_0 | 0 | pypi | pypi |
| pytz | 2026.2 | pypi_0 | 0 | pypi | pypi |
| pyyaml | 6.0.3 | pypi_0 | 0 | pypi | pypi |
| readline | 8.3 | hc2a1206_0 | 0 | pkgs/main | linux-64 |
| regex | 2026.6.28 | pypi_0 | 0 | pypi | pypi |
| requests | 2.34.2 | pypi_0 | 0 | pypi | pypi |
| rich | 15.0.0 | pypi_0 | 0 | pypi | pypi |
| ruff | 0.15.20 | pypi_0 | 0 | pypi | pypi |
| safehttpx | 0.1.7 | pypi_0 | 0 | pypi | pypi |
| safetensors | 0.8.0 | pypi_0 | 0 | pypi | pypi |
| scipy | 1.18.0 | pypi_0 | 0 | pypi | pypi |
| semantic-version | 2.10.0 | pypi_0 | 0 | pypi | pypi |
| sentencepiece | 0.2.1 | pypi_0 | 0 | pypi | pypi |
| setuptools | 81.0.0 | pypi_0 | 0 | pypi | pypi |
| shellingham | 1.5.4 | pypi_0 | 0 | pypi | pypi |
| shtab | 1.8.1 | pypi_0 | 0 | pypi | pypi |
| six | 1.17.0 | pypi_0 | 0 | pypi | pypi |
| sqlite | 3.53.2 | h795bf6d_0 | 0 | pkgs/main | linux-64 |
| sse-starlette | 3.4.5 | pypi_0 | 0 | pypi | pypi |
| starlette | 0.52.1 | pypi_0 | 0 | pypi | pypi |
| sympy | 1.14.0 | pypi_0 | 0 | pypi | pypi |
| sysroot_linux-64 | 2.34 | h087de78_3 | 3 | conda-forge | noarch |
| termcolor | 3.3.0 | pypi_0 | 0 | pypi | pypi |
| tiktoken | 0.13.0 | pypi_0 | 0 | pypi | pypi |
| tk | 8.6.15 | h54e0aa7_0 | 0 | pkgs/main | linux-64 |
| tokenizers | 0.22.2 | pypi_0 | 0 | pypi | pypi |
| tomlkit | 0.13.3 | pypi_0 | 0 | pypi | pypi |
| torch | 2.9.1 | pypi_0 | 0 | pypi | pypi |
| torchaudio | 2.9.1+cu128 | pypi_0 | 0 | pypi | pypi |
| torchdata | 0.11.0 | pypi_0 | 0 | pypi | pypi |
| torchvision | 0.24.1+cu128 | pypi_0 | 0 | pypi | pypi |
| tqdm | 4.68.3 | pypi_0 | 0 | pypi | pypi |
| transformers | 5.6.0 | pypi_0 | 0 | pypi | pypi |
| triton | 3.5.1 | pypi_0 | 0 | pypi | pypi |
| trl | 0.24.0 | pypi_0 | 0 | pypi | pypi |
| typer | 0.26.8 | pypi_0 | 0 | pypi | pypi |
| typing-extensions | 4.16.0 | pypi_0 | 0 | pypi | pypi |
| typing-inspection | 0.4.2 | pypi_0 | 0 | pypi | pypi |
| tyro | 0.8.14 | pypi_0 | 0 | pypi | pypi |
| tzdata | 2026.2 | pypi_0 | 0 | pypi | pypi |
| urllib3 | 2.7.0 | pypi_0 | 0 | pypi | pypi |
| uvicorn | 0.50.1 | pypi_0 | 0 | pypi | pypi |
| websockets | 15.0.1 | pypi_0 | 0 | pypi | pypi |
| wheel | 0.47.0 | py312h06a4308_0 | 0 | pkgs/main | linux-64 |
| xorg-libx11 | 1.8.12 | h9b100fa_1 | 1 | pkgs/main | linux-64 |
| xorg-libxau | 1.0.12 | h9b100fa_0 | 0 | pkgs/main | linux-64 |
| xorg-libxdmcp | 1.1.5 | h9b100fa_0 | 0 | pkgs/main | linux-64 |
| xorg-xorgproto | 2024.1 | h47b2149_2 | 2 | pkgs/main | linux-64 |
| xxhash | 3.8.0 | pypi_0 | 0 | pypi | pypi |
| xz | 5.8.2 | h448239c_0 | 0 | pkgs/main | linux-64 |
| yarl | 1.24.2 | pypi_0 | 0 | pypi | pypi |
| zlib | 1.3.2 | h47b2149_0 | 0 | pkgs/main | linux-64 |

## Refresh command

Re-run the following commands after either environment changes, then regenerate the tables from the JSON output:

```bash
conda list -n Kllama --json
conda list -n Deepspeed --json
```
