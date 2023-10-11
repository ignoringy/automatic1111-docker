# Based https://github.com/camenduru/stable-diffusion-webui-docker/blob/main/Dockerfile.v2.6.gpu
FROM nvidia/cuda:12.2.0-base-ubuntu22.04
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y && \
    apt-get install -y git && \
    adduser --disabled-password --gecos '' user && \
    mkdir /content && \
    chown -R user:user /content && \
    apt-get install -y aria2 libgl1 libglib2.0-0 wget git git-lfs python3-pip python-is-python3 && \
    pip install -q torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 torchtext==0.15.2 torchdata==0.6.1 --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip install packaging==23.1

WORKDIR /content
USER user

RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui /content/webui && \
    git clone https://github.com/Iyashinouta/sd-model-downloader /content/webui/extensions/sd-model-downloader && \
    git clone https://github.com/etherealxx/batchlinks-webui /content/webui/extensions/batchlinks-webui && \
    git clone https://github.com/Mikubill/sd-webui-controlnet /content/webui/extensions/sd-webui-controlnet && \
    git clone https://github.com/fkunn1326/openpose-editor /content/webui/extensions/openpose-editor

#ADD --chown=user https://huggingface.co/ckpt/ControlNet/resolve/main/control_canny-fp16.safetensors /content/webui/extensions/sd-webui-controlnet/models/control_canny-fp16.safetensors
#ADD --chown=user https://huggingface.co/ckpt/ControlNet/resolve/main/control_depth-fp16.safetensors /content/webui/extensions/sd-webui-controlnet/models/control_depth-fp16.safetensors
#ADD --chown=user https://huggingface.co/ckpt/ControlNet/resolve/main/control_hed-fp16.safetensors /content/webui/extensions/sd-webui-controlnet/models/control_hed-fp16.safetensors
#ADD --chown=user https://huggingface.co/ckpt/ControlNet/resolve/main/control_mlsd-fp16.safetensors /content/webui/extensions/sd-webui-controlnet/models/control_mlsd-fp16.safetensors
#ADD --chown=user https://huggingface.co/ckpt/ControlNet/resolve/main/control_normal-fp16.safetensors /content/webui/extensions/sd-webui-controlnet/models/control_normal-fp16.safetensors
#ADD --chown=user https://huggingface.co/ckpt/ControlNet/resolve/main/control_openpose-fp16.safetensors /content/webui/extensions/sd-webui-controlnet/models/control_openpose-fp16.safetensors
#ADD --chown=user https://huggingface.co/ckpt/ControlNet/resolve/main/control_scribble-fp16.safetensors /content/webui/extensions/sd-webui-controlnet/models/control_scribble-fp16.safetensors
#ADD --chown=user https://huggingface.co/ckpt/ControlNet/resolve/main/control_seg-fp16.safetensors /content/webui/extensions/sd-webui-controlnet/models/control_seg-fp16.safetensors
#ADD --chown=user https://huggingface.co/ckpt/ControlNet/resolve/main/hand_pose_model.pth /content/webui/extensions/sd-webui-controlnet/annotator/openpose/hand_pose_model.pth
#ADD --chown=user https://huggingface.co/ckpt/ControlNet/resolve/main/body_pose_model.pth /content/webui/extensions/sd-webui-controlnet/annotator/openpose/body_pose_model.pth
#ADD --chown=user https://huggingface.co/ckpt/ControlNet/resolve/main/dpt_hybrid-midas-501f0c75.pt /content/webui/extensions/sd-webui-controlnet/annotator/midas/dpt_hybrid-midas-501f0c75.pt
#ADD --chown=user https://huggingface.co/ckpt/ControlNet/resolve/main/mlsd_large_512_fp32.pth /content/webui/extensions/sd-webui-controlnet/annotator/mlsd/mlsd_large_512_fp32.pth
#ADD --chown=user https://huggingface.co/ckpt/ControlNet/resolve/main/mlsd_tiny_512_fp32.pth /content/webui/extensions/sd-webui-controlnet/annotator/mlsd/mlsd_tiny_512_fp32.pth
#ADD --chown=user https://huggingface.co/ckpt/ControlNet/resolve/main/network-bsds500.pth /content/webui/extensions/sd-webui-controlnet/annotator/hed/network-bsds500.pth
#ADD --chown=user https://huggingface.co/ckpt/ControlNet/resolve/main/upernet_global_small.pth /content/webui/extensions/sd-webui-controlnet/annotator/uniformer/upernet_global_small.pth
#ADD --chown=user https://huggingface.co/ckpt/ControlNet/resolve/main/t2iadapter_style_sd14v1.pth /content/webui/extensions/sd-webui-controlnet/models/t2iadapter_style_sd14v1.pth
#ADD --chown=user https://huggingface.co/ckpt/ControlNet/resolve/main/t2iadapter_sketch_sd14v1.pth /content/webui/extensions/sd-webui-controlnet/models/t2iadapter_sketch_sd14v1.pth
#ADD --chown=user https://huggingface.co/ckpt/ControlNet/resolve/main/t2iadapter_seg_sd14v1.pth /content/webui/extensions/sd-webui-controlnet/models/t2iadapter_seg_sd14v1.pth
#ADD --chown=user https://huggingface.co/ckpt/ControlNet/resolve/main/t2iadapter_openpose_sd14v1.pth /content/webui/extensions/sd-webui-controlnet/models/t2iadapter_openpose_sd14v1.pth
#ADD --chown=user https://huggingface.co/ckpt/ControlNet/resolve/main/t2iadapter_keypose_sd14v1.pth /content/webui/extensions/sd-webui-controlnet/models/t2iadapter_keypose_sd14v1.pth
#ADD --chown=user https://huggingface.co/ckpt/ControlNet/resolve/main/t2iadapter_depth_sd14v1.pth /content/webui/extensions/sd-webui-controlnet/models/t2iadapter_depth_sd14v1.pth
#ADD --chown=user https://huggingface.co/ckpt/ControlNet/resolve/main/t2iadapter_color_sd14v1.pth /content/webui/extensions/sd-webui-controlnet/models/t2iadapter_color_sd14v1.pth
#ADD --chown=user https://huggingface.co/ckpt/ControlNet/resolve/main/t2iadapter_canny_sd14v1.pth /content/webui/extensions/sd-webui-controlnet/models/t2iadapter_canny_sd14v1.pth


RUN sed -i -e 's/    start()/    #start()/g' /content/webui/launch.py && \
	cd /content/webui && \
    python launch.py --skip-torch-cuda-test && \
    git reset --hard

#ADD --chown=user https://huggingface.co/ckpt/sd15/resolve/main/v1-5-pruned-emaonly.ckpt /content/webui/models/Stable-diffusion/v1-5-pruned-emaonly.ckpt
EXPOSE 7860

CMD cd /content/webui && python webui.py --xformers --listen --enable-insecure-extension-access --gradio-queue