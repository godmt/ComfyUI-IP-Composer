# (c) godmt || MIT License (https://mit-license.org/)
# The original project page (https://ip-composer.github.io/IP-Composer/)
import torch
import safetensors.torch as st
import numpy as np
import open_clip
import folder_paths
from pathlib import Path

MODEL_DIR_NAME = "ip_concept_matrices"
models_dir = Path(__file__).parent.parent.parent / "models"
root_path = models_dir.resolve() / MODEL_DIR_NAME
if not root_path.exists():
    root_path.mkdir(parents=True, exist_ok=True)
folder_paths.folder_names_and_paths[MODEL_DIR_NAME] = ([str(root_path)], [".safetensors"])


CLIP_VISIONS = {
    "ViT-H-14": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "ViT-bigG-14": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
}


class IPLoadCLIPVision:
    @classmethod
    def INPUT_TYPES(cls):
        clip_vision_names = list(CLIP_VISIONS.keys())
        return {"required": {
            "clip_vision": (clip_vision_names,),
            "device": ("STRING", {"default": "cuda:0"})
        }}
    RETURN_TYPES = ("OPEN_CLIP", )
    CATEGORY = "ip_composer"
    TITLE = "IP-Comp Load CLIP Vision"
    FUNCTION = "execute"

    def execute(self, clip_vision, device):
        hf_name = CLIP_VISIONS[clip_vision]
        model, _, preprocess = open_clip.create_model_and_transforms(f"hf-hub:{hf_name}")
        model.eval()
        model.to(device)
        tokenizer = open_clip.get_tokenizer(f"hf-hub:{hf_name}")
        return ({
            "hf_name": hf_name,
            "device": device,
            "model": model,
            "preprocess": preprocess,
            "tokenizer": tokenizer
        },)


class IPConceptMatrix:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "open_clip_model": ("OPEN_CLIP",),
            "descriptions": ("STRING", {"multiline": True}),
            "rank": ("INT", {"default": 30, "min": 1, "max": 1024}),
            "batch_size": ("INT", {"default": 100, "min": 1, "max": 512})
        }}
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("concept_matrix",)
    CATEGORY = "ip_composer"
    TITLE = "IP-Comp Concept Matrix"
    FUNCTION = "execute"

    def execute(self, open_clip_model, descriptions, rank, batch_size):
        device = open_clip_model["device"]
        model = open_clip_model["model"]
        tokenizer = open_clip_model["tokenizer"]
        # 1. 行リストを作成
        lines = [l.strip() for l in descriptions.split("\n") if l.strip()]
        # 2. encode concept descriptions
        with torch.no_grad():
            embs = []
            # batch encoding
            for i in range(0, len(lines), batch_size):
                batch_desc = lines[i:i + batch_size]
                texts = tokenizer(batch_desc).to(device)
                batch_embeddings = model.encode_text(texts)
                batch_embeddings = batch_embeddings.detach().cpu().numpy()
                embs.append(batch_embeddings)
                del texts, batch_embeddings
                torch.cuda.empty_cache()
            embs = np.vstack(embs)
            # Perform SVD on the combined matrix
            _, _, v = np.linalg.svd(embs, full_matrices=False)
        # Select the top `rank` singular vectors to construct the projection matrix for the concept direction
        v = v[:rank]
        # project via (Vᵀ V) ≈ identity on subspace
        P = v.T @ v  # (1024, 1024)
        P = torch.from_numpy(P)
        return (P,)


class IPSaveConceptMatrix:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "open_clip_model": ("OPEN_CLIP",),
            "concept_matrix": ("TENSOR",),
            "concept_name": ("STRING", {"default": "concept"})
        }}
    RETURN_TYPES = ()
    CATEGORY = "ip_composer"
    TITLE = "IP-Comp Save Concept Matrix"
    FUNCTION = "execute"
    OUTPUT_NODE = True

    def execute(self, open_clip_model, concept_matrix, concept_name):
        meta = {"clip_vision_model": open_clip_model["hf_name"]}
        file_path = str(root_path / f"{concept_name}.safetensors")
        st.save_file({"concept_matrix": concept_matrix.cpu()}, file_path, metadata=meta)
        return ()


class IPLoadConceptMatrix:
    @classmethod
    def INPUT_TYPES(cls):
        matrix_names = folder_paths.get_filename_list(MODEL_DIR_NAME)
        return {"required": {
            "matrix_name": (matrix_names,),
        }}
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("concept_matrix",)
    CATEGORY = "ip_composer"
    TITLE = "IP-Comp Load Concept Matrix"
    FUNCTION = "execute"

    def execute(self, matrix_name):
        file_path = folder_paths.get_full_path(MODEL_DIR_NAME, matrix_name)
        data = st.load_file(file_path)
        P = data["concept_matrix"]
        return (P,)


class IPConceptMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "ref_embed": ("EMBEDS",),
            "concept_embed": ("EMBEDS",),
            "concept_matrix": ("TENSOR",)
        }}
    RETURN_TYPES = ("EMBEDS",)  # merged embeddings
    CATEGORY = "ip_composer"
    TITLE = "IP-Comp Concept Merge"
    FUNCTION = "execute"

    def execute(self, ref_embed, concept_embed, concept_matrix):
        P = concept_matrix
        ref_embed = ref_embed
        concept_embed = concept_embed
        # merge concept        
        e_mix = ref_embed - ref_embed @ P + concept_embed @ P
        return (e_mix,)


NODE_CLASS_MAPPINGS = {
    "IPCompLoadOpenCLIP": IPLoadCLIPVision,
    "IPCompConceptSubspace": IPConceptMatrix,
    "IPCompConceptMerge": IPConceptMerge,
    "IPSaveConceptMatrix": IPSaveConceptMatrix,
    "IPLoadConceptMatrix": IPLoadConceptMatrix
}
