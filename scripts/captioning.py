import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from qwen_vl_utils import process_vision_info

from prompts import PROMPTS, COMPLETIONS

flash_attn_available = False
try:
    import flash_attn

    print(f"Flash Attention version: {flash_attn.__version__}")
    flash_attn_available = True
except ImportError:
    print("Flash Attention is not installed. Install it to increase throughput.")


def convert_to_pil_image(image_path):
    with rasterio.open(image_path) as src:
        image_array = src.read()  # shape: (bands, height, width)

    arr = np.transpose(image_array, (1, 2, 0))  # shape: (height, width, bands)
    if arr.dtype != np.uint8:
        arr_min, arr_max = arr.min(), arr.max()
        arr = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)

    return Image.fromarray(arr)


def get_args():
    parser = argparse.ArgumentParser(description="Captioning script")
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["ben", "s1", "s2"],
        required=True,
        help="Dataset to use: ben, s1, or s2",
    )
    parser.add_argument(
        "--ratio",
        "-r",
        type=float,
        default=None,
        help="Ratio of images to process (0.1 for 10%, 1 for all, or an integer for a fixed number)",
    )
    parser.add_argument(
        "--prompt-type",
        "-p",
        type=str,
        choices=["base", "specific"],
        default="base",
        help="Prompt to use: base or specific",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Set paths
    BASE_DIR = Path(__file__).parent.parent.resolve()
    PARQUET_PATH = BASE_DIR / "data/selected_data_12_v2k.parquet"
    MODEL = "Qwen/Qwen3-VL-8B-Instruct"

    data_paths = {
        "ben": BASE_DIR / "data/12k_BigEarthNetRGB",
        "s1": BASE_DIR / "data/S1",
        "s2": BASE_DIR / "data/S2_multi_12k_norm",
    }

    # Get optional parameters from command line
    args = get_args()

    # Parse arguments
    image_dir = data_paths[args.mode]
    prompt = PROMPTS[args.prompt_type]

    if args.prompt_type == "specific":
        prompt = prompt.format(*COMPLETIONS[args.mode])

    ratio = int(args.ratio) if args.ratio is not None and args.ratio.is_integer() else args.ratio

    # Set output directory
    output_dir = BASE_DIR / f"outputs/prompt_{args.prompt_type}/{args.mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Prompt: {prompt}")
    print("Output directory:", output_dir)

    # Load model and processor
    print(f"Loading model {MODEL}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL,
        dtype=torch.bfloat16 if flash_attn_available else "auto",
        attn_implementation="flash_attention_2" if flash_attn_available else "eager",
        device_map="auto",
    ).eval()

    min_pixels = max_pixels = 120 * 120 if args.mode != "s2" else 60 * 60
    processor = Qwen3VLProcessor.from_pretrained(
        MODEL, min_pixels=min_pixels, max_pixels=max_pixels
    )

    # Load reference test dataframe
    df = pd.read_parquet(PARQUET_PATH)
    df_test = df[df["set"] == "test"].reset_index(drop=True)
    num_total = len(df_test)
    print(f"Total test images: {num_total}")

    # Select subset of images based on ratio
    if isinstance(ratio, int):
        df_selected = df_test.iloc[:ratio]
    elif isinstance(ratio, float):
        df_selected = df_test.sample(frac=ratio, random_state=42)
    else:
        df_selected = df_test

    print(f"Images selected: {len(df_selected)}")

    # Inference loop
    for idx, row in tqdm(df_selected.iterrows(), total=len(df_selected)):
        image_index = row["image_index"]
        file_name = row["file_name"] if args.mode != "s1" else row["s1_name"]

        image_path: Path = image_dir / file_name

        if not image_path.suffix:
            image_path = image_path.with_suffix(".tif")

        if not image_path.exists():
            tqdm.write(f"Image not found: {image_path}")
            continue

        tqdm.write(f"Processing {image_index}")

        image = Image.open(image_path) if args.mode == "ben" else convert_to_pil_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,  # type: ignore
            add_generation_prompt=True,  # type: ignore
        )
        image_inputs, video_inputs = process_vision_info(messages)  # type: ignore
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,  # type: ignore
            return_tensors="pt",  # type: ignore
        ).to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        output_path = output_dir / f"{image_index}.txt"

        with open(output_path, "w") as f:
            f.write(output_text)

    print("Done.")
