import os
from typing import List, Dict, Union
import numpy as np
import torch
from PIL import Image
import cv2

from EmotionPipeline import EmotionSVDPipeline
from diffusers.utils import load_image, export_to_video
from utils.event_utils import process_image, mkdir


def get_evaluation_data(data_path: str, temp_length: int = 4) -> Dict[str, Union[torch.Tensor, Image.Image]]:
    """
    Load and preprocess evaluation data.

    Args:
        data_path (str): Path to the .npy file containing the event sequence.
        temp_length (int): Number of temporal frames to use as conditioning.

    Returns:
        dict: A dictionary containing pixel values, input image, and temporal conditioning data.
    """
    event_sequence_npy = torch.from_numpy(np.load(data_path))
    event_input = (event_sequence_npy[0] + 1) / 2
    temp_cond = event_sequence_npy[1:1 + temp_length]
    return {
        "pixel_values": event_sequence_npy,
        "image": event_input,
        "temp_cond": temp_cond
    }


def concatenate_image_lists(list1: List[Image.Image], list2: List[Image.Image]) -> List[Image.Image]:
    """
    Concatenate two lists of images side by side.

    Args:
        list1 (List[Image.Image]): List of first set of images.
        list2 (List[Image.Image]): List of second set of images.

    Returns:
        List[Image.Image]: A list of concatenated images.
    """
    concatenated_images = []
    for img1, img2 in zip(list1, list2):
        img1 = process_image(img1, 1)
        img2 = process_image(img2, 3)
        new_image = Image.new('RGB', (img1.width + img2.width, img1.height))
        new_image.paste(img1, (0, 0))
        new_image.paste(img2, (img1.width, 0))
        concatenated_images.append(new_image)
    return concatenated_images


def load_pipeline(checkpoint_path: str, device: str = "cuda") -> EmotionSVDPipeline:
    """
    Load the EmotionSVDPipeline model from the given checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint.

    Returns:
        EmotionSVDPipeline: The loaded pipeline.
    """
    pipe = EmotionSVDPipeline.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float32,
        variant="fp16"
    )
    pipe.to(device)
    pipe.unet.enable_forward_chunking()
    return pipe


def process_event_sequence(data_path: str, output_path: str, pipeline: EmotionSVDPipeline, seed: int) -> None:
    """
    Process an event sequence and export the results as a video.

    Args:
        data_path (str): Path to the event sequence data (.npy file).
        pipeline (EmotionSVDPipeline): Loaded EmotionSVDPipeline model.
        seed (int): Seed value for reproducibility.
    """
    seq_name = os.path.splitext(os.path.basename(data_path))[0]
    batch = get_evaluation_data(data_path, temp_length=5)

    # Prepare input image
    image_tensor = batch['image']
    input_image = (image_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    input_image_pil = Image.fromarray(input_image)
    input_images = [load_image(input_image_pil)]

    # Prepare ground truth images
    ground_truth_images = [
        Image.fromarray(((frame.permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8))
        for frame in batch["pixel_values"]
    ]

    # Generate frames using the pipeline
    generator = torch.manual_seed(seed)
    generated_frames = pipeline(
        input_images,
        decode_chunk_size=8,
        generator=generator,
        temp_cond=batch['temp_cond']
    ).frames[0]

    # Combine input and generated frames
    generated_frames.insert(0, input_images[0])
    concatenated_frames = concatenate_image_lists(ground_truth_images, generated_frames)

    mkdir(output_path)
    # Export the results to a video
    export_to_video(concatenated_frames, os.path.join(output_path, f"{seq_name}.mp4"), fps=7)



if __name__ == "__main__":
    import argparse

    # Define argument parser
    parser = argparse.ArgumentParser(description="Process event sequences with EmotionSVDPipeline.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the event sequence data (.npy file).")
    parser.add_argument("--output_path", type=str, default=".", help="Directory to save the output video.")
    parser.add_argument("--seed", type=int, default=87, help="Seed for reproducibility.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on ('cuda' or 'cpu').")

    # Parse arguments
    args = parser.parse_args()

    # Load pipeline and process event sequence
    pipeline = load_pipeline(args.model_path, device=args.device)
    process_event_sequence(
        data_path=args.data_path,
        pipeline=pipeline,
        output_path=args.output_path,
        seed=args.seed,
    )