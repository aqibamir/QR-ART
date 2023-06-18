import os
import qrcode
import requests
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DDIMScheduler


# Generate a QR code for a given text or URL with the highest error correction level
def generate_qr(data, id):
    filename = f"{id}-qrcode.png"

    qr = qrcode.QRCode(version=1,
                       error_correction=qrcode.constants.ERROR_CORRECT_H,
                       box_size=10,
                       border=4)
    qr.add_data(data)
    qr.make(fit=True)

    # Create an image from the QR code
    qr_image = qr.make_image(fill_color="black", back_color="white")

    # Save the image file as PNG format
    qr_image.save(filename, "PNG")


# Function to resize an image to a given resolution without adding padding
def resize_image(input_image: Image, resolution: int) -> Image:
    input_image = input_image.convert("RGB")
    resized_image = input_image.resize((resolution, resolution), resample=Image.LANCZOS)
    return resized_image


# Function to download an image from a URL and save it as a PNG file
def download_image(image_url, save_path):
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()

        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        # Open the downloaded image using PIL
        image = Image.open(save_path)

        # Save the image as PNG format
        image.save(save_path, "PNG")

        return True

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return False


# Function to download the model and get the pipeline
def get_pipeline():
    try:
        # Load the ControlNet model from a pretrained checkpoint
        controlnet = ControlNetModel.from_pretrained(
            "DionTimmer/controlnet_qrcode-control_v11p_sd21",
            torch_dtype=torch.float16)

        # Create a StableDiffusionControlNetImg2ImgPipeline with the loaded ControlNet model
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=torch.float16)

        # Enable memory-efficient attention for the pipeline
        pipe.enable_xformers_memory_efficient_attention()

        # Set the scheduler for the pipeline to DDIMScheduler with its current configuration
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        # Enable CPU offload for the model
        pipe.enable_model_cpu_offload()

        return pipe

    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return None


# Generate QR code art using the provided inputs
def generate_qr_art(id, url, reference_image_url, prompt, pipe):
    try:
        # Generate the QR code image
        generate_qr(url, id)

        # Download the reference image
        reference_image_path = f"{id}_reference.png"
        if not download_image(reference_image_url, reference_image_path):
            return False

        # Load the QR code image from local storage
        source_image_path = f"{id}-qrcode.png"
        source_image = Image.open(source_image_path)
        source_image = resize_image(source_image, 768)

        # Load the initial image
        init_image = Image.open(reference_image_path)
        init_image = resize_image(init_image, 768)

        generator = torch.manual_seed(123121231)

        # Generate the image using the pipeline
        image = pipe(
            prompt=prompt,
            negative_prompt="ugly, disfigured, low quality, blurry",
            image=source_image,
            control_image=init_image,
            width=768,
            height=768,
            guidance_scale=7.5,
            controlnet_conditioning_scale=1.5,
            generator=generator,
            strength=0.9,
            num_inference_steps=150
        )

        # Save the generated image
        output_path = "output.png"
        image.images[0].save(output_path)

        # Clean up temporary files
        os.remove(source_image_path)
        os.remove(reference_image_path)

        return True

    except Exception as e:
        print(f"Error generating QR art: {e}")
        return False
