import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
# --------------------------------------------------------------------------
# 1. Загрузка модели Stable Diffusion (ИСПРАВЛЕННАЯ ВЕРСИЯ)
# --------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"
print(f"The device used: {device}")

if device == "cuda":
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    print("The model is loaded in float16 mode (GPU optimization).")
else:
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    print("The model is loaded in float32 mode (for CPU).")

pipe = pipe.to(device)
print("The model is successfully ready for work.")
# --------------------------------------------------------------------------
# 2. Создание функции для генерации изображений
# --------------------------------------------------------------------------
def generate_image(prompt, negative_prompt, guidance_scale):
    """
    Generates an image based on text prompts and settings.
    """
    print(f"Generating image")
    print(f"  - Prompt: {prompt}")
    print(f"  - Negative Prompt: {negative_prompt}")
    print(f"  - Guidance Scale: {guidance_scale}")

    generator = torch.Generator(device).manual_seed(42)
    
    with torch.autocast(device):
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
    print("Image generated successfully.")
    return image
# --------------------------------------------------------------------------
# 3. Создание и запуск интерфейса Gradio
# --------------------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("🎨 Stable Diffusion Image Generator")
    gr.Markdown(
        "This demo app lets you create images from text using the Stable Diffusion model."
        "Enter what you want to see, add what you don't want to see (negative prompt),"
        "and adjust the prompt-following strength using the slider."
    )
    
    with gr.Row():
        with gr.Column(scale=4):
            prompt_input = gr.Textbox(label="Prompt (what do you  want to generate?)", placeholder="For example: 'Photograph of an astronaut riding a horse on Mars'")
            negative_prompt_input = gr.Textbox(label="Negative Prompt (What to Avoid?)", placeholder="'For example: 'Poor anatomy, blurriness, extra limbs'")
            guidance_slider = gr.Slider(minimum=1.0, maximum=20.0, step=0.5, value=7.5, label="Guidance Scale")
            submit_button = gr.Button("Generate", variant="primary")
            
        with gr.Column(scale=4):
            image_output = gr.Image(label="Result")
    
    submit_button.click(
        fn=generate_image,
        inputs=[prompt_input, negative_prompt_input, guidance_slider],
        outputs=image_output
    )

if __name__ == "__main__":
    demo.launch(share=True)


# def greet(name):
#     return "Hello, " + name

# demo = gr.Interface(fn=greet, inputs="text", outputs="text")
# demo.launch()
