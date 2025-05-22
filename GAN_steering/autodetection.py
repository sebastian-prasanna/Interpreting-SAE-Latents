import torch
from transformers import Blip2Processor, Blip2Model, Blip2ForConditionalGeneration
from PIL import Image
import torchvision.transforms as T

# Load BLIP-2 model and processor
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
decoder_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

# Move models to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
decoder_model = decoder_model.to(device)

def embed_images(images):
    """Embed a batch of images using BLIP-2.
    
    Args:
        images (torch.Tensor): Batch of images with shape (B, C, H, W)
        
    Returns:
        torch.Tensor: Image embeddings
    """
    # Convert images to range [0, 1] if they're in [-1, 1]
    if images.min() < 0:
        images = (images + 1) / 2
    
    # Convert to PIL Images
    transform = T.ToPILImage()
    pil_images = [transform(img) for img in images]
    
    # Process images
    inputs = processor(images=pil_images, return_tensors="pt").to(device)
    
    # Get image embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeddings = outputs.image_embeds
    
    return image_embeddings

def decode_embeddings(image_embeddings, max_length=50):
    """Decode image embeddings into text descriptions using BLIP-2.
    
    Args:
        image_embeddings (torch.Tensor): Image embeddings from BLIP-2
        max_length (int): Maximum length of generated text
        
    Returns:
        list: List of generated text descriptions
    """
    with torch.no_grad():
        outputs = decoder_model.generate(
            vision_hidden_states=image_embeddings,
            max_length=max_length,
            num_beams=5,
            min_length=5,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
        )
    
    # Decode the generated tokens to text
    generated_texts = processor.batch_decode(outputs, skip_special_tokens=True)
    return generated_texts

if __name__ == "__main__":
    # Load and process test images
    test_images = torch.load('test_imgs.pt')
    print(f"Loaded images shape: {test_images.shape}")

    # Get embeddings
    embeddings = embed_images(test_images)
    print(f"Generated embeddings shape: {embeddings.shape}")

    # Generate descriptions for a few examples
    descriptions = decode_embeddings(embeddings[:5])  # Process first 5 images as example
    for i, desc in enumerate(descriptions):
        print(f"Image {i}: {desc}") 