import torch
import timm
from transformers import BertModel, BertTokenizer
from transformers import GPT2LMHeadModel, GPT2Config
from torch import nn
from transformers import GPT2Tokenizer, top_k_top_p_filtering
from torch.nn import functional as F

class ImageDecoder(nn.Module):
    def __init__(self, latent_dim, output_channels=3):
        super(ImageDecoder, self).__init__()
        # Simple linear layer to start the decoding process
        self.fc = nn.Linear(latent_dim, 256)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=5)
        self.deconv2 = nn.ConvTranspose2d(128, output_channels, kernel_size=5)
        
    def forward(self, z):
        z = F.relu(self.fc(z))
        z = z.view(z.size(0), 256, 1, 1)  # Reshape for deconvolution
        z = F.relu(self.deconv1(z))
        reconstruction = torch.sigmoid(self.deconv2(z))  # Image pixels are usually bounded [0, 1]
        return reconstruction

class TransformerTextDecoder(nn.Module):
    def __init__(self, latent_dim, gpt2_model_name='gpt2'):
        super(TransformerTextDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        # Linear layer to project latent vector to GPT-2 input dimension if they do not match
        self.latent_to_gpt2 = nn.Linear(latent_dim, self.gpt2.config.n_embd)

    def forward(self, z, input_ids=None, attention_mask=None):
        # Project the latent vector to match GPT-2's expected input dimension
        gpt2_input = self.latent_to_gpt2(z)
        # Reshape to (batch_size, sequence_length, hidden_size)
        gpt2_input = gpt2_input.unsqueeze(1).expand(-1, input_ids.size(1), -1)
        
        # Get the GPT-2 outputs
        gpt2_outputs = self.gpt2(inputs_embeds=gpt2_input, attention_mask=attention_mask)
        
        # The output tuple includes the language modeling logits
        lm_logits = gpt2_outputs.logits
        return lm_logits

class MultimodalVAEDecoder(nn.Module):
    def __init__(self, latent_dim, img_channels, vocab_size):
        super(MultimodalVAEDecoder, self).__init__()
        self.image_decoder = ImageDecoder(latent_dim, img_channels)
        self.text_decoder =  TransformerTextDecoder(latent_dim)
    
    def forward(self, z):
        img_recon = self.image_decoder(z)
        text_logits = self.text_decoder(z)
        return img_recon, text_logits

class MultimodalVAE(nn.Module):

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def __init__(self, image_model_name, text_model_name, latent_dim, img_channels, vocab_size):
        super(MultimodalVAE, self).__init__()

        # Load the pretrained models
        self.image_encoder = timm.create_model(image_model_name, pretrained=True)
        self.text_encoder = BertModel.from_pretrained(text_model_name)

        # Assuming the image encoder and text encoder have different output dimensions,
        # we'll map them to a common latent space dimension `latent_dim`
        self.image_latent = nn.Linear(self.image_encoder.get_classifier().in_features, latent_dim)
        self.text_latent = nn.Linear(self.text_encoder.config.hidden_size, latent_dim)

        # Replace the classifier head with an identity mapping, we don't need classification here
        self.image_encoder.reset_classifier(0, '')

        # Decoder
        self.decoder = MultimodalVAEDecoder(latent_dim, img_channels, vocab_size)

    def forward(self, images, input_ids, attention_mask):
        # Encoder steps
        image_features = self.image_encoder(images)
        image_mu = self.image_latent(image_features)
        # Placeholder for image log variance - should be produced by an encoder
        image_logvar = torch.zeros_like(image_mu)  

        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_mu = self.text_latent(text_features)
        # Placeholder for text log variance - should be produced by an encoder
        text_logvar = torch.zeros_like(text_mu)  

        # Product of Experts to combine latent representations
        combined_mu, combined_logvar = self.product_of_experts((image_mu, image_logvar), (text_mu, text_logvar))

        # Reparameterization step using combined mu and logvar
        z = self.reparameterize(combined_mu, combined_logvar)

        # Decode the latent vector into the respective modalities
        img_recon, text_logits = self.decoder(z)

        return img_recon, text_logits, combined_mu, combined_logvar


    def product_of_experts(self, image_latent, text_latent):
        # Unpack the mean and logvar from the latent variables
        image_mean, image_logvar = image_latent
        text_mean, text_logvar = text_latent

        # Convert log variance to precision
        image_precision = torch.exp(-image_logvar)
        text_precision = torch.exp(-text_logvar)

        # Combine precisions and means for each expert
        combined_precision = image_precision + text_precision
        combined_mean = (image_mean * image_precision + text_mean * text_precision) / combined_precision

        # Calculate combined logvar
        combined_logvar = torch.log(1.0 / combined_precision)

        return combined_mean, combined_logvar

    def generate_text_autoregressively(self, z, tokenizer, max_length=50):
        """
        Generates text from the latent vector autoregressively using nucleus sampling.
        """
        device = z.device
        generated_sequence = torch.full((z.size(0), 1), tokenizer.bos_token_id, dtype=torch.long).to(device)
        
        for _ in range(max_length):
            # Decode the latent vector concatenated with the generated tokens so far
            text_logits = self.decoder.text_decoder(z, generated_sequence).logits[:, -1, :]
            
            # Convert logits to probabilities
            probabilities = F.softmax(text_logits, dim=-1)
            
            # Apply top-p (nucleus) filtering
            filtered_probabilities = top_k_top_p_filtering(probabilities, top_p=0.9)
            
            # Sample the next token from the filtered distribution
            next_token = torch.multinomial(F.softmax(filtered_probabilities, dim=-1), num_samples=1)
            
            # Append sampled token to the generated sequence
            generated_sequence = torch.cat((generated_sequence, next_token), dim=1)
            
            # Check if the end of sequence token was generated
            if next_token in tokenizer.all_special_ids:
                break

        # Decode the generated sequence into text
        generated_text = tokenizer.batch_decode(generated_sequence, skip_special_tokens=True)
        return generated_text
    
# Training loop
def train(model, data_loader, optimizer, epochs, tokenizer):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (images, texts) in enumerate(data_loader):
            text_inputs = tokenizer(texts, padding=True, return_tensors='pt')

            optimizer.zero_grad()

            # Forward pass through the model
            img_recon, text_logits, latent, image_mu, image_logvar, text_mu, text_logvar = model(images, text_inputs['input_ids'], text_inputs['attention_mask'])

            # Compute the reconstruction losses
            img_loss = F.mse_loss(img_recon, images)
            text_loss = F.cross_entropy(text_logits.view(-1, text_logits.size(-1)), text_inputs['input_ids'].view(-1))

            # Compute the KL divergence for each latent representation
            image_kl_divergence = -0.5 * torch.sum(1 + image_logvar - image_mu.pow(2) - image_logvar.exp())
            text_kl_divergence = -0.5 * torch.sum(1 + text_logvar - text_mu.pow(2) - text_logvar.exp())

            # Total loss
            loss = img_loss + text_loss + image_kl_divergence + text_kl_divergence

            # Backpropagation
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')


if __name__=="__main__":
    # Example data
    images = torch.randn(2, 3, 224, 224)
    texts = ["hello world", "variational autoencoder"]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    multi_modal_vae = MultimodalVAE(
        image_model_name='resnet50',
        text_model_name='bert-base-uncased',
        latent_dim=512,
        img_channels=3,
        vocab_size=tokenizer.vocab_size
    )
    # Assume data_loader and optimizer are defined
    # train(multi_modal_vae, data_loader, optimizer, epochs=10, tokenizer=tokenizer)

    # For demonstration purposes, we'll just do a forward pass here
    text_inputs = tokenizer(texts, padding=True, return_tensors='pt')
    img_recon, text_logits, latent, image_mu, image_logvar, text_mu, text_logvar = multi_modal_vae(images, text_inputs['input_ids'], text_inputs['attention_mask'])
    print("Latent representation:", latent)