import torch
from torch import nn

from diffusers import StableDiffusionPipeline

class VAELoss(nn.Module):
  def __init__(self):
    super(VAELoss, self).__init__()
  
  def forward(self, predict, target, mu, logvar):
    recon_loss = nn.functional.binary_cross_entropy(predict, target, reduction='sum')
    kl_loss = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + kl_loss

    return loss

class SDSLoss(nn.Module):
  def __init__(self, prompt, device):
    super(SDSLoss, self).__init__()
    self.prompt = prompt
    self.device = device

    self.pipe = StableDiffusionPipeline.from_pretrained(
      "runwayml/stable-diffusion-v1-5",
      torch_dtype=torch.float16,
    )
    self.pipe = self.pipe.to(self.device)
    self.alphas = self.pipe.scheduler.alphas_cumprod.to(self.device)
    self.sigmas = (1 - self.pipe.scheduler.alphas_cumprod).to(self.device)

    self.text_embeddings = None
    self.embed_text()

  def embed_text(self):
    text_input = self.pipe.tokenizer(
      self.prompt,
      padding='max_length',
      max_length=self.pipe.tokenizer.model_max_length,
      truncation=True,
      return_tensors='pt'
    )
    uncond_input = self.pipe.tokenizer(
      [''],
      padding='max_length',
      max_length=text_input.input_ids.shape[-1],
      return_tensors='pt'
    )

    with torch.no_grad():
      text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]
      uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]
    self.text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    self.text_embeddings = self.text_embeddings.repeat_interleave(1, 0)
    
    del self.pipe.tokenizer
    del self.pipe.text_encoder

  def forward(self, image):
    sds_loss = 0

    # encode rendered image
    x = image.repeat(1, 3, 1, 1) * 2.0 - 1.0
    with torch.cuda.amp.autocast():
      init_latent_z = (self.pipe.vae.encode(x).latent_dist.sample())
    latent_z = 0.18215 * init_latent_z  # scaling_factor * init_latents

    with torch.inference_mode():
      # sample timesteps
      timestep = torch.randint(
        low=50,
        high=950,
        size=(latent_z.shape[0],),
        device=self.device,
        dtype=torch.long
      )

      # add noise
      eps = torch.randn_like(latent_z)
      # zt = alpha_t * latent_z + sigma_t * eps
      noised_latent_zt = self.pipe.scheduler.add_noise(latent_z, eps, timestep)

      # denoise
      z_in = torch.cat([noised_latent_zt] * 2)  # expand latents for classifier free guidance
      timestep_in = torch.cat([timestep] * 2)
      with torch.autocast(device_type="cuda", dtype=torch.float16):
        eps_t_uncond, eps_t = self.pipe.unet(
          z_in, 
          timestep_in, 
          encoder_hidden_states=self.text_embeddings).sample.float().chunk(2)
        
      eps_t = eps_t_uncond + 100.0 * (eps_t - eps_t_uncond)

      # w = alphas[timestep]^0.5 * (1 - alphas[timestep]) = alphas[timestep]^0.5 * sigmas[timestep]
      grad_z = self.alphas[timestep]**0.5 * self.sigmas[timestep] * (eps_t - eps)
      assert torch.isfinite(grad_z).all()
      grad_z = torch.nan_to_num(grad_z.detach().float(), 0.0, 0.0, 0.0)

    sds_loss = grad_z.clone() * latent_z
    del grad_z

    sds_loss = sds_loss.sum(1).mean()
    return sds_loss
