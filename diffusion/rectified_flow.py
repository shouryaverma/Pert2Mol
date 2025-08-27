import torch as th
import numpy as np
from typing import Dict, Any, Optional


class RectifiedFlow:
    """
    Rectified Flow implementation for straight-line flow from noise to data.
    
    Key differences from diffusion:
    - Uses linear interpolation: x_t = (1-t)*x_0 + t*x_1 where x_0~N(0,I), x_1 is data
    - Model predicts velocity field: v_t = x_1 - x_0
    - No noise schedule - just uniform time sampling
    """
    
    def __init__(self, num_timesteps: int = 1000):
        self.num_timesteps = num_timesteps
        
    def sample_time(self, batch_size: int, device: th.device) -> th.Tensor:
        """Sample random timesteps uniformly from [0, 1]"""
        return th.rand(batch_size, device=device)
    
    def interpolate(self, x_0: th.Tensor, x_1: th.Tensor, t: th.Tensor) -> th.Tensor:
        """
        Linear interpolation between noise x_0 and data x_1.
        x_t = (1-t)*x_0 + t*x_1
        
        Args:
            x_0: noise tensor [B, C, H, W]  
            x_1: data tensor [B, C, H, W]
            t: time tensor [B] with values in [0, 1]
        """
        t = t.view(-1, 1, 1, 1)  # Reshape for broadcasting
        return (1 - t) * x_0 + t * x_1
    
    def compute_velocity(self, x_0: th.Tensor, x_1: th.Tensor) -> th.Tensor:
        """
        Compute the true velocity field: v = x_1 - x_0
        
        Args:
            x_0: noise tensor [B, C, H, W]
            x_1: data tensor [B, C, H, W]  
        """
        return x_1 - x_0
    
    def training_losses(self, model, x_start: th.Tensor, t: Optional[th.Tensor] = None, 
                       model_kwargs: Optional[Dict[str, Any]] = None, 
                       noise: Optional[th.Tensor] = None) -> Dict[str, th.Tensor]:
        """
        Compute rectified flow training loss.
        
        Args:
            model: velocity prediction model
            x_start: clean data tensor [B, C, H, W]
            t: time tensor [B] (if None, will be sampled)
            model_kwargs: conditioning arguments for model
            noise: noise tensor [B, C, H, W] (if None, will be sampled)
            
        Returns:
            Dictionary with 'loss' key containing MSE loss
        """
        if model_kwargs is None:
            model_kwargs = {}
            
        if noise is None:
            noise = th.randn_like(x_start)
            
        if t is None:
            t = self.sample_time(x_start.shape[0], x_start.device)
        
        # Convert t to discrete timesteps for model (if model expects discrete steps)
        t_discrete = (t * (self.num_timesteps - 1)).long()
        
        # Linear interpolation: x_t = (1-t)*noise + t*data
        x_t = self.interpolate(noise, x_start, t)
        
        # True velocity field: v = data - noise  
        true_velocity = self.compute_velocity(noise, x_start)
        
        # Model prediction
        model_output = model(x_t, t_discrete, **model_kwargs)
        # Handle learn_sigma case - split output if it's double the input channels
        if model_output.shape[1] == 2 * x_start.shape[1]:
            predicted_velocity, _ = th.split(model_output, x_start.shape[1], dim=1)
        else:
            predicted_velocity = model_output
        
        # MSE loss between predicted and true velocity
        mse_loss = th.nn.functional.mse_loss(predicted_velocity, true_velocity, reduction='none')
        mse_loss = mse_loss.mean(dim=list(range(1, len(mse_loss.shape))))  # Mean over all dims except batch
        
        return {
            'loss': mse_loss,
            'mse': mse_loss  # For compatibility with diffusion interface
        }
    
    def sample_step(self, model, x_t: th.Tensor, t: th.Tensor, dt: float,
                   model_kwargs: Optional[Dict[str, Any]] = None) -> th.Tensor:
        """
        Single Euler step for sampling: x_{t+dt} = x_t + dt * v_theta(x_t, t)
        
        Args:
            model: velocity prediction model
            x_t: current state [B, C, H, W]
            t: current time [B]
            dt: step size
            model_kwargs: conditioning arguments
        """
        if model_kwargs is None:
            model_kwargs = {}
            
        # Convert continuous time to discrete timesteps  
        t_discrete = (t * (self.num_timesteps - 1)).long()
        
        # Predict velocity
        with th.no_grad():
            velocity = model(x_t, t_discrete, **model_kwargs)
        
        # Euler step
        x_next = x_t + dt * velocity
        return x_next
    
    def sample(self, model, shape: tuple, num_steps: int = 50, 
              model_kwargs: Optional[Dict[str, Any]] = None,
              device: Optional[th.device] = None) -> th.Tensor:
        """
        Generate samples using Euler method integration.
        
        Args:
            model: velocity prediction model
            shape: output shape [B, C, H, W]
            num_steps: number of integration steps
            model_kwargs: conditioning arguments
            device: device for computation
        """
        if device is None:
            device = next(model.parameters()).device
            
        if model_kwargs is None:
            model_kwargs = {}
        
        # Start from noise
        x = th.randn(*shape, device=device)
        
        # Integration from t=0 to t=1
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = th.full((shape[0],), i * dt, device=device)
            x = self.sample_step(model, x, t, dt, model_kwargs)
            
        return x
    
    def sample_loop(self, model, shape: tuple, num_steps: int = 50,
                   model_kwargs: Optional[Dict[str, Any]] = None,
                   device: Optional[th.device] = None,
                   progress: bool = False) -> th.Tensor:
        """Sample with optional progress bar (for compatibility with diffusion interface)"""
        if progress:
            try:
                from tqdm.auto import tqdm
                steps = tqdm(range(num_steps))
            except ImportError:
                steps = range(num_steps)
        else:
            steps = range(num_steps)
            
        if device is None:
            device = next(model.parameters()).device
            
        if model_kwargs is None:
            model_kwargs = {}
            
        x = th.randn(*shape, device=device) 
        dt = 1.0 / num_steps
        
        for i in steps:
            t = th.full((shape[0],), i * dt, device=device)
            x = self.sample_step(model, x, t, dt, model_kwargs)
            
        return x


def create_rectified_flow(num_timesteps: int = 1000) -> RectifiedFlow:
    """Factory function to create RectifiedFlow (matches diffusion create_diffusion interface)"""
    return RectifiedFlow(num_timesteps=num_timesteps)