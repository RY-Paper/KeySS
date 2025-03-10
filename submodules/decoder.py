import torch
import torch.nn as nn
import torch.nn.functional as F
#for single secret hiding 
class decoder_all_prompt2(nn.Module):
    def __init__(self, fealist=["sh", "rotation", "scale", "opacity", "xyz"], mean_scaling=0.03):
        super(decoder_all_prompt2, self).__init__()
        fea_dict = {
            "sh": 48,
            "rotation": 4,
            "scale": 3, 
            "opacity": 1,
            "xyz": 3
        }
        self.fealist = fealist
        input_dim = 0
        for f in fea_dict:
            input_dim += fea_dict[f]
        input_dim += 768 # Add 1 dimension for constant feature
        
        # self.prompt_embedding = nn.Conv1d(in_channels=768, out_channels=10, kernel_size=1)
        
        # Define 1D convolutional layers with ReLU
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)

        #scale
        if "scale" in fealist:
            self.scale = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)
            self.scale_out = nn.Conv1d(in_channels=128, out_channels=3, kernel_size=1)
            self.scaling_activation = torch.exp
        
        #opacity
        if "opacity" in fealist:
            self.opacity = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)
            self.opacity_out = nn.Conv1d(in_channels=128, out_channels=1, kernel_size=1)
            self.opacity_activation = torch.sigmoid
        
        #sh
        if "sh" in fealist:
            self.sh = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)
            self.sh_out = nn.Conv1d(in_channels=128, out_channels=48, kernel_size=1)
        
        #rotation
        if "rotation" in fealist:
            self.rotation = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)
            self.rotation_out = nn.Conv1d(in_channels=128, out_channels=4, kernel_size=1)
            self.rotation_activation = lambda x: F.normalize(x, dim=1, eps=1e-16)
            
        #xyz
        if "xyz" in fealist:
            self.xyz = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)
            self.xyz_out = nn.Conv1d(in_channels=128, out_channels=3, kernel_size=1)
        
        #activation
        self.activation = F.relu

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, prompt, x, scales, opacity, sh, rotation, xyz, sh_mask=None):
        try:
            # Process prompt - input shape (1, 768), output shape (1, 10, 1)
            constant_tensor = prompt #self.activation(self.prompt_embedding(prompt))  # (1, 10, 1)
            
            # Concatenate features along channel dimension
            # x shape: (N, 59, 1)
            # constant_tensor shape: (1, 10, 1) -> expand to (N, 10, 1)
            features = torch.cat([
                x,  # (N, 59, 1)
                constant_tensor.expand(x.shape[0], -1, -1)  # (N, 10, 1)
            ], dim=1)  # Result: (N, 69, 1)
            
            # Shared backbone
            features = self.activation(self.conv1(features))  # (N, 128, 1)
            features = self.activation(self.conv2(features))  # (N, 128, 1)
            
            output_dict = {}
            
            # Process each feature independently
            if "scale" in self.fealist:
                x_scale = self.activation(self.scale(features))
                x_scale = self.activation(self.scale_out(x_scale))
                output_dict["scale"] = self.scaling_activation(x_scale.squeeze(-1) + scales)
            
            if "opacity" in self.fealist:
                x_opacity = self.activation(self.opacity(features))
                x_opacity = self.opacity_out(x_opacity)
                output_dict["opacity"] = self.opacity_activation(x_opacity.squeeze(-1) + opacity)
            
            if "sh" in self.fealist:
                x_sh = self.activation(self.sh(features))
                x_sh = self.activation(self.sh_out(x_sh)) #*sh_mask.view(-1,48,1)
                output_dict["sh"] = x_sh.squeeze(-1) + sh
                # #test
                # output_dict["sh"] = x_sh
            
            if "rotation" in self.fealist:
                x_rotation = self.activation(self.rotation(features))
                x_rotation = self.activation(self.rotation_out(x_rotation))
                output_dict["rotation"] = self.rotation_activation(x_rotation.squeeze(-1) + rotation)
            
            if "xyz" in self.fealist:
                x_xyz = self.activation(self.xyz(features))
                x_xyz = self.activation(self.xyz_out(x_xyz))
                output_dict["xyz"] = x_xyz.squeeze(-1) + xyz
            
            return output_dict
            
        except Exception as e:
            print("\nError in decoder forward pass:", str(e))
            print(f"Shapes: prompt={prompt.shape}, x={x.shape}, scales={scales.shape}, "
                  f"opacity={opacity.shape}, sh={sh.shape}, rotation={rotation.shape}, xyz={xyz.shape}")
            raise e

        
#for single secret hiding with sh feature, due to gpu limit, please run the following command:
class decoder_all_forloop(decoder_all_prompt2):
    def __init__(self, fealist=["sh", "rotation", "scale", "opacity", "xyz"], mean_scaling=0.03, batchsize=200000):
        super().__init__(fealist, mean_scaling)
        self.batchsize = batchsize
    
    def single_forward(self, prompt, x, scales, opacity, sh, rotation, xyz, sh_mask=None):
        # try:
        # Process prompt - input shape (1, 768), output shape (1, 10, 1)
        constant_tensor = prompt #self.activation(self.prompt_embedding(prompt))  # (1, 10, 1)
        
        # Concatenate features along channel dimension
        # x shape: (N, 59, 1)
        # constant_tensor shape: (1, 10, 1) -> expand to (N, 10, 1)
        features = torch.cat([
            x,  # (N, 59, 1)
            constant_tensor.expand(x.shape[0], -1, -1)  # (N, 10, 1)
        ], dim=1)  # Result: (N, 69, 1)
        
        # Shared backbone
        features = self.activation(self.conv1(features))  # (N, 128, 1)
        features = self.activation(self.conv2(features))  # (N, 128, 1)
        
        # output_scale, output_opacity, output_rotation, output_xyz, output_sh = None,None,None,None,None
        output_dict = {}
        
        # Process each feature independently
        if "scale" in self.fealist:
            x_scale = self.activation(self.scale(features))
            x_scale = self.activation(self.scale_out(x_scale))
            output_dict["scale"] = self.scaling_activation(x_scale.squeeze(-1) + scales)
        
        if "opacity" in self.fealist:
            x_opacity = self.activation(self.opacity(features))
            x_opacity = self.opacity_out(x_opacity)
            output_dict["opacity"] = self.opacity_activation(x_opacity.squeeze(-1) + opacity)
        
        if "sh" in self.fealist:
            x_sh = self.activation(self.sh(features))
            x_sh = self.sh_out(x_sh) #*sh_mask.view(-1,48,1)
            output_dict["sh"] = x_sh.squeeze(-1) + sh
            # #test
            # output_dict["sh"] = x_sh
        
        if "rotation" in self.fealist:
            x_rotation = self.activation(self.rotation(features))
            x_rotation = self.activation(self.rotation_out(x_rotation))
            output_dict["rotation"] = self.rotation_activation(x_rotation.squeeze(-1) + rotation)
        
        if "xyz" in self.fealist:
            x_xyz = self.activation(self.xyz(features))
            x_xyz = self.activation(self.xyz_out(x_xyz))
            output_dict["xyz"] = x_xyz.squeeze(-1) + xyz
        
        return output_dict

    def forward(self, prompt, x, scales, opacity, sh, rotation, xyz, shs_masked=None):
        n = x.shape[0]
        # Process batches sequentially to save memory
        batches = torch.arange(n).split(self.batchsize)
        
        # Initialize outputs as empty lists to save memory
        outputs = {
            "scale": [],
            "opacity": [],
            "rotation": [],
            "xyz": [],
            "sh": []
        }
        
        for _, batch_indices in enumerate(batches):
            # Process batch
            batch_outputs = self.single_forward(prompt,
                               x[batch_indices],
                               scales[batch_indices], 
                               opacity[batch_indices],
                               sh[batch_indices], 
                               rotation[batch_indices],
                               xyz[batch_indices]
                               )
            
            # Append batch results to lists
            for key in outputs:
                if key in batch_outputs and batch_outputs[key] is not None:
                    outputs[key].append(batch_outputs[key])
            
            # Clear CUDA cache after each batch
            torch.cuda.empty_cache()
        
        # Concatenate results only at the end
        final_outputs = {}
        for key in outputs:
            if outputs[key]:  # Only concatenate if we have outputs for this key
                final_outputs[key] = torch.cat(outputs[key], dim=0)
        
        return final_outputs
