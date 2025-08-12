import cv2
import numpy as np
from config.config import (
    TERRAIN_CONFIG, CUSTOM_MOUNTAINS, SMOOTHING_CONFIG, 
    EXPORT_CONFIG, DISPLAY_CONFIG, HIGH_RES_CONFIG, ADVANCED_CONFIG, RIVER_CONFIG, ROAD_CONFIG
)

def use_convolution_cv2(data, kernel_type='gaussian', kernel_size=5, sigma=1.0, custom_kernel=None):
    """
    Apply convolution to smooth the image data using OpenCV.

    Parameters:
    - data: 2D array-like structure to be convolved.
    - kernel_type: Type of smoothing kernel ('gaussian', 'box', 'mean', 'custom').
    - kernel_size: Size of the kernel (must be odd number).
    - sigma: Standard deviation for Gaussian kernel.
    - custom_kernel: Custom kernel array (used when kernel_type='custom').

    Returns:
    - Smoothed data as a 2D array.
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Convert to float32 for OpenCV processing
    data_float = data.astype(np.float32)
    
    if custom_kernel is not None:
        kernel = custom_kernel.astype(np.float32)
        smoothed_data = cv2.filter2D(data_float, -1, kernel)
    elif kernel_type == 'gaussian':
        # Use OpenCV's Gaussian blur
        smoothed_data = cv2.GaussianBlur(data_float, (kernel_size, kernel_size), sigma)
    elif kernel_type == 'box' or kernel_type == 'mean':
        # Use OpenCV's box filter
        smoothed_data = cv2.boxFilter(data_float, -1, (kernel_size, kernel_size))
    elif kernel_type == 'bilateral':
        # Bilateral filter for edge-preserving smoothing
        smoothed_data = cv2.bilateralFilter(data_float, kernel_size, sigma*50, sigma*50)
    elif kernel_type == 'median':
        # Median filter for noise reduction
        smoothed_data = cv2.medianBlur(data_float, kernel_size)
    else:
        raise ValueError("Unsupported kernel type. Use 'gaussian', 'box', 'mean', 'bilateral', 'median', or 'custom'.")
    
    return smoothed_data

def create_gaussian_kernel_cv2(size, sigma):
    """
    Create a 2D Gaussian kernel using OpenCV.
    
    Parameters:
    - size: Size of the kernel (must be odd).
    - sigma: Standard deviation of the Gaussian.
    
    Returns:
    - 2D Gaussian kernel normalized to sum to 1.
    """
    return cv2.getGaussianKernel(size, sigma) @ cv2.getGaussianKernel(size, sigma).T

def create_simple_mountain_cv2(X, Y, x_center, y_center, height, width, mountain_type='varied'):
    """
    Create simple mountains with basic shapes (original method).
    
    Parameters:
    - X, Y: Coordinate meshgrids
    - x_center, y_center: Mountain center coordinates
    - height: Mountain height
    - width: Base width of mountain
    - mountain_type: Type of mountain ('peaked', 'ridge', 'mesa', 'volcano', 'varied')
    
    Returns:
    - Mountain height data with basic shapes
    """
    # Basic distance calculations
    dx = X - x_center
    dy = Y - y_center
    distance = np.sqrt(dx**2 + dy**2)
    
    # Normalize distance for shape calculations
    norm_distance = distance / width
    
    if mountain_type == 'varied':
        # Random selection of mountain type
        mountain_type = np.random.choice(['peaked', 'ridge', 'mesa', 'volcano', 'asymmetric'])
    
    if mountain_type == 'peaked':
        # Sharp peaked mountain with steep cliffs (simple version)
        base_shape = np.exp(-norm_distance**1.2)
        
        # Add cliff faces on random sides
        cliff_angle = np.random.uniform(0, 2*np.pi)
        cliff_direction = np.cos(cliff_angle) * dx + np.sin(cliff_angle) * dy
        cliff_mask = cliff_direction > 0
        base_shape[cliff_mask] *= np.exp(-norm_distance[cliff_mask]**1.0)
        
        # Add rocky texture
        rocky_noise = np.random.normal(0, 0.05, X.shape)
        rocky_texture = cv2.GaussianBlur(rocky_noise.astype(np.float32), (3, 3), 0.8)
        base_shape += rocky_texture * base_shape * 0.1
        
    elif mountain_type == 'ridge':
        # Ridge-like mountain with elongated shape
        ridge_angle = np.random.uniform(0, np.pi)
        ridge_dx = np.cos(ridge_angle) * dx + np.sin(ridge_angle) * dy
        ridge_dy = -np.sin(ridge_angle) * dx + np.cos(ridge_angle) * dy
        
        # Create ridge profile
        ridge_distance = np.sqrt((ridge_dx/width)**2 + (ridge_dy/(width*0.4))**2)
        base_shape = np.exp(-ridge_distance**1.0)
        
        # Add sharp edges along ridge
        ridge_sharpness = np.exp(-np.abs(ridge_dy)/(width*0.15))
        base_shape += ridge_sharpness * 0.2 * np.exp(-ridge_distance)
        
    elif mountain_type == 'mesa':
        # Mesa/plateau with steep sides
        plateau_distance = norm_distance
        
        # Flat top with steep edges
        base_shape = np.where(plateau_distance < 0.7,
                             1.0 - (plateau_distance/0.7)**3,
                             np.exp(-(plateau_distance-0.7)**2 * 5))
        
        # Add erosion channels
        erosion_angle = np.random.uniform(0, 2*np.pi)
        erosion_channels = np.sin(4 * (np.arctan2(dy, dx) + erosion_angle))
        channel_mask = (erosion_channels > 0.8) & (plateau_distance > 0.5)
        base_shape[channel_mask] *= 0.5
        
    elif mountain_type == 'volcano':
        # Volcanic cone with crater
        base_shape = np.exp(-norm_distance**1.3)
        
        # Add crater at top
        crater_mask = norm_distance < 0.2
        base_shape[crater_mask] *= (1 - np.exp(-(norm_distance[crater_mask]/0.1)**2))
        
        # Add lava flow patterns
        flow_angle = np.random.uniform(0, 2*np.pi)
        flow_direction = np.cos(flow_angle) * dx + np.sin(flow_angle) * dy
        flow_mask = (flow_direction > 0) & (norm_distance < 1.5)
        base_shape[flow_mask] += 0.2 * np.exp(-norm_distance[flow_mask]**0.5)
        
    elif mountain_type == 'asymmetric':
        # Asymmetric mountain with different slopes on each side
        base_shape = np.exp(-norm_distance**1.2)
        
        # Create different slopes for different directions
        angle_from_center = np.arctan2(dy, dx)
        slope_variation = 1 + 0.8 * np.sin(3 * angle_from_center + np.random.uniform(0, 2*np.pi))
        base_shape *= np.exp(-norm_distance * slope_variation * 0.5)
        
        # Add avalanche/scree slopes
        scree_angle = np.random.uniform(0, 2*np.pi)
        scree_direction = np.cos(scree_angle) * dx + np.sin(scree_angle) * dy
        scree_mask = (scree_direction > 0) & (norm_distance > 0.5)
        base_shape[scree_mask] *= 0.7
    
    else:
        # Default Gaussian shape
        base_shape = np.exp(-norm_distance**2)
    
    # Apply height and add final details
    mountain = height * base_shape
    
    # Add fine detail noise
    if np.max(mountain) > 0.1:
        detail_intensity = np.clip(height / 3.0, 0.02, 0.1)
        
        # Create detail pattern
        detail_x = np.linspace(-2, 2, X.shape[1])
        detail_y = np.linspace(-2, 2, X.shape[0])
        detail_X, detail_Y = np.meshgrid(detail_x, detail_y)
        
        # Use multiple scales of detail
        detail_noise = (
            0.4 * np.sin(detail_X * 8 + np.random.uniform(0, 2*np.pi)) * np.cos(detail_Y * 8 + np.random.uniform(0, 2*np.pi)) +
            0.3 * np.sin(detail_X * 16 + np.random.uniform(0, 2*np.pi)) * np.cos(detail_Y * 16 + np.random.uniform(0, 2*np.pi)) +
            0.3 * np.random.normal(0, 0.5, X.shape)
        )
        
        # Apply light smoothing
        detail_texture = cv2.GaussianBlur(detail_noise.astype(np.float32), (3, 3), 0.3)
        
        # Apply detail only where mountain exists
        mountain_mask = base_shape > 0.05
        mountain[mountain_mask] += detail_texture[mountain_mask] * mountain[mountain_mask] * detail_intensity
    
    # Ensure no negative values
    mountain = np.maximum(mountain, 0)
    
    return mountain

def create_complex_mountain_cv2(X, Y, x_center, y_center, height, width, mountain_type='varied'):
    """
    Create realistic mountains with proper geological features and natural proportions.
    
    Parameters:
    - X, Y: Coordinate meshgrids
    - x_center, y_center: Mountain center coordinates
    - height: Mountain height
    - width: Base width of mountain
    - mountain_type: Type of mountain ('peaked', 'ridge', 'mesa', 'volcano', 'varied')
    
    Returns:
    - Mountain height data with realistic geological features
    """
    # Basic distance calculations
    dx = X - x_center
    dy = Y - y_center
    distance = np.sqrt(dx**2 + dy**2)
    
    # Ensure minimum width for natural-looking mountains
    effective_width = max(width, 1.0)  # Minimum width of 1.0 unit
    
    # Normalize distance for shape calculations
    norm_distance = distance / effective_width
    
    if mountain_type == 'varied':
        # Random selection of mountain type with weights for better results
        mountain_type = np.random.choice(['peaked', 'ridge', 'mesa', 'volcano', 'asymmetric'], 
                                       p=[0.35, 0.25, 0.15, 0.10, 0.15])
    
    # Create base mountain shape with natural curves and irregularities
    angle = np.arctan2(dy, dx)
    
    # Create natural, curved mountain boundaries using multiple harmonics
    # This creates organic, flowing shapes instead of straight lines
    boundary_curves = (
        0.20 * np.sin(2 * angle + np.random.uniform(0, 2*np.pi)) +
        0.15 * np.sin(3 * angle + np.random.uniform(0, 2*np.pi)) +
        0.10 * np.sin(5 * angle + np.random.uniform(0, 2*np.pi)) +
        0.08 * np.sin(7 * angle + np.random.uniform(0, 2*np.pi)) +
        0.05 * np.sin(11 * angle + np.random.uniform(0, 2*np.pi))
    )
    
    # Add some noise for natural irregularity
    fine_noise = np.random.normal(0, 0.05, X.shape)
    fine_noise = cv2.GaussianBlur(fine_noise.astype(np.float32), (5, 5), 1.0)
    
    # Apply gentle distortion to break up circular patterns and create curves
    distorted_distance = norm_distance * (1 + 0.25 * boundary_curves + 0.1 * fine_noise)
    
    if mountain_type == 'peaked':
        # Alpine peak with natural slopes
        # Create main mountain body
        base_shape = np.exp(-distorted_distance**1.0)  # Gentler falloff
        
        # Add subtle ridges from peak
        num_ridges = np.random.randint(2, 4)
        for i in range(num_ridges):
            ridge_angle = (2 * np.pi * i / num_ridges) + np.random.uniform(-0.3, 0.3)
            ridge_direction = angle - ridge_angle
            ridge_alignment = np.cos(ridge_direction)
            
            # Ridge effect only in positive direction and near center
            ridge_mask = (ridge_alignment > 0.7) & (distorted_distance < 0.8)
            if np.any(ridge_mask):
                ridge_strength = 0.3 * ridge_alignment[ridge_mask] * np.exp(-distorted_distance[ridge_mask] * 2)
                base_shape[ridge_mask] += ridge_strength
        
        # Add one steeper cliff face
        cliff_angle = np.random.uniform(0, 2*np.pi)
        cliff_direction = angle - cliff_angle
        cliff_mask = (np.abs(cliff_direction) < np.pi/4) & (distorted_distance > 0.4) & (distorted_distance < 1.2)
        base_shape[cliff_mask] *= np.exp(-distorted_distance[cliff_mask]**0.6)  # Steeper on cliff side
        
    elif mountain_type == 'ridge':
        # Mountain ridge with elongated shape
        ridge_angle = np.random.uniform(0, np.pi)
        
        # Transform coordinates to ridge-aligned system
        ridge_dx = np.cos(ridge_angle) * dx + np.sin(ridge_angle) * dy
        ridge_dy = -np.sin(ridge_angle) * dx + np.cos(ridge_angle) * dy
        
        # Create ridge profile - elongated in one direction
        ridge_length_factor = np.random.uniform(2.0, 4.0)  # How elongated the ridge is
        ridge_distance = np.sqrt((ridge_dx/(effective_width * ridge_length_factor))**2 + 
                                (ridge_dy/(effective_width * 0.6))**2)
        
        base_shape = np.exp(-ridge_distance**0.9)
        
        # Add sharp crest line along the ridge
        crest_width = effective_width * 0.15
        crest_mask = np.abs(ridge_dy) < crest_width
        crest_enhancement = 0.4 * np.exp(-np.abs(ridge_dy) / crest_width) * np.exp(-ridge_distance)
        base_shape[crest_mask] += crest_enhancement[crest_mask]
        
        # Add side slopes with different steepness
        left_side = ridge_dy < 0
        right_side = ridge_dy > 0
        base_shape[left_side] *= np.exp(-0.1 * (np.abs(ridge_dy[left_side]) / effective_width))
        base_shape[right_side] *= np.exp(-0.2 * (np.abs(ridge_dy[right_side]) / effective_width))
        
    elif mountain_type == 'mesa':
        # Mesa/plateau with steep sides
        plateau_size = np.random.uniform(0.4, 0.7)  # Size of flat top relative to base
        
        # Create flat-topped mountain
        base_shape = np.where(distorted_distance < plateau_size,
                             1.0,  # Flat top
                             np.where(distorted_distance < plateau_size + 0.3,
                                     # Steep sides - avoid negative values under square root
                                     np.exp(-np.abs((distorted_distance - plateau_size) / 0.3)**0.5),
                                     0))
        
        # Add some erosion channels cutting into the mesa
        num_channels = np.random.randint(1, 3)
        for _ in range(num_channels):
            channel_angle = np.random.uniform(0, 2*np.pi)
            channel_width = np.random.uniform(np.pi/8, np.pi/6)
            
            angle_diff = np.abs(((angle - channel_angle + np.pi) % (2*np.pi)) - np.pi)
            channel_mask = (angle_diff < channel_width) & (distance > effective_width * plateau_size)
            
            if np.any(channel_mask):
                # Gentle channel cut
                channel_depth = 0.4 * (1 - np.clip((distance[channel_mask] - effective_width * plateau_size) / 
                                                  (effective_width * 0.5), 0, 1))
                base_shape[channel_mask] *= (1 - channel_depth)
        
    elif mountain_type == 'volcano':
        # Volcanic cone with crater
        base_shape = np.exp(-distorted_distance**1.1)
        
        # Create crater at summit
        crater_size = np.random.uniform(0.08, 0.15)
        crater_mask = distorted_distance < crater_size
        if np.any(crater_mask):
            crater_depth = 0.5 * (1 - (distorted_distance[crater_mask] / crater_size)**2)
            base_shape[crater_mask] *= (1 - crater_depth)
        
        # Add some lava flow ridges
        num_flows = np.random.randint(1, 3)
        for _ in range(num_flows):
            flow_angle = np.random.uniform(0, 2*np.pi)
            flow_width = np.random.uniform(np.pi/8, np.pi/5)
            
            angle_diff = np.abs(((angle - flow_angle + np.pi) % (2*np.pi)) - np.pi)
            flow_mask = (angle_diff < flow_width) & (distorted_distance > crater_size) & (distorted_distance < 1.0)
            
            if np.any(flow_mask):
                flow_elevation = 0.2 * np.exp(-distorted_distance[flow_mask]**0.8) * \
                               np.exp(-(angle_diff[flow_mask] / flow_width)**2)
                base_shape[flow_mask] += flow_elevation
        
    elif mountain_type == 'asymmetric':
        # Asymmetric mountain with different slopes (natural weathering)
        base_shape = np.exp(-distorted_distance**0.9)
        
        # Create asymmetry based on prevailing wind direction
        wind_angle = np.random.uniform(0, 2*np.pi)
        wind_alignment = np.cos(angle - wind_angle)
        
        # Windward side: more eroded, gentler slopes
        windward_mask = wind_alignment > 0
        base_shape[windward_mask] *= np.exp(-0.2 * distorted_distance[windward_mask])
        
        # Leeward side: steeper, more protected
        leeward_mask = wind_alignment <= 0
        base_shape[leeward_mask] *= np.exp(-0.05 * distorted_distance[leeward_mask]**0.7)
        
        # Add scree accumulation on leeward side
        scree_mask = leeward_mask & (distorted_distance > 0.6) & (distorted_distance < 1.2)
        if np.any(scree_mask):
            scree_accumulation = 0.15 * np.exp(-(distorted_distance[scree_mask] - 0.6) / 0.3)
            base_shape[scree_mask] += scree_accumulation
    
    else:
        # Default: natural mountain with gentle irregular slopes
        base_shape = np.exp(-distorted_distance**1.0)
    
    # Apply height scaling
    mountain = height * base_shape
    
    # Add realistic surface detail based on elevation zones
    if np.max(mountain) > 0.1:
        # Scale detail intensity with mountain height and size
        detail_scale = min(0.1, height * 0.02)
        
        # High elevation: exposed rock with fine detail
        high_elevation_mask = mountain > height * 0.75
        if np.any(high_elevation_mask):
            rock_detail = np.random.normal(0, detail_scale * 0.5, X.shape)
            rock_detail = cv2.GaussianBlur(rock_detail.astype(np.float32), (3, 3), 0.8)
            mountain[high_elevation_mask] += rock_detail[high_elevation_mask] * mountain[high_elevation_mask] * 0.1
        
        # Mid elevation: mixed terrain
        mid_elevation_mask = (mountain > height * 0.4) & (mountain <= height * 0.75)
        if np.any(mid_elevation_mask):
            mixed_detail = np.random.normal(0, detail_scale * 0.3, X.shape)
            mixed_detail = cv2.GaussianBlur(mixed_detail.astype(np.float32), (5, 5), 1.2)
            mountain[mid_elevation_mask] += mixed_detail[mid_elevation_mask] * mountain[mid_elevation_mask] * 0.05
        
        # Lower slopes: smoother, soil-covered
        low_elevation_mask = mountain <= height * 0.4
        if np.any(low_elevation_mask):
            smooth_detail = np.random.normal(0, detail_scale * 0.2, X.shape)
            smooth_detail = cv2.GaussianBlur(smooth_detail.astype(np.float32), (7, 7), 1.8)
            mountain[low_elevation_mask] += smooth_detail[low_elevation_mask] * mountain[low_elevation_mask] * 0.02
    
    # Ensure no negative values and smooth transition at edges
    mountain = np.maximum(mountain, 0)
    
    # Gentle edge transition to avoid sharp cutoffs
    edge_mask = norm_distance > 1.0
    if np.any(edge_mask):
        fade_factor = np.exp(-(norm_distance[edge_mask] - 1.0) * 3)
        mountain[edge_mask] *= fade_factor
    
    return mountain

def create_fractal_noise_cv2(size, octaves=4, persistence=0.5, scale=0.1):
    """
    Generate improved fractal noise for terrain detail using OpenCV with natural patterns.
    
    Parameters:
    - size: Size of the noise map
    - octaves: Number of noise layers
    - persistence: How much each octave contributes
    - scale: Scale of the noise
    
    Returns:
    - 2D fractal noise array with natural, non-circular patterns
    """
    noise = np.zeros((size, size), dtype=np.float32)
    frequency = scale
    amplitude = 1.0
    max_value = 0.0
    
    for octave in range(octaves):
        # Create more natural noise using gradient-based approach
        # Generate random gradient vectors at grid points
        grid_size = max(8, int(size * frequency / 2))  # Larger grid for more natural patterns
        if grid_size >= size:
            grid_size = size // 2
            
        # Create directional noise instead of purely random
        # This creates more natural flow patterns
        angle_noise = np.random.uniform(0, 2*np.pi, (grid_size, grid_size))
        
        # Create gradient field with natural flow patterns
        grad_x = np.cos(angle_noise) * np.random.uniform(0.5, 1.0, (grid_size, grid_size))
        grad_y = np.sin(angle_noise) * np.random.uniform(0.5, 1.0, (grid_size, grid_size))
        
        # Combine gradients to create height field
        height_field = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # Create natural patterns by integrating gradients
        for i in range(1, grid_size):
            for j in range(1, grid_size):
                # Simple integration to create height from gradients
                height_field[i, j] = (height_field[i-1, j] + grad_x[i, j] * 0.1 +
                                    height_field[i, j-1] + grad_y[i, j] * 0.1) * 0.5
        
        # Add some random variation to break up patterns
        random_component = np.random.normal(0, 0.3, (grid_size, grid_size))
        combined_noise = height_field + random_component
        
        # Resize to full size using bicubic interpolation for smoother results
        layer_noise = cv2.resize(combined_noise.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
        
        # Apply minimal directional smoothing to create natural flow patterns
        if octave == 0:  # Only apply directional smoothing to base layer
            # Create directional kernel for natural erosion patterns
            kernel_angle = np.random.uniform(0, 2*np.pi)
            kernel = np.array([
                [0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2], 
                [0.1, 0.1, 0.1]
            ], dtype=np.float32)
            layer_noise = cv2.filter2D(layer_noise, -1, kernel)
        elif octave < 3:  # Light smoothing for mid-range octaves
            layer_noise = cv2.GaussianBlur(layer_noise, (3, 3), 0.5)
        
        # Normalize layer
        if np.std(layer_noise) > 0:
            layer_noise = (layer_noise - np.mean(layer_noise)) / np.std(layer_noise)
        
        noise += layer_noise * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= 2
    
    # Final normalization with slight bias toward positive values (creates more natural terrain)
    if max_value > 0:
        noise = noise / max_value
        # Add slight upward bias for more natural terrain distribution
        noise = noise * 0.9 + 0.1
    
    return noise

def fix_terrain_artifacts_cv2(Z, artifact_threshold=0.95):
    """
    Detect and fix circular/ring artifacts and other unnatural patterns in terrain.
    
    Parameters:
    - Z: Height map data
    - artifact_threshold: Threshold for detecting artifacts (0-1)
    
    Returns:
    - Fixed height map
    """
    Z_fixed = Z.copy().astype(np.float32)
    
    # Detect potential circular artifacts using gradient analysis
    # Calculate gradients
    grad_x = cv2.Sobel(Z_fixed, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(Z_fixed, cv2.CV_32F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude and direction
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Detect circular patterns by looking for high gradient variance in local regions
    kernel = np.ones((7, 7), np.float32) / 49
    local_grad_mean = cv2.filter2D(grad_magnitude, -1, kernel)
    local_grad_variance = cv2.filter2D((grad_magnitude - local_grad_mean)**2, -1, kernel)
    
    # Identify potential artifact regions
    artifact_mask = local_grad_variance > np.percentile(local_grad_variance, artifact_threshold * 100)
    
    if np.any(artifact_mask):
        print(f"Detected and fixing {np.sum(artifact_mask)} potential artifact pixels...")
        
        # Fix artifacts by smoothing only the problematic regions
        Z_smoothed = cv2.GaussianBlur(Z_fixed, (5, 5), 1.5)
        
        # Blend smoothed version only in artifact areas
        Z_fixed[artifact_mask] = (0.7 * Z_smoothed[artifact_mask] + 
                                 0.3 * Z_fixed[artifact_mask])
    
    # Additional check for extreme outliers
    mean_height = np.mean(Z_fixed)
    std_height = np.std(Z_fixed)
    outlier_mask = np.abs(Z_fixed - mean_height) > 4 * std_height
    
    if np.any(outlier_mask):
        print(f"Fixing {np.sum(outlier_mask)} extreme outlier pixels...")
        # Cap extreme values
        Z_fixed[outlier_mask] = np.clip(Z_fixed[outlier_mask], 
                                       mean_height - 3 * std_height,
                                       mean_height + 3 * std_height)
    
    return Z_fixed

def add_ridges_and_valleys_cv2(X, Y, Z, config):
    """
    Add prominent ridge lines and valleys with natural geological features using OpenCV operations.
    
    Parameters:
    - X, Y: Coordinate meshgrids
    - Z: Height map data
    - config: Terrain configuration
    
    Returns:
    - Enhanced height map with natural ridges and valleys
    """
    if not config.get('add_ridges', False) and not config.get('add_valleys', False):
        return Z
    
    ridge_strength = config.get('ridge_strength', 0.4) * 2.0  # Increase strength for visibility
    valley_depth = config.get('valley_depth', 0.3) * 1.5     # Increase depth for visibility
    
    Z_float = Z.astype(np.float32)
    
    # Create prominent ridge systems
    if config.get('add_ridges', False):
        ridge_map = np.zeros_like(Z_float)
        
        # Generate multiple ridge systems across the terrain
        num_ridge_systems = np.random.randint(4, 8)
        
        for system_idx in range(num_ridge_systems):
            # Create main ridge line
            ridge_start_x = np.random.uniform(X.min(), X.max())
            ridge_start_y = np.random.uniform(Y.min(), Y.max())
            
            # Ridge direction with natural variation
            main_direction = np.random.uniform(0, 2*np.pi)
            ridge_length = np.random.uniform(8, 15)  # Length in coordinate units
            
            # Generate curved ridge path
            ridge_points = []
            current_x, current_y = ridge_start_x, ridge_start_y
            current_direction = main_direction
            
            for step in range(int(ridge_length * 10)):  # More points for smoother curves
                # Add natural meandering
                direction_change = np.random.normal(0, 0.1)
                current_direction += direction_change
                
                # Step along ridge
                step_size = ridge_length / 100
                current_x += step_size * np.cos(current_direction)
                current_y += step_size * np.sin(current_direction)
                
                # Keep within bounds
                if X.min() <= current_x <= X.max() and Y.min() <= current_y <= Y.max():
                    ridge_points.append((current_x, current_y))
                else:
                    break
            
            # Draw ridge system
            for i, (ridge_x, ridge_y) in enumerate(ridge_points):
                # Convert to grid coordinates
                grid_x = int((ridge_x - X.min()) / (X.max() - X.min()) * (X.shape[1] - 1))
                grid_y = int((ridge_y - Y.min()) / (Y.max() - Y.min()) * (X.shape[0] - 1))
                
                # Skip if out of bounds
                if not (0 <= grid_x < X.shape[1] and 0 <= grid_y < X.shape[0]):
                    continue
                
                # Create ridge profile around this point
                ridge_width = np.random.uniform(2, 5)  # Width in coordinate units
                ridge_width_pixels = int(ridge_width / (X.max() - X.min()) * X.shape[1])
                
                y_min = max(0, grid_y - ridge_width_pixels)
                y_max = min(X.shape[0], grid_y + ridge_width_pixels + 1)
                x_min = max(0, grid_x - ridge_width_pixels)
                x_max = min(X.shape[1], grid_x + ridge_width_pixels + 1)
                
                for y in range(y_min, y_max):
                    for x in range(x_min, x_max):
                        dist_pixels = np.sqrt((y - grid_y)**2 + (x - grid_x)**2)
                        if dist_pixels <= ridge_width_pixels:
                            # Create sharp ridge profile
                            ridge_factor = 1 - (dist_pixels / ridge_width_pixels)**0.5
                            ridge_height = ridge_strength * ridge_factor
                            ridge_map[y, x] = max(ridge_map[y, x], ridge_height)
                
                # Add secondary ridges branching off
                if i % 20 == 0 and np.random.random() < 0.6:  # Branch every 20 points, 60% chance
                    branch_direction = current_direction + np.random.uniform(-np.pi/3, np.pi/3)
                    branch_length = np.random.uniform(2, 6)
                    
                    branch_x, branch_y = ridge_x, ridge_y
                    for branch_step in range(int(branch_length * 10)):
                        branch_x += (branch_length / 100) * np.cos(branch_direction)
                        branch_y += (branch_length / 100) * np.sin(branch_direction)
                        
                        if X.min() <= branch_x <= X.max() and Y.min() <= branch_y <= Y.max():
                            # Convert to grid coordinates
                            branch_grid_x = int((branch_x - X.min()) / (X.max() - X.min()) * (X.shape[1] - 1))
                            branch_grid_y = int((branch_y - Y.min()) / (Y.max() - Y.min()) * (X.shape[0] - 1))
                            
                            if 0 <= branch_grid_x < X.shape[1] and 0 <= branch_grid_y < X.shape[0]:
                                branch_width = ridge_width * 0.6
                                branch_width_pixels = int(branch_width / (X.max() - X.min()) * X.shape[1])
                                
                                for dy in range(-branch_width_pixels, branch_width_pixels + 1):
                                    for dx in range(-branch_width_pixels, branch_width_pixels + 1):
                                        by, bx = branch_grid_y + dy, branch_grid_x + dx
                                        if 0 <= by < X.shape[0] and 0 <= bx < X.shape[1]:
                                            dist = np.sqrt(dy**2 + dx**2)
                                            if dist <= branch_width_pixels:
                                                branch_factor = 1 - (dist / branch_width_pixels)**0.5
                                                branch_height = ridge_strength * 0.6 * branch_factor
                                                ridge_map[by, bx] = max(ridge_map[by, bx], branch_height)
        
        # Apply ridge map to terrain
        Z_float += ridge_map
    
    # Create prominent valley systems
    if config.get('add_valleys', False):
        valley_map = np.zeros_like(Z_float)
        
        # Generate multiple valley systems
        num_valley_systems = np.random.randint(5, 10)
        
        for valley_idx in range(num_valley_systems):
            # Create main valley
            valley_start_x = np.random.uniform(X.min() + 2, X.max() - 2)
            valley_start_y = np.random.uniform(Y.min() + 2, Y.max() - 2)
            
            # Valley flows toward edge or lower elevation
            if np.random.random() < 0.7:  # 70% flow to edge
                # Flow toward nearest edge
                edges = [
                    (X.min(), valley_start_y),  # Left edge
                    (X.max(), valley_start_y),  # Right edge
                    (valley_start_x, Y.min()),  # Bottom edge
                    (valley_start_x, Y.max())   # Top edge
                ]
                end_x, end_y = edges[np.random.randint(4)]
            else:
                # Flow toward lower elevation
                end_x = np.random.uniform(X.min(), X.max())
                end_y = np.random.uniform(Y.min(), Y.max())
            
            # Generate curved valley path
            valley_points = []
            steps = 50
            for step in range(steps + 1):
                t = step / steps
                
                # Add natural meandering using sine waves
                meander_x = np.sin(t * np.pi * 3 + np.random.uniform(0, 2*np.pi)) * 1.5
                meander_y = np.cos(t * np.pi * 2.5 + np.random.uniform(0, 2*np.pi)) * 1.2
                
                # Interpolate with meandering
                valley_x = valley_start_x + t * (end_x - valley_start_x) + meander_x
                valley_y = valley_start_y + t * (end_y - valley_start_y) + meander_y
                
                # Keep within bounds
                valley_x = np.clip(valley_x, X.min(), X.max())
                valley_y = np.clip(valley_y, Y.min(), Y.max())
                
                valley_points.append((valley_x, valley_y))
            
            # Draw valley system
            for i, (valley_x, valley_y) in enumerate(valley_points):
                # Convert to grid coordinates
                grid_x = int((valley_x - X.min()) / (X.max() - X.min()) * (X.shape[1] - 1))
                grid_y = int((valley_y - Y.min()) / (Y.max() - Y.min()) * (X.shape[0] - 1))
                
                # Valley gets wider toward the end
                valley_width = 1.5 + (i / len(valley_points)) * 3  # Width in coordinate units
                valley_width_pixels = int(valley_width / (X.max() - X.min()) * X.shape[1])
                
                y_min = max(0, grid_y - valley_width_pixels)
                y_max = min(X.shape[0], grid_y + valley_width_pixels + 1)
                x_min = max(0, grid_x - valley_width_pixels)
                x_max = min(X.shape[1], grid_x + valley_width_pixels + 1)
                
                for y in range(y_min, y_max):
                    for x in range(x_min, x_max):
                        dist_pixels = np.sqrt((y - grid_y)**2 + (x - grid_x)**2)
                        if dist_pixels <= valley_width_pixels:
                            # Create U-shaped valley profile
                            valley_factor = 1 - (dist_pixels / valley_width_pixels)**0.8
                            valley_cut = valley_depth * valley_factor
                            valley_map[y, x] = min(valley_map[y, x], -valley_cut)
                
                # Add tributary valleys
                if i % 15 == 0 and np.random.random() < 0.5:  # Tributary every 15 points, 50% chance
                    tributary_direction = np.random.uniform(0, 2*np.pi)
                    tributary_length = np.random.uniform(2, 5)
                    
                    trib_x, trib_y = valley_x, valley_y
                    for trib_step in range(int(tributary_length * 8)):
                        trib_x += (tributary_length / 80) * np.cos(tributary_direction)
                        trib_y += (tributary_length / 80) * np.sin(tributary_direction)
                        
                        if X.min() <= trib_x <= X.max() and Y.min() <= trib_y <= Y.max():
                            # Convert to grid coordinates
                            trib_grid_x = int((trib_x - X.min()) / (X.max() - X.min()) * (X.shape[1] - 1))
                            trib_grid_y = int((trib_y - Y.min()) / (Y.max() - Y.min()) * (X.shape[0] - 1))
                            
                            if 0 <= trib_grid_x < X.shape[1] and 0 <= trib_grid_y < X.shape[0]:
                                trib_width = valley_width * 0.4
                                trib_width_pixels = int(trib_width / (X.max() - X.min()) * X.shape[1])
                                
                                for dy in range(-trib_width_pixels, trib_width_pixels + 1):
                                    for dx in range(-trib_width_pixels, trib_width_pixels + 1):
                                        ty, tx = trib_grid_y + dy, trib_grid_x + dx
                                        if 0 <= ty < X.shape[0] and 0 <= tx < X.shape[1]:
                                            dist = np.sqrt(dy**2 + dx**2)
                                            if dist <= trib_width_pixels:
                                                trib_factor = 1 - (dist / trib_width_pixels)**0.8
                                                trib_cut = valley_depth * 0.5 * trib_factor
                                                valley_map[ty, tx] = min(valley_map[ty, tx], -trib_cut)
        
        # Apply valley map to terrain
        Z_float += valley_map
        
        # Smooth valley transitions to prevent harsh edges
        valley_mask = valley_map < 0
        if np.any(valley_mask):
            # Apply selective smoothing only to valley areas
            Z_smoothed = cv2.GaussianBlur(Z_float, (3, 3), 0.8)
            Z_float[valley_mask] = (0.7 * Z_float[valley_mask] + 0.3 * Z_smoothed[valley_mask])
    
    # Final cleanup - ensure terrain remains realistic
    Z_float = np.maximum(Z_float, 0)  # No negative elevations
    
    return Z_float
    
    return Z_float

def generate_river_path(start_edge, start_pos, end_edge, end_pos, map_range, map_size, meander=0.2):
    """
    Generate a river path from one edge to another with optional meandering.
    
    Parameters:
    - start_edge: Starting edge ('top', 'bottom', 'left', 'right')
    - start_pos: Position along starting edge (0.0 to 1.0)
    - end_edge: Ending edge ('top', 'bottom', 'left', 'right')
    - end_pos: Position along ending edge (0.0 to 1.0)
    - map_range: (min, max) coordinate range
    - map_size: Size of the map
    - meander: Amount of river meandering (0.0 = straight, 0.5 = very curvy)
    
    Returns:
    - Array of (x, y) coordinates along the river path
    """
    # Convert edge positions to coordinates
    min_coord, max_coord = map_range
    coord_range = max_coord - min_coord
    
    edge_coords = {
        'top': lambda pos: (min_coord + pos * coord_range, max_coord),
        'bottom': lambda pos: (min_coord + pos * coord_range, min_coord),
        'left': lambda pos: (min_coord, min_coord + pos * coord_range),
        'right': lambda pos: (max_coord, min_coord + pos * coord_range),
    }
    
    start_x, start_y = edge_coords[start_edge](start_pos)
    end_x, end_y = edge_coords[end_edge](end_pos)
    
    # Number of points along the river path
    num_points = max(50, int(map_size * 0.1))
    
    # Base linear path
    t = np.linspace(0, 1, num_points)
    base_x = start_x + t * (end_x - start_x)
    base_y = start_y + t * (end_y - start_y)
    
    if meander > 0:
        # Add meandering using sine waves
        river_length = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        meander_amplitude = meander * coord_range * 0.1
        
        # Calculate perpendicular direction for meandering
        dx = end_x - start_x
        dy = end_y - start_y
        length = np.sqrt(dx**2 + dy**2)
        
        if length > 0:
            # Perpendicular vector
            perp_x = -dy / length
            perp_y = dx / length
            
            # Add meandering oscillations
            meander_x = meander_amplitude * np.sin(t * np.pi * 4) * perp_x
            meander_y = meander_amplitude * np.sin(t * np.pi * 4) * perp_y
            
            # Add secondary meandering for more natural curves
            meander_x += meander_amplitude * 0.5 * np.sin(t * np.pi * 7 + 1) * perp_x
            meander_y += meander_amplitude * 0.5 * np.sin(t * np.pi * 7 + 1) * perp_y
            
            base_x += meander_x
            base_y += meander_y
    
    return list(zip(base_x, base_y))

def create_tributaries(main_path, num_tributaries, map_range, meander=0.1):
    """
    Create tributary paths branching from the main river.
    
    Parameters:
    - main_path: List of (x, y) coordinates for main river
    - num_tributaries: Number of tributaries to create
    - map_range: (min, max) coordinate range
    - meander: Amount of tributary meandering
    
    Returns:
    - List of tributary paths
    """
    tributaries = []
    min_coord, max_coord = map_range
    coord_range = max_coord - min_coord
    
    for i in range(num_tributaries):
        # Choose a random point along the main river (avoid endpoints)
        branch_index = np.random.randint(len(main_path) // 4, 3 * len(main_path) // 4)
        branch_point = main_path[branch_index]
        
        # Generate a random endpoint for the tributary
        side = np.random.choice(['left', 'right', 'up', 'down'])
        tributary_length = np.random.uniform(0.2, 0.6) * coord_range
        
        angle = np.random.uniform(0, 2 * np.pi)
        end_x = branch_point[0] + tributary_length * np.cos(angle)
        end_y = branch_point[1] + tributary_length * np.sin(angle)
        
        # Ensure endpoint is within map bounds
        end_x = np.clip(end_x, min_coord, max_coord)
        end_y = np.clip(end_y, min_coord, max_coord)
        
        # Create tributary path
        num_points = max(20, len(main_path) // 3)
        t = np.linspace(0, 1, num_points)
        
        trib_x = branch_point[0] + t * (end_x - branch_point[0])
        trib_y = branch_point[1] + t * (end_y - branch_point[1])
        
        # Add light meandering to tributaries
        if meander > 0:
            dx = end_x - branch_point[0]
            dy = end_y - branch_point[1]
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 0:
                perp_x = -dy / length
                perp_y = dx / length
                
                meander_amplitude = meander * coord_range * 0.05
                meander_x = meander_amplitude * np.sin(t * np.pi * 3) * perp_x
                meander_y = meander_amplitude * np.sin(t * np.pi * 3) * perp_y
                
                trib_x += meander_x
                trib_y += meander_y
        
        tributary_path = list(zip(trib_x, trib_y))
        tributaries.append(tributary_path)
    
    return tributaries

def follow_terrain_flow(X, Y, Z, start_point, end_point, meander=0.2):
    """
    Generate a river path that follows terrain elevation, flowing downhill and avoiding mountains.
    
    Parameters:
    - X, Y: Coordinate meshgrids
    - Z: Height map data
    - start_point: (x, y) starting coordinates
    - end_point: (x, y) ending coordinates
    - meander: Amount of meandering allowed
    
    Returns:
    - List of (x, y) coordinates following terrain flow
    """
    # Enhanced path finding that strongly avoids mountains and prefers valleys
    path = [start_point]
    current_pos = np.array(start_point)
    target_pos = np.array(end_point)
    
    # Calculate terrain statistics for better mountain avoidance
    elevation_mean = np.mean(Z)
    elevation_std = np.std(Z)
    mountain_threshold = elevation_mean + 0.5 * elevation_std  # Heights above this are considered mountains
    valley_threshold = elevation_mean - 0.3 * elevation_std    # Heights below this are preferred valleys
    
    # Number of steps to reach target (more steps for better path finding)
    num_steps = max(80, int(np.linalg.norm(target_pos - current_pos) * 30))
    
    for step in range(num_steps):
        if np.linalg.norm(current_pos - target_pos) < 0.08:
            break
            
        # Direction towards target
        direction = target_pos - current_pos
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction = direction / direction_norm
        
        # Sample more points in wider area for better mountain avoidance
        sample_radius = 0.8  # Increased radius for better path options
        num_samples = 16     # More samples for better choices
        angles = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
        
        best_pos = current_pos.copy()
        best_score = float('inf')
        
        for angle in angles:
            # Sample point in this direction
            sample_dir = np.array([np.cos(angle), np.sin(angle)])
            sample_pos = current_pos + sample_dir * sample_radius * (0.3 + 0.7 * np.random.random())
            
            # Add bias towards target (but less aggressive to allow mountain avoidance)
            target_bias = 0.4  # Reduced from 0.7 to allow more deviation
            sample_pos = sample_pos * (1 - target_bias) + (current_pos + direction * sample_radius) * target_bias
            
            # Check if position is within bounds
            if (sample_pos[0] >= X.min() and sample_pos[0] <= X.max() and
                sample_pos[1] >= Y.min() and sample_pos[1] <= Y.max()):
                
                # Interpolate elevation at this position
                x_idx = np.clip(int((sample_pos[0] - X.min()) / (X.max() - X.min()) * (Z.shape[1] - 1)), 0, Z.shape[1] - 1)
                y_idx = np.clip(int((sample_pos[1] - Y.min()) / (Y.max() - Y.min()) * (Z.shape[0] - 1)), 0, Z.shape[0] - 1)
                elevation = Z[y_idx, x_idx]
                
                # Calculate score with strong mountain avoidance
                distance_to_target = np.linalg.norm(sample_pos - target_pos)
                
                # Base score from elevation and distance
                score = elevation * 0.5 + distance_to_target * 0.1
                
                # Heavy penalty for mountains
                if elevation > mountain_threshold:
                    mountain_penalty = (elevation - mountain_threshold) * 10  # Strong penalty
                    score += mountain_penalty
                
                # Bonus for valleys
                if elevation < valley_threshold:
                    valley_bonus = (valley_threshold - elevation) * 2  # Moderate bonus
                    score -= valley_bonus
                
                # Additional penalty for steep slopes (gradient calculation)
                gradient_radius = 0.2
                nearby_elevations = []
                for grad_angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
                    grad_pos = sample_pos + gradient_radius * np.array([np.cos(grad_angle), np.sin(grad_angle)])
                    if (grad_pos[0] >= X.min() and grad_pos[0] <= X.max() and
                        grad_pos[1] >= Y.min() and grad_pos[1] <= Y.max()):
                        grad_x_idx = np.clip(int((grad_pos[0] - X.min()) / (X.max() - X.min()) * (Z.shape[1] - 1)), 0, Z.shape[1] - 1)
                        grad_y_idx = np.clip(int((grad_pos[1] - Y.min()) / (Y.max() - Y.min()) * (Z.shape[0] - 1)), 0, Z.shape[0] - 1)
                        nearby_elevations.append(Z[grad_y_idx, grad_x_idx])
                
                if nearby_elevations:
                    gradient = np.std(nearby_elevations)  # High std = steep slopes
                    slope_penalty = gradient * 3  # Penalty for steep slopes
                    score += slope_penalty
                
                if score < best_score:
                    best_score = score
                    best_pos = sample_pos
        
        # Move towards best position with adaptive step size
        remaining_distance = np.linalg.norm(target_pos - current_pos)
        step_size = max(0.08, min(remaining_distance / (num_steps - step + 1), 0.3))
        
        move_direction = best_pos - current_pos
        if np.linalg.norm(move_direction) > 0:
            move_direction = move_direction / np.linalg.norm(move_direction)
        
        # Add meandering (but less in mountainous areas)
        current_elevation = Z[np.clip(int((current_pos[1] - Y.min()) / (Y.max() - Y.min()) * (Z.shape[0] - 1)), 0, Z.shape[0] - 1),
                             np.clip(int((current_pos[0] - X.min()) / (X.max() - X.min()) * (Z.shape[1] - 1)), 0, Z.shape[1] - 1)]
        
        meander_factor = meander
        if current_elevation > mountain_threshold:
            meander_factor *= 0.3  # Reduce meandering in mountains (straighter path to get out faster)
        elif current_elevation < valley_threshold:
            meander_factor *= 1.5  # Increase meandering in valleys (more natural flow)
        
        if meander_factor > 0:
            perpendicular = np.array([-move_direction[1], move_direction[0]])
            meander_offset = perpendicular * meander_factor * (np.random.random() - 0.5) * step_size
            move_direction += meander_offset * 0.2
            # Normalize again
            if np.linalg.norm(move_direction) > 0:
                move_direction = move_direction / np.linalg.norm(move_direction)
        
        current_pos += move_direction * step_size
        path.append(tuple(current_pos))
    
    # Ensure we end at the target
    path.append(end_point)
    return path

def carve_river_cv2(X, Y, Z, river_path, width, depth):
    """
    Carve a river into the terrain using OpenCV operations.
    Creates a river with consistent water level (flat surface) that cuts through terrain.
    
    Parameters:
    - X, Y: Coordinate meshgrids
    - Z: Height map data
    - river_path: List of (x, y) coordinates along the river
    - width: River width
    - depth: River depth multiplier
    
    Returns:
    - Modified height map with carved river
    """
    Z_carved = Z.copy().astype(np.float32)
    
    # Create river mask
    river_mask = np.zeros_like(Z, dtype=np.float32)
    
    # Calculate elevations along river path to determine water level
    path_elevations = []
    for rx, ry in river_path:
        # Get elevation at river point
        x_idx = np.clip(int((rx - X.min()) / (X.max() - X.min()) * (Z.shape[1] - 1)), 0, Z.shape[1] - 1)
        y_idx = np.clip(int((ry - Y.min()) / (Y.max() - Y.min()) * (Z.shape[0] - 1)), 0, Z.shape[0] - 1)
        elevation = Z_carved[y_idx, x_idx]
        path_elevations.append(elevation)
    
    # Determine water level - use a consistent level across entire map
    min_elevation = min(path_elevations)
    
    # Use a flat water level across the entire river for consistent appearance
    flat_water_level = min_elevation - depth * 0.2  # Single consistent level
    
    # All points along the river use the same water level
    water_levels = [flat_water_level] * len(river_path)
    
    print(f"    River water level: {flat_water_level:.3f} (flat across entire river)")
    
    for i, (rx, ry) in enumerate(river_path):
        # Calculate distance from each point to river center
        distance = np.sqrt((X - rx)**2 + (Y - ry)**2)
        
        # Create river cross-section with smoother, more gradual profile
        # Use a combination of Gaussian and smoother falloff for natural banks
        river_profile_main = np.exp(-(distance / width)**2)
        river_profile_extended = np.exp(-(distance / (width * 2.5))**1.5)  # Gentler, wider falloff
        
        # Combine profiles for smoother banks
        river_profile = 0.7 * river_profile_main + 0.3 * river_profile_extended
        
        # Set water level for this point
        water_level = water_levels[i]
        
        # Update river mask with smooth profile
        river_mask = np.maximum(river_mask, river_profile)
        
        # Carve the river to maintain consistent water level
        # Use a gentler mask for smoother transitions
        river_bed_mask = river_profile > 0.05  # Lower threshold for gentler edges
        
        # Calculate how deep to carve at each point
        current_terrain_height = Z_carved
        carving_depth = np.maximum(0, current_terrain_height - water_level)
        
        # Apply carving with very smooth falloff from river center
        carving_amount = carving_depth * river_profile
        Z_carved = np.where(river_bed_mask, 
                           current_terrain_height - carving_amount, 
                           Z_carved)
        
        # Ensure river bed doesn't go above water level
        Z_carved = np.where(river_bed_mask,
                           np.minimum(Z_carved, water_level),
                           Z_carved)
    
    # Smooth the river banks using OpenCV for more natural transitions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    river_mask_smooth = cv2.morphologyEx(river_mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply multiple levels of smoothing for very smooth banks
    river_area = river_mask_smooth > 0.05
    if np.any(river_area):
        # First level: Strong Gaussian smoothing for the main river area
        Z_river_smooth = cv2.GaussianBlur(Z_carved, (9, 9), 2.0)
        
        # Second level: Gentle smoothing for wider area around river
        extended_area = river_mask_smooth > 0.02
        if np.any(extended_area):
            Z_extended_smooth = cv2.GaussianBlur(Z_carved, (15, 15), 3.0)
            
            # Create smooth transition zones
            for i in range(3):  # Multiple blending passes
                # Inner river area gets strongest smoothing
                inner_blend = river_mask_smooth[river_area] ** 0.5  # Softer falloff
                Z_carved[river_area] = (Z_river_smooth[river_area] * inner_blend + 
                                       Z_carved[river_area] * (1 - inner_blend))
                
                # Extended area gets gentle smoothing
                if np.any(extended_area):
                    extended_blend = (river_mask_smooth[extended_area] * 0.3) ** 0.8
                    Z_carved[extended_area] = (Z_extended_smooth[extended_area] * extended_blend + 
                                              Z_carved[extended_area] * (1 - extended_blend))
        else:
            # Fallback if extended area is empty
            blend_factor = river_mask_smooth[river_area] ** 0.6  # Smooth power curve
            Z_carved[river_area] = (Z_river_smooth[river_area] * blend_factor + 
                                   Z_carved[river_area] * (1 - blend_factor))
    
    # Final gentle smoothing pass over the entire carved area
    final_smooth_area = river_mask > 0.01
    if np.any(final_smooth_area):
        Z_final_smooth = cv2.GaussianBlur(Z_carved, (7, 7), 1.5)
        final_blend = (river_mask[final_smooth_area] * 0.2) ** 1.5
        Z_carved[final_smooth_area] = (Z_final_smooth[final_smooth_area] * final_blend + 
                                      Z_carved[final_smooth_area] * (1 - final_blend))
    
    return Z_carved

def add_rivers_to_terrain_cv2(X, Y, Z, river_config):
    """
    Add rivers to terrain with flat water levels and simple meandering paths.
    
    Parameters:
    - X, Y: Coordinate meshgrids
    - Z: Height map data
    - river_config: River configuration dictionary
    
    Returns:
    - Modified height map with rivers
    """
    if not river_config.get('enabled', False):
        print("Rivers disabled - skipping river generation")
        return Z
    
    Z_with_rivers = Z.copy()
    map_range = (X.min(), X.max())
    
    num_rivers = len(river_config['rivers'])
    print(f"🌊 Starting river generation - {num_rivers} river(s) configured")
    print(f"   Map bounds: {X.min():.2f} to {X.max():.2f}")
    
    if num_rivers == 0:
        print("   No rivers configured - skipping river generation")
        return Z
    
    for i, river_data in enumerate(river_config['rivers']):
        print(f"🏞️  Generating river {i+1}/{num_rivers}:")
        print(f"     Start: {river_data['start_edge']} edge, position {river_data['start_position']:.2f}")
        print(f"     End: {river_data['end_edge']} edge, position {river_data['end_position']:.2f}")
        print(f"     Width: {river_data.get('width', 0.1):.3f}, Depth: {river_data.get('depth', 0.2):.3f}")
        print(f"     Meander factor: {river_data.get('meander', 0.2):.2f}")
        
        # Convert edge positions to coordinates
        min_coord, max_coord = map_range
        coord_range = max_coord - min_coord
        
        edge_coords = {
            'top': lambda pos: (min_coord + pos * coord_range, max_coord),
            'bottom': lambda pos: (min_coord + pos * coord_range, min_coord),
            'left': lambda pos: (min_coord, min_coord + pos * coord_range),
            'right': lambda pos: (max_coord, min_coord + pos * coord_range),
        }
        
        start_x, start_y = edge_coords[river_data['start_edge']](river_data['start_position'])
        end_x, end_y = edge_coords[river_data['end_edge']](river_data['end_position'])
        
        print(f"     Coordinates: ({start_x:.2f}, {start_y:.2f}) → ({end_x:.2f}, {end_y:.2f})")
        
        # Generate simple meandering path
        print(f"     Generating meandering path...")
        river_path = create_simple_meandering_path(
            (start_x, start_y), (end_x, end_y),
            river_data.get('meander', 0.2)
        )
        
        print(f"     Path generated with {len(river_path)} points")
        
        # Carve river with flat water level
        print(f"     Carving river channel...")
        Z_with_rivers = carve_simple_river(
            X, Y, Z_with_rivers, river_path,
            river_data.get('width', 0.1),
            river_data.get('depth', 0.2)
        )
        
        print(f"     ✅ River {i+1} completed successfully")
    
    print(f"🌊 All rivers generated successfully!")
    return Z_with_rivers

def create_simple_meandering_path(start_point, end_point, meander_factor):
    """
    Create a simple meandering path between two points.
    
    Parameters:
    - start_point: (x, y) starting coordinates
    - end_point: (x, y) ending coordinates
    - meander_factor: Amount of meandering (0.0 = straight, 1.0 = very curved)
    
    Returns:
    - List of (x, y) coordinates forming the river path
    """
    start_x, start_y = start_point
    end_x, end_y = end_point
    
    # Calculate path parameters
    distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
    num_points = max(20, int(distance * 10))  # More points for smoother curves
    
    # Create base straight line
    t_values = np.linspace(0, 1, num_points)
    base_x = start_x + t_values * (end_x - start_x)
    base_y = start_y + t_values * (end_y - start_y)
    
    # Add meandering using sine waves
    if meander_factor > 0:
        # Direction perpendicular to the main flow
        flow_angle = np.arctan2(end_y - start_y, end_x - start_x)
        perp_angle = flow_angle + np.pi / 2
        
        # Multiple sine waves for more natural meandering
        meander_amplitude = distance * meander_factor * 0.3
        
        # Primary meander
        meander1 = np.sin(t_values * np.pi * 4) * meander_amplitude
        # Secondary smaller meander
        meander2 = np.sin(t_values * np.pi * 8 + 1.5) * meander_amplitude * 0.4
        # Tertiary very small meander
        meander3 = np.sin(t_values * np.pi * 16 + 2.8) * meander_amplitude * 0.2
        
        total_meander = meander1 + meander2 + meander3
        
        # Apply fade-in and fade-out to reduce meandering at endpoints
        fade = np.sin(t_values * np.pi)**0.5  # Smooth fade
        total_meander *= fade
        
        # Apply meandering perpendicular to flow direction
        meander_x = total_meander * np.cos(perp_angle)
        meander_y = total_meander * np.sin(perp_angle)
        
        base_x += meander_x
        base_y += meander_y
    
    # Combine into path
    path = list(zip(base_x, base_y))
    return path

def carve_simple_river(X, Y, Z, river_path, width, depth):
    """
    Carve a river with consistent flat water level.
    
    Parameters:
    - X, Y: Coordinate meshgrids
    - Z: Height map data
    - river_path: List of (x, y) coordinates for river path
    - width: River width
    - depth: River depth
    
    Returns:
    - Modified height map with carved river
    """
    Z_carved = Z.copy()
    original_min_height = Z.min()
    original_max_height = Z.max()
    
    print(f"       Analyzing terrain for river carving...")
    print(f"       Original terrain range: {original_min_height:.3f} to {original_max_height:.3f}")
    
    # Find the minimum elevation along the entire path for consistent water level
    min_elevation = float('inf')
    path_elevations = []
    
    for rx, ry in river_path:
        x_idx = np.clip(int((rx - X.min()) / (X.max() - X.min()) * (Z.shape[1] - 1)), 0, Z.shape[1] - 1)
        y_idx = np.clip(int((ry - Y.min()) / (Y.max() - Y.min()) * (Z.shape[0] - 1)), 0, Z.shape[0] - 1)
        elevation = Z_carved[y_idx, x_idx]
        path_elevations.append(elevation)
        min_elevation = min(min_elevation, elevation)
    
    # Set consistent water level below the minimum terrain elevation
    water_level = min_elevation - depth
    avg_path_elevation = np.mean(path_elevations)
    
    print(f"       River path analysis:")
    print(f"         Minimum path elevation: {min_elevation:.3f}")
    print(f"         Average path elevation: {avg_path_elevation:.3f}")
    print(f"         Maximum path elevation: {max(path_elevations):.3f}")
    print(f"         Water level set to: {water_level:.3f} (flat across entire river)")
    
    # Carve the river channel
    points_processed = 0
    total_points = len(river_path)
    
    for rx, ry in river_path:
        points_processed += 1
        if points_processed % max(1, total_points // 10) == 0:
            progress = (points_processed / total_points) * 100
            print(f"       Carving progress: {progress:.1f}% ({points_processed}/{total_points} points)")
        
        # Calculate distance from each grid point to river center
        distance = np.sqrt((X - rx)**2 + (Y - ry)**2)
        
        # Create smooth river profile (Gaussian-like)
        river_mask = np.exp(-(distance / width)**2)
        
        # Apply stronger carving in the center, gentler at edges
        carving_strength = river_mask * depth * 2.0  # Multiply for deeper carving
        
        # Set water level where the river influence is significant
        significant_area = river_mask > 0.01
        
        # Carve below water level for clear visibility
        riverbed_level = water_level - carving_strength
        Z_carved = np.where(significant_area,
                           np.minimum(Z_carved, riverbed_level),
                           Z_carved)
    
    print(f"       Applying river bank smoothing...")
    # Smooth the river banks for natural appearance
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Create river mask for smoothing
    river_area_mask = np.zeros_like(Z_carved)
    affected_points = 0
    
    for rx, ry in river_path:
        distance = np.sqrt((X - rx)**2 + (Y - ry)**2)
        river_mask = distance < width * 2.0  # Wider area for smoothing
        river_area_mask = np.logical_or(river_area_mask, river_mask)
        affected_points += np.sum(river_mask)
    
    print(f"       River affects {affected_points} terrain points")
    
    # Apply gentle smoothing only to river area
    if np.any(river_area_mask):
        print(f"       Smoothing river banks...")
        Z_river_smooth = cv2.GaussianBlur(Z_carved, (7, 7), 1.5)
        Z_carved = np.where(river_area_mask, Z_river_smooth, Z_carved)
    
    final_min_height = Z_carved.min()
    final_max_height = Z_carved.max()
    height_change = original_min_height - final_min_height
    
    print(f"       River carving complete!")
    print(f"       Final terrain range: {final_min_height:.3f} to {final_max_height:.3f}")
    print(f"       Maximum depth carved: {height_change:.3f}")
    
    return Z_carved
    
    return Z_with_rivers

def create_terrain_following_tributaries(X, Y, Z, main_path, num_tributaries, map_range, meander=0.1):
    """
    Create tributary paths that follow terrain and flow into the main river.
    
    Parameters:
    - X, Y: Coordinate meshgrids
    - Z: Height map data
    - main_path: List of (x, y) coordinates for main river
    - num_tributaries: Number of tributaries to create
    - map_range: (min, max) coordinate range
    - meander: Amount of tributary meandering
    
    Returns:
    - List of tributary paths
    """
    tributaries = []
    min_coord, max_coord = map_range
    coord_range = max_coord - min_coord
    
    for i in range(num_tributaries):
        # Choose a random point along the main river (avoid endpoints)
        branch_index = np.random.randint(len(main_path) // 4, 3 * len(main_path) // 4)
        branch_point = main_path[branch_index]
        
        # Find a higher elevation point to start the tributary
        tributary_length = np.random.uniform(0.2, 0.6) * coord_range
        
        # Search for a higher starting point
        best_start = None
        best_elevation_diff = 0
        
        # Try multiple random directions to find a good starting point
        for attempt in range(20):
            angle = np.random.uniform(0, 2 * np.pi)
            start_x = branch_point[0] + tributary_length * np.cos(angle)
            start_y = branch_point[1] + tributary_length * np.sin(angle)
            
            # Ensure starting point is within map bounds
            if (start_x >= min_coord and start_x <= max_coord and
                start_y >= min_coord and start_y <= max_coord):
                
                # Get elevation at starting point
                x_idx = np.clip(int((start_x - X.min()) / (X.max() - X.min()) * (Z.shape[1] - 1)), 0, Z.shape[1] - 1)
                y_idx = np.clip(int((start_y - Y.min()) / (Y.max() - Y.min()) * (Z.shape[0] - 1)), 0, Z.shape[0] - 1)
                start_elevation = Z[y_idx, x_idx]
                
                # Get elevation at branch point
                branch_x_idx = np.clip(int((branch_point[0] - X.min()) / (X.max() - X.min()) * (Z.shape[1] - 1)), 0, Z.shape[1] - 1)
                branch_y_idx = np.clip(int((branch_point[1] - Y.min()) / (Y.max() - Y.min()) * (Z.shape[0] - 1)), 0, Z.shape[0] - 1)
                branch_elevation = Z[branch_y_idx, branch_x_idx]
                
                elevation_diff = start_elevation - branch_elevation
                
                # Prefer starting points that are higher than the branch point
                if elevation_diff > best_elevation_diff:
                    best_elevation_diff = elevation_diff
                    best_start = (start_x, start_y)
        
        # If we found a good starting point, create the tributary
        if best_start is not None and best_elevation_diff > 0.01:
            tributary_path = follow_terrain_flow(
                X, Y, Z, best_start, branch_point, meander
            )
            tributaries.append(tributary_path)
        elif best_start is not None:
            # Even if elevation difference is small, create a short tributary
            # but make sure it flows downhill
            tributary_path = follow_terrain_flow(
                X, Y, Z, best_start, branch_point, meander * 0.5
            )
            if len(tributary_path) > 5:  # Only add if path is reasonable
                tributaries.append(tributary_path)
    
    return tributaries

def generate_road_path(start_edge, start_pos, end_edge, end_pos, map_range, map_size, follow_contours=True):
    """
    Generate an efficient road path that follows terrain, preferring gentler slopes.
    
    Parameters:
    - start_edge: Starting edge ('top', 'bottom', 'left', 'right')
    - start_pos: Position along starting edge (0.0 to 1.0)
    - end_edge: Ending edge ('top', 'bottom', 'left', 'right')
    - end_pos: Position along ending edge (0.0 to 1.0)
    - map_range: (min, max) coordinate range
    - map_size: Size of the map
    - follow_contours: Whether to follow elevation contours
    
    Returns:
    - List of (x, y) coordinates along the road path
    """
    # Convert edge positions to coordinates
    min_coord, max_coord = map_range
    coord_range = max_coord - min_coord
    
    edge_coords = {
        'top': lambda pos: (min_coord + pos * coord_range, max_coord),
        'bottom': lambda pos: (min_coord + pos * coord_range, min_coord),
        'left': lambda pos: (min_coord, min_coord + pos * coord_range),
        'right': lambda pos: (max_coord, min_coord + pos * coord_range),
    }
    
    start_x, start_y = edge_coords[start_edge](start_pos)
    end_x, end_y = edge_coords[end_edge](end_pos)
    
    if not follow_contours:
        # Direct path with smooth curves - optimized point count
        num_points = min(50, max(20, int(map_size * 0.1)))  # Much fewer points
        t = np.linspace(0, 1, num_points)
        
        # Basic linear interpolation
        path_x = start_x + t * (end_x - start_x)
        path_y = start_y + t * (end_y - start_y)
        
        # Add gentle curves for more natural road appearance
        curve_amplitude = coord_range * 0.01  # Reduced curve
        path_x += curve_amplitude * np.sin(t * np.pi * 2)
        path_y += curve_amplitude * np.cos(t * np.pi * 1.5)
        
        return list(zip(path_x, path_y))
    else:
        # More efficient path that considers terrain
        path = [(start_x, start_y)]
        current_pos = np.array([start_x, start_y])
        target_pos = np.array([end_x, end_y])
        
        # Calculate total distance and appropriate number of steps - optimized
        total_distance = np.linalg.norm(target_pos - current_pos)
        num_steps = min(80, max(30, int(total_distance * 20)))  # Much fewer steps
        
        for step in range(num_steps):
            if np.linalg.norm(current_pos - target_pos) < 0.1:  # Larger threshold
                break
            
            # Direction towards target
            direction = target_pos - current_pos
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                direction = direction / direction_norm
            
            # Step size that gets smaller as we approach target
            step_size = min(0.15, np.linalg.norm(target_pos - current_pos) / 8)  # Larger steps
            
            # Move towards target with slight randomness for natural appearance
            next_pos = current_pos + direction * step_size
            
            # Add small random deviation for natural road curves (reduced)
            perpendicular = np.array([-direction[1], direction[0]])
            deviation = perpendicular * np.random.normal(0, step_size * 0.03)  # Less deviation
            next_pos += deviation
            
            # Ensure we stay within bounds
            next_pos[0] = np.clip(next_pos[0], min_coord, max_coord)
            next_pos[1] = np.clip(next_pos[1], min_coord, max_coord)
            
            current_pos = next_pos
            path.append(tuple(current_pos))
        
        # Ensure we end at the target
        path.append((end_x, end_y))
        return path

def carve_road_cv2(X, Y, Z, road_path, width, cut_depth, smoothness=0.8, banking=True):
    """
    Efficiently carve a road into the terrain using optimized algorithms.
    
    Parameters:
    - X, Y: Coordinate meshgrids
    - Z: Height map data
    - road_path: List of (x, y) coordinates along the road
    - width: Road width
    - cut_depth: Maximum depth to cut into terrain
    - smoothness: How smooth to make the road surface (0.0-1.0)
    - banking: Whether to add banking/embankments
    
    Returns:
    - Modified height map with carved road
    """
    Z_carved = Z.copy().astype(np.float32)
    original_min_height = Z.min()
    original_max_height = Z.max()
    
    print(f"       Analyzing terrain for optimized road carving...")
    print(f"       Original terrain range: {original_min_height:.3f} to {original_max_height:.3f}")
    print(f"       Road parameters: width={width:.3f}, cut_depth={cut_depth:.3f}, smoothness={smoothness:.2f}")
    
    # Convert world coordinates to grid indices once
    map_width = X.max() - X.min()
    map_height = Y.max() - Y.min()
    grid_width = Z.shape[1]
    grid_height = Z.shape[0]
    
    def world_to_grid(wx, wy):
        """Convert world coordinates to grid indices"""
        gx = int((wx - X.min()) / map_width * (grid_width - 1))
        gy = int((wy - Y.min()) / map_height * (grid_height - 1))
        return np.clip(gx, 0, grid_width - 1), np.clip(gy, 0, grid_height - 1)
    
    # Convert road path to grid coordinates and sample elevations
    print(f"       Converting road path to grid coordinates...")
    grid_path = []
    road_elevations = []
    
    for wx, wy in road_path:
        gx, gy = world_to_grid(wx, wy)
        grid_path.append((gx, gy))
        road_elevations.append(Z_carved[gy, gx])
    
    # Smooth road elevations efficiently
    if smoothness > 0 and len(road_elevations) > 2:
        print(f"       Smoothing road elevations...")
        smoothed_elevations = np.array(road_elevations)
        smoothing_passes = int(smoothness * 3)  # Reduced passes
        for _ in range(smoothing_passes):
            smoothed_elevations[1:-1] = (smoothed_elevations[:-2] + smoothed_elevations[1:-1] + smoothed_elevations[2:]) / 3
        road_elevations = smoothed_elevations.tolist()
    
    # Calculate road width in grid units
    grid_width_pixels = int(width / map_width * grid_width)
    grid_width_pixels = max(2, grid_width_pixels)  # Minimum 2 pixels wide
    
    print(f"       Road width in pixels: {grid_width_pixels}")
    print(f"       Processing {len(grid_path)} road points...")
    
    # Create road mask using efficient line drawing
    road_mask = np.zeros_like(Z, dtype=np.float32)
    
    # Use OpenCV line drawing for efficient road creation
    road_image = np.zeros_like(Z, dtype=np.uint8)
    
    # Draw road lines between consecutive points
    for i in range(len(grid_path) - 1):
        gx1, gy1 = grid_path[i]
        gx2, gy2 = grid_path[i + 1]
        cv2.line(road_image, (gx1, gy1), (gx2, gy2), 255, grid_width_pixels)
    
    # Convert line drawing to distance-based mask
    road_binary = (road_image > 0).astype(np.uint8)
    
    # Create distance transform for smooth road profile
    if np.any(road_binary):
        # Use distance transform for efficient distance calculations
        dist_transform = cv2.distanceTransform(255 - road_binary * 255, cv2.DIST_L2, 5)
        
        # Create smooth road profile using distance transform
        max_dist = grid_width_pixels * 1.5
        road_mask = np.exp(-(dist_transform / max_dist)**2)
        road_mask[road_binary == 0] = 0  # Only apply to road area
        
        # Banking mask (wider area)
        if banking:
            banking_max_dist = grid_width_pixels * 3
            banking_mask = np.exp(-(dist_transform / banking_max_dist)**2) * 0.4
            banking_mask[dist_transform > banking_max_dist] = 0
    else:
        road_mask = np.zeros_like(Z)
        banking_mask = np.zeros_like(Z)
    
    # Apply road carving efficiently
    print(f"       Applying road cuts...")
    
    # Find all road pixels
    road_pixels = np.where(road_mask > 0.01)
    
    if len(road_pixels[0]) > 0:
        # Calculate target elevations for each road pixel
        road_pixel_elevations = []
        
        for py, px in zip(road_pixels[0], road_pixels[1]):
            # Find closest road point for elevation reference
            min_dist = float('inf')
            closest_elevation = road_elevations[0]
            
            for i, (gx, gy) in enumerate(grid_path):
                dist = (px - gx)**2 + (py - gy)**2
                if dist < min_dist:
                    min_dist = dist
                    closest_elevation = road_elevations[i]
            
            road_pixel_elevations.append(closest_elevation)
        
        road_pixel_elevations = np.array(road_pixel_elevations)
        
        # Apply cutting
        current_elevations = Z_carved[road_pixels]
        cutting_needed = np.maximum(0, current_elevations - road_pixel_elevations)
        actual_cut = np.minimum(cutting_needed, cut_depth)
        
        # Apply cuts with road mask intensity
        road_mask_values = road_mask[road_pixels]
        cut_amounts = actual_cut * road_mask_values
        
        Z_carved[road_pixels] = current_elevations - cut_amounts
        
        print(f"       Modified {len(road_pixels[0])} road pixels")
    
    # Apply banking if enabled
    if banking and 'banking_mask' in locals():
        print(f"       Applying road banking...")
        banking_pixels = np.where(banking_mask > 0.02)
        
        if len(banking_pixels[0]) > 0:
            banking_adjustment = banking_mask[banking_pixels] * cut_depth * 0.6
            Z_carved[banking_pixels] += banking_adjustment
            print(f"       Applied banking to {len(banking_pixels[0])} pixels")
    
    # Final smoothing for integration
    print(f"       Final smoothing...")
    if np.any(road_mask > 0.01):
        # Light Gaussian blur for natural integration
        Z_smooth = cv2.GaussianBlur(Z_carved, (5, 5), 1.0)
        
        # Blend only in road areas
        blend_mask = road_mask * 0.3
        Z_carved = Z_smooth * blend_mask + Z_carved * (1 - blend_mask)
    
    final_min_height = Z_carved.min()
    final_max_height = Z_carved.max()
    max_cut = original_max_height - final_min_height if final_min_height < original_min_height else 0
    
    print(f"       ✅ Road carving complete!")
    print(f"       Final terrain range: {final_min_height:.3f} to {final_max_height:.3f}")
    print(f"       Maximum cut depth: {max_cut:.3f}")
    
    return Z_carved

def add_roads_to_terrain_cv2(X, Y, Z, road_config):
    """
    Add roads to terrain based on configuration.
    
    Parameters:
    - X, Y: Coordinate meshgrids
    - Z: Height map data
    - road_config: Road configuration dictionary
    
    Returns:
    - Modified height map with roads
    """
    if not road_config.get('enabled', False):
        print("Roads disabled - skipping road generation")
        return Z
    
    Z_with_roads = Z.copy()
    map_range = (X.min(), X.max())
    map_size = Z.shape[0]
    
    num_roads = len(road_config['roads'])
    print(f"🛣️  Starting road generation - {num_roads} road(s) configured")
    print(f"   Map bounds: {X.min():.2f} to {X.max():.2f}, size: {map_size}x{map_size}")
    
    if num_roads == 0:
        print("   No roads configured - skipping road generation")
        return Z
    
    for i, road_data in enumerate(road_config['roads']):
        print(f"🚗  Generating road {i+1}/{num_roads}:")
        print(f"     Start: {road_data['start_edge']} edge, position {road_data['start_position']:.2f}")
        print(f"     End: {road_data['end_edge']} edge, position {road_data['end_position']:.2f}")
        print(f"     Width: {road_data.get('width', 0.05):.3f}, Cut depth: {road_data.get('cut_depth', 0.1):.3f}")
        print(f"     Follow contours: {road_data.get('follow_contours', True)}")
        print(f"     Smoothness: {road_data.get('smoothness', 0.8):.2f}, Banking: {road_data.get('banking', True)}")
        
        # Generate road path
        print(f"     Generating road path...")
        road_path = generate_road_path(
            road_data['start_edge'], road_data['start_position'],
            road_data['end_edge'], road_data['end_position'],
            map_range, map_size, road_data.get('follow_contours', True)
        )
        
        print(f"     Road path generated with {len(road_path)} points")
        
        # Carve road into terrain
        print(f"     Carving road into terrain...")
        Z_with_roads = carve_road_cv2(
            X, Y, Z_with_roads, road_path,
            road_data.get('width', 0.05),
            road_data.get('cut_depth', 0.1),
            road_data.get('smoothness', 0.8),
            road_data.get('banking', True)
        )
        
        print(f"     ✅ Road {i+1} completed successfully")
    
    print(f"🛣️  All roads generated successfully!")
    return Z_with_roads

def apply_erosion_simulation_cv2(Z, iterations=3, strength=0.15):
    """
    Simple erosion simulation using OpenCV morphological operations.
    
    Parameters:
    - Z: Height map data
    - iterations: Number of erosion passes
    - strength: Erosion strength
    
    Returns:
    - Eroded height map
    """
    Z_float = Z.astype(np.float32)
    
    # Create morphological kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    for _ in range(iterations):
        # Use morphological operations for erosion simulation
        dilated = cv2.dilate(Z_float, kernel, iterations=1)
        eroded_morph = cv2.erode(Z_float, kernel, iterations=1)
        
        # Calculate slope-based erosion
        slope = dilated - Z_float
        erosion_mask = slope > 0.1
        
        # Apply erosion
        Z_float[erosion_mask] -= strength * slope[erosion_mask] * 0.5
        
        # Redistribute material to lower areas
        deposition_mask = slope < -0.05
        Z_float[deposition_mask] += strength * np.abs(slope[deposition_mask]) * 0.2
        
        # Apply slight smoothing
        Z_float = cv2.GaussianBlur(Z_float, (3, 3), 0.5)
    
    return Z_float

def apply_advanced_filters_cv2(Z, filter_type='edge_enhance'):
    """
    Apply advanced OpenCV filters for terrain enhancement.
    
    Parameters:
    - Z: Height map data
    - filter_type: Type of filter ('edge_enhance', 'sharpen', 'emboss')
    
    Returns:
    - Filtered height map
    """
    Z_float = Z.astype(np.float32)
    
    if filter_type == 'edge_enhance':
        # Edge enhancement kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]], dtype=np.float32)
        enhanced = cv2.filter2D(Z_float, -1, kernel)
        return 0.7 * Z_float + 0.3 * enhanced
    
    elif filter_type == 'sharpen':
        # Sharpening kernel
        kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(Z_float, -1, kernel)
    
    elif filter_type == 'emboss':
        # Emboss kernel
        kernel = np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [ 0,  1, 2]], dtype=np.float32)
        embossed = cv2.filter2D(Z_float, -1, kernel)
        return Z_float + 0.3 * embossed
    
    return Z_float

def generate_terrain_cv2(config=None, custom_config=None):
    """
    Generate complex terrain with multiple mountains using OpenCV optimizations.
    
    Parameters:
    - config: Configuration dictionary (uses TERRAIN_CONFIG if None)
    - custom_config: Custom mountain configuration (uses CUSTOM_MOUNTAINS if None)
    
    Returns:
    - X, Y: Coordinate meshgrids
    - Z: Height map data
    """
    if config is None:
        config = TERRAIN_CONFIG
    if custom_config is None:
        custom_config = CUSTOM_MOUNTAINS
    
    # Set random seed if specified
    if ADVANCED_CONFIG['random_seed'] is not None:
        np.random.seed(ADVANCED_CONFIG['random_seed'])
    
    # Extract configuration values
    num_mountains = config['num_mountains']
    map_size = config['map_size']
    map_range = config['map_range']
    noise_level = config['noise_level']
    
    x = np.linspace(map_range[0], map_range[1], map_size)
    y = np.linspace(map_range[0], map_range[1], map_size)
    X, Y = np.meshgrid(x, y)
    
    # Initialize terrain with base noise
    Z = np.random.normal(0, noise_level * 0.5, X.shape).astype(np.float32)
    
    # Add fractal noise for detail
    if config.get('add_fractal_noise', True):
        fractal_octaves = config.get('fractal_octaves', 4)
        fractal_persistence = config.get('fractal_persistence', 0.5)
        fractal_noise = create_fractal_noise_cv2(map_size, fractal_octaves, fractal_persistence)
        Z += fractal_noise * noise_level * 2
    
    # Use custom mountains if enabled
    if custom_config['enabled']:
        mountain_positions = custom_config['positions']
        mountain_heights = custom_config['heights']
        mountain_widths = custom_config['widths']
        mountain_types = custom_config.get('types', ['varied'] * len(mountain_positions))  # Get custom types or default to varied
        num_mountains = len(mountain_positions)
    else:
        # Generate random mountain properties with varied distribution
        mountain_positions = []
        height_distribution = ADVANCED_CONFIG.get('height_distribution', 'varied')
        
        for i in range(num_mountains):
            x_pos = np.random.uniform(map_range[0], map_range[1])
            y_pos = np.random.uniform(map_range[0], map_range[1])
            mountain_positions.append((x_pos, y_pos))
        
        height_range = config['mountain_height_range']
        width_range = config['mountain_width_range']
        
        # Ensure minimum width for better-looking mountains
        min_width = max(width_range[0], 1.2)  # Minimum width of 1.2 units
        max_width = max(width_range[1], min_width + 0.5)  # Ensure max > min
        
        if height_distribution == 'varied':
            # Create varied heights with some dominant peaks
            mountain_heights = []
            mountain_widths = []
            for i in range(num_mountains):
                if i < 2:  # First two are dominant peaks
                    height = np.random.uniform(height_range[1] * 0.8, height_range[1])
                    width = np.random.uniform(max_width * 0.8, max_width)  # Larger widths for tall mountains
                elif i < num_mountains // 2:  # Medium peaks
                    height = np.random.uniform(height_range[0] * 1.2, height_range[1] * 0.7)
                    width = np.random.uniform(min_width * 1.2, max_width * 0.8)
                else:  # Smaller hills
                    height = np.random.uniform(height_range[0], height_range[1] * 0.5)
                    width = np.random.uniform(min_width, max_width * 0.6)
                mountain_heights.append(height)
                mountain_widths.append(width)
        else:
            mountain_heights = np.random.uniform(height_range[0], height_range[1], num_mountains)
            mountain_widths = np.random.uniform(min_width, max_width, num_mountains)
    
    # Add each mountain with complex shapes
    for i in range(num_mountains):
        x_center, y_center = mountain_positions[i]
        height = mountain_heights[i]
        width = mountain_widths[i]
        
        # Determine mountain type based on configuration
        config_mountain_type = config.get('mountain_type', 'varied')
        
        # Debug: Print the mountain type configuration
        if i == 0:  # Only print once
            print(f"Mountain type config: {config_mountain_type}")
        
        # Check if using custom mountains with individual types
        if custom_config['enabled'] and 'mountain_types' in locals() and i < len(mountain_types):
            custom_mountain_type = mountain_types[i]
            print(f"Using custom mountain type: {custom_mountain_type} for mountain {i+1}")
            if custom_mountain_type == 'varied':
                # Use height-based logic for this specific mountain
                if height > height_range[1] * 0.7:
                    mountain_type = np.random.choice(['peaked', 'ridge', 'volcano', 'asymmetric'], p=[0.3, 0.3, 0.2, 0.2])
                elif height > height_range[1] * 0.4:
                    mountain_type = np.random.choice(['asymmetric', 'ridge', 'mesa', 'peaked'], p=[0.4, 0.3, 0.2, 0.1])
                else:
                    mountain_type = np.random.choice(['asymmetric', 'ridge'], p=[0.8, 0.2])
            else:
                mountain_type = custom_mountain_type
        elif config_mountain_type == 'varied':
            # Use original logic based on height and position (favor softer shapes)
            if height > height_range[1] * 0.7:
                # Tall mountains get dramatic shapes (but reduced sharp peaks)
                mountain_type = np.random.choice(['peaked', 'ridge', 'volcano', 'asymmetric'], p=[0.3, 0.3, 0.2, 0.2])
            elif height > height_range[1] * 0.4:
                # Medium mountains get varied shapes (favor gentler shapes)
                mountain_type = np.random.choice(['asymmetric', 'ridge', 'mesa', 'peaked'], p=[0.4, 0.3, 0.2, 0.1])
            else:
                # Smaller hills get gentler shapes
                mountain_type = np.random.choice(['asymmetric', 'ridge'], p=[0.8, 0.2])
        else:
            # Use the specific mountain type from configuration
            mountain_type = config_mountain_type
        
        # Create mountain shape based on generation method
        generation_method = config.get('mountain_generation_method', 'geological')  # 'geological' or 'simple'
        
        if generation_method == 'simple':
            # Use simple/original mountain generation
            mountain = create_simple_mountain_cv2(X, Y, x_center, y_center, height, width, mountain_type)
        else:
            # Use geological/realistic mountain generation (default)
            mountain = create_complex_mountain_cv2(X, Y, x_center, y_center, height, width, mountain_type)
        
        # Add additional shape variation
        variation_scale = width * 0.5
        angle_variation = np.random.uniform(0, 2*np.pi)
        shape_variation = 0.15 * height * np.sin((X - x_center) / variation_scale + angle_variation) * \
                         np.cos((Y - y_center) / variation_scale + angle_variation) * \
                         np.exp(-((X - x_center)**2 + (Y - y_center)**2) / (width**2 * 2))
        mountain += shape_variation
        
        # Add mountain to terrain using maximum to preserve peaks
        Z = np.maximum(Z, mountain.astype(np.float32))
        
        print(f"    Mountain {i+1}: {mountain_type} at ({x_center:.1f}, {y_center:.1f}), height {height:.2f}")
    
    # Add ridges and valleys for complexity
    Z = add_ridges_and_valleys_cv2(X, Y, Z, config)
    
    # Apply erosion simulation if enabled
    if SMOOTHING_CONFIG.get('erosion_simulation', {}).get('enabled', False):
        erosion_config = SMOOTHING_CONFIG['erosion_simulation']
        Z = apply_erosion_simulation_cv2(Z, erosion_config['iterations'], erosion_config['strength'])
    
    # Fix any artifacts that may have been introduced
    print("Checking for and fixing terrain artifacts...")
    Z = fix_terrain_artifacts_cv2(Z, artifact_threshold=0.95)
    
    # Apply final gentle smoothing to ensure natural appearance
    print("Applying final terrain smoothing...")
    Z_smooth = cv2.GaussianBlur(Z.astype(np.float32), (3, 3), 0.5)
    # Blend original with smoothed (preserve detail while reducing artifacts)
    Z = 0.85 * Z + 0.15 * Z_smooth
    
    # Add roads first
    Z = add_roads_to_terrain_cv2(X, Y, Z, ROAD_CONFIG)
    
    # Add rivers after roads (so rivers can flow over/around roads)
    Z = add_rivers_to_terrain_cv2(X, Y, Z, RIVER_CONFIG)
    
    # Ensure no negative values (but allow rivers to go below original terrain)
    Z = np.maximum(Z, np.min(Z))
    
    return X, Y, Z

def save_heightmap_as_tiff_cv2(data, filename, config=None):
    """
    Save height map data as a TIFF image using OpenCV.
    
    Parameters:
    - data: 2D array containing height values
    - filename: Output filename (will be processed based on config)
    - config: Export configuration dictionary (uses EXPORT_CONFIG if None)
    """
    import os
    from datetime import datetime
    
    if config is None:
        config = EXPORT_CONFIG
    
    # Create maps folder if it doesn't exist
    maps_folder = "maps"
    if not os.path.exists(maps_folder):
        os.makedirs(maps_folder)
        print(f"Created folder: {maps_folder}")
    
    # Process filename
    if config['include_timestamp']:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext}"
    
    # Add prefix if specified
    if config['prefix']:
        filename = config['prefix'] + filename
    
    # Ensure the filename has .tif extension
    if not filename.lower().endswith(('.tif', '.tiff')):
        filename += '.tif'
    
    # Add maps folder to path
    filepath = os.path.join(maps_folder, filename)
    
    bit_depth = config['bit_depth']
    normalize = ADVANCED_CONFIG['normalize_output']
    
    if normalize:
        if bit_depth == 16:
            data_norm = ((data - data.min()) / (data.max() - data.min()) * 65535).astype(np.uint16)
        else:
            data_norm = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
    else:
        if bit_depth == 16:
            data_norm = data.astype(np.uint16)
        else:
            data_norm = data.astype(np.uint8)
    
    # Save using OpenCV
    success = cv2.imwrite(filepath, data_norm)
    if success:
        print(f"Height map saved as: {os.path.abspath(filepath)} ({bit_depth}-bit)")
    else:
        print(f"Failed to save {filepath}")

def create_height_map_cv2(data, title='Height Map', save_as_tiff=None, filename=None, show_image=True):
    """
    Create and display a height map using OpenCV.

    Parameters:
    - data: 2D array-like structure containing height values.
    - title: Title of the image window.
    - save_as_tiff: Whether to save TIFF (uses EXPORT_CONFIG if None).
    - filename: Filename for saving (optional, will auto-generate if not provided).
    - show_image: Whether to display the image.
    """
    if save_as_tiff is None:
        save_as_tiff = EXPORT_CONFIG['save_tiff']
    
    # Normalize data for display (0-255)
    data_display = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
    
    # Apply colormap for better visualization
    # Use COLORMAP_JET as it provides good terrain-like colors
    colored = cv2.applyColorMap(data_display, cv2.COLORMAP_JET)
    
    # Display image if enabled
    if show_image and DISPLAY_CONFIG['show_plots']:
        # Resize for display if too large
        display_data = colored
        if data.shape[0] > 800:
            scale = 800 / data.shape[0]
            new_width = int(data.shape[1] * scale)
            new_height = int(data.shape[0] * scale)
            display_data = cv2.resize(colored, (new_width, new_height))
        
        cv2.imshow(title, display_data)
        cv2.waitKey(0)  # Wait for key press
        cv2.destroyAllWindows()
        
        # Save plot as PNG if enabled
        if DISPLAY_CONFIG['save_plot_images']:
            import os
            # Create maps folder if it doesn't exist
            maps_folder = "maps"
            if not os.path.exists(maps_folder):
                os.makedirs(maps_folder)
            
            plot_filename = title.replace(' ', '_').replace('(', '').replace(')', '').lower() + '_plot.png'
            plot_filepath = os.path.join(maps_folder, plot_filename)
            cv2.imwrite(plot_filepath, colored)
            print(f"Plot saved as: {plot_filepath}")
    
    # Save as TIFF if requested
    if save_as_tiff:
        if filename is None:
            # Auto-generate filename from title
            filename = title.replace(' ', '_').replace('(', '').replace(')', '').lower() + '.tif'
        save_heightmap_as_tiff_cv2(data, filename)

if __name__ == "__main__":
    print("=== Height Map Generator (OpenCV Version) ===")
    print(f"Configuration: {TERRAIN_CONFIG['num_mountains']} mountains, {TERRAIN_CONFIG['map_size']}x{TERRAIN_CONFIG['map_size']} resolution")
    
    # Generate terrain using configuration
    print("\nGenerating terrain from config...")
    X, Y, Z = generate_terrain_cv2()
    
    # Show original terrain
    create_height_map_cv2(Z, title='Original Terrain CV2', filename='original_terrain_cv2.tif')
    
    # Apply OpenCV smoothing if enabled in config
    if SMOOTHING_CONFIG['gaussian']['enabled']:
        print("Applying Gaussian smoothing with OpenCV...")
        gaussian_config = SMOOTHING_CONFIG['gaussian']
        Z_smooth_gaussian = use_convolution_cv2(
            Z, 
            kernel_type='gaussian', 
            kernel_size=gaussian_config['kernel_size'], 
            sigma=gaussian_config['sigma']
        )
        create_height_map_cv2(Z_smooth_gaussian, title='Gaussian Smoothed Terrain CV2', 
                        filename='gaussian_smoothed_terrain_cv2.tif')
    
    # Apply bilateral filter for edge-preserving smoothing
    print("Applying bilateral filter (edge-preserving)...")
    Z_bilateral = use_convolution_cv2(Z, kernel_type='bilateral', kernel_size=9, sigma=2.0)
    create_height_map_cv2(Z_bilateral, title='Bilateral Filtered Terrain CV2', 
                    filename='bilateral_filtered_terrain_cv2.tif')
    
    # Apply advanced filters
    print("Applying edge enhancement...")
    Z_enhanced = apply_advanced_filters_cv2(Z, 'edge_enhance')
    create_height_map_cv2(Z_enhanced, title='Edge Enhanced Terrain CV2', 
                    filename='edge_enhanced_terrain_cv2.tif')
    
    if SMOOTHING_CONFIG['box_filter']['enabled']:
        print("Applying box filter smoothing with OpenCV...")
        box_config = SMOOTHING_CONFIG['box_filter']
        Z_smooth_box = use_convolution_cv2(
            Z, 
            kernel_type='box', 
            kernel_size=box_config['kernel_size']
        )
        create_height_map_cv2(Z_smooth_box, title='Box Filter Smoothed Terrain CV2', 
                        filename='box_smoothed_terrain_cv2.tif')
    
    # Generate high-resolution terrain if enabled
    if HIGH_RES_CONFIG['enabled']:
        print(f"\nGenerating high-resolution terrain ({HIGH_RES_CONFIG['map_size']}x{HIGH_RES_CONFIG['map_size']})...")
        
        # Create temporary config for high-res
        high_res_terrain_config = TERRAIN_CONFIG.copy()
        high_res_terrain_config.update({
            'map_size': HIGH_RES_CONFIG['map_size'],
            'num_mountains': HIGH_RES_CONFIG['num_mountains'],
            'noise_level': HIGH_RES_CONFIG['noise_level']
        })
        
        X3, Y3, Z3 = generate_terrain_cv2(config=high_res_terrain_config)
        
        # Save without displaying if show_plots is False
        if DISPLAY_CONFIG['show_plots']:
            create_height_map_cv2(Z3, title='High Resolution Terrain CV2', 
                            filename='high_res_terrain_cv2.tif')
        else:
            save_heightmap_as_tiff_cv2(Z3, 'high_res_terrain_cv2.tif')
            print("High-resolution terrain saved without displaying.")
    
    print("\n=== Generation Complete ===")
    print("Check the generated TIFF files in the 'maps' folder.")
    print("Files are saved with '_cv2' suffix to distinguish from matplotlib version.")
    print("Press any key in the image windows to proceed to the next image.")
    print("Modify config.py to customize terrain generation settings.")
