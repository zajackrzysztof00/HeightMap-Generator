"""
UI-compatible wrapper for the height map generator.
This module provides functions that can send progress updates and messages
to the GUI interface.
"""

import sys
import io
import contextlib
import queue
import numpy as np
import cv2
from config.config import (
    TERRAIN_CONFIG, CUSTOM_MOUNTAINS, SMOOTHING_CONFIG, 
    EXPORT_CONFIG, DISPLAY_CONFIG, HIGH_RES_CONFIG, ADVANCED_CONFIG, 
    RIVER_CONFIG, ROAD_CONFIG
)

# Global queues for UI communication
message_queue = None
progress_queue = None

def set_ui_queues(msg_queue, prog_queue):
    """Set the queues for UI communication"""
    global message_queue, progress_queue
    message_queue = msg_queue
    progress_queue = prog_queue

def send_message(message):
    """Send a message to the UI"""
    if message_queue:
        message_queue.put(message)
    print(message)  # Also print to console

def send_progress(progress):
    """Send progress update to the UI"""
    if progress_queue:
        progress_queue.put(progress)

@contextlib.contextmanager
def capture_output():
    """Capture print statements and send them to UI"""
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    try:
        yield captured_output
    finally:
        sys.stdout = old_stdout
        output = captured_output.getvalue()
        if output.strip():
            send_message(output.strip())

def generate_terrain_with_progress():
    """Generate terrain with progress updates for UI"""
    
    send_progress(5)
    send_message("Starting terrain generation...")
    
    # Import the main generation function
    from .main_cv2 import generate_terrain_cv2
    
    send_progress(10)
    send_message("Configuration loaded...")
    
    # Extract configuration values
    num_mountains = TERRAIN_CONFIG['num_mountains']
    map_size = TERRAIN_CONFIG['map_size']
    map_range = TERRAIN_CONFIG['map_range']
    noise_level = TERRAIN_CONFIG['noise_level']
    
    send_progress(15)
    send_message(f"Generating {num_mountains} mountains on {map_size}x{map_size} map...")
    
    # Set random seed if specified
    if ADVANCED_CONFIG['random_seed'] is not None:
        np.random.seed(ADVANCED_CONFIG['random_seed'])
        send_message(f"Using random seed: {ADVANCED_CONFIG['random_seed']}")
    
    send_progress(20)
    send_message("Creating coordinate meshgrids...")
    
    x = np.linspace(map_range[0], map_range[1], map_size)
    y = np.linspace(map_range[0], map_range[1], map_size)
    X, Y = np.meshgrid(x, y)
    
    send_progress(25)
    send_message("Initializing terrain with base noise...")
    
    # Initialize terrain with base noise
    Z = np.random.normal(0, noise_level * 0.5, X.shape).astype(np.float32)
    
    # Add fractal noise for detail
    if TERRAIN_CONFIG.get('add_fractal_noise', True):
        send_progress(30)
        send_message("Adding fractal noise for detail...")
        from .main_cv2 import create_fractal_noise_cv2
        fractal_octaves = TERRAIN_CONFIG.get('fractal_octaves', 4)
        fractal_persistence = TERRAIN_CONFIG.get('fractal_persistence', 0.5)
        fractal_noise = create_fractal_noise_cv2(map_size, fractal_octaves, fractal_persistence)
        Z += fractal_noise * noise_level * 2
    
    send_progress(35)
    send_message("Generating mountain configurations...")
    
    # Use custom mountains if enabled
    if CUSTOM_MOUNTAINS['enabled']:
        mountain_positions = CUSTOM_MOUNTAINS['positions']
        mountain_heights = CUSTOM_MOUNTAINS['heights']
        mountain_widths = CUSTOM_MOUNTAINS['widths']
        mountain_types = CUSTOM_MOUNTAINS.get('types', ['varied'] * len(mountain_positions))
        num_mountains = len(mountain_positions)
        send_message(f"Using {num_mountains} custom mountain positions")
    else:
        # Generate random mountain properties
        mountain_positions = []
        mountain_types = None  # Will use global config
        height_distribution = ADVANCED_CONFIG.get('height_distribution', 'varied')
        
        for i in range(num_mountains):
            x_pos = np.random.uniform(map_range[0], map_range[1])
            y_pos = np.random.uniform(map_range[0], map_range[1])
            mountain_positions.append((x_pos, y_pos))
        
        height_range = TERRAIN_CONFIG['mountain_height_range']
        width_range = TERRAIN_CONFIG['mountain_width_range']
        
        if height_distribution == 'varied':
            # Create varied heights with some dominant peaks
            mountain_heights = []
            for i in range(num_mountains):
                if i < 2:  # First two are dominant peaks
                    height = np.random.uniform(height_range[1] * 0.8, height_range[1])
                elif i < num_mountains // 2:  # Medium peaks
                    height = np.random.uniform(height_range[0] * 1.2, height_range[1] * 0.7)
                else:  # Smaller hills
                    height = np.random.uniform(height_range[0], height_range[1] * 0.5)
                mountain_heights.append(height)
        else:
            mountain_heights = np.random.uniform(height_range[0], height_range[1], num_mountains)
        
        mountain_widths = np.random.uniform(width_range[0], width_range[1], num_mountains)
    
    # Add each mountain with method selection
    from .main_cv2 import create_complex_mountain_cv2, create_simple_mountain_cv2
    
    progress_step = 35 / num_mountains  # Progress from 40 to 75
    
    # Get the mountain generation method from config
    generation_method = TERRAIN_CONFIG.get('mountain_generation_method', 'geological')
    send_message(f"Using {generation_method} mountain generation method")
    
    for i in range(num_mountains):
        x_center, y_center = mountain_positions[i]
        height = mountain_heights[i]
        width = mountain_widths[i]
        
        # Determine mountain type based on configuration
        config_mountain_type = TERRAIN_CONFIG.get('mountain_type', 'varied')
        
        # Check if using custom mountains with individual types
        if CUSTOM_MOUNTAINS['enabled'] and mountain_types and i < len(mountain_types):
            custom_mountain_type = mountain_types[i]
            if custom_mountain_type == 'varied':
                # Use height-based logic for this specific mountain
                height_range = TERRAIN_CONFIG['mountain_height_range']
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
            height_range = TERRAIN_CONFIG['mountain_height_range']
            if height > height_range[1] * 0.7:
                mountain_type = np.random.choice(['peaked', 'ridge', 'volcano', 'asymmetric'], p=[0.3, 0.3, 0.2, 0.2])
            elif height > height_range[1] * 0.4:
                mountain_type = np.random.choice(['asymmetric', 'ridge', 'mesa', 'peaked'], p=[0.4, 0.3, 0.2, 0.1])
            else:
                mountain_type = np.random.choice(['asymmetric', 'ridge'], p=[0.8, 0.2])
        else:
            # Use the specific mountain type from configuration
            mountain_type = config_mountain_type
        
        send_progress(40 + i * progress_step)
        send_message(f"Creating mountain {i+1}/{num_mountains}: {mountain_type} at ({x_center:.1f}, {y_center:.1f}), height {height:.2f}")
        
        # Create mountain shape based on generation method
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
    
    send_progress(75)
    send_message("Adding ridges and valleys for complexity...")
    
    # Add ridges and valleys for complexity
    from .main_cv2 import add_ridges_and_valleys_cv2
    Z = add_ridges_and_valleys_cv2(X, Y, Z, TERRAIN_CONFIG)
    
    # Apply erosion simulation if enabled
    if SMOOTHING_CONFIG.get('erosion_simulation', {}).get('enabled', False):
        send_progress(78)
        send_message("Applying erosion simulation...")
        from .main_cv2 import apply_erosion_simulation_cv2
        erosion_config = SMOOTHING_CONFIG['erosion_simulation']
        Z = apply_erosion_simulation_cv2(Z, erosion_config['iterations'], erosion_config['strength'])
    
    # Add roads first
    if ROAD_CONFIG.get('enabled', False):
        send_progress(80)
        send_message(f"Adding {len(ROAD_CONFIG['roads'])} road(s) to terrain...")
        from .main_cv2 import add_roads_to_terrain_cv2
        with capture_output():
            Z = add_roads_to_terrain_cv2(X, Y, Z, ROAD_CONFIG)
    
    # Add rivers after roads
    if RIVER_CONFIG.get('enabled', False):
        send_progress(85)
        send_message(f"Adding {len(RIVER_CONFIG['rivers'])} river(s) to terrain...")
        from .main_cv2 import add_rivers_to_terrain_cv2
        with capture_output():
            Z = add_rivers_to_terrain_cv2(X, Y, Z, RIVER_CONFIG)
    
    send_progress(90)
    send_message("Finalizing terrain...")
    
    # Ensure no negative values (but allow rivers to go below original terrain)
    Z = np.maximum(Z, np.min(Z))
    
    send_progress(95)
    send_message("Terrain generation complete!")
    
    # Save the generated terrain
    if EXPORT_CONFIG.get('save_tiff', True):
        send_progress(98)
        send_message("Saving terrain as TIFF...")
        from .main_cv2 import save_heightmap_as_tiff_cv2
        save_heightmap_as_tiff_cv2(Z, 'ui_generated_terrain.tif')
    
    send_progress(100)
    send_message("Map generation finished successfully!")
    
    return X, Y, Z

def generate_with_ui_integration(terrain_config, custom_mountains, river_config, road_config, 
                                smoothing_config, export_config, advanced_config, 
                                msg_queue, prog_queue):
    """
    Generate terrain with full UI integration.
    Updates global configurations and generates terrain with progress reporting.
    """
    
    # Set up UI communication
    set_ui_queues(msg_queue, prog_queue)
    
    # Update global configurations
    TERRAIN_CONFIG.update(terrain_config)
    CUSTOM_MOUNTAINS.update(custom_mountains)
    RIVER_CONFIG.update(river_config)
    ROAD_CONFIG.update(road_config)
    SMOOTHING_CONFIG.update(smoothing_config)
    EXPORT_CONFIG.update(export_config)
    ADVANCED_CONFIG.update(advanced_config)
    
    # Generate terrain with progress updates
    return generate_terrain_with_progress()

def apply_smoothing_with_progress(Z, smoothing_config):
    """Apply smoothing operations with progress updates"""
    
    results = {'original': Z}
    
    # Apply Gaussian smoothing if enabled
    if smoothing_config['gaussian']['enabled']:
        send_progress(20)
        send_message("Applying Gaussian smoothing...")
        from .main_cv2 import use_convolution_cv2
        gaussian_config = smoothing_config['gaussian']
        Z_smooth_gaussian = use_convolution_cv2(
            Z, 
            kernel_type='gaussian', 
            kernel_size=gaussian_config['kernel_size'], 
            sigma=gaussian_config['sigma']
        )
        results['gaussian'] = Z_smooth_gaussian
    
    # Apply bilateral filter
    send_progress(40)
    send_message("Applying bilateral filter (edge-preserving)...")
    from .main_cv2 import use_convolution_cv2
    Z_bilateral = use_convolution_cv2(Z, kernel_type='bilateral', kernel_size=9, sigma=2.0)
    results['bilateral'] = Z_bilateral
    
    # Apply edge enhancement
    send_progress(60)
    send_message("Applying edge enhancement...")
    from .main_cv2 import apply_advanced_filters_cv2
    Z_enhanced = apply_advanced_filters_cv2(Z, 'edge_enhance')
    results['enhanced'] = Z_enhanced
    
    # Apply box filter if enabled
    if smoothing_config['box_filter']['enabled']:
        send_progress(80)
        send_message("Applying box filter smoothing...")
        from .main_cv2 import use_convolution_cv2
        box_config = smoothing_config['box_filter']
        Z_smooth_box = use_convolution_cv2(
            Z, 
            kernel_type='box', 
            kernel_size=box_config['kernel_size']
        )
        results['box'] = Z_smooth_box
    
    send_progress(100)
    send_message("Smoothing operations complete!")
    
    return results
