# Height Map Generator Configuration
# Modify these settings to customize your terrain generation

# Terrain Generation Settings
TERRAIN_CONFIG = {
    # Basic terrain parameters
    'num_mountains': 16,          # Number of mountains to generate
    'map_size': 1025,           # Resolution of the height map (map_size x map_size)
    'map_range': (-10, 10),       # Coordinate range (min, max) for the map
    'noise_level': 0.06,        # Amount of random noise (0.0 to 0.3)
    
    # Mountain generation method
    'mountain_generation_method': 'geological',  # 'geological' for realistic terrain or 'simple' for basic shapes
    
    # Mountain type selection
    'mountain_type': 'varied',    # Mountain type: 'varied' (random mix), 'peaked', 'ridge', 'mesa', 'volcano', 'asymmetric'
    
    # Mountain generation ranges (used when positions not specified)
    'mountain_height_range': (0.8, 5.0),   # Min/max heights for random mountains
    'mountain_width_range': (1.2, 4.0),    # Min/max widths for random mountains (increased minimum)
    
    # Complex terrain features
    'add_ridges': True,         # Add mountain ridges
    'add_valleys': True,        # Add valleys between mountains
    'add_fractal_noise': True,  # Add fractal noise for detail
    'ridge_strength': 0.25,     # Strength of ridge lines (increased for visibility)
    'valley_depth': 0.35,       # Depth of valleys (increased for visibility)
    'fractal_octaves': 2,       # Number of fractal noise layers (reduced for smoother terrain)
    'fractal_persistence': 0.4, # Fractal noise persistence (reduced for less sharp detail)
}

# Custom Mountain Positions (optional - leave None for random placement)
CUSTOM_MOUNTAINS = {
    'enabled': False,  # Set to True to use custom mountain placement
    'positions': [(-5, -5), (-7, 1), (1, -7), (8, 2), (-7, -1)],  # (x, y) coordinates
    'heights': [1.8, 1.2, 1.5, 1.0, 0.8],                       # Height multipliers
    'widths': [1.0, 1.5, 1.8, 1.2, 1.0],                        # Width multipliers
    'types': ['peaked', 'ridge', 'mesa', 'volcano', 'asymmetric'], # Mountain types for each position
}

# Convolution/Smoothing Settings
SMOOTHING_CONFIG = {
    'gaussian': {
        'enabled': True,
        'kernel_size': 5,
        'sigma': 1.2,
    },
    'box_filter': {
        'enabled': False,
        'kernel_size': 7,
    },
    'erosion_simulation': {
        'enabled': True,        # Simulate natural erosion
        'iterations': 3,
        'strength': 0.15,
    },
}

# River Generation Settings
# Rivers automatically follow terrain elevation, flowing from higher to lower areas
# Enhanced mountain avoidance: Rivers will actively avoid high elevations and seek valleys
# If specified end point is higher than start point, the river direction will be automatically swapped
RIVER_CONFIG = {
    'enabled': True,            # Set to True to generate rivers
    'flat_water_surface': True, # If True, rivers maintain consistent water level (realistic)
                               # If False, rivers follow terrain elevation changes
    'rivers': [
        # River 1: Top to bottom with tributaries
        {
            'start_edge': 'top',        # 'top', 'bottom', 'left', 'right'
            'start_position': 0.55,      # Position along edge (0.0 to 1.0)
            'end_edge': 'right',       # 'top', 'bottom', 'left', 'right'
            'end_position': 0.2,        # Position along edge (0.0 to 1.0)
            'width': 0.2,              # River width (0.05 to 0.5)
            'depth': 0.2,               # River depth multiplier
            'meander': 0.3,             # River meandering amount (increased for mountain avoidance)
            'tributaries': 0,           # Number of tributaries (0-5)
        },
        # River 2: Left to right
        {
            'start_edge': 'top',
            'start_position': 0.15,
            'end_edge': 'bottom',
            'end_position': 0.2,
            'width': 0.2,
            'depth': 0.3,
            'meander': 0.3,             # Increased meandering for mountain avoidance
            'tributaries': 0,
        },
        # Add more rivers as needed...
        # Example River 3: Diagonal flow
        # {
        #     'start_edge': 'top',
        #     'start_position': 0.8,
        #     'end_edge': 'right',
        #     'end_position': 0.2,
        #     'width': 0.08,
        #     'depth': 0.2,
        #     'meander': 0.4,
        #     'tributaries': 0,
        # },
    ]
}

# Road Generation Settings
# Roads follow terrain elevation, creating paths that cut through landscape
# Roads can connect different edges or create networks across the terrain
ROAD_CONFIG = {
    'enabled': True,           # Set to True to generate roads
    'roads': [
        # Road 1: Mountain pass road
        {
            'start_edge': 'left',       # 'top', 'bottom', 'left', 'right'
            'start_position': 0.3,      # Position along edge (0.0 to 1.0)
            'end_edge': 'right',        # 'top', 'bottom', 'left', 'right'
            'end_position': 0.7,        # Position along edge (0.0 to 1.0)
            'width': 0.12,              # Road width (increased for visibility)
            'cut_depth': 0.25,          # How deep to cut into terrain (increased)
            'smoothness': 0.8,          # Road smoothness (0.0 = rough, 1.0 = very smooth)
            'banking': True,            # Add banking/embankments on sides
            'follow_contours': True,    # Try to follow elevation contours when possible
        },
        # Road 2: Valley road
        {
            'start_edge': 'top',
            'start_position': 0.2,
            'end_edge': 'bottom',
            'end_position': 0.5,
            'width': 0.10,              # Road width (increased for visibility)
            'cut_depth': 0.20,          # How deep to cut into terrain (increased)
            'smoothness': 0.9,
            'banking': True,
            'follow_contours': False,   # Direct route
        },
        # Example Road 3: Connecting road (enabled for better visibility)
        {
            'start_edge': 'top',
            'start_position': 0.8,
            'end_edge': 'right',
            'end_position': 0.4,
            'width': 0.08,              # Medium width road
            'cut_depth': 0.15,          # Medium cut depth
            'smoothness': 0.7,
            'banking': True,
            'follow_contours': True,
        },
    ]
}

# Export Settings
EXPORT_CONFIG = {
    'save_tiff': True,          # Whether to save TIFF files
    'bit_depth': 16,            # 8 or 16 bit depth
    'dpi': 72,                  # DPI resolution
    'compression': 'lzw',       # TIFF compression
    
    # File naming
    'prefix': 'complex_terrain_',  # Prefix for auto-generated filenames
    'include_timestamp': False, # Add timestamp to filenames
}

# Display Settings
DISPLAY_CONFIG = {
    'show_plots': True,         # Whether to display matplotlib plots
    'figure_size': (10, 8),     # Size of the plot figures
    'colormap': 'terrain',      # Default colormap ('terrain', 'viridis', 'binary', etc.)
    'save_plot_images': False,  # Save plots as PNG images
}

# High Resolution Export Settings (for export-only generation)
HIGH_RES_CONFIG = {
    'enabled': False,
    'map_size': 2049,           # Higher resolution for export
    'num_mountains': 12,        # More mountains for complexity
    'noise_level': 0.06,
    'filename': 'complex_high_res_terrain.tif',
}

# Advanced Settings
ADVANCED_CONFIG = {
    'random_seed': 2,         #2 # Set to integer for reproducible results (None for random)
    'normalize_output': True,   # Normalize height data to full range
    'boundary_mode': 'wrap',    # Convolution boundary mode ('wrap', 'fill', 'symm')
    'height_distribution': 'varied',  # 'uniform', 'varied', 'peaked'
}
