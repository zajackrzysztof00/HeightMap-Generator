import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font
import threading
import queue
import numpy as np
import cv2
from PIL import Image, ImageTk
import sys
import os
import subprocess
from io import StringIO
import contextlib
import traceback
import datetime

# Import configuration
try:
    from config.config import (
        TERRAIN_CONFIG, CUSTOM_MOUNTAINS, SMOOTHING_CONFIG, 
        EXPORT_CONFIG, DISPLAY_CONFIG, HIGH_RES_CONFIG, ADVANCED_CONFIG, 
        RIVER_CONFIG, ROAD_CONFIG
    )
except ImportError as e:
    print(f"Error importing configuration: {e}")
    # Create default configurations if import fails
    TERRAIN_CONFIG = {'num_mountains': 16, 'map_size': 1025, 'noise_level': 0.08, 
                     'mountain_generation_method': 'geological', 'mountain_type': 'varied',
                     'mountain_height_range': (0.8, 5.0), 'mountain_width_range': (1.2, 4.0),
                     'add_ridges': True, 'add_valleys': True, 'add_fractal_noise': True,
                     'ridge_strength': 0.25, 'valley_depth': 0.35}
    CUSTOM_MOUNTAINS = {'enabled': False, 'positions': [], 'heights': [], 'widths': [], 'types': []}
    SMOOTHING_CONFIG = {'gaussian': {'enabled': True, 'kernel_size': 5, 'sigma': 1.2},
                       'box_filter': {'enabled': False, 'kernel_size': 7},
                       'erosion_simulation': {'enabled': True, 'iterations': 3, 'strength': 0.15}}
    RIVER_CONFIG = {'enabled': True, 'rivers': []}
    ROAD_CONFIG = {'enabled': True, 'roads': []}
    EXPORT_CONFIG = {'save_tiff': True, 'bit_depth': 16}
    ADVANCED_CONFIG = {'random_seed': None}

class HeightMapGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Height Map Generator - Configuration & Preview")
        self.root.geometry("1800x1000")  # Increased width for 3-column layout
        self.root.minsize(1400, 800)     # Increased minimum width for 3 columns
        
        # Enable window resizing
        self.root.resizable(True, True)
        
        # Configure modern styling
        self.setup_styles()
        
        # Threading
        self.generation_thread = None
        self.message_queue = queue.Queue()
        self.progress_queue = queue.Queue()
        self.is_generating = False
        
        # Configuration copies (to avoid modifying original while editing)
        self.terrain_config = TERRAIN_CONFIG.copy()
        self.custom_mountains = CUSTOM_MOUNTAINS.copy()
        self.smoothing_config = SMOOTHING_CONFIG.copy()
        self.river_config = RIVER_CONFIG.copy()
        self.road_config = ROAD_CONFIG.copy()
        self.export_config = EXPORT_CONFIG.copy()
        self.advanced_config = ADVANCED_CONFIG.copy()
        
        # Current map data
        self.current_map = None
        
        self.setup_ui()
        self.update_ui_from_config()
        
        # Start message polling
        self.root.after(100, self.check_messages)
    
    def setup_styles(self):
        """Setup modern styling for the UI"""
        # Configure style for ttk widgets
        style = ttk.Style()
        
        # Set a modern theme
        style.theme_use('clam')
        
        # Configure custom styles with better visibility
        style.configure('Header.TLabel', 
                       font=('Segoe UI', 14, 'bold'),
                       foreground='#1a252f',
                       background='#f8f9fa')
        
        style.configure('Subheader.TLabel', 
                       font=('Segoe UI', 11, 'bold'),
                       foreground='#2c3e50')
        
        style.configure('Info.TLabel', 
                       font=('Segoe UI', 10),
                       foreground='#495057')
        
        style.configure('Success.TLabel', 
                       font=('Segoe UI', 10, 'bold'),
                       foreground='#198754')
        
        style.configure('Error.TLabel', 
                       font=('Segoe UI', 10, 'bold'),
                       foreground='#dc3545')
        
        # Enhanced button styles
        style.configure('Generate.TButton',
                       font=('Segoe UI', 11, 'bold'),
                       padding=(25, 12))
        
        style.configure('Action.TButton',
                       font=('Segoe UI', 10),
                       padding=(15, 8))
        
        # Enhanced notebook tabs
        style.configure('TNotebook.Tab',
                       font=('Segoe UI', 10),
                       padding=(15, 10))
        
        # Default label and widget fonts
        style.configure('TLabel',
                       font=('Segoe UI', 10))
        
        style.configure('TEntry',
                       font=('Segoe UI', 10))
        
        style.configure('TCombobox',
                       font=('Segoe UI', 10))
        
        style.configure('TCheckbutton',
                       font=('Segoe UI', 10))
        
        # Enhanced Treeview (table) styles
        style.configure('Treeview',
                       font=('Segoe UI', 11),
                       rowheight=25)
        
        style.configure('Treeview.Heading',
                       font=('Segoe UI', 11, 'bold'),
                       foreground='#1a252f',
                       background='#e9ecef',
                       relief='flat')
        
        # Progress bar
        style.configure('TProgressbar',
                       thickness=20)
        
        # Configure colors
        self.colors = {
            'bg_primary': '#f8f9fa',
            'bg_secondary': '#e9ecef',
            'accent': '#007bff',
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545',
            'text_primary': '#212529',
            'text_secondary': '#6c757d',
            'border': '#dee2e6'
        }
        
        # Configure root background
        self.root.configure(bg=self.colors['bg_primary'])
        
        # Create formatted value variables for display
        self.formatted_vars = {}
    
    def create_simple_scale(self, parent, label_text, var_name, var_value, from_val, to_val, decimal_places=2):
        """Create a scale with simple layout like Advanced tab"""
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=2)
        
        # Label on left
        ttk.Label(frame, text=label_text).pack(side="left")
        
        # Create the variable and formatted display variable
        self.terrain_vars[var_name] = tk.DoubleVar(value=var_value)
        self.formatted_vars[var_name] = tk.StringVar()
        
        # Create formatted value label on right
        value_label = ttk.Label(frame, textvariable=self.formatted_vars[var_name])
        value_label.pack(side="right")
        
        # Create the scale in the middle
        scale = ttk.Scale(frame, from_=from_val, to=to_val, 
                         variable=self.terrain_vars[var_name], 
                         orient="horizontal")
        scale.pack(side="right", fill="x", expand=True, padx=(10, 10))
        
        # Update formatted value initially and on change
        def update_formatted_value(*args):
            value = self.terrain_vars[var_name].get()
            self.formatted_vars[var_name].set(f"{value:.{decimal_places}f}")
        
        self.terrain_vars[var_name].trace('w', update_formatted_value)
        update_formatted_value()  # Initial update
        
        return scale
    
    def create_formatted_smoothing_scale(self, parent, label_text, var_name, var_value, from_val, to_val, decimal_places=2):
        """Create a scale for smoothing parameters with properly formatted value display using grid layout"""
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=2)
        
        # Configure frame grid for responsive behavior
        frame.grid_columnconfigure(1, weight=1)  # Scale column expands
        
        # Label
        ttk.Label(frame, text=label_text).grid(row=0, column=0, sticky="w", padx=(0, 10))
        
        # Create the variable and formatted display variable
        self.smoothing_vars[var_name] = tk.DoubleVar(value=var_value)
        if not hasattr(self, 'smoothing_formatted_vars'):
            self.smoothing_formatted_vars = {}
        self.smoothing_formatted_vars[var_name] = tk.StringVar()
        
        # Create the scale with grid
        scale = ttk.Scale(frame, from_=from_val, to=to_val, 
                         variable=self.smoothing_vars[var_name], 
                         orient="horizontal")
        scale.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        
        # Create formatted value label
        value_label = ttk.Label(frame, textvariable=self.smoothing_formatted_vars[var_name], style='Info.TLabel')
        value_label.grid(row=0, column=2, sticky="e")
        
        # Update formatted value initially and on change
        def update_formatted_value(*args):
            value = self.smoothing_vars[var_name].get()
            self.smoothing_formatted_vars[var_name].set(f"{value:.{decimal_places}f}")
        
        self.smoothing_vars[var_name].trace('w', update_formatted_value)
        update_formatted_value()  # Initial update
        
        return scale
    
    def setup_ui(self):
        """Setup the main UI with 3-column grid layout"""
        # Configure root window for responsive behavior
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create main container frame
        main_container = ttk.Frame(self.root)
        main_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Configure main container grid for 3 columns
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1, minsize=350)  # Config panel
        main_container.grid_columnconfigure(1, weight=2, minsize=500)  # Preview & progress panel
        main_container.grid_columnconfigure(2, weight=1, minsize=400)  # Console panel
        
        # Column 1: Configuration panel (left)
        config_frame = ttk.Frame(main_container, padding=5)
        config_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        # Configure config frame for internal responsiveness
        config_frame.grid_rowconfigure(1, weight=1)  # Notebook expands
        config_frame.grid_columnconfigure(0, weight=1)
        
        # Column 2: Preview and progress panel (middle)
        preview_frame = ttk.Frame(main_container, padding=5)
        preview_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 5))
        
        # Configure preview frame for internal responsiveness
        preview_frame.grid_rowconfigure(1, weight=1)  # Preview gets most space
        preview_frame.grid_rowconfigure(2, weight=0)  # Progress/controls fixed
        preview_frame.grid_columnconfigure(0, weight=1)
        
        # Column 3: Console panel (right)
        console_frame = ttk.Frame(main_container, padding=5)
        console_frame.grid(row=0, column=2, sticky="nsew")
        
        # Configure console frame
        console_frame.grid_rowconfigure(1, weight=1)  # Console expands
        console_frame.grid_columnconfigure(0, weight=1)
        
        # Setup panels
        self.setup_config_panel(config_frame)
        self.setup_preview_and_progress_panel(preview_frame)
        self.setup_console_panel(console_frame)
    
    def setup_config_panel(self, parent):
        """Setup configuration panel with responsive layout"""
        # Add header with grid layout
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header_frame.grid_columnconfigure(0, weight=1)
        
        header_label = ttk.Label(header_frame, text="Configuration", style='Header.TLabel')
        header_label.grid(row=0, column=0, sticky="w")
        
        # Create notebook for configuration tabs with flexible sizing
        self.config_notebook = ttk.Notebook(parent)
        self.config_notebook.grid(row=1, column=0, sticky="nsew")
        
        # Setup tabs
        self.setup_terrain_tab()
        self.setup_mountains_tab()
        self.setup_rivers_tab()
        self.setup_roads_tab()
        self.setup_advanced_tab()
    
    def setup_terrain_tab(self):
        terrain_frame = ttk.Frame(self.config_notebook)
        self.config_notebook.add(terrain_frame, text="Terrain")
        
        # Basic terrain parameters - using direct frame like Advanced tab
        header_frame = ttk.Frame(terrain_frame)
        header_frame.pack(fill="x", pady=(5, 5))
        
        ttk.Label(header_frame, text="Basic Terrain Parameters", style='Header.TLabel').pack(anchor="w")
        ttk.Separator(header_frame, orient='horizontal').pack(fill="x", pady=(2, 0))
        
        self.terrain_vars = {}
        
        # Number of mountains - compact layout
        frame = ttk.Frame(terrain_frame)
        frame.pack(fill="x", pady=2)
        
        ttk.Label(frame, text="Mountains:", style='Subheader.TLabel').pack(side="left")
        ttk.Label(frame, text="(1-50)").pack(side="right")
        
        self.terrain_vars['num_mountains'] = tk.IntVar(value=self.terrain_config['num_mountains'])
        spinbox = ttk.Spinbox(frame, from_=1, to=50, 
                             textvariable=self.terrain_vars['num_mountains'],
                             width=8, font=('Segoe UI', 11))
        spinbox.pack(side="right", padx=(0, 10))
        
        # Map size - compact layout
        frame = ttk.Frame(terrain_frame)
        frame.pack(fill="x", pady=2)
        
        ttk.Label(frame, text="Map Size:").pack(side="left")
        self.terrain_vars['map_size'] = tk.IntVar(value=self.terrain_config['map_size'])
        size_combo = ttk.Combobox(frame, textvariable=self.terrain_vars['map_size'], 
                                  values=[256, 512, 1025, 2049, 4097], state="readonly", width=10)
        size_combo.pack(side="right")
        
        # Mountain Generation Method
        frame = ttk.Frame(terrain_frame)
        frame.pack(fill="x", pady=2)
        
        ttk.Label(frame, text="Generation Method:", style='Subheader.TLabel').pack(side="left")
        ttk.Label(frame, text="(geological=realistic, simple=basic)").pack(side="right")
        self.terrain_vars['mountain_generation_method'] = tk.StringVar(value=self.terrain_config.get('mountain_generation_method', 'geological'))
        method_combo = ttk.Combobox(frame, textvariable=self.terrain_vars['mountain_generation_method'], 
                                   values=['geological', 'simple'], state="readonly", width=12)
        method_combo.pack(side="right", padx=(0, 10))
        
        # Mountain Type Selection
        frame = ttk.Frame(terrain_frame)
        frame.pack(fill="x", pady=2)
        
        ttk.Label(frame, text="Mountain Type:", style='Subheader.TLabel').pack(side="left")
        ttk.Label(frame, text="(varied=random mix)").pack(side="right")
        self.terrain_vars['mountain_type'] = tk.StringVar(value=self.terrain_config.get('mountain_type', 'varied'))
        type_combo = ttk.Combobox(frame, textvariable=self.terrain_vars['mountain_type'], 
                                 values=['varied', 'peaked', 'ridge', 'mesa', 'volcano', 'asymmetric'], 
                                 state="readonly", width=12)
        type_combo.pack(side="right", padx=(0, 10))
        
        # Noise level
        self.create_simple_scale(terrain_frame, "Noise Level:", 'noise_level', 
                                   self.terrain_config['noise_level'], 0.0, 0.5, 3)
        
        # Mountain height range
        ttk.Label(terrain_frame, text="Mountain Height Range", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(10, 5))
        
        self.create_simple_scale(terrain_frame, "Min Height:", 'height_min', 
                                   self.terrain_config['mountain_height_range'][0], 0.1, 10.0, 1)
        
        self.create_simple_scale(terrain_frame, "Max Height:", 'height_max', 
                                   self.terrain_config['mountain_height_range'][1], 0.1, 10.0, 1)
        
        # Mountain width range
        ttk.Label(terrain_frame, text="Mountain Width Range", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(10, 5))
        
        self.create_simple_scale(terrain_frame, "Min Width:", 'width_min', 
                                   self.terrain_config['mountain_width_range'][0], 0.1, 5.0, 1)
        
        self.create_simple_scale(terrain_frame, "Max Width:", 'width_max', 
                                   self.terrain_config['mountain_width_range'][1], 0.1, 5.0, 1)
        
        # Complex terrain features
        ttk.Label(terrain_frame, text="Terrain Features", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(10, 5))
        
        self.terrain_vars['add_ridges'] = tk.BooleanVar(value=self.terrain_config['add_ridges'])
        ttk.Checkbutton(terrain_frame, text="Add Ridges", variable=self.terrain_vars['add_ridges']).pack(anchor="w")
        
        self.terrain_vars['add_valleys'] = tk.BooleanVar(value=self.terrain_config['add_valleys'])
        ttk.Checkbutton(terrain_frame, text="Add Valleys", variable=self.terrain_vars['add_valleys']).pack(anchor="w")
        
        self.terrain_vars['add_fractal_noise'] = tk.BooleanVar(value=self.terrain_config['add_fractal_noise'])
        ttk.Checkbutton(terrain_frame, text="Add Fractal Noise", variable=self.terrain_vars['add_fractal_noise']).pack(anchor="w")
        
        # Ridge and valley parameters
        self.create_simple_scale(terrain_frame, "Ridge Strength:", 'ridge_strength', 
                                   self.terrain_config['ridge_strength'], 0.0, 1.0, 2)
        
        self.create_simple_scale(terrain_frame, "Valley Depth:", 'valley_depth', 
                                   self.terrain_config['valley_depth'], 0.0, 1.0, 2)
    
    def setup_mountains_tab(self):
        mountains_frame = ttk.Frame(self.config_notebook)
        self.config_notebook.add(mountains_frame, text="Mountains")
        
        # Custom mountains toggle
        self.custom_mountains_enabled = tk.BooleanVar(value=self.custom_mountains['enabled'])
        ttk.Checkbutton(mountains_frame, text="Use Custom Mountain Positions", 
                       variable=self.custom_mountains_enabled,
                       command=self.toggle_custom_mountains).pack(anchor="w", pady=10)
        
        # Mountains list frame
        self.mountains_frame = ttk.LabelFrame(mountains_frame, text="Custom Mountains")
        self.mountains_frame.pack(fill="both", expand=True, pady=10)
        
        # Mountains list
        self.mountains_tree = ttk.Treeview(self.mountains_frame, columns=("x", "y", "height", "width", "type"), show="headings")
        self.mountains_tree.heading("#1", text="X Position")
        self.mountains_tree.heading("#2", text="Y Position")
        self.mountains_tree.heading("#3", text="Height")
        self.mountains_tree.heading("#4", text="Width")
        self.mountains_tree.heading("#5", text="Type")
        
        # Configure column widths for better readability
        self.mountains_tree.column("#1", width=80, anchor="center")
        self.mountains_tree.column("#2", width=80, anchor="center")
        self.mountains_tree.column("#3", width=70, anchor="center")
        self.mountains_tree.column("#4", width=70, anchor="center")
        self.mountains_tree.column("#5", width=90, anchor="center")
        
        self.mountains_tree.pack(fill="both", expand=True, pady=5)
        
        # Mountain controls
        mountain_controls = ttk.Frame(self.mountains_frame)
        mountain_controls.pack(fill="x", pady=5)
        
        ttk.Button(mountain_controls, text="Add Mountain", command=self.add_mountain).pack(side="left", padx=5)
        ttk.Button(mountain_controls, text="Edit Mountain", command=self.edit_mountain).pack(side="left", padx=5)
        ttk.Button(mountain_controls, text="Remove Mountain", command=self.remove_mountain).pack(side="left", padx=5)
        
        self.update_mountains_list()
    
    def setup_rivers_tab(self):
        rivers_frame = ttk.Frame(self.config_notebook)
        self.config_notebook.add(rivers_frame, text="Rivers")
        
        # Rivers enabled toggle
        self.rivers_enabled = tk.BooleanVar(value=self.river_config['enabled'])
        ttk.Checkbutton(rivers_frame, text="Enable Rivers", variable=self.rivers_enabled).pack(anchor="w", pady=10)
        
        # Rivers list frame
        self.rivers_frame = ttk.LabelFrame(rivers_frame, text="Rivers Configuration")
        self.rivers_frame.pack(fill="both", expand=True, pady=10)
        
        # Rivers list
        self.rivers_tree = ttk.Treeview(self.rivers_frame, columns=("start", "end", "width", "depth", "meander"), show="headings")
        self.rivers_tree.heading("#1", text="Start")
        self.rivers_tree.heading("#2", text="End")
        self.rivers_tree.heading("#3", text="Width")
        self.rivers_tree.heading("#4", text="Depth")
        self.rivers_tree.heading("#5", text="Meander")
        
        # Configure column widths for better readability
        self.rivers_tree.column("#1", width=120, anchor="center")
        self.rivers_tree.column("#2", width=120, anchor="center")
        self.rivers_tree.column("#3", width=80, anchor="center")
        self.rivers_tree.column("#4", width=80, anchor="center")
        self.rivers_tree.column("#5", width=80, anchor="center")
        
        self.rivers_tree.pack(fill="both", expand=True, pady=5)
        
        # River controls
        river_controls = ttk.Frame(self.rivers_frame)
        river_controls.pack(fill="x", pady=5)
        
        ttk.Button(river_controls, text="Add River", command=self.add_river).pack(side="left", padx=5)
        ttk.Button(river_controls, text="Edit River", command=self.edit_river).pack(side="left", padx=5)
        ttk.Button(river_controls, text="Remove River", command=self.remove_river).pack(side="left", padx=5)
        
        self.update_rivers_list()
    
    def setup_roads_tab(self):
        roads_frame = ttk.Frame(self.config_notebook)
        self.config_notebook.add(roads_frame, text="Roads")
        
        # Roads enabled toggle
        self.roads_enabled = tk.BooleanVar(value=self.road_config['enabled'])
        ttk.Checkbutton(roads_frame, text="Enable Roads", variable=self.roads_enabled).pack(anchor="w", pady=10)
        
        # Roads list frame
        self.roads_frame = ttk.LabelFrame(roads_frame, text="Roads Configuration")
        self.roads_frame.pack(fill="both", expand=True, pady=10)
        
        # Roads list
        self.roads_tree = ttk.Treeview(self.roads_frame, columns=("start", "end", "width", "depth", "smoothness"), show="headings")
        self.roads_tree.heading("#1", text="Start")
        self.roads_tree.heading("#2", text="End")
        self.roads_tree.heading("#3", text="Width")
        self.roads_tree.heading("#4", text="Cut Depth")
        self.roads_tree.heading("#5", text="Smoothness")
        
        # Configure column widths for better readability
        self.roads_tree.column("#1", width=120, anchor="center")
        self.roads_tree.column("#2", width=120, anchor="center")
        self.roads_tree.column("#3", width=80, anchor="center")
        self.roads_tree.column("#4", width=90, anchor="center")
        self.roads_tree.column("#5", width=90, anchor="center")
        
        self.roads_tree.pack(fill="both", expand=True, pady=5)
        
        # Road controls
        road_controls = ttk.Frame(self.roads_frame)
        road_controls.pack(fill="x", pady=5)
        
        ttk.Button(road_controls, text="Add Road", command=self.add_road).pack(side="left", padx=5)
        ttk.Button(road_controls, text="Edit Road", command=self.edit_road).pack(side="left", padx=5)
        ttk.Button(road_controls, text="Remove Road", command=self.remove_road).pack(side="left", padx=5)
        
        self.update_roads_list()
    
    def setup_advanced_tab(self):
        advanced_frame = ttk.Frame(self.config_notebook)
        self.config_notebook.add(advanced_frame, text="Advanced")
        
        # Advanced settings
        self.advanced_vars = {}
        
        # Random seed
        frame = ttk.Frame(advanced_frame)
        frame.pack(fill="x", pady=5)
        ttk.Label(frame, text="Random Seed:").pack(side="left")
        self.advanced_vars['random_seed'] = tk.IntVar(value=self.advanced_config.get('random_seed', 0) or 0)
        ttk.Entry(frame, textvariable=self.advanced_vars['random_seed']).pack(side="right")
        
        # Export settings
        ttk.Label(advanced_frame, text="Export Settings", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(20, 5))
        
        self.export_vars = {}
        
        # Bit depth
        frame = ttk.Frame(advanced_frame)
        frame.pack(fill="x", pady=2)
        ttk.Label(frame, text="Bit Depth:").pack(side="left")
        self.export_vars['bit_depth'] = tk.IntVar(value=self.export_config['bit_depth'])
        bit_combo = ttk.Combobox(frame, textvariable=self.export_vars['bit_depth'], 
                                values=[8, 16], state="readonly")
        bit_combo.pack(side="right")
        
        # Smoothing settings
        ttk.Label(advanced_frame, text="Smoothing Settings", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(20, 5))
        
        self.smoothing_vars = {}
        
        # Gaussian smoothing
        self.smoothing_vars['gaussian_enabled'] = tk.BooleanVar(value=self.smoothing_config['gaussian']['enabled'])
        ttk.Checkbutton(advanced_frame, text="Enable Gaussian Smoothing", 
                       variable=self.smoothing_vars['gaussian_enabled']).pack(anchor="w")
        
        frame = ttk.Frame(advanced_frame)
        frame.pack(fill="x", pady=2)
        ttk.Label(frame, text="Gaussian Kernel Size:").pack(side="left")
        self.smoothing_vars['gaussian_kernel'] = tk.IntVar(value=self.smoothing_config['gaussian']['kernel_size'])
        spinbox = ttk.Spinbox(frame, from_=3, to=15, 
                             textvariable=self.smoothing_vars['gaussian_kernel'],
                             width=5, font=('Segoe UI', 10))
        spinbox.pack(side="right", padx=(0, 10))
        ttk.Label(frame, text="(3-15, odd numbers)", style='Info.TLabel').pack(side="right")
        
        # Erosion simulation
        self.smoothing_vars['erosion_enabled'] = tk.BooleanVar(value=self.smoothing_config['erosion_simulation']['enabled'])
        ttk.Checkbutton(advanced_frame, text="Enable Erosion Simulation", 
                       variable=self.smoothing_vars['erosion_enabled']).pack(anchor="w")
        
        # Erosion iterations
        frame = ttk.Frame(advanced_frame)
        frame.pack(fill="x", pady=2)
        ttk.Label(frame, text="Erosion Iterations:").pack(side="left")
        self.smoothing_vars['erosion_iterations'] = tk.IntVar(value=self.smoothing_config['erosion_simulation']['iterations'])
        spinbox = ttk.Spinbox(frame, from_=1, to=10, 
                             textvariable=self.smoothing_vars['erosion_iterations'],
                             width=5, font=('Segoe UI', 10))
        spinbox.pack(side="right", padx=(0, 10))
        ttk.Label(frame, text="(1-10 iterations)", style='Info.TLabel').pack(side="right")
        
        # Erosion strength
        self.create_formatted_smoothing_scale(advanced_frame, "Erosion Strength:", 'erosion_strength', 
                                             self.smoothing_config['erosion_simulation']['strength'], 0.0, 1.0, 2)
    
    def setup_preview_and_progress_panel(self, parent):
        """Setup preview and progress panel (middle column)"""
        # Configure parent grid weights
        parent.grid_rowconfigure(1, weight=1)  # Preview gets most space
        parent.grid_rowconfigure(2, weight=0)  # Controls fixed height
        parent.grid_columnconfigure(0, weight=1)
        
        # Header
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header_frame.grid_columnconfigure(0, weight=1)
        
        header_label = ttk.Label(header_frame, text="Live Preview", style='Header.TLabel')
        header_label.grid(row=0, column=0, sticky="w")
        
        # Grayscale preview toggle button
        self.grayscale_preview_var = tk.BooleanVar(value=False)
        grayscale_toggle = ttk.Checkbutton(header_frame, text="Grayscale", 
                                         variable=self.grayscale_preview_var,
                                         command=self.toggle_grayscale_preview)
        grayscale_toggle.grid(row=0, column=1, sticky="e", padx=(10, 0))
        
        # Binary text preview button
        binary_text_btn = ttk.Button(header_frame, text="Binary Data", 
                                    command=self.show_binary_data,
                                    style='Action.TButton')
        binary_text_btn.grid(row=0, column=2, sticky="e", padx=(5, 0))
        
        # Preview frame with border - responsive sizing
        preview_frame = ttk.LabelFrame(parent, text="Terrain Preview", padding=10)
        preview_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)
        
        # Canvas for preview with responsive sizing
        self.preview_canvas = tk.Canvas(preview_frame, 
                                      bg='white',
                                      relief=tk.SUNKEN,
                                      borderwidth=2)
        self.preview_canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Bind canvas resize event for responsive preview
        self.preview_canvas.bind('<Configure>', self.on_canvas_configure)
        
        # Progress and controls frame - fixed height
        controls_frame = ttk.LabelFrame(parent, text="Generation Controls", padding=10)
        controls_frame.grid(row=2, column=0, sticky="ew")
        controls_frame.grid_columnconfigure(0, weight=1)
        
        # Progress section with grid
        progress_frame = ttk.Frame(controls_frame)
        progress_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        progress_frame.grid_columnconfigure(0, weight=1)
        
        ttk.Label(progress_frame, text="Generation Progress:", style='Subheader.TLabel').grid(row=0, column=0, sticky="w")
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, 
                                          variable=self.progress_var,
                                          maximum=100,
                                          style='TProgressbar')
        self.progress_bar.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        
        self.status_label = ttk.Label(progress_frame, text="Ready to generate", style='Info.TLabel')
        self.status_label.grid(row=2, column=0, sticky="w", pady=(5, 0))
        
        # Generation buttons with grid layout
        button_frame = ttk.Frame(controls_frame)
        button_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        button_frame.grid_columnconfigure(0, weight=1)
        
        # Left side buttons
        left_buttons = ttk.Frame(button_frame)
        left_buttons.grid(row=0, column=0, sticky="w")
        
        self.generate_button = ttk.Button(left_buttons, 
                                        text="Generate Terrain",
                                        command=self.generate_map,
                                        style='Generate.TButton')
        self.generate_button.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_button = ttk.Button(left_buttons, 
                                    text="Stop Generation",
                                    command=self.stop_generation,
                                    state="disabled",
                                    style='Action.TButton')
        self.stop_button.grid(row=0, column=1, padx=(0, 10))
        
        # Right side buttons
        right_buttons = ttk.Frame(button_frame)
        right_buttons.grid(row=0, column=1, sticky="e")
        
        ttk.Button(right_buttons, text="Save Config", 
                  command=self.save_config,
                  style='Action.TButton').grid(row=0, column=0, padx=(0, 5))
        
        ttk.Button(right_buttons, text="Load Config", 
                  command=self.load_config,
                  style='Action.TButton').grid(row=0, column=1)
    
    def setup_console_panel(self, parent):
        """Setup console panel (right column)"""
        # Configure parent grid
        parent.grid_rowconfigure(1, weight=1)  # Console text expands
        parent.grid_columnconfigure(0, weight=1)
        
        # Header for console
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header_frame.grid_columnconfigure(0, weight=1)
        
        header_label = ttk.Label(header_frame, text="📋 Generation Console", 
                                font=('Segoe UI', 13, 'bold'), foreground='#1a252f')
        header_label.grid(row=0, column=0, sticky="w")
        
        # Clear button
        clear_btn = ttk.Button(header_frame, text="Clear Log", 
                              command=self.clear_messages,
                              style='Action.TButton')
        clear_btn.grid(row=0, column=1, sticky="e")
        
        # Console frame
        console_frame = ttk.LabelFrame(parent, text="Live Generation Output", padding=15)
        console_frame.grid(row=1, column=0, sticky="nsew")
        console_frame.grid_rowconfigure(0, weight=1)  # Text area expands
        console_frame.grid_columnconfigure(0, weight=1)
        
        # Text widget for messages with scrollbar - responsive sizing
        text_frame = ttk.Frame(console_frame)
        text_frame.grid(row=0, column=0, sticky="nsew")
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)
        
        # Determine best monospace font available
        try:
            # Try to use JetBrains Mono if available
            test_font = tk.font.Font(family='JetBrains Mono', size=10)
            console_font = ('JetBrains Mono', 10)
            console_font_bold = ('JetBrains Mono', 10, 'bold')
            console_font_small = ('JetBrains Mono', 9)
        except:
            try:
                # Fallback to Consolas
                test_font = tk.font.Font(family='Consolas', size=10)
                console_font = ('Consolas', 10)
                console_font_bold = ('Consolas', 10, 'bold')
                console_font_small = ('Consolas', 9)
            except:
                # Final fallback to Courier
                console_font = ('Courier', 10)
                console_font_bold = ('Courier', 10, 'bold')
                console_font_small = ('Courier', 9)
        
        self.messages_text = tk.Text(text_frame, 
                                   wrap=tk.WORD,
                                   font=console_font,  # Use determined font
                                   bg='#1e1e2e',  # Darker, more modern background
                                   fg='#cdd6f4',  # Softer text color
                                   insertbackground='#f5e0dc',  # Cursor color
                                   relief=tk.FLAT,
                                   padx=15,  # More padding
                                   pady=12,
                                   selectbackground='#585b70',  # Selection background
                                   selectforeground='#cdd6f4',  # Selection text
                                   state=tk.DISABLED)  # Start disabled to prevent editing
        
        # Enhanced scrollbar with grid layout
        messages_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.messages_text.yview)
        self.messages_text.configure(yscrollcommand=messages_scrollbar.set)
        
        self.messages_text.grid(row=0, column=0, sticky="nsew")
        messages_scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Enhanced text tags with better colors and styling
        self.messages_text.tag_configure("info", 
                                        foreground="#89b4fa",  # Light blue
                                        font=console_font)
        self.messages_text.tag_configure("warning", 
                                        foreground="#f9e2af",  # Yellow
                                        font=console_font_bold)
        self.messages_text.tag_configure("error", 
                                        foreground="#f38ba8",  # Pink/red
                                        font=console_font_bold)
        self.messages_text.tag_configure("success", 
                                        foreground="#a6e3a1",  # Green
                                        font=console_font_bold)
        self.messages_text.tag_configure("timestamp", 
                                        foreground="#6c7086",  # Gray
                                        font=console_font_small)
        self.messages_text.tag_configure("river", 
                                        foreground="#74c7ec",  # Cyan for rivers
                                        font=console_font_bold)
        self.messages_text.tag_configure("road", 
                                        foreground="#fab387",  # Orange for roads
                                        font=console_font_bold)
        
        # Add welcome message
        self.add_welcome_message()
    
    def on_canvas_configure(self, event):
        """Handle canvas resize events for responsive preview"""
        # This allows the preview to maintain aspect ratio while being responsive
        if hasattr(self, 'current_map') and self.current_map is not None:
            self.update_preview()
    
    def add_welcome_message(self):
        """Add a welcome message to the console"""
        self.messages_text.configure(state=tk.NORMAL)
        welcome_msg = """
                    Height Map Generator Console                      
                                                                      
  🏔️  Configure terrain parameters in the left panel                 
  🎯  Click 'Generate Terrain' to start the generation process       
  📊  Watch real-time progress and detailed logging here             

"""
        self.messages_text.insert(tk.END, welcome_msg, "info")
        self.messages_text.configure(state=tk.DISABLED)
        self.messages_text.see(tk.END)
    
    def clear_messages(self):
        """Clear the messages console"""
        self.messages_text.configure(state=tk.NORMAL)
        self.messages_text.delete(1.0, tk.END)
        self.add_welcome_message()
        
    def insert_message(self, message, tag="info"):
        """Insert a message into the console with proper formatting"""
        self.messages_text.configure(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
        
        # Determine special tags for rivers and roads
        if "🌊" in message or "river" in message.lower():
            tag = "river"
        elif "🛣️" in message or "road" in message.lower():
            tag = "road"
        elif "error" in message.lower() or "failed" in message.lower():
            tag = "error"
        elif "warning" in message.lower():
            tag = "warning"
        elif "✅" in message or "success" in message.lower() or "complete" in message.lower():
            tag = "success"
        
        self.messages_text.insert(tk.END, message + "\n", tag)
        self.messages_text.configure(state=tk.DISABLED)
        self.messages_text.see(tk.END)
    
    def update_ui_from_config(self):
        """Update UI elements from current configuration"""
        # Update terrain variables
        for var_name, var in self.terrain_vars.items():
            if var_name in ['height_min', 'height_max']:
                if var_name == 'height_min':
                    var.set(self.terrain_config['mountain_height_range'][0])
                else:
                    var.set(self.terrain_config['mountain_height_range'][1])
            elif var_name in ['width_min', 'width_max']:
                if var_name == 'width_min':
                    var.set(self.terrain_config['mountain_width_range'][0])
                else:
                    var.set(self.terrain_config['mountain_width_range'][1])
            elif var_name in self.terrain_config:
                var.set(self.terrain_config[var_name])
        
        # Update other configurations
        self.update_mountains_list()
        self.update_rivers_list()
        self.update_roads_list()
    
    def get_config_from_ui(self):
        """Extract configuration from UI elements"""
        # Update terrain config
        self.terrain_config['num_mountains'] = self.terrain_vars['num_mountains'].get()
        self.terrain_config['map_size'] = self.terrain_vars['map_size'].get()
        self.terrain_config['mountain_generation_method'] = self.terrain_vars['mountain_generation_method'].get()
        self.terrain_config['mountain_type'] = self.terrain_vars['mountain_type'].get()
        self.terrain_config['noise_level'] = self.terrain_vars['noise_level'].get()
        self.terrain_config['mountain_height_range'] = (
            self.terrain_vars['height_min'].get(),
            self.terrain_vars['height_max'].get()
        )
        self.terrain_config['mountain_width_range'] = (
            self.terrain_vars['width_min'].get(),
            self.terrain_vars['width_max'].get()
        )
        self.terrain_config['add_ridges'] = self.terrain_vars['add_ridges'].get()
        self.terrain_config['add_valleys'] = self.terrain_vars['add_valleys'].get()
        self.terrain_config['add_fractal_noise'] = self.terrain_vars['add_fractal_noise'].get()
        self.terrain_config['ridge_strength'] = self.terrain_vars['ridge_strength'].get()
        self.terrain_config['valley_depth'] = self.terrain_vars['valley_depth'].get()
        
        # Update rivers and roads enabled status
        self.river_config['enabled'] = self.rivers_enabled.get()
        self.road_config['enabled'] = self.roads_enabled.get()
        
        # Update custom mountains
        self.custom_mountains['enabled'] = self.custom_mountains_enabled.get()
        
        # Update advanced settings
        if hasattr(self, 'advanced_vars'):
            seed_val = self.advanced_vars['random_seed'].get()
            self.advanced_config['random_seed'] = seed_val if seed_val != 0 else None
        
        if hasattr(self, 'export_vars'):
            self.export_config['bit_depth'] = self.export_vars['bit_depth'].get()
        
        if hasattr(self, 'smoothing_vars'):
            self.smoothing_config['gaussian']['enabled'] = self.smoothing_vars['gaussian_enabled'].get()
            self.smoothing_config['gaussian']['kernel_size'] = self.smoothing_vars['gaussian_kernel'].get()
            self.smoothing_config['erosion_simulation']['enabled'] = self.smoothing_vars['erosion_enabled'].get()
            self.smoothing_config['erosion_simulation']['iterations'] = self.smoothing_vars['erosion_iterations'].get()
            self.smoothing_config['erosion_simulation']['strength'] = self.smoothing_vars['erosion_strength'].get()
    
    def toggle_custom_mountains(self):
        """Toggle custom mountains functionality"""
        enabled = self.custom_mountains_enabled.get()
        if enabled:
            self.mountains_frame.config(state="normal")
        else:
            self.mountains_frame.config(state="disabled")
    
    def update_mountains_list(self):
        """Update the mountains treeview"""
        # Clear existing items
        for item in self.mountains_tree.get_children():
            self.mountains_tree.delete(item)
        
        # Add mountains from config
        if 'positions' in self.custom_mountains:
            positions = self.custom_mountains['positions']
            heights = self.custom_mountains.get('heights', [1.0] * len(positions))
            widths = self.custom_mountains.get('widths', [1.0] * len(positions))
            types = self.custom_mountains.get('types', ['varied'] * len(positions))
            
            for i, (pos, height, width, mountain_type) in enumerate(zip(positions, heights, widths, types)):
                self.mountains_tree.insert("", "end", values=(pos[0], pos[1], height, width, mountain_type))
    
    def update_rivers_list(self):
        """Update the rivers treeview"""
        # Clear existing items
        for item in self.rivers_tree.get_children():
            self.rivers_tree.delete(item)
        
        # Add rivers from config
        for river in self.river_config.get('rivers', []):
            start_str = f"{river['start_edge']} ({river['start_position']:.2f})"
            end_str = f"{river['end_edge']} ({river['end_position']:.2f})"
            self.rivers_tree.insert("", "end", values=(
                start_str, end_str, 
                river.get('width', 0.1),
                river.get('depth', 0.2),
                river.get('meander', 0.3)
            ))
    
    def update_roads_list(self):
        """Update the roads treeview"""
        # Clear existing items
        for item in self.roads_tree.get_children():
            self.roads_tree.delete(item)
        
        # Add roads from config
        for road in self.road_config.get('roads', []):
            start_str = f"{road['start_edge']} ({road['start_position']:.2f})"
            end_str = f"{road['end_edge']} ({road['end_position']:.2f})"
            self.roads_tree.insert("", "end", values=(
                start_str, end_str,
                road.get('width', 0.1),
                road.get('cut_depth', 0.2),
                road.get('smoothness', 0.8)
            ))
    
    def add_mountain(self):
        """Add a new mountain"""
        dialog = MountainDialog(self.root, "Add Mountain")
        if dialog.result:
            x, y, height, width, mountain_type = dialog.result
            if 'positions' not in self.custom_mountains:
                self.custom_mountains['positions'] = []
                self.custom_mountains['heights'] = []
                self.custom_mountains['widths'] = []
                self.custom_mountains['types'] = []
            
            self.custom_mountains['positions'].append((x, y))
            self.custom_mountains['heights'].append(height)
            self.custom_mountains['widths'].append(width)
            self.custom_mountains['types'].append(mountain_type)
            self.update_mountains_list()
    
    def edit_mountain(self):
        """Edit selected mountain"""
        selection = self.mountains_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a mountain to edit")
            return
        
        item = selection[0]
        values = self.mountains_tree.item(item, "values")
        
        # Ensure we have all 5 values (including type)
        mountain_type = values[4] if len(values) > 4 else 'varied'
        
        dialog = MountainDialog(self.root, "Edit Mountain", 
                               (float(values[0]), float(values[1]), 
                                float(values[2]), float(values[3]), mountain_type))
        
        if dialog.result:
            x, y, height, width, mountain_type = dialog.result
            index = self.mountains_tree.index(item)
            
            # Ensure types list exists
            if 'types' not in self.custom_mountains:
                self.custom_mountains['types'] = ['varied'] * len(self.custom_mountains['positions'])
            
            self.custom_mountains['positions'][index] = (x, y)
            self.custom_mountains['heights'][index] = height
            self.custom_mountains['widths'][index] = width
            self.custom_mountains['types'][index] = mountain_type
            self.update_mountains_list()
    
    def remove_mountain(self):
        """Remove selected mountain"""
        selection = self.mountains_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a mountain to remove")
            return
        
        item = selection[0]
        index = self.mountains_tree.index(item)
        
        self.custom_mountains['positions'].pop(index)
        self.custom_mountains['heights'].pop(index)
        self.custom_mountains['widths'].pop(index)
        if 'types' in self.custom_mountains and index < len(self.custom_mountains['types']):
            self.custom_mountains['types'].pop(index)
        self.update_mountains_list()
    
    def add_river(self):
        """Add a new river"""
        dialog = RiverDialog(self.root, "Add River")
        if dialog.result:
            self.river_config['rivers'].append(dialog.result)
            self.update_rivers_list()
    
    def edit_river(self):
        """Edit selected river"""
        selection = self.rivers_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a river to edit")
            return
        
        item = selection[0]
        index = self.rivers_tree.index(item)
        river_data = self.river_config['rivers'][index]
        
        dialog = RiverDialog(self.root, "Edit River", river_data)
        if dialog.result:
            self.river_config['rivers'][index] = dialog.result
            self.update_rivers_list()
    
    def remove_river(self):
        """Remove selected river"""
        selection = self.rivers_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a river to remove")
            return
        
        item = selection[0]
        index = self.rivers_tree.index(item)
        self.river_config['rivers'].pop(index)
        self.update_rivers_list()
    
    def add_road(self):
        """Add a new road"""
        dialog = RoadDialog(self.root, "Add Road")
        if dialog.result:
            self.road_config['roads'].append(dialog.result)
            self.update_roads_list()
    
    def edit_road(self):
        """Edit selected road"""
        selection = self.roads_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a road to edit")
            return
        
        item = selection[0]
        index = self.roads_tree.index(item)
        road_data = self.road_config['roads'][index]
        
        dialog = RoadDialog(self.root, "Edit Road", road_data)
        if dialog.result:
            self.road_config['roads'][index] = dialog.result
            self.update_roads_list()
    
    def remove_road(self):
        """Remove selected road"""
        selection = self.roads_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a road to remove")
            return
        
        item = selection[0]
        index = self.roads_tree.index(item)
        self.road_config['roads'].pop(index)
        self.update_roads_list()
    
    def generate_map(self):
        """Start map generation in a separate thread"""
        if self.is_generating:
            return
        
        # Get configuration from UI
        self.get_config_from_ui()
        
        # Clear messages and reset progress
        self.messages_text.delete(1.0, tk.END)
        self.progress_var.set(0)
        self.status_label.config(text="Starting generation...")
        
        # Update button states
        self.generate_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.is_generating = True
        
        # Start generation thread
        self.generation_thread = threading.Thread(target=self.run_generation)
        self.generation_thread.daemon = True
        self.generation_thread.start()
    
    def run_generation(self):
        """Run the map generation process"""
        try:
            # Import generation function
            from src.ui_generator import generate_with_ui_integration
            
            self.progress_queue.put(5)
            self.message_queue.put("Starting map generation...")
            
            # Generate terrain with UI integration
            X, Y, Z = generate_with_ui_integration(
                self.terrain_config,
                self.custom_mountains,
                self.river_config,
                self.road_config,
                self.smoothing_config,
                self.export_config,
                self.advanced_config,
                self.message_queue,
                self.progress_queue
            )
            
            # Update preview
            self.current_map = Z
            self.update_preview(Z)
            
            self.message_queue.put("Map generation completed successfully!")
            
        except Exception as e:
            import traceback
            error_msg = f"Error during generation: {str(e)}\n{traceback.format_exc()}"
            self.message_queue.put(error_msg)
        finally:
            # Reset generation state
            self.is_generating = False
            self.root.after(0, self.generation_complete)
    
    def generation_complete(self):
        """Called when generation is complete"""
        self.generate_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Generation complete!")
    
    def stop_generation(self):
        """Stop the current generation"""
        if self.generation_thread and self.generation_thread.is_alive():
            # Note: This is a simple implementation. For proper cancellation,
            # you'd need to implement cancellation checks in the generation code
            self.message_queue.put("Generation stopped by user")
            self.is_generating = False
            self.generation_complete()
    
    def toggle_grayscale_preview(self):
        """Toggle between colormap and grayscale preview formats"""
        if hasattr(self, 'current_map') and self.current_map is not None:
            self.update_preview()  # Refresh with current toggle state
    
    def show_binary_data(self):
        """Show binary data in a separate window"""
        if not hasattr(self, 'current_map') or self.current_map is None:
            messagebox.showwarning("Warning", "No heightmap data available. Generate a heightmap first.")
            return
        
        # Create binary data window
        binary_window = tk.Toplevel(self.root)
        binary_window.title("Binary Heightmap Data")
        binary_window.geometry("800x600")
        binary_window.transient(self.root)
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(binary_window)
        text_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap="none", font=("Consolas", 9))
        v_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        h_scrollbar = ttk.Scrollbar(text_frame, orient="horizontal", command=text_widget.xview)
        
        text_widget.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Grid layout
        text_widget.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)
        
        # Generate binary data representation
        try:
            # Sample a smaller region for display (e.g., 32x32 from center)
            h, w = self.current_map.shape
            sample_size = min(32, h, w)
            start_h = (h - sample_size) // 2
            start_w = (w - sample_size) // 2
            sample_data = self.current_map[start_h:start_h+sample_size, start_w:start_w+sample_size]
            
            # Normalize to 0-255 range
            normalized = ((sample_data - sample_data.min()) / 
                         (sample_data.max() - sample_data.min()) * 255).astype(np.uint8)
            
            # Create binary representation
            binary_text = f"Binary Heightmap Data (Sample {sample_size}x{sample_size} from center)\n"
            binary_text += f"Full map size: {h}x{w}\n"
            binary_text += "=" * 60 + "\n\n"
            
            # Header with column numbers
            binary_text += "   "
            for j in range(sample_size):
                binary_text += f"{j:2d} "
            binary_text += "\n"
            
            # Binary data rows
            for i in range(sample_size):
                binary_text += f"{i:2d} "
                for j in range(sample_size):
                    value = normalized[i, j]
                    binary_str = format(value, '08b')
                    binary_text += binary_str + " "
                binary_text += "\n"
            
            # Add hexadecimal representation
            binary_text += "\n" + "=" * 60 + "\n"
            binary_text += "Hexadecimal Representation:\n\n"
            
            binary_text += "   "
            for j in range(sample_size):
                binary_text += f"{j:2d} "
            binary_text += "\n"
            
            for i in range(sample_size):
                binary_text += f"{i:2d} "
                for j in range(sample_size):
                    value = normalized[i, j]
                    hex_str = format(value, '02X')
                    binary_text += hex_str + " "
                binary_text += "\n"
            
            # Add decimal values for reference
            binary_text += "\n" + "=" * 60 + "\n"
            binary_text += "Decimal Values (0-255):\n\n"
            
            binary_text += "   "
            for j in range(sample_size):
                binary_text += f"{j:2d} "
            binary_text += "\n"
            
            for i in range(sample_size):
                binary_text += f"{i:2d} "
                for j in range(sample_size):
                    value = normalized[i, j]
                    binary_text += f"{value:3d} "
                binary_text += "\n"
            
            text_widget.insert("1.0", binary_text)
            text_widget.configure(state="disabled")  # Make read-only
            
        except Exception as e:
            text_widget.insert("1.0", f"Error generating binary data: {str(e)}")
            text_widget.configure(state="disabled")
    
    def create_binary_visualization(self, data, size):
        """Create binary format visualization of the heightmap data"""
        try:
            # Create a text-based binary representation
            binary_img = np.zeros((size, size, 3), dtype=np.uint8)
            
            # Convert each pixel value to binary string
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.2
            color = (255, 255, 255)  # White text
            thickness = 1
            
            # Calculate grid size for binary text display
            cell_size = max(8, size // 64)  # Minimum 8 pixels per cell
            grid_rows = size // cell_size
            grid_cols = size // cell_size
            
            # Sample data points for binary display
            for row in range(grid_rows):
                for col in range(grid_cols):
                    # Get pixel value from data
                    data_row = min(row * cell_size, data.shape[0] - 1)
                    data_col = min(col * cell_size, data.shape[1] - 1)
                    pixel_value = data[data_row, data_col]
                    
                    # Convert to 8-bit binary string
                    binary_str = format(pixel_value, '08b')
                    
                    # Position for text
                    x = col * cell_size + 1
                    y = row * cell_size + cell_size - 2
                    
                    # Ensure we don't go outside image bounds
                    if y < size and x < size - 40:  # Leave space for text
                        # Add binary text (show only first 4 bits for space)
                        short_binary = binary_str[:4]
                        cv2.putText(binary_img, short_binary, (x, y), font, font_scale, color, thickness)
            
            # If the grid approach doesn't work well, fall back to a simpler visualization
            if grid_rows < 4:  # Too few cells for meaningful display
                # Create a pattern-based binary visualization
                binary_img = np.zeros((size, size, 3), dtype=np.uint8)
                
                # Use the data values to create binary patterns
                for i in range(0, size, 4):
                    for j in range(0, size, 4):
                        if i < data.shape[0] and j < data.shape[1]:
                            value = data[i, j]
                            # Create binary pattern based on value
                            if value & 1:   # LSB
                                binary_img[i:i+2, j:j+2] = [255, 0, 0]    # Red
                            if value & 2:   # 2nd bit
                                binary_img[i:i+2, j+2:j+4] = [0, 255, 0]  # Green  
                            if value & 4:   # 3rd bit
                                binary_img[i+2:i+4, j:j+2] = [0, 0, 255]  # Blue
                            if value & 8:   # 4th bit
                                binary_img[i+2:i+4, j+2:j+4] = [255, 255, 0]  # Yellow
            
            return binary_img
            
        except Exception as e:
            # Fallback to simple binary pattern if text rendering fails
            binary_img = np.zeros((size, size, 3), dtype=np.uint8)
            
            # Simple bit visualization using colors
            for i in range(0, size, 2):
                for j in range(0, size, 2):
                    if i < data.shape[0] and j < data.shape[1]:
                        value = data[i, j]
                        # Map different bits to different colors
                        r = (value & 0x07) * 32  # Lower 3 bits for red
                        g = ((value >> 3) & 0x07) * 32  # Middle 3 bits for green  
                        b = ((value >> 6) & 0x03) * 64  # Upper 2 bits for blue
                        binary_img[i:i+2, j:j+2] = [r, g, b]
            
            return binary_img
    
    def update_preview(self, map_data=None):
        """Update the preview canvas with map data (responsive to canvas size)"""
        if map_data is not None:
            self.current_map = map_data
        
        if not hasattr(self, 'current_map') or self.current_map is None:
            return
            
        try:
            # Get current canvas dimensions
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            
            # Use smaller dimension to maintain square aspect ratio
            if canvas_width > 1 and canvas_height > 1:  # Ensure canvas is visible
                preview_size = min(canvas_width - 20, canvas_height - 20)  # Account for padding
                preview_size = max(preview_size, 200)  # Minimum size
            else:
                preview_size = 400  # Default size
            
            # Normalize data for display
            normalized = ((self.current_map - self.current_map.min()) / 
                         (self.current_map.max() - self.current_map.min()) * 255).astype(np.uint8)
            
            # Resize for responsive preview
            resized = cv2.resize(normalized, (preview_size, preview_size))
            
            # Check if grayscale format is enabled
            if self.grayscale_preview_var.get():
                # Grayscale format display
                colored_rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            else:
                # Normal colormap display
                try:
                    colored = cv2.applyColorMap(resized, cv2.COLORMAP_TERRAIN)
                except:
                    try:
                        colored = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
                    except:
                        # Fallback to grayscale if colormap fails
                        colored = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
                
                colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image and then to PhotoImage
            pil_image = Image.fromarray(colored_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update canvas
            self.root.after(0, lambda: self._update_canvas(photo))
            
        except Exception as e:
            self.message_queue.put(f"Error updating preview: {str(e)}")
            print(f"Preview error: {e}")
            print(traceback.format_exc())
    
    def _update_canvas(self, photo):
        """Update canvas in main thread (responsive positioning)"""
        self.preview_canvas.delete("all")
        
        # Get canvas dimensions for centering
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        
        # Center the image on the canvas
        center_x = canvas_width // 2 if canvas_width > 1 else 200
        center_y = canvas_height // 2 if canvas_height > 1 else 200
        
        self.preview_canvas.create_image(center_x, center_y, image=photo)
        self.preview_canvas.image = photo  # Keep a reference
    
    def check_messages(self):
        """Check for messages from generation thread"""
        try:
            while True:
                message = self.message_queue.get_nowait()
                # Use the enhanced message insertion method
                self.insert_message(message)
        except queue.Empty:
            pass
        
        try:
            while True:
                progress_data = self.progress_queue.get_nowait()
                if isinstance(progress_data, dict):
                    # Handle progress with status message
                    progress = progress_data.get('progress', 0)
                    status = progress_data.get('status', 'Processing...')
                    self.progress_var.set(progress)
                    self.status_label.configure(text=status)
                else:
                    # Handle simple progress value
                    self.progress_var.set(progress_data)
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_messages)
    
    def save_config(self):
        """Save current configuration to file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".py",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        
        if filename:
            self.get_config_from_ui()
            # Save configuration to file
            # This would involve writing the configuration dictionaries to a Python file
            messagebox.showinfo("Info", "Configuration saved successfully!")
    
    def load_config(self):
        """Load configuration from file"""
        filename = filedialog.askopenfilename(
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        
        if filename:
            # Load configuration from file
            # This would involve importing the configuration from the selected file
            self.update_ui_from_config()
            messagebox.showinfo("Info", "Configuration loaded successfully!")


class MountainDialog:
    def __init__(self, parent, title, initial_values=None):
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("320x240")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Variables
        self.x_var = tk.DoubleVar(value=initial_values[0] if initial_values else 0.0)
        self.y_var = tk.DoubleVar(value=initial_values[1] if initial_values else 0.0)
        self.height_var = tk.DoubleVar(value=initial_values[2] if initial_values else 1.0)
        self.width_var = tk.DoubleVar(value=initial_values[3] if initial_values else 1.0)
        self.type_var = tk.StringVar(value=initial_values[4] if initial_values and len(initial_values) > 4 else 'varied')
        
        # Create form
        ttk.Label(self.dialog, text="X Position:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        ttk.Entry(self.dialog, textvariable=self.x_var).grid(row=0, column=1, padx=10, pady=5)
        
        ttk.Label(self.dialog, text="Y Position:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        ttk.Entry(self.dialog, textvariable=self.y_var).grid(row=1, column=1, padx=10, pady=5)
        
        ttk.Label(self.dialog, text="Height:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        ttk.Entry(self.dialog, textvariable=self.height_var).grid(row=2, column=1, padx=10, pady=5)
        
        ttk.Label(self.dialog, text="Width:").grid(row=3, column=0, sticky="w", padx=10, pady=5)
        ttk.Entry(self.dialog, textvariable=self.width_var).grid(row=3, column=1, padx=10, pady=5)
        
        ttk.Label(self.dialog, text="Mountain Type:").grid(row=4, column=0, sticky="w", padx=10, pady=5)
        type_combo = ttk.Combobox(self.dialog, textvariable=self.type_var, 
                                 values=['varied', 'peaked', 'ridge', 'mesa', 'volcano', 'asymmetric'], 
                                 state="readonly", width=12)
        type_combo.grid(row=4, column=1, padx=10, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(self.dialog)
        button_frame.grid(row=5, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="OK", command=self.ok_clicked).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).pack(side="left", padx=5)
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")
        
        self.dialog.wait_window()
    
    def ok_clicked(self):
        self.result = (self.x_var.get(), self.y_var.get(), self.height_var.get(), self.width_var.get(), self.type_var.get())
        self.dialog.destroy()
    
    def cancel_clicked(self):
        self.dialog.destroy()


class RiverDialog:
    def __init__(self, parent, title, initial_data=None):
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("350x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Variables
        edges = ['top', 'bottom', 'left', 'right']
        
        self.start_edge_var = tk.StringVar(value=initial_data.get('start_edge', 'top') if initial_data else 'top')
        self.start_pos_var = tk.DoubleVar(value=initial_data.get('start_position', 0.5) if initial_data else 0.5)
        self.end_edge_var = tk.StringVar(value=initial_data.get('end_edge', 'bottom') if initial_data else 'bottom')
        self.end_pos_var = tk.DoubleVar(value=initial_data.get('end_position', 0.5) if initial_data else 0.5)
        self.width_var = tk.DoubleVar(value=initial_data.get('width', 0.1) if initial_data else 0.1)
        self.depth_var = tk.DoubleVar(value=initial_data.get('depth', 0.2) if initial_data else 0.2)
        self.meander_var = tk.DoubleVar(value=initial_data.get('meander', 0.3) if initial_data else 0.3)
        self.tributaries_var = tk.IntVar(value=initial_data.get('tributaries', 0) if initial_data else 0)
        
        # Create form
        row = 0
        ttk.Label(self.dialog, text="Start Edge:").grid(row=row, column=0, sticky="w", padx=10, pady=5)
        ttk.Combobox(self.dialog, textvariable=self.start_edge_var, values=edges, state="readonly").grid(row=row, column=1, padx=10, pady=5)
        
        row += 1
        ttk.Label(self.dialog, text="Start Position:").grid(row=row, column=0, sticky="w", padx=10, pady=5)
        ttk.Entry(self.dialog, textvariable=self.start_pos_var).grid(row=row, column=1, padx=10, pady=5)
        
        row += 1
        ttk.Label(self.dialog, text="End Edge:").grid(row=row, column=0, sticky="w", padx=10, pady=5)
        ttk.Combobox(self.dialog, textvariable=self.end_edge_var, values=edges, state="readonly").grid(row=row, column=1, padx=10, pady=5)
        
        row += 1
        ttk.Label(self.dialog, text="End Position:").grid(row=row, column=0, sticky="w", padx=10, pady=5)
        ttk.Entry(self.dialog, textvariable=self.end_pos_var).grid(row=row, column=1, padx=10, pady=5)
        
        row += 1
        ttk.Label(self.dialog, text="Width:").grid(row=row, column=0, sticky="w", padx=10, pady=5)
        ttk.Entry(self.dialog, textvariable=self.width_var).grid(row=row, column=1, padx=10, pady=5)
        
        row += 1
        ttk.Label(self.dialog, text="Depth:").grid(row=row, column=0, sticky="w", padx=10, pady=5)
        ttk.Entry(self.dialog, textvariable=self.depth_var).grid(row=row, column=1, padx=10, pady=5)
        
        row += 1
        ttk.Label(self.dialog, text="Meander:").grid(row=row, column=0, sticky="w", padx=10, pady=5)
        ttk.Entry(self.dialog, textvariable=self.meander_var).grid(row=row, column=1, padx=10, pady=5)
        
        row += 1
        ttk.Label(self.dialog, text="Tributaries:").grid(row=row, column=0, sticky="w", padx=10, pady=5)
        ttk.Entry(self.dialog, textvariable=self.tributaries_var).grid(row=row, column=1, padx=10, pady=5)
        
        # Buttons
        row += 1
        button_frame = ttk.Frame(self.dialog)
        button_frame.grid(row=row, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="OK", command=self.ok_clicked).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).pack(side="left", padx=5)
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")
        
        self.dialog.wait_window()
    
    def ok_clicked(self):
        self.result = {
            'start_edge': self.start_edge_var.get(),
            'start_position': self.start_pos_var.get(),
            'end_edge': self.end_edge_var.get(),
            'end_position': self.end_pos_var.get(),
            'width': self.width_var.get(),
            'depth': self.depth_var.get(),
            'meander': self.meander_var.get(),
            'tributaries': self.tributaries_var.get()
        }
        self.dialog.destroy()
    
    def cancel_clicked(self):
        self.dialog.destroy()


class RoadDialog:
    def __init__(self, parent, title, initial_data=None):
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("350x350")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Variables
        edges = ['top', 'bottom', 'left', 'right']
        
        self.start_edge_var = tk.StringVar(value=initial_data.get('start_edge', 'left') if initial_data else 'left')
        self.start_pos_var = tk.DoubleVar(value=initial_data.get('start_position', 0.5) if initial_data else 0.5)
        self.end_edge_var = tk.StringVar(value=initial_data.get('end_edge', 'right') if initial_data else 'right')
        self.end_pos_var = tk.DoubleVar(value=initial_data.get('end_position', 0.5) if initial_data else 0.5)
        self.width_var = tk.DoubleVar(value=initial_data.get('width', 0.1) if initial_data else 0.1)
        self.cut_depth_var = tk.DoubleVar(value=initial_data.get('cut_depth', 0.2) if initial_data else 0.2)
        self.smoothness_var = tk.DoubleVar(value=initial_data.get('smoothness', 0.8) if initial_data else 0.8)
        self.banking_var = tk.BooleanVar(value=initial_data.get('banking', True) if initial_data else True)
        self.follow_contours_var = tk.BooleanVar(value=initial_data.get('follow_contours', True) if initial_data else True)
        
        # Create form
        row = 0
        ttk.Label(self.dialog, text="Start Edge:").grid(row=row, column=0, sticky="w", padx=10, pady=5)
        ttk.Combobox(self.dialog, textvariable=self.start_edge_var, values=edges, state="readonly").grid(row=row, column=1, padx=10, pady=5)
        
        row += 1
        ttk.Label(self.dialog, text="Start Position:").grid(row=row, column=0, sticky="w", padx=10, pady=5)
        ttk.Entry(self.dialog, textvariable=self.start_pos_var).grid(row=row, column=1, padx=10, pady=5)
        
        row += 1
        ttk.Label(self.dialog, text="End Edge:").grid(row=row, column=0, sticky="w", padx=10, pady=5)
        ttk.Combobox(self.dialog, textvariable=self.end_edge_var, values=edges, state="readonly").grid(row=row, column=1, padx=10, pady=5)
        
        row += 1
        ttk.Label(self.dialog, text="End Position:").grid(row=row, column=0, sticky="w", padx=10, pady=5)
        ttk.Entry(self.dialog, textvariable=self.end_pos_var).grid(row=row, column=1, padx=10, pady=5)
        
        row += 1
        ttk.Label(self.dialog, text="Width:").grid(row=row, column=0, sticky="w", padx=10, pady=5)
        ttk.Entry(self.dialog, textvariable=self.width_var).grid(row=row, column=1, padx=10, pady=5)
        
        row += 1
        ttk.Label(self.dialog, text="Cut Depth:").grid(row=row, column=0, sticky="w", padx=10, pady=5)
        ttk.Entry(self.dialog, textvariable=self.cut_depth_var).grid(row=row, column=1, padx=10, pady=5)
        
        row += 1
        ttk.Label(self.dialog, text="Smoothness:").grid(row=row, column=0, sticky="w", padx=10, pady=5)
        ttk.Entry(self.dialog, textvariable=self.smoothness_var).grid(row=row, column=1, padx=10, pady=5)
        
        row += 1
        ttk.Checkbutton(self.dialog, text="Banking", variable=self.banking_var).grid(row=row, column=0, columnspan=2, sticky="w", padx=10, pady=5)
        
        row += 1
        ttk.Checkbutton(self.dialog, text="Follow Contours", variable=self.follow_contours_var).grid(row=row, column=0, columnspan=2, sticky="w", padx=10, pady=5)
        
        # Buttons
        row += 1
        button_frame = ttk.Frame(self.dialog)
        button_frame.grid(row=row, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="OK", command=self.ok_clicked).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).pack(side="left", padx=5)
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")
        
        self.dialog.wait_window()
    
    def ok_clicked(self):
        self.result = {
            'start_edge': self.start_edge_var.get(),
            'start_position': self.start_pos_var.get(),
            'end_edge': self.end_edge_var.get(),
            'end_position': self.end_pos_var.get(),
            'width': self.width_var.get(),
            'cut_depth': self.cut_depth_var.get(),
            'smoothness': self.smoothness_var.get(),
            'banking': self.banking_var.get(),
            'follow_contours': self.follow_contours_var.get()
        }
        self.dialog.destroy()
    
    def cancel_clicked(self):
        self.dialog.destroy()


def main():
    root = tk.Tk()
    app = HeightMapGeneratorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
