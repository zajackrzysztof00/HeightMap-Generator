# Height Map Generator GUI

A comprehensive graphical user interface for generating complex terrain height maps with mountains, rivers, and roads.

## Features

### 🎛️ **Complete Configuration Control**
- **Terrain Parameters**: Adjust mountain count, map size, noise levels, and terrain features
- **Mountain Configuration**: Set height/width ranges, enable ridges and valleys
- **Custom Mountains**: Add, edit, and remove specific mountains with precise positioning
- **Rivers & Roads**: Full control over waterways and transportation networks
- **Advanced Settings**: Random seeds, export options, and smoothing parameters

### 🗺️ **Real-time Preview**
- Live map preview with terrain coloring
- Updates automatically after generation
- Zoom and pan capabilities

### 📊 **Progress Monitoring**
- Real-time progress bar during generation
- Detailed status messages
- Live output from generation process

### 🎨 **Multi-threaded Operation**
- UI remains responsive during generation
- Background processing with progress updates
- Ability to stop generation in progress

## Getting Started

### Prerequisites
```bash
pip install tkinter pillow opencv-python numpy
```

### Launch Options

#### GUI Interface (Recommended)
```bash
python launcher.py --gui
# or simply
python launcher.py
# or directly
python ui.py
```

#### Command Line Interface
```bash
python launcher.py --cli
# or directly  
python main_cv2.py
```

## Using the GUI

### 1. **Terrain Configuration Tab**
Configure basic terrain parameters:
- **Number of Mountains**: 1-50 mountains
- **Map Size**: Choose from preset resolutions (256x256 to 4097x4097)
- **Noise Level**: Base terrain roughness (0.0-0.5)
- **Mountain Ranges**: Set min/max heights and widths
- **Terrain Features**: Enable/disable ridges, valleys, and fractal noise
- **Feature Strength**: Adjust ridge strength and valley depth

### 2. **Mountains Tab**
Manage custom mountain placement:
- **Toggle Custom Positions**: Use predefined positions vs random
- **Add Mountain**: Specify exact coordinates, height, and width
- **Edit/Remove**: Modify or delete existing mountains
- **Mountain List**: View all configured mountains

### 3. **Rivers Tab**
Configure water features:
- **Enable Rivers**: Toggle river generation
- **Add River**: Create new rivers with start/end positions
- **River Properties**: Set width, depth, and meandering
- **Edge Connections**: Connect rivers between map edges
- **Tributaries**: Add branching waterways (future feature)

### 4. **Roads Tab**
Design transportation networks:
- **Enable Roads**: Toggle road generation
- **Add Road**: Create new roads between map edges
- **Road Properties**: Configure width, cut depth, and smoothness
- **Advanced Options**: Banking, contour following
- **Road Networks**: Connect multiple roads

### 5. **Advanced Tab**
Fine-tune generation:
- **Random Seed**: Set for reproducible results
- **Export Settings**: Bit depth, file formats
- **Smoothing Options**: Gaussian, erosion simulation
- **Performance**: Kernel sizes and iterations

## Generation Process

1. **Configure**: Set up terrain parameters using the tabs
2. **Generate**: Click "Generate Map" to start
3. **Monitor**: Watch progress bar and messages
4. **Preview**: View the generated terrain in real-time
5. **Export**: Maps are automatically saved to the `maps/` folder

## Key Features Explained

### **Flat River Levels** 🌊
Rivers maintain consistent water elevation across the entire map for realistic appearance, eliminating the need for complex flow direction calculations.

### **Complex Mountain Types** 🏔️
- **Peaked**: Sharp mountain peaks with cliff faces
- **Ridge**: Elongated mountain ridges
- **Mesa**: Flat-topped plateaus with steep sides
- **Volcano**: Cone-shaped with crater
- **Asymmetric**: Varied slopes on different sides

### **Intelligent Road Placement** 🛣️
Roads can follow terrain contours or take direct routes, with proper banking and elevation smoothing for realistic appearance.

### **Multi-threaded Architecture** ⚡
The UI runs on a separate thread from generation, ensuring responsiveness and allowing real-time progress updates.

## File Outputs

Generated files are saved in the `maps/` folder:
- `ui_generated_terrain.tif`: Main terrain file
- Additional smoothed/filtered versions
- 16-bit TIFF format for maximum detail

## Configuration Management

### Save/Load Configurations
- **Save Config**: Export current settings to a Python file
- **Load Config**: Import settings from saved files
- **Preset Configurations**: Quick setup for common scenarios

### Real-time Updates
All UI changes immediately update the internal configuration, ready for the next generation cycle.

## Tips for Best Results

### 🎯 **Terrain Design**
- Start with fewer mountains (8-16) for faster generation
- Use varied mountain heights for natural appearance
- Enable ridges and valleys for complex geology

### 🌊 **River Placement**
- Connect different edges for long rivers
- Use higher meandering (0.6-0.9) for natural curves
- Increase width/depth if rivers aren't visible enough

### 🛣️ **Road Networks**
- Enable contour following for mountain roads
- Use banking for realistic elevated roads
- Adjust cut depth based on terrain complexity

### ⚡ **Performance**
- Larger map sizes (2049+) take significantly longer
- Reduce fractal octaves for faster generation
- Use lower resolution for testing, higher for final output

## Troubleshooting

### Common Issues

**UI Won't Start**
```bash
# Install required packages
pip install tkinter pillow opencv-python numpy
```

**Generation Errors**
- Check that all numeric inputs are valid
- Ensure position values are between 0.0 and 1.0
- Verify map size is a valid preset value

**Preview Not Updating**
- Allow generation to complete fully
- Check console for error messages
- Try generating a smaller map first

**Memory Issues**
- Reduce map size for large terrains
- Close other applications during generation
- Consider using 8-bit export instead of 16-bit

### Performance Optimization

**For Faster Generation:**
- Reduce number of mountains
- Lower map resolution
- Disable erosion simulation
- Reduce fractal octaves

**For Higher Quality:**
- Increase map resolution
- Add more mountains with varied types
- Enable all terrain features
- Use 16-bit export depth

## Advanced Usage

### Batch Generation
Use the CLI interface with scripting for batch operations:
```bash
python launcher.py --cli --config my_config.py
```

### Custom Scripts
Import the UI generator for custom workflows:
```python
from ui_generator import generate_with_ui_integration
# Your custom generation code here
```

## Contributing

The GUI is designed to be extensible. Key areas for enhancement:
- Additional mountain types
- River tributary systems  
- Road intersection networks
- Custom export formats
- Batch processing interface

---

**Happy Terrain Generating!** 🏔️🌊🛣️
