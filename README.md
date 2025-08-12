# 🏔️ Heightmap Generator

A sophisticated terrain heightmap generator with multiple generation methods and a user-friendly interface.

## 📁 Project Structure

```
hightMapGen/
├── src/                    # Core source code
│   ├── main_cv2.py        # Main terrain generation engine
│   └── ui_generator.py    # UI-integrated generator
├── ui/                    # User interface files
│   ├── ui.py             # Main tkinter UI
│   └── view_terrain.py   # Terrain viewer
├── config/               # Configuration files
│   └── config.py        # Settings and parameters
├── tests/                # Test files
│   ├── test_*.py        # Various test scripts
├── launchers/            # Quick launcher scripts
│   ├── quick_ui_launcher.py
│   ├── command_line_generator.py
│   └── run_tests.py
├── output/               # Generated content
│   ├── maps/            # Generated heightmaps (.tif)
│   └── images/          # Preview images (.png)
├── docs/                 # Documentation
│   └── README_*.md      # Various documentation files
└── run_ui.py            # Main application launcher
```

## 🚀 Quick Start

### Option 1: Run the UI (Recommended)
```bash
python run_ui.py
```

### Option 2: Use the enhanced launcher
```bash
python launchers/quick_ui_launcher.py
```

### Option 3: Command line generation
```bash
python launchers/command_line_generator.py
```

## 🛠️ Features

### Terrain Generation
- **Multiple Mountain Types**: Varied, Peaked, Ridge, Mesa, Volcano, Asymmetric
- **Advanced Algorithms**: Geological-inspired terrain generation
- **Custom Parameters**: Configurable terrain properties
- **High Resolution**: Support for large heightmaps

### User Interface
- **Modern UI**: Clean tkinter interface with multiple tabs
- **Real-time Preview**: Grayscale and color preview modes
- **Mountain Management**: Add, edit, remove custom mountains
- **River & Road Systems**: Planned features for complex terrain

### Export Options
- **Multiple Formats**: TIFF, PNG support
- **High Quality**: 16-bit grayscale export
- **Batch Processing**: Generate multiple variations

## 📦 Dependencies

```bash
pip install numpy opencv-python Pillow tkinter
```

## 🧪 Testing

Run all tests:
```bash
python launchers/run_tests.py
```

Run specific test:
```bash
python tests/test_mountain_types.py
```

## ⚙️ Configuration

Edit `config/config.py` to customize:
- Terrain parameters
- Mountain types
- Export settings
- UI preferences

## 🎯 Mountain Types

1. **Varied** - Mixed terrain with diverse features
2. **Peaked** - Sharp, pointed mountain peaks
3. **Ridge** - Long mountain ridges and valleys
4. **Mesa** - Flat-topped mountains with steep sides
5. **Volcano** - Cone-shaped volcanic mountains
6. **Asymmetric** - Irregular, non-symmetrical peaks

## 📝 Usage Examples

### Basic Terrain Generation
1. Launch the UI: `python run_ui.py`
2. Go to the "Terrain" tab
3. Select mountain type and parameters
4. Click "Generate Heightmap"
5. Preview and export your terrain

### Custom Mountain Configuration
1. Edit mountains in the "Mountains" tab
2. Set custom positions and properties
3. Use "Custom" mountain type in terrain generation

## 🤝 Contributing

1. Place new features in appropriate folders (`src/`, `ui/`, etc.)
2. Add tests in the `tests/` folder
3. Update configuration in `config/config.py`
4. Create launchers for new tools in `launchers/`

## 📄 License

This project is open source. See individual files for specific license information.
