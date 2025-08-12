# Height Map Generator - Enhanced UI Guide

## Overview
The Height Map Generator now features a modern, enhanced GUI with improved styling, larger preview, and split-screen layout for optimal workflow.

## New UI Features

### Split-Screen Layout
- **Left Panel (50% width)**: Configuration tabs and parameters
- **Right Panel (50% width)**: Large terrain preview and generation controls

### Enhanced Styling
- Modern color scheme with professional appearance
- Custom styled buttons, labels, and progress bars
- Improved typography using Segoe UI font family
- Color-coded message system for better feedback

### Large Preview Canvas
- **600x600 pixel preview** (increased from 400x400)
- High-quality terrain visualization with TERRAIN colormap
- Real-time updates during generation
- Centered display with proper padding

### Improved Message System
- **Timestamped messages** with format `[HH:MM:SS] message`
- **Color-coded feedback**:
  - 🟢 Green: Success/completion messages
  - 🔵 Blue: General information
  - 🟡 Yellow: Warnings
  - 🔴 Red: Errors
- Dark terminal-style background for better readability
- Auto-scroll to latest messages

### Professional Controls
- **Generate.TButton**: Large, prominent generation button
- **Action.TButton**: Smaller utility buttons (Save, Load, Stop)
- Enhanced progress bar with 20px thickness
- Real-time status updates with detailed progress information

## UI Sections

### Configuration Panel (Left)
1. **Header Section**: Clear "Configuration" title with separator
2. **Tabbed Interface**:
   - Terrain: Basic parameters with styled sliders and value displays
   - Mountains: Custom mountain placement and configuration
   - Rivers: River system parameters
   - Roads: Road network settings
   - Advanced: Export and advanced options

### Preview & Generation Panel (Right)
1. **Live Preview Section**:
   - Large 600x600 canvas with sunken border
   - High-quality terrain rendering
   - Real-time updates during generation

2. **Generation Controls**:
   - Progress bar with detailed status
   - Generate/Stop buttons with clear styling
   - Save/Load configuration buttons

3. **Generation Messages**:
   - Dark console-style display
   - Color-coded message types
   - Timestamps for all messages
   - Auto-scrolling message history

## Color Scheme
- **Primary Background**: `#f8f9fa` (Light gray)
- **Secondary Background**: `#e9ecef` (Slightly darker gray)
- **Accent Color**: `#007bff` (Professional blue)
- **Success**: `#28a745` (Green)
- **Warning**: `#ffc107` (Yellow)
- **Error**: `#dc3545` (Red)
- **Text Primary**: `#212529` (Dark gray)
- **Text Secondary**: `#6c757d` (Medium gray)

## Typography
- **Headers**: Segoe UI, 12pt, Bold
- **Subheaders**: Segoe UI, 10pt, Bold
- **Body Text**: Segoe UI, 9pt
- **Console**: Consolas, 9pt (monospace for messages)

## Window Layout
- **Default Size**: 1400x900 pixels
- **Minimum Size**: 1200x800 pixels
- **Resizable**: Yes, with proper scaling
- **Split Ratio**: 50/50 between configuration and preview

## Usage Tips

### Navigation
- Use the configuration tabs on the left to adjust terrain parameters
- Watch the live preview update in real-time as you make changes
- Monitor generation progress in the controls section
- Review detailed messages in the console-style message area

### Generation Workflow
1. Configure terrain parameters in the left panel tabs
2. Click "Generate Terrain" to start the process
3. Monitor progress in the progress bar and status label
4. Watch detailed messages in the message console
5. View the live preview update as generation proceeds
6. Save your configuration when satisfied with results

### Message Interpretation
- **Green messages**: Generation steps completed successfully
- **Blue messages**: General information and progress updates
- **Yellow messages**: Non-critical warnings or notifications
- **Red messages**: Errors that need attention

## Performance Notes
- The larger preview (600x600) provides better detail visualization
- Real-time updates are optimized to prevent UI freezing
- Background generation ensures responsive interface
- Message queue system prevents bottlenecks

## Customization
The UI styling can be further customized by modifying the `setup_styles()` method in `ui.py`. Key style elements include:
- Color definitions in `self.colors` dictionary
- TTK style configurations for widgets
- Font specifications for different text elements
- Layout parameters for spacing and sizing

## Troubleshooting
- **Slow preview updates**: Reduce map size in terrain configuration
- **UI freezing**: Ensure generation is running in background thread
- **Display issues**: Check OpenCV colormap compatibility
- **Layout problems**: Verify minimum window size requirements

## Future Enhancements
- Customizable color schemes
- Adjustable preview size
- Additional preview modes (wireframe, contour lines)
- Export preview as image
- Undo/redo functionality for configuration changes
