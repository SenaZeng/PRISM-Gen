# Assets Directory

This directory contains visual assets for the PRISM-Gen Demo skill.

## Required Assets for ClawHub

### 1. Skill Icon
- **File**: `icon.png` or `icon.svg`
- **Size**: 512x512 pixels recommended
- **Format**: PNG or SVG
- **Content**: Representative icon for drug discovery/data analysis

### 2. Screenshots
- **File**: `screenshot1.png`, `screenshot2.png`, etc.
- **Size**: 1280x720 pixels or similar
- **Format**: PNG
- **Content**: 
  - Command line interface in action
  - Generated visualizations
  - Data analysis results

### 3. Banner Image (Optional)
- **File**: `banner.png`
- **Size**: 1200x300 pixels
- **Format**: PNG
- **Content**: Attractive banner for skill listing

## Current Assets

### Generated Screenshots
The following screenshots are automatically generated during testing:

1. **Distribution Plots**: `plots/step4a_admet_final_pIC50__distribution_*.png`
   - Shows pIC50 distribution analysis
   - Includes histogram, box plot, Q-Q plot, CDF

2. **Scatter Plots**: `plots/step4a_admet_final_scatter_pIC50__vs_QED__*.png`
   - Shows correlation between pIC50 and QED
   - Includes trend line and statistical analysis

### How to Create Screenshots

```bash
# Generate example screenshots
cd ~/.openclaw/workspace/skills/prism-gen-demo

# 1. Generate distribution plot
bash scripts/demo_plot_distribution.sh step4a_admet_final.csv pIC50

# 2. Generate scatter plot with trendline
bash scripts/demo_plot_scatter.sh step4a_admet_final.csv pIC50 QED --trendline --correlation

# 3. Copy to assets directory
cp plots/*.png assets/
```

## Design Guidelines

### Colors
- Primary: Blue (#4A90E2) - for data/science theme
- Secondary: Green (#50C878) - for success/positive results
- Accent: Orange (#FF6B35) - for warnings/attention
- Background: White or light gray

### Typography
- Headers: Bold, clear
- Code: Monospace font
- Body: Readable sans-serif

### Icons
- Use the 🧪 (test tube) emoji as base
- Consider molecular structures or data charts
- Keep it simple and recognizable

## Creating Custom Assets

### Using Python (Matplotlib)
```python
import matplotlib.pyplot as plt
import numpy as np

# Create simple icon
fig, ax = plt.subplots(figsize=(5, 5))
# Add your design here
plt.savefig('assets/icon.png', dpi=300, transparent=True)
```

### Using External Tools
- **Inkscape**: For vector graphics (SVG)
- **GIMP**: For raster graphics (PNG)
- **Canva**: For banners and promotional graphics

## Asset Checklist for Release

- [ ] Skill icon (512x512 PNG)
- [ ] At least 2 screenshots (1280x720 PNG)
- [ ] Optional banner (1200x300 PNG)
- [ ] All assets in correct format
- [ ] Assets reflect actual skill functionality
- [ ] No copyrighted material used

## Notes
- Keep file sizes reasonable (< 1MB per image)
- Use compression for PNG files
- Ensure good contrast for readability
- Test assets on different screen sizes