# 📁 Assets Folder

This folder contains all static assets for the **Mathematics for Data Science & Machine Learning** project.

---

## 📂 Folder Structure

```
assets/
│
├── images/              # Images and graphics
│   ├── logo.png         # Project logo
│   ├── favicon.ico      # Website favicon
│   ├── hero-bg.svg      # Hero section background
│   └── screenshots/     # Project screenshots
│       ├── desktop.png
│       ├── mobile.png
│       └── dark-mode.png
│
├── fonts/               # Custom fonts (if needed)
│   └── README.md        # Font licensing info
│
├── icons/               # SVG icons and graphics
│   ├── linear-algebra.svg
│   ├── calculus.svg
│   ├── neural-net.svg
│   └── optimization.svg
│
├── styles/              # Additional stylesheets (optional)
│   ├── print.css        # Print-specific styles
│   └── themes.css       # Additional theme variations
│
├── scripts/             # Additional JavaScript files (optional)
│   ├── analytics.js     # Analytics tracking
│   └── utils.js         # Utility functions
│
└── data/                # JSON data files (optional)
    ├── formulas.json    # Structured formula data
    └── categories.json  # Category metadata
```

---

## 🖼️ Images Folder (`images/`)

### Recommended Images:

#### 1. **Logo** (`logo.png`)
- **Size:** 512x512px or 1024x1024px
- **Format:** PNG with transparency
- **Usage:** Header, social media sharing

#### 2. **Favicon** (`favicon.ico`)
- **Size:** 16x16px, 32x32px, 48x48px (multi-size ICO)
- **Usage:** Browser tab icon
- **Alternative:** Use `favicon.png` (32x32px)

#### 3. **Open Graph Image** (`og-image.png`)
- **Size:** 1200x630px
- **Format:** PNG or JPG
- **Usage:** Social media previews (Twitter, Facebook, LinkedIn)

#### 4. **Hero Background** (`hero-bg.svg` or `hero-bg.png`)
- **Type:** Gradient patterns, geometric shapes
- **Usage:** Hero section background enhancement

#### 5. **Screenshots**
- Desktop view (1920x1080px)
- Mobile view (375x812px)
- Dark mode examples
- **Usage:** README.md, documentation

---

## 🎨 Icons Folder (`icons/`)

### Recommended Icon Set:

Create or download SVG icons for:

- 📊 Linear Algebra
- 📐 Calculus
- 🎲 Probability
- 📈 Statistics
- 📉 Regression
- 🧠 Neural Networks
- ⚡ Optimization
- ✅ Metrics
- 🎯 Clustering
- 🔥 Deep Learning

### Recommended Sources:
- [Heroicons](https://heroicons.com/) - Free MIT licensed icons
- [Feather Icons](https://feathericons.com/) - Minimalist icons
- [Font Awesome](https://fontawesome.com/) - Comprehensive icon library

---

## 🔤 Fonts Folder (`fonts/`)

### When to Add Custom Fonts:

Only add custom fonts if:
- You need specific mathematical symbols not in standard fonts
- You want a unique brand font
- You need offline support without Google Fonts

**Note:** Current project uses Google Fonts (Roboto) via CDN, which is faster and doesn't require local files.

### If adding fonts:

```css
@font-face {
  font-family: 'CustomFont';
  src: url('../fonts/CustomFont-Regular.woff2') format('woff2');
  font-weight: 400;
  font-display: swap;
}
```

---

## 📊 Data Folder (`data/`)

### Example: `formulas.json`

Structure formula data for dynamic loading:

```json
{
  "linearAlgebra": {
    "title": "Linear Algebra",
    "formulas": [
      {
        "name": "Dot Product",
        "latex": "\\mathbf{a}\\cdot\\mathbf{b}=\\sum_{i=1}^n a_i b_i",
        "description": "The dot product of two vectors"
      }
    ]
  }
}
```

### Example: `categories.json`

```json
{
  "categories": [
    {
      "id": "linear-algebra",
      "name": "Linear Algebra",
      "icon": "📊",
      "color": "#667eea",
      "order": 1
    }
  ]
}
```

---

## 📜 Scripts Folder (`scripts/`)

### Optional Enhancement Scripts:

#### 1. **Analytics** (`analytics.js`)

```javascript
// Google Analytics or custom tracking
function trackPageView() {
  // Implementation
}
```

#### 2. **Utils** (`utils.js`)

```javascript
// Utility functions
export function copyToClipboard(text) {
  // Implementation
}
```

#### 3. **Search** (`search.js`)

```javascript
// Formula search functionality
function searchFormulas(query) {
  // Implementation
}
```

---

## 🎨 Styles Folder (`styles/`)

### Optional Additional Stylesheets:

#### 1. **Print Styles** (`print.css`)

```css
@media print {
  .nav-wrapper,
  .back-to-top,
  .dark-mode-toggle {
    display: none;
  }
  
  body {
    background: white;
    color: black;
  }
}
```

#### 2. **Theme Variations** (`themes.css`)

```css
/* Alternative color schemes */
.theme-blue {
  --primary: #0066cc;
  --secondary: #004d99;
}
```

---

## 📝 Current Usage in Project

Currently, the main `index.html` file is **self-contained** and doesn't require assets folder. However, you can enhance it by:

### Adding to HTML `<head>`:

```html
<!-- Favicon -->
<link rel="icon" type="image/png" href="assets/images/favicon.png">

<!-- Open Graph Meta Tags -->
<meta property="og:image" content="assets/images/og-image.png">
<meta property="og:title" content="Mathematics for DS & ML">
<meta property="og:description" content="Comprehensive math formulas reference">

<!-- Additional Stylesheet -->
<link rel="stylesheet" href="assets/styles/print.css" media="print">
```

---

## 🚀 Quick Start

### **Option 1: No Assets Needed (Current)**
The project works perfectly without any assets folder. Everything is embedded or loaded from CDN.

### **Option 2: Add Assets for Enhancement**

1. **Create the folder structure:**
   ```bash
   mkdir -p assets/{images,icons,fonts,styles,scripts,data}
   ```

2. **Add your files** according to the structure above

3. **Update `index.html`** with asset references

4. **Optimize images** before adding:
   - Use WebP format for better compression
   - Compress PNG/JPG files
   - Use SVG for icons and logos

---

## 📦 Asset Optimization Tips

### **Images:**
- **Compress:** Use TinyPNG, ImageOptim, or Squoosh
- **Format:** WebP > PNG > JPG
- **Lazy loading:** Add `loading="lazy"` to `<img>` tags
- **Responsive:** Use `srcset` for multiple sizes

### **SVG:**
- **Optimize:** Use SVGO or SVGOMG
- **Inline:** Small SVGs can be embedded in HTML
- **Icon sets:** Use SVG sprites for multiple icons

### **Fonts:**
- **Subset:** Only include characters you need
- **Format:** WOFF2 is best, with WOFF fallback
- **Preload:** Critical fonts for faster loading

---

## 🔒 License Notes

When adding assets:

✅ Use license-compatible images (CC0, MIT, Apache)  
✅ Credit photographers/designers if required  
✅ Don't include copyrighted material  
✅ Document sources in this README

---

## 📚 Recommended Resources

### **Free Image Sources:**
- [Unsplash](https://unsplash.com/) - High-quality photos
- [Pexels](https://www.pexels.com/) - Free stock photos
- [unDraw](https://undraw.co/) - SVG illustrations

### **Icon Libraries:**
- [Heroicons](https://heroicons.com/)
- [Feather Icons](https://feathericons.com/)
- [Lucide](https://lucide.dev/)

### **Optimization Tools:**
- [TinyPNG](https://tinypng.com/) - Image compression
- [SVGOMG](https://jakearchibald.github.io/svgomg/) - SVG optimization
- [Squoosh](https://squoosh.app/) - Image converter

---

## 📧 Questions?

If you have questions about assets or need help adding specific files, please open an issue in the repository.

---

**Note:** The current project is fully functional without any assets folder. This guide is for enhancement and future expansion.
