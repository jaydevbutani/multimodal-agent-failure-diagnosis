---
name: Executive Warm Canvas
colors:
  surface: '#fef9f0'
  surface-dim: '#ded9d1'
  surface-bright: '#fef9f0'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f8f3ea'
  surface-container: '#f2ede4'
  surface-container-high: '#ece8df'
  surface-container-highest: '#e7e2d9'
  on-surface: '#1d1c16'
  on-surface-variant: '#4f4542'
  inverse-surface: '#32302a'
  inverse-on-surface: '#f5f0e7'
  outline: '#807471'
  outline-variant: '#d2c3bf'
  surface-tint: '#6c5b55'
  primary: '#120805'
  on-primary: '#ffffff'
  primary-container: '#2b1e1a'
  on-primary-container: '#98847e'
  inverse-primary: '#d8c2bb'
  secondary: '#a23e18'
  on-secondary: '#ffffff'
  secondary-container: '#fe8357'
  on-secondary-container: '#6f2000'
  tertiary: '#130800'
  on-tertiary: '#ffffff'
  tertiary-container: '#311d00'
  on-tertiary-container: '#b67d27'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#f5ddd7'
  primary-fixed-dim: '#d8c2bb'
  on-primary-fixed: '#251915'
  on-primary-fixed-variant: '#53433e'
  secondary-fixed: '#ffdbcf'
  secondary-fixed-dim: '#ffb59c'
  on-secondary-fixed: '#390c00'
  on-secondary-fixed-variant: '#822801'
  tertiary-fixed: '#ffddb5'
  tertiary-fixed-dim: '#fcba5f'
  on-tertiary-fixed: '#2a1800'
  on-tertiary-fixed-variant: '#643f00'
  background: '#fef9f0'
  on-background: '#1d1c16'
  surface-variant: '#e7e2d9'
typography:
  display-lg:
    fontFamily: Space Grotesk
    fontSize: 36px
    fontWeight: '600'
    lineHeight: 44px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Space Grotesk
    fontSize: 26px
    fontWeight: '600'
    lineHeight: 34px
    letterSpacing: -0.015em
  headline-md:
    fontFamily: Space Grotesk
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: 0.02em
  headline-sm:
    fontFamily: Plus Jakarta Sans
    fontSize: 15px
    fontWeight: '600'
    lineHeight: 20px
    letterSpacing: 0.01em
  body-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 22px
  body-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 19px
  body-sm:
    fontFamily: Plus Jakarta Sans
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.04em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.06em
  code-sm:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '400'
    lineHeight: 17px
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 1rem
  gutter-compact: 0.5rem
  margin: 1.25rem
  margin-panel: 0.75rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1.25rem
  space-xl: 1.75rem
  space-2xl: 2.5rem
---

## Brand & Style

This design system serves an enterprise-grade AI reliability and telemetry platform focused on multimodal agent failure diagnosis, execution graphs, and automated root-cause attribution. The target audience comprises AI architects, engineering executives, and reliability specialists who require intense data density without cognitive fatigue.

The style merges **warm editorial luxury** with **technical telemetry precision**. Rather than relying on standard cold blue or dark cybernetic tropes, the interface employs a warm ivory and cream workspace contrasted with deep espresso/mocha shell navigation (sidebar and top masthead). The emotional response should be authoritative, analytical, calm, and unmistakably premium—balancing complex visual computation (node execution graphs, timeline swimlanes, code-level stack attribution) with tactile clarity and surgical typography.

## Colors

The palette is anchored by a dual-tone environmental model:
- **Navigation & Structural Shell:** Deep espresso mocha (`#2B1E1A`) and dark roasted umber (`#1F1512`) wrap the top bar, left vertical rails, and floating contextual command menus. Text on these panels defaults to warm off-white (`#F8F4ED`) and muted sepia dust (`#A4978D`).
- **Workspace & Canvas:** Subtle ivory wash (`#F4EFE6`) and porcelain card containers (`#FAF7F2`) with crisp sepia borders (`#E2D7C8`). This eliminates glare while providing extreme definition for complex graphs.
- **Diagnostics & Status Accents:**
  - **Critical Failure / Error Attribution:** Deep terracotta vermilion (`#C85A32`).
  - **Warning / Degraded Agent Node:** Warm amber ochre (`#D99B43`).
  - **Healthy Execution / Validation:** Muted celadon sage (`#4A6B53`).
  - **Telemetric Highlight / Tool Calls:** Burnt sienna brown (`#824628`).
- **Authentication & Action State:** Primary actionable buttons and login triggers use the focused vermilion terracotta (`#C85A32`) or solid espresso (`#2B1E1A`), maintaining unified visual weight across the landing interface and internal diagnostics workspace.

## Typography

Typography balances technical rigor with modern editorial polish:
- **Headings (`Space Grotesk`):** Used for platform banners, panel labels, and major section designations. Rendered uppercase with generous tracking (`0.02em` to `0.05em`) in technical panels, and natural title case for analytical dashboards.
- **Data & Body (`Plus Jakarta Sans`):** Selected for high legibility at micro sizes, used in agent execution descriptions, user prompts, assistant summaries, and form fields.
- **Telemetry & Machine Output (`JetBrains Mono`):** Applied to telemetry readouts, token consumption meters, agent tool IDs (e.g. `DB_Tool`, `API_Auth`), stack traces, and status badges.

## Layout & Spacing

The platform uses an **integrated split-shell architecture**:
1. **Masthead:** Fixed 52px top bar in deep espresso (`#2B1E1A`) encompassing platform breadcrumbs, agent health state chips, and global environment actions.
2. **Left Primary Navigation:** Compact 64px to 200px fixed dark espresso sidebar spanning directly underneath or flush with the top masthead.
3. **Multi-Pane Analytical Canvas:** A three-zone responsive workspace:
   - **Main Telemetry/Graph Canvas (60% to 65% width):** Interactive node orchestration graph stacked atop temporal time-series swimlanes.
   - **Root-Cause & Attribution Inspector (35% to 40% width):** Vertical column with card modules for fault type, localized components, evidence chains, and code traces.
   - **Collapsible Diagnostic Copilot Drawer:** Docked bottom-right overlay or embedded inspector card.

Spacing favors analytical density (`0.75rem` component gaps) with clean margin boundaries (`1.25rem`) to maintain calm organization despite high information density.

## Elevation & Depth

Visual hierarchy is maintained through **layered tonal planes and low-contrast borders** rather than diffuse black drop shadows:
- **Base Level (Canvas):** Warm ivory wash (`#F4EFE6`).
- **Layer 1 (Cards & Data Tiles):** Crisp linen white (`#FAF7F2`) framed with a 1px solid hairline border in muted sepia sand (`#DFD4C5`).
- **Layer 2 (Fault Banners & Code Enclosures):** Tinted parchment fills (`#EFE8DC`) with subtle 1px border accents (`#DACDBB`).
- **Active / Alert Highlight:** Node halos and active error cards receive a directional 2px solid border in terracotta (`#C85A32`) with a soft atmospheric halo (`box-shadow: 0 4px 18px rgba(200, 90, 50, 0.16)`).
- **Floating Overlays & Modals:** Floating assistant windows and diagnostic tooltips feature a 1px mocha border (`#42322C`) with a crisp, high-density drop shadow (`box-shadow: 0 12px 32px rgba(43, 30, 26, 0.22)`).

## Shapes

The shape system adopts a **softened precision profile (`roundedness: 1`)**:
- Outer workspace container panels and analytical modules: `0.5rem` (`rounded-lg`).
- Internal nodes, code blocks, input elements, and buttons: `0.25rem` (`rounded`).
- Interactive execution graph nodes: circular (`9999px`) or pill tags (`9999px`) to distinguish fluid agent routing paths from structural data grids.
- Role selector cards and authentication panels: `0.375rem` to `0.5rem` with micro-borders.

## Components

### 1. Navigation Masthead & Sidebar
- **Background:** Deep espresso (`#2B1E1A`).
- **Items:** Monospace or clean geometric labels (`Space Grotesk`, 12px uppercase) with warm cream hover states.
- **Active State:** Mocha highlight container (`#3D2C27`) with a 2px left or bottom indicator strip in vermilion terracotta (`#C85A32`).

### 2. Failure & Root-Cause Cards
- **Container:** Warm linen (`#FAF7F2`) with 1px border (`#DFD4C5`).
- **Header:** Uppercase monospace eyebrow (`JetBrains Mono`, 11px) paired with bold failure taxonomy.
- **Alert Variations:** 
  - Fault type badges use warm sand/terracotta fills (`#F3DDD3`) with burnt rust text (`#973B1B`).
  - Warning badges use warm ochre tinted backgrounds (`#F6EDD9`) with dark honey text (`#8A5E17`).

### 3. Execution Graph Nodes
- **Standard Execution Node:** Cream interior (`#FAF7F2`), warm sepia stroke (`#C5B8A5`), dark espresso icon/label.
- **Failing Node:** Terracotta accent fill (`#F9EAE1`), terracotta ring (`#C85A32`), dark espresso alert tag.
- **Connection Lines:** Hairline bezier curves (`#C7BBAA`), turning dashed red/terracotta (`#C85A32`) along the identified failure propagation route.

### 4. Interactive Telemetry & Swimlane Timelines
- **Canvas:** Ivory gridlines on linen background.
- **Waveforms:** Gradient fill beneath line strokes shifting from burnt sienna to muted parchment.
- **Execution Spans:** Rounded timeline bars with color differentiation by sub-agent tool call.

### 5. Authentication & Quick-Access Cards
- **Card Shell:** Centered, structured white linen card on ivory field, 1px perimeter border (`#DFD4C5`).
- **Quick-Access Role Cards:** Side-by-side split cards with icon, role title, user handle (`JetBrains Mono`), and high-contrast action button.
- **Action Buttons:**
  - Primary / Failure Action: Terracotta vermilion (`#C85A32`) with crisp white typography. Hover: `#B14A24`.
  - Secondary / Evaluation Action: Warm linen button with 1px border (`#CFC3B0`) and espresso typography (`#2B1E1A`).
- **Form Inputs:** Flush sand-tinted inputs (`#EFEBE4`), dark espresso text, hairline focus ring (`#C85A32`).

### 6. Code & Trace Inspector
- **Container:** Recessed warm neutral (`#ECE5DA`) with copy-to-clipboard trigger top-right.
- **Typography:** `JetBrains Mono` 11px with syntax highlighting geared toward warm earth tones: chocolate brown keys, terracotta strings, and ochre functions.