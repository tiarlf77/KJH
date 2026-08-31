---
name: Kinetic Enterprise AI
colors:
  surface: '#f9f9f9'
  surface-dim: '#dadada'
  surface-bright: '#f9f9f9'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f3f3f3'
  surface-container: '#eeeeee'
  surface-container-high: '#e8e8e8'
  surface-container-highest: '#e2e2e2'
  on-surface: '#1a1c1c'
  on-surface-variant: '#424656'
  inverse-surface: '#2f3131'
  inverse-on-surface: '#f1f1f1'
  outline: '#737687'
  outline-variant: '#c3c6d8'
  surface-tint: '#0052dd'
  primary: '#004ccd'
  on-primary: '#ffffff'
  primary-container: '#0f62fe'
  on-primary-container: '#f3f3ff'
  inverse-primary: '#b4c5ff'
  secondary: '#5d5f5f'
  on-secondary: '#ffffff'
  secondary-container: '#e2e2e2'
  on-secondary-container: '#636465'
  tertiary: '#304db9'
  on-tertiary: '#ffffff'
  tertiary-container: '#4b67d3'
  on-tertiary-container: '#f3f3ff'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#dbe1ff'
  primary-fixed-dim: '#b4c5ff'
  on-primary-fixed: '#00174c'
  on-primary-fixed-variant: '#003da9'
  secondary-fixed: '#e2e2e2'
  secondary-fixed-dim: '#c6c6c6'
  on-secondary-fixed: '#1a1c1c'
  on-secondary-fixed-variant: '#454747'
  tertiary-fixed: '#dde1ff'
  tertiary-fixed-dim: '#b8c4ff'
  on-tertiary-fixed: '#001453'
  on-tertiary-fixed-variant: '#1a3ca8'
  background: '#f9f9f9'
  on-background: '#1a1c1c'
  surface-variant: '#e2e2e2'
typography:
  display-lg:
    fontFamily: Hanken Grotesk
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Hanken Grotesk
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-lg-mobile:
    fontFamily: Hanken Grotesk
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  headline-md:
    fontFamily: Hanken Grotesk
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
  body-lg:
    fontFamily: Hanken Grotesk
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: Hanken Grotesk
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-sm:
    fontFamily: Hanken Grotesk
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-md:
    fontFamily: Hanken Grotesk
    fontSize: 14px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.05em
  code-sm:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  base: 8px
  container-max-width: 1280px
  gutter: 24px
  margin-desktop: 40px
  margin-mobile: 16px
  stack-xs: 4px
  stack-sm: 12px
  stack-md: 24px
  stack-lg: 48px
---

## Brand & Style
The design system is rooted in **Corporate Modernism**, prioritizing clarity, efficiency, and a sense of institutional stability. The target audience includes HR administrators and employees navigating complex benefits landscapes. The UI must evoke a "quietly capable" emotional response—reducing the cognitive load of dense information through ample whitespace and a systematic information hierarchy. 

The aesthetic avoids unnecessary decoration, focusing instead on high-quality typography and a disciplined grid. It borrows subtle cues from **Minimalism** to ensure the AI assistant feels like a native, integrated tool rather than a disruptive overlay. The goal is to transform "benefits administration" from a chore into a seamless, guided experience.

## Colors
The palette is dominated by "Blue 60" (#0F62FE) as the primary action color, signaling trust and technical precision. Surface colors are strictly neutral, utilizing a range of cool-toned greys to define containment without creating visual clutter.

- **Primary:** Used for main actions, active states, and key AI-driven highlights.
- **Secondary:** Used for subtle borders and secondary UI elements.
- **Tertiary:** A deep navy used for high-contrast text and grounding headers.
- **Neutral:** The foundation for all background surfaces, using subtle shifts in value to create hierarchy.
- **Semantic Badges:** "Eligible" uses a soft green tint; "Requires Review" uses a warm amber to indicate priority without causing alarm.

## Typography
This design system utilizes **Hanken Grotesk** across all roles to maintain a unified, contemporary, and sharp appearance. The typeface was chosen for its excellent legibility in data-heavy SaaS environments and its professional, precise character.

- **Headlines:** Set with tight tracking and semi-bold weights to command attention.
- **Body Text:** Uses "Body MD" for general interface copy and "Body LG" for AI chat responses to prioritize readability.
- **Labels:** Small caps and increased letter spacing are used for metadata and status indicators to distinguish them from actionable content.
- **Hierarchy:** Use weight over color to differentiate information levels, ensuring accessibility standards are met.

## Layout & Spacing
The layout follows a **12-column fluid grid** for desktop and a **4-column grid** for mobile. The spacing rhythm is based on an 8px base unit.

- **Chat Interface:** The main interaction hub uses a centered "focused" column model (max-width 800px) to prevent long line lengths in AI responses.
- **Dashboards:** Use a 24px gutter to separate data cards and action panels.
- **Margins:** Desktop views utilize generous 40px outer margins to create a premium, "breathable" feel. On mobile, margins shrink to 16px to maximize real estate for chat and document snippets.

## Elevation & Depth
The design system employs **Tonal Layering** supplemented by extremely subtle shadows. 

- **Surface Level 0:** The main background, set in a very light grey or white.
- **Surface Level 1:** Secondary panels or containers (e.g., chat sidebars), slightly darker/lighter than Level 0.
- **Cards & Bubbles:** Use a 1px border (#E0E0E0) instead of heavy shadows. AI-generated cards may use a faint Blue-tinted glow (4px blur, 5% opacity) to signify the "intelligence" layer.
- **Modals:** Use a medium-diffusion "Ambient Shadow" (16px blur, 12% opacity) to separate them from the work surface without creating harsh contrast.

## Shapes
The shape language is **Soft (0.25rem)**. This provides a balance between the rigid precision of enterprise software and the approachability required for a "helpful assistant."

- **Buttons & Inputs:** 4px (0.25rem) corner radius.
- **Cards & Chat Bubbles:** 8px (0.5rem) corner radius.
- **Status Badges:** Fully rounded (pill-shaped) to distinguish them from interactive buttons.
- **Document Icons:** 2px radius for a sharper, "paper-like" feel in previews.

## Components
- **Buttons:** Primary buttons are solid Blue 60 with white text. Secondary buttons are outlined. Ghost buttons are used for low-priority actions in document previews.
- **Chat Bubbles:** User bubbles are neutral grey; AI bubbles are white with a subtle 1px primary border. Text is left-aligned with a max-width of 85% to maintain a conversational flow.
- **Status Badges:** Use a "Light Fill" style—a pale background tint with a high-contrast text version of the same hue (e.g., Light Green background with Dark Green text for "Eligible").
- **Quick Action Cards:** Horizontal cards with an icon, title, and "chevron-right" affordance. These should change to a light blue tint on hover.
- **Document Snippets:** Compact cards that show a file-type icon, filename, and a "last modified" timestamp. A "Preview" button should appear on hover.
- **Input Fields:** Flat design with a 1px bottom border that transforms into a 2px primary blue border on focus.