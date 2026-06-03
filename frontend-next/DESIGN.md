# DocQuery Design System

> Monochrome "precision instrument" aesthetic. Black (`#0A0A0A`) is the only accent. Trust UI is the differentiator. Generated from the codebase after the `emil-design-eng` + `impeccable` polish passes.

---

## Motion system

### Easing tokens (globals.css)

| Token | Curve | Use |
| --- | --- | --- |
| `--ease-out` | `cubic-bezier(0.23, 1, 0.32, 1)` | Enter / exit, button press, popovers |
| `--ease-in-out` | `cubic-bezier(0.77, 0, 0.175, 1)` | On-screen movement |
| `--ease-drawer` | `cubic-bezier(0.32, 0.72, 0, 1)` | Sheet / drawer slide |
| `--ease-spring` | `cubic-bezier(0.34, 1.56, 0.64, 1)` | Drag / gesture surfaces only |

### Duration tokens

| Token | Value | Use |
| --- | --- | --- |
| `--dur-fast` | `120ms` | Buttons, chips, icon hovers |
| `--dur-base` | `200ms` | Tooltips, small popovers, dialogs |
| `--dur-slow` | `300ms` | Ceiling for UI animations |

### Pressable elements

All interactive pressables use **`active:scale-[0.97]`** — uniform system-wide. No `scale-[0.95]` or `scale-[0.98]`. Transitions are explicit (never `transition-all`):

```
transition-[transform,background-color,border-color,color,box-shadow]
duration-[120ms] ease-[cubic-bezier(0.23,1,0.32,1)]
```

### Popover / modal rule

- **Popovers** scale from their trigger side (`transformOrigin` set to the trigger edge, not `center`).
- **Modals / dialogs** stay `transformOrigin: center` — they appear centered in the viewport, not anchored to a trigger.
- **CommandPalette (⌘K)**: no enter/exit animation — keyboard shortcuts used 100+ times/day must not animate.

### Asymmetric timing

- Enter: 150–200ms with `--ease-out`
- Exit: faster than enter (100–150ms or smaller `y` offset)
- Drawers: `--ease-drawer` at 280ms enter / 150ms exit

### Reduced motion

All framer-motion components (`Dialog`, `Sheet`, `ThinkingStream`) use `useReducedMotion()`:
- Keep opacity transitions.
- Drop `transform` / `x` / `y` / `scale` motion.
- `globals.css` kills `animation-duration` for CSS keyframes under reduced-motion (shimmer, cursor-blink).

### Hover gating

All CSS hover rules in `globals.css` are gated behind `@media (hover: hover) and (pointer: fine)` to prevent false-positive activation on touch devices.

---

## Component conventions

- **No `transition-all` anywhere.** Always name the properties you animate.
- **No `ease-in` on UI.** `ease-in` delays the first frame (the moment users are watching most closely).
- **No `scale(0)` entry.** Minimum `scale(0.97)` for enters.
- **No spring for standard UI.** Spring reserved for drag / gesture / decorative mouse-tracking.
- **Stagger delays**: 50–60ms between items (not 80+ ms, which makes the UI feel slow).

---

## Absolute bans (impeccable)

- `border-left` / `border-right` accent stripe on cards or callouts — use full border or bg tint.
- `background-clip: text` gradient text — use a solid color.
- Glassmorphism on app surfaces — landing-only (`AuroraBackground`, `GlassCard`) and intentional.
- `border-radius > 16px` on cards — current tokens cap at `xl:12px` / `2xl:16px`. Buttons use `rounded-[10px]`.
- `border: 1px solid X` + `box-shadow ≥16px blur` on the same element (ghost-card). `--shadow-sm` (2px) and `--shadow-md` (8px) stay within limit.

---

## Copy rules

- Button labels: **verb + object** ("Send message", "Stop generating", "Upload Documents").
- No em dashes in user-facing copy. Use periods, colons, or commas.
- No marketing buzzwords (seamless, empower, transform, leverage, etc.).
- Empty states: state the situation, then the action ("No documents yet. Upload one above.").
- Error messages: state what failed, then what to do ("Processing failed. Delete and re-upload.").
