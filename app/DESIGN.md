# ClauseLens Document Review Desk: Design Specification

## Palette (6 Named Hex Colors)
- Canvas Background: `#fcfbfa` (warm off-white editorial paper)
- Card / Panel Surface: `#f5f2eb` (linen desk surface)
- Primary Text / Deep Slate: `#1f2937` (high-contrast neutral charcoal)
- Secondary Text / Slate Muted: `#596573` (neutral body secondary)
- Border & Separation: `#dcd7cc` (subtle physical divider border)
- Flagged Critical Stroke: `#a83232` (crimson warning mark, paired with text label)
- Flagged Caution Stroke: `#b87333` (warm copper caution mark)
- Highlight Pale Amber: `#fbe8b5` (marker highlight for dates & key terms)
- Highlight Pale Blue: `#dce8f5` (marker highlight for parties & governance)

## Typography (Google Fonts)
- Headings & Contract Prose: `Newsreader` (clean literary text serif)
- System Interface & Metadata: `Plus Jakarta Sans` (clear contemporary neutral sans-serif)
- Monospace / Offsets: `JetBrains Mono`

## Wireframe (Desktop Review Desk)
```
+--------------------------------------------------------------------------+
|  ClauseLens                                       Model: CUAD (US Only)  |
+--------------------------------------------------------------------------+
|  [Document: Lease Agreement] [Parties: Acme & Beta] [Tenure: 11 mo]      |
+----------------------------------------------------+---------------------+
|  Contract Reading Pane                             | Findings Column     |
|                                                    |                     |
|  1. PARTIES                                        | Worth checking (2)  |
|  This Agreement is made on 1st April 2024 by...    | - Auto renewal      |
|                                                    | - Foreign courts    |
|  2. TERM & RENT                                    |                     |
|  The term shall be for eleven (11) months.         | Important to read(1)|
|  Rent shall be paid monthly...                     | - Stamp duty ref    |
|                                                    |                     |
|  3. GOVERNING LAW                                  | Not checked yet (9) |
|  This agreement is governed by the laws of India.  | [collapsed list]    |
|                                                    |                     |
+----------------------------------------------------+---------------------+
```

## Five Guiding Principles
1. **Physical Review Desk Metaphor**: The contract is the primary visual centerpiece, not a secondary artifact tucked behind cards. Findings operate like marginal annotations written by a reviewer.
2. **Accessible Semantics Over Color Alone**: Every highlight, severity tag, and status badge carries an explicit text label and structural styling (double underlines, brackets, border styles), ensuring full accessibility for color-blind readers.
3. **Restraint and Quiet Confidence**: Zero neon gradients, glassmorphism, floating drop-shadows, or cartoon emojis. Space and crisp borders delineate visual hierarchy.
4. **Factual and Non-Alarmist Copy**: Findings avoid inflammatory or legally absolute claims (e.g. never "illegal" or "invalid"). They state what was detected, the specific sentence, and the practical reason to verify it.
5. **Zero Layout Shifts**: Heights and columns maintain consistent bounding boxes so users never experience jumpy scrolling while documents load.

## Forbidden Tells Review & Explicit Decisions
- **No eyebrow badges**: Replaced uppercase micro-caps with quiet serif section headings.
- **No middle-dot meta lines**: Replaced `・` delimiters with clean table rows or labeled key-value metadata tags.
- **No "WORD - fragment" labels**: Replaced robotic compound tags with standard sentence-case descriptors.
- **No button arrows**: Removed `->` and `→` from button labels; used unambiguous verbs like "Analyze document" and "Load sample".
- **No unearned sequence numbers**: Avoided `01 / 02 / 03` on non-sequential finding cards.
- **No italic/accented single words in headlines**: Headlines are rendered in uniform neutral weights.
- **No big-number stat blocks**: Avoided generic grid dashboards with large numbers and small labels.
- **No card hover animations**: Findings remain stationary with crisp border changes on focus/active state.

