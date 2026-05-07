# Hardware Setup

This document describes the physical components, wiring, and configuration required to reproduce the experimental setup.

---

## System Overview

The smart air pillow platform consists of five integrated subsystems:

1. **Layered mechanical structure** — Memory foam / silicone / air balloon / rigid plate
2. **Pneumatic actuation** — 10-channel solenoid valve array + air tank + pressure regulator
3. **Pressure sensing** — 32×16 piezoresistive force array (512 points)
4. **Physiological acquisition** — Garmin smartwatch + smartphone OCR pipeline
5. **Embedded control** — Raspberry Pi 4 (high-level) + Raspberry Pi Pico (low-level)

---

## Pillow Mechanical Layers (top → bottom)

| Layer | Material | Function |
|-------|----------|----------|
| 1 — Surface | Memory foam | Viscoelastic compliance, load redistribution |
| 2 — Sensing | Piezoresistive pressure mat | 512-point spatial force sensing |
| 3 — Silicone | Silicone sheet | Mechanical coupling between sensing and actuation |
| 4 — Actuation | TPU air balloons (×10) | Pneumatic chambers for localized deformation |
| 5 — Base | Rigid 3D-printed plate | Structural support and component alignment |

**Foam thickness trade-off:** Thinner layers preserve spatial resolution (sharper pressure gradients); thicker layers improve comfort but attenuate signal. The selected thickness balances these requirements.

---

## Pneumatic Actuation System

| Component | Specification |
|-----------|--------------|
| Air source | Compressed air tank |
| Pressure regulator | Adjustable, nominal operating range: low kPa |
| Solenoid valves | 24 V DC, N/C type, ×10 channels |
| Chamber material | Flexible TPU |
| Chamber layout | 2-row × 5-column array |
| Actuation mode | Discrete inflate / hold / deflate |

**Operating pressure:** Determined by head weight (approx. 4–5 kg) divided by effective chamber area. Low-pressure regime (~several kPa) to maintain compliance and safety.

**Inter-channel coupling note:** A shared air supply introduces transient pressure redistribution during simultaneous multi-channel actuation. Sequential actuation is used to mitigate this.

---

## Pressure Sensing

| Component | Specification |
|-----------|--------------|
| Sensor type | Piezoresistive flexible array |
| Grid resolution | 32 × 16 = 512 sensing points |
| Interface | Serial (UART / Bluetooth) to Raspberry Pi 4 |
| Sampling rate | ~5 Hz |
| Output | 2D pressure matrix (normalized 0–1) |

---

## Physiological Acquisition

| Component | Specification |
|-----------|--------------|
| Wearable | Garmin smartwatch (model: Garmin Forerunner/Venu series) |
| Signals | HR (bpm), RR (brpm), SpO₂ (%) |
| App | Garmin Connect companion app on Android/iOS |
| Screen mirroring | MSS (Python) on paired laptop |
| OCR engine | Tesseract v5 |
| Output latency | ~1 s (display refresh + OCR processing) |

### OCR Pipeline Steps

```
Smartwatch → Companion App (smartphone) → Screen Mirror (MSS) →
Frame Capture → ROI Extraction → Preprocessing (grayscale / threshold / denoise) →
Tesseract OCR → Value Parsing → JSON/CSV Logging
```

---

## Embedded Control Architecture

### High-Level Unit — Raspberry Pi 4

| Role | Details |
|------|---------|
| AI inference | TCN-Transformer model (PyTorch / ONNX) |
| Sensor fusion | Pressure + physiological signal integration |
| Decision-making | Apnea detection → region selection → command generation |
| Communication | Serial (UART) to Pico; Socket (TCP/IP) for physiological data |

### Low-Level Unit — Raspberry Pi Pico

| Role | Details |
|------|---------|
| Actuation control | GPIO → MOSFET driver → solenoid valve |
| Execution | Deterministic real-time valve switching |
| Interface | Serial commands from RPi 4 |

---

## Driver and Power Electronics

| Component | Specification |
|-----------|--------------|
| MOSFET type | Logic-level N-channel MOSFET |
| Valve voltage | 24 V DC |
| Control voltage | 3.3 V (Pico GPIO) |
| Protection | Flyback diode per channel |
| PCB | Custom designed, housed in 3D-printed enclosure |
| Power supply | 24 V regulated supply (valves) + 5 V (control electronics) |

**Safety features:**
- Flyback diodes suppress inductive switching transients
- Local decoupling capacitors on each channel
- Separate power domains for control and actuation

---

## System Timing

| Parameter | Value |
|-----------|-------|
| Pressure data acquisition rate | ~5 Hz |
| Physiological data update rate | ~1 Hz (OCR-limited) |
| AI inference cycle | ~30 s (60-s window, 50% overlap) |
| Actuation response time | 1–3 s |
| Inter-actuation delay | ~15 s (allows physical settling + sensor update) |
| Closed-loop update period | 3–5 s |

---

## Wiring Diagram

```
[Air Tank] → [Pressure Regulator] → [Solenoid Valves ×10]
                                            ↑
                              [PCB: MOSFET Driver ×10]
                                            ↑
                              [RPi Pico GPIO ×10]
                                            ↑ (UART Serial)
                              [RPi 4 — Control Unit]
                              ↑                    ↑
                 [Pressure Mat]           [Physiological Socket]
                 (Bluetooth/UART)         (TCP/IP from laptop OCR)
```

---

## Reproducing the Experimental Setup

1. Assemble the pillow layers in order (base → chambers → silicone → pressure mat → foam)
2. Connect each solenoid valve to its corresponding MOSFET driver channel
3. Set pressure regulator to operating pressure (~5–10 kPa; adjust per head weight)
4. Flash `scripts/control/pico_firmware.py` to the Raspberry Pi Pico
5. Start the pressure sensing acquisition service on RPi 4
6. Start the OCR physiological acquisition pipeline on the laptop
7. Launch the main closed-loop controller: `python scripts/control/closed_loop_main.py`

See [`HARDWARE_SETUP.md`](HARDWARE_SETUP.md) for photographs and PCB schematics (to be uploaded).
