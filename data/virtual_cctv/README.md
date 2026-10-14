# Virtual multi-camera demo (four views cut from one recording)

**VIRTUAL MULTI-CAMERA DEMO - 4 crops of ONE real recording, not 4 physical cameras.**

Source: `samples/originals/WhatsApp_Video_2026-10-07_at_2.36.05_PM.mp4` (848x480, 29.96 fps, 21.63 s, sha1 `7871cc5791e4`); the editor end card from 21.63 s (logo on black) is left out.

| camera | crop [x0, y0, x1, y1] in source pixels |
|---|---|
| CAM01 Left wall | [0, 120, 480, 480] |
| CAM02 Centre pool | [184, 120, 664, 480] |
| CAM03 Far side | [368, 0, 848, 360] |
| CAM04 Elevated wide | [0, 0, 848, 480] |

What this demo can show: the real detector, tracker and behaviour rules running per camera, the multi-camera
index and query, Re-ID candidate ranking, and links between views that are certain because the crop geometry
is known exactly.

What it can NOT show: real multi-camera Re-ID (all views share lighting, clock and viewpoint), real camera
synchronisation, or accuracy (no human-annotated ground truth exists). Which side is the deep end is not
known from the footage; the user has to tell the system.

Rebuild: `python tools/make_virtual_cameras.py`
