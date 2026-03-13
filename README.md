<!-- # TTApp

# for analysis:
# assumptions: the white lines of the table are all mostly visible. for this, the camera needs to be at a height of approx. 150cm?
# a side angle of the table (perpendicular to the net) would be best
# only one ball in the frame
# players aren't switching sides
# legal serves
# limit glare on the table if possible
# for editing (and generally for table detection): after an allowed 20s setup period, the camera doesn't move. the table doesn't blend in with the background and when a point is not being played, nothing is in covering the table (ie the whole table is visible) -->

<!-- static camera after initial setup -->

<!-- need ffmpeg, only tested for win?  -->

<!-- side and 45 degree angles for editing, only side for analysis -->

<!-- explain limitations of the method, which was used for speed. doesn't catch net balls and tosses across table, leading to extra points etc -->
<!-- white balls, avoid wearing white if possible -->
<!-- winning bounce displayed -->

# TTApp

TTApp is a tool for analyzing and editing table tennis matches from video recordings. It is designed for use with a **static camera setup** and focuses on extracting structured information from rallies while also providing tools for preparing and cleaning footage for analysis.

The application assumes a controlled recording environment and works best when the table and ball are clearly visible throughout the video.

---

## Overview

TTApp provides two primary capabilities:

* **Analysis** – Detect and analyze table tennis rallies from video.
* **Editing** – Prepare recorded footage for analysis (table detection, trimming setup periods, etc.).

Because the system relies on visual detection of the table and ball, **camera placement and recording conditions are critical** for reliable results.

---

## Requirements

### Software

* **FFmpeg** (required for video processing)
  https://ffmpeg.org/download.html

* **FFmpeg preset documentation**
  https://trac.ffmpeg.org/wiki/Encode/H.264

FFmpeg must be available in your system PATH or accessible by the application.

---

python edit_video.py "path\to\video" --preset fast

## Recording Setup

TTApp assumes a consistent recording environment. Follow these guidelines when recording footage.

### Camera Position

* Camera height: **approximately 150 cm**
* The **white lines of the table must be mostly visible**
* The **entire table must remain within the frame**

### Camera Angles

Two camera angles are supported depending on the task:

| Task     | Supported Angles                      |
| -------- | ------------------------------------- |
| Analysis | Side angle (perpendicular to the net) |
| Editing  | Side angle or ~45° angle              |

**Side angle definition:** camera placed perpendicular to the net along the long side of the table.

---

## Recording Assumptions

For reliable analysis, recordings should follow these assumptions:

* **Static camera after setup**
* **Only one ball visible in the frame**
* **Players do not switch sides during the recording**
* **Serves are legal**
* **Minimal glare on the table surface**
* **Table contrasts clearly with the background**

---

## Setup Period

TTApp allows a **20 second setup window** at the beginning of the video.

During this time you may:

* Adjust the camera
* Center the table in the frame
* Ensure lighting is correct

After this period:

* The **camera must remain completely static**
* The **table position must not change**

---

## Table Visibility Requirements

For correct table detection:

* The **entire table must be visible** when a point is not being played
* **No objects or players should block the table** during idle periods
* The table should **not blend into the background**
* Table boundary lines should be **clearly distinguishable**

These conditions are required for reliable segmentation and detection.

---

## Ball Visibility Requirements

* Only **one ball** should appear in the frame
* The ball should **contrast with the background**
* Avoid lighting conditions that cause glare or motion blur

---


## Limitations

TTApp may fail or produce unreliable results when:

* The camera moves after the setup period
* Table surface is not visible
* Players or objects block the table
* Multiple balls appear in the frame
* The table color blends with the environment
* Excessive glare or reflections are present

---

## Future Improvements

Potential future features include:

* Support for additional camera angles
* Support for matches where players switch sides

---

## License

Add your project license here.
