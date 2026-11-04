---
name: linear-dataset
description: Defines the fields used to manage battle power and other image features alongside DINOv3 image embeddings, providing guidelines to populate dataset values according to this skill.
---

# 1. Training Data
A CSV file containing embeddings, battle power, and other image metadata. The file name follows this format:
- embeddings_yyyymmdd_num.csv

# 2. Fields
Descriptions for the 7 feature fields. Since image embeddings are extracted and recorded by the model, these image characteristics are mainly entered manually. Where possible, they should be automatically populated by an LLM or VLM.

1. person_id
2. power
3. body_direction
4. take
5. distance
6. posture
7. gender

### 2-1. person_id
An ID representing an individual. The identifiers below are fixed, assuming the person identifier prefix is in the file name. IDs from 100 onwards are sequential numbers for arbitrary datasets, so values of 100 or greater are assigned by default.

| **person_id** | **Name** | **Identifier** |
| --------- | ----------- | --------- |
| 1 | me | hide |
| 2+ | Other | - |

### 2-2. power
Battle power. Enter an integer value from 1 to 99. This value is displayed in the app.

### 2-3. body_direction
The facing direction of the person in the photo. Judge by the direction the face is pointing relative to the viewer, not by the person's location. Only the following values are allowed. Enter only a single value.

- front
- left
- right
- back

### 2-4. take
Specifies whether the person in the photo is shown full-body or half-body. Basically, if about 80% or more of the human body is visible in the photo, set to `full`; otherwise, set to `half`. Allowed values:

- full
- half

### 2-5. distance
Specifies the perceived distance to the person in the photo. If the face or body occupies most of the photo, set to `near`. If the person accounts for roughly 20% and the rest is predominantly scenery/background, set to `far`. Otherwise, set to `medium`. Allowed values:

- near
- medium
- far

### 2-6. posture
Specifies the state or posture of the person in the photo. For sleeping or lying down, enter `lying`. For running, jumping, etc., enter `moving`. Allowed values:

- standing
- sitting
- lying
- moving

### 2-7. gender
Specifies the gender. Allowed values:

- male
- female
