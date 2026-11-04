---
name: linear-dataset
description: DINO3の画像のembeddingの値を取った後にと合わせて管理する戦闘力及びその他画像の特徴を示す項目を定義し、このskillに沿って値datasetの値を埋めてもらうためのスキル。
---

# 1. 学習活用データ
embedding及び戦闘力およびその他画像データのあるcsv。ファイル名はこの形式
- embeddings_yyyymmdd_num.csv

# 2. 項目
７つの項目内容の説明を記載。画像のembeddingはモデルから記録されるため主に画像の特徴を手動で入れている。可能であればLLM、VLMに自動で補って欲しい。

1. person_id
2. power
3. body_direction
4. take
5. distance
6. posture
7. gender

### 2-1. person_id
人を表すid。以下は固定でファイル名の先頭に人物識別値がついている想定。100以降はファイル名適当なデータセットの連番なので基本は100以降のperson_idを付与する

| **person_id** | **人物名**     | **人物識別子** |
| --------- | ----------- | --------- |
| 1         | me   | hide      |
| 2以降     | その他         | -  | 

### 2-2. power
戦闘力。1から99までの数値を記入。アプリにはこれを表示する。

### 2-3. body_direction
写っている人物写真の向きを記入。人物の位置ではなく見る人から顔が向いている方向で判断すること。記入値は下記のみ。値は一つのみ記入する。

- front
- left
- right
- back

### 2-4. take
写っている人物写真が全身か半身かを記入。基本的に8割くらい写真に人間の体が入っていればfullにし、それ以外はhalfにする。記入値は下記のみ

- full
- half

### 2-5. distance
写っている人物写真の距離感を記入。顔や体が写真のほとんどを占めている時はnear,人が2割くらいであとは大半が風景の時はfarにし、それ以外はmediumにする。記入値は下記のみ

- near
- medium
- far

### 2-6. posture
写っている人物写真の様子、状況を記入。寝ている、横たわっている様子はlying。走っている、ジャンプしているなどはmovingを記入する。記入値は下記のみ

- standing
- sitting
- lying
- moving 

### 2-7. gender
性別を記入。記入値は下記のみ

- male
- female